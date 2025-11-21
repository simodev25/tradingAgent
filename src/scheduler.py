"""
Scheduler pour exécution automatique du trading agent aux quarts d'heure
(00, 15, 30, 45) avec gestion de timezone et écriture atomique des résultats.
"""

import asyncio
import json
import os
import signal
import sys
import tempfile
from datetime import datetime, time
from typing import Dict, List, Any

import pytz
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

# Import du trading agent
from agents import trading_agent
import importlib.util as _imp_util


class TradingScheduler:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()

        # Timezone depuis la config (ex: "Europe/Paris")
        timezone_str = self.config.get("scheduler", {}).get("timezone", "UTC")
        self.tz = pytz.timezone(timezone_str)

        # Scheduler avec timezone cohérente
        self.scheduler = AsyncIOScheduler(timezone=self.tz)

        self.running = False
        self.results: List[Dict[str, Any]] = []

        # Configuration du logging
        self._setup_logging()

        # Cap de prix pour filtrer certains symboles (ex: ≤ 1.5)
        gs = self.config.get("global_settings", {})
        self.price_cap_enabled: bool = bool(gs.get("price_cap_enabled", True))
        try:
            self.price_cap_max: float = float(gs.get("price_cap_max", 1.5))
        except Exception:
            self.price_cap_max = 1.5

        # Chargement dynamique de meta_api pour récupérer les quotes
        self._meta_api = self._load_meta_api()

    def _load_meta_api(self):
        try:
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp", "data_fetch", "meta_api.py"))
            spec = _imp_util.spec_from_file_location("meta_api_scheduler", path)
            if not spec or not spec.loader:
                return None
            mod = _imp_util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod
        except Exception as e:
            logger.warning(f"[Scheduler] meta_api load failed: {e}")
            return None

    def _get_mid_price(self, symbol: str) -> float | None:
        if not self._meta_api or not hasattr(self._meta_api, "get_current_price"):
            return None
        try:
            raw = self._meta_api.get_current_price(symbol)
            data_or_raw = None
            if isinstance(raw, str):
                try:
                    data_or_raw = json.loads(raw)
                except Exception:
                    data_or_raw = None
            data = data_or_raw.get("data") if isinstance(data_or_raw, dict) and "data" in data_or_raw else data_or_raw
            if isinstance(data, dict):
                bid = data.get("bid") or data.get("Bid")
                ask = data.get("ask") or data.get("Ask")
                px = data.get("price") or data.get("Price")
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
                    return (float(bid) + float(ask)) / 2.0
                if isinstance(px, (int, float)):
                    return float(px)
        except Exception as e:
            logger.debug(f"[Scheduler] mid price fetch failed for {symbol}: {e}")
        return None

    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier JSON"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Fichier de configuration {self.config_path} non trouvé")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON dans {self.config_path}: {e}")
            sys.exit(1)

    def _setup_logging(self):
        """Configure le système de logging"""
        log_config = self.config.get("logging", {})
        level = log_config.get("level", "INFO")
        log_file = log_config.get("file", "trading_agent.log")
        max_size = log_config.get("max_size", "10MB")
        backup_count = log_config.get("backup_count", 5)

        # Supprime les handlers existants
        logger.remove()

        # Handler console
        logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>"
        )

        # Handler fichier
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=max_size,
            retention=backup_count,
            compression="zip"
        )

    async def _process_symbol(self, symbol_config: Dict[str, Any]) -> Dict[str, Any]:
        """Traite un symbole individuel"""
        symbol = symbol_config["symbol"]
        logger.info(f"🔄 Traitement du symbole: {symbol}")

        try:
            # Pré-filtre prix (stabilité)
            if self.price_cap_enabled:
                mid = self._get_mid_price(symbol)
                if mid is not None and mid > self.price_cap_max:
                    logger.info(f"⏭️ Skip {symbol}: mid={mid} > cap={self.price_cap_max}")
                    return {
                        "symbol": symbol,
                        "timestamp": datetime.now(self.tz).isoformat(),
                        "status": "skipped",
                        "reason": f"price_cap mid={mid} > {self.price_cap_max}",
                    }

            # Préparation des paramètres
            params = {
                "symbol": symbol,
                "period": symbol_config.get("period", "5d"),
                "interval": symbol_config.get("interval", "15m"),
                "horizon": symbol_config.get("horizon", "scalping"),
                "risk_level": symbol_config.get("risk_level", "high"),
                "default_volume": symbol_config.get("default_volume", 0.01),
                "min_confidence": symbol_config.get("min_confidence", 40),
                "honor_hold": self.config["global_settings"].get("honor_hold", True),
                "dry_run": self.config["global_settings"].get("dry_run", True),
                "news_top_n": self.config["global_settings"].get("news_top_n", 10),
                "include_columns": self.config["global_settings"].get("include_columns", "Open,High,Low,Close,Volume"),
            }

            # Exécution du trading agent
            result = await trading_agent(params)

            logger.success(f"✅ Symbole {symbol} result ok ")
            # Ajout de métadonnées
            result["timestamp"] = datetime.now(self.tz).isoformat()
            result["status"] = "success"

            logger.success(f"✅ Symbole {symbol} traité avec succès")

            return result

        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now(self.tz).isoformat(),
                "status": "error",
                "error": str(e),
            }

    async def _run_trading_cycle(self):
        """Exécute un cycle complet de trading pour tous les symboles"""
        logger.info("🚀 Début du cycle de trading")
        start_time = datetime.now(self.tz)

        symbols = self.config["symbols"]
        max_concurrent = self.config["global_settings"].get("max_concurrent_symbols", 3)

        # Traitement par lots pour limiter la concurrence
        results: List[Dict[str, Any]] = []
        for i in range(0, len(symbols), max_concurrent):
            batch = symbols[i:i + max_concurrent]
            logger.info(f"📊 Traitement du lot {i // max_concurrent + 1}: {[s['symbol'] for s in batch]}")

            tasks = [self._process_symbol(symbol_config) for symbol_config in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"❌ Exception dans le lot: {result}")
                else:
                    results.append(result)

            # Petite pause entre lots
            if i + max_concurrent < len(symbols):
                await asyncio.sleep(2)

        # Sauvegarde des résultats
        self.results.extend(results)
        self._save_results()

        # Statistiques
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = len(results) - success_count
        duration = (datetime.now(self.tz) - start_time).total_seconds()

        logger.info(f"📈 Cycle terminé: {success_count} succès, {error_count} erreurs en {duration:.2f}s")

    def _save_results(self):
        """Sauvegarde les résultats dans un fichier JSON, de manière atomique"""
        results_file = f"results_{datetime.now(self.tz).strftime('%Y%m%d')}.json"
        try:
            dirname = os.path.dirname(results_file) or "."
            os.makedirs(dirname, exist_ok=True)

            with tempfile.NamedTemporaryFile('w', delete=False, dir=dirname, encoding='utf-8') as tf:
                json.dump(self.results, tf, indent=2, ensure_ascii=False)
                temp_name = tf.name

            os.replace(temp_name, results_file)
            logger.debug(f"💾 Résultats sauvegardés dans {results_file}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")

    def _is_trading_time(self) -> bool:
        """Vérifie si c'est le bon moment pour trader (gère fenêtres overnight)."""
        scheduler_config = self.config.get("scheduler", {})
        if not scheduler_config.get("enabled", True):
            return True

        start_time_str = scheduler_config.get("start_time", "09:00")
        end_time_str = scheduler_config.get("end_time", "17:00")

        try:
            now_t = datetime.now(self.tz).time()
            start_t = time.fromisoformat(start_time_str)
            end_t = time.fromisoformat(end_time_str)

            if start_t == end_t:
                # fenêtre 24/7 si mêmes heures
                return True
            if start_t < end_t:
                # fenêtre dans la même journée
                return start_t <= now_t <= end_t
            else:
                # fenêtre overnight (ex: 22:00 -> 06:00)
                return now_t >= start_t or now_t <= end_t
        except Exception as e:
            logger.error(f"❌ Erreur de vérification de l'heure: {e}")
            return True

    async def _scheduled_task(self):
        """Tâche programmée principale"""
        if not self._is_trading_time():
            logger.info("⏰ Hors des heures de trading, cycle ignoré")
            return

        try:
            await self._run_trading_cycle()
        except Exception as e:
            logger.error(f"❌ Erreur dans la tâche programmée: {e}")

    def start(self):
        """Démarre le scheduler avec alignement exact 00/15/30/45"""
        if self.running:
            logger.warning("⚠️ Le scheduler est déjà en cours d'exécution")
            return

        scheduler_config = self.config.get("scheduler", {})
        if not scheduler_config.get("enabled", True):
            logger.info("⏸️ Scheduler désactivé dans la configuration")
            return

        # (Re)configure explicitement la TZ au cas où
        self.scheduler.configure(timezone=self.tz)

        # Déclencheur aligné sur 0,15,30,45 (seconde = 0), configurable via config.json
        cron_expr = scheduler_config.get("cron_trigger", "0,15,30,45")
        cron_trigger = CronTrigger(
            minute=cron_expr,
            second=0,
            timezone=self.tz
        )

        # Job récurrent strictement aux quarts d'heure
        self.scheduler.add_job(
            self._scheduled_task,
            trigger=cron_trigger,
            id="trading_cycle",
            name="Trading Cycle (*/15 aligned)",
            replace_existing=True,
            max_instances=1,        # évite chevauchement si une run dure longtemps
            coalesce=True,          # regroupe les exécutions manquées en une seule
            misfire_grace_time=120  # tolérance (s) si le process se réveille en retard
        )

        # Option : exécuter une fois immédiatement au démarrage
        run_on_start = scheduler_config.get("run_on_start", True)
        if run_on_start:
            self.scheduler.add_job(
                self._scheduled_task,
                trigger=DateTrigger(run_date=datetime.now(self.tz)),
                id="trading_cycle_boot",
                name="Trading Cycle (boot)",
                replace_existing=True,
            )

        self.scheduler.start()
        self.running = True

        logger.info(f"🚀 Scheduler démarré - Exécution alignée aux {cron_expr}")
        logger.info(f"🌍 Timezone: {self.tz.zone}")
        logger.info(f"⏰ Heures de trading: {scheduler_config.get('start_time', '09:00')} - {scheduler_config.get('end_time', '17:00')}")
        if run_on_start:
            logger.info("▶️ Exécution immédiate au démarrage activée")

    def stop(self):
        """Arrête le scheduler"""
        if not self.running:
            logger.warning("⚠️ Le scheduler n'est pas en cours d'exécution")
            return

        self.scheduler.shutdown(wait=True)
        self.running = False
        logger.info("🛑 Scheduler arrêté")

    async def run_once(self):
        """Exécute un cycle unique (pour les tests)"""
        logger.info("🧪 Exécution d'un cycle unique")
        await self._run_trading_cycle()


def signal_handler(signum, frame):
    """Gestionnaire de signaux pour arrêt propre"""
    logger.info(f"📡 Signal {signum} reçu, arrêt en cours...")
    if 'scheduler' in globals():
        scheduler.stop()
    sys.exit(0)


async def main():
    """Fonction principale"""
    global scheduler

    # Configuration des signaux (Unix)
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except Exception:
        # Environnements qui ne supportent pas signal (ex: Windows/threads)
        pass

    # Création et démarrage du scheduler
    scheduler = TradingScheduler()

    try:
        # Vérification des arguments de ligne de commande
        if len(sys.argv) > 1 and sys.argv[1] == "--once":
            # Exécution unique
            await scheduler.run_once()
        else:
            # Mode scheduler
            scheduler.start()

            # Boucle infinie
            while True:
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("⌨️ Interruption clavier détectée")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
