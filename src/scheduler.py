"""
Scheduler pour exécution automatique du trading agent toutes les 15 minutes
"""
import asyncio
import json
import os
import signal
import sys
from datetime import datetime, time
from typing import Dict, List, Any
import pytz
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Import du trading agent
from agents import trading_agent

class TradingScheduler:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.scheduler = AsyncIOScheduler()
        self.running = False
        self.results = []
        
        # Configuration du logging
        self._setup_logging()
        
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
        
        # Handler pour console
        logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Handler pour fichier
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
                "include_columns": self.config["global_settings"].get("include_columns", "Open,High,Low,Close,Volume")
            }
            
            # Exécution du trading agent
            result = await trading_agent(params)
     
            logger.success(f"✅ Symbole {symbol} result {result} ")
            # Ajout de métadonnées
            result["timestamp"] = datetime.now().isoformat()
            result["status"] = "success"
            
            logger.success(f"✅ Symbole {symbol} traité avec succès")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    async def _run_trading_cycle(self):
        """Exécute un cycle complet de trading pour tous les symboles"""
        logger.info("🚀 Début du cycle de trading")
        start_time = datetime.now()
        
        symbols = self.config["symbols"]
        max_concurrent = self.config["global_settings"].get("max_concurrent_symbols", 3)
        
        # Traitement par lots pour éviter la surcharge
        results = []
        for i in range(0, len(symbols), max_concurrent):
            batch = symbols[i:i + max_concurrent]
            logger.info(f"📊 Traitement du lot {i//max_concurrent + 1}: {[s['symbol'] for s in batch]}")
            
            # Exécution parallèle du lot
            tasks = [self._process_symbol(symbol_config) for symbol_config in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Traitement des résultats
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"❌ Exception dans le lot: {result}")
                else:
                    results.append(result)
            
            # Pause entre les lots
            if i + max_concurrent < len(symbols):
                await asyncio.sleep(2)
        
        # Sauvegarde des résultats
        self.results.extend(results)
        self._save_results()
        
        # Statistiques
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = len(results) - success_count
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"📈 Cycle terminé: {success_count} succès, {error_count} erreurs en {duration:.2f}s")
    
    def _save_results(self):
        """Sauvegarde les résultats dans un fichier JSON"""
        try:
            results_file = f"results_{datetime.now().strftime('%Y%m%d')}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.debug(f"💾 Résultats sauvegardés dans {results_file}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")
    
    def _is_trading_time(self) -> bool:
        """Vérifie si c'est le bon moment pour trader"""
        scheduler_config = self.config.get("scheduler", {})
        if not scheduler_config.get("enabled", True):
            return True
        
        # Récupération des heures de trading
        start_time_str = scheduler_config.get("start_time", "09:00")
        end_time_str = scheduler_config.get("end_time", "17:00")
        timezone_str = scheduler_config.get("timezone", "UTC")
        
        try:
            tz = pytz.timezone(timezone_str)
            now = datetime.now(tz).time()
            start_time = time.fromisoformat(start_time_str)
            end_time = time.fromisoformat(end_time_str)
            
            return start_time <= now <= end_time
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
        """Démarre le scheduler"""
        if self.running:
            logger.warning("⚠️ Le scheduler est déjà en cours d'exécution")
            return
        
        scheduler_config = self.config.get("scheduler", {})
        if not scheduler_config.get("enabled", True):
            logger.info("⏸️ Scheduler désactivé dans la configuration")
            return
        
        # Configuration du trigger
        interval_minutes = scheduler_config.get("interval_minutes", 15)
        
        # Ajout de la tâche programmée
        self.scheduler.add_job(
            self._scheduled_task,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id="trading_cycle",
            name="Trading Cycle",
            replace_existing=True,
            next_run_time=datetime.now()  # déclenche une exécution immédiate au démarrage
        )
        
        # Démarrage du scheduler
        self.scheduler.start()
        self.running = True
        
        logger.info(f"🚀 Scheduler démarré - Exécution toutes les {interval_minutes} minutes")
        logger.info(f"⏰ Heures de trading: {scheduler_config.get('start_time', '09:00')} - {scheduler_config.get('end_time', '17:00')}")
    
    def stop(self):
        """Arrête le scheduler"""
        if not self.running:
            logger.warning("⚠️ Le scheduler n'est pas en cours d'exécution")
            return
        
        self.scheduler.shutdown()
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
    
    # Configuration des signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
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
