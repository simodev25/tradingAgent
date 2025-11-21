"""
Scheduler (1 minute) pour l'agent Positions Guard MCP.
Exécute un audit et une éventuelle mise à jour des SL pour sécuriser des sorties positives.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from loguru import logger

from agents import run_positions_guard_mcp


class PositionsGuardScheduler:
    def __init__(self, config_path: str = "config.json") -> None:
        self.config_path = config_path
        self.config = self._load_config()

        tz_str = (
            self.config.get("positions_guard", {}).get("timezone")
            or self.config.get("scheduler", {}).get("timezone")
            or "UTC"
        )
        self.tz = pytz.timezone(tz_str)
        self.scheduler = AsyncIOScheduler(timezone=self.tz)
        self.running = False
        self.results: List[Dict[str, Any]] = []

        # Logging basique (réutilise logger global)
        logger.remove()
        logger.add(sys.stdout, level=self.config.get("logging", {}).get("level", "INFO"))

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[GuardScheduler] config load failed: {e}")
            return {}

    async def _run_guard_cycle(self):
        cfg = self.config.get("positions_guard", {})
        dry_run = bool(cfg.get("dry_run", True))
        strategy = cfg.get(
            "strategy",
            "Toujours sécuriser: SL > BE si en gain; trailing progressif",
        )
        rules = cfg.get("rules") or {}

        logger.info("🛡️ Positions Guard: cycle")
        start = datetime.now(self.tz)
        try:
            res = await run_positions_guard_mcp(
                strategy=strategy, rules_overrides=rules, dry_run=dry_run
            )
            out = {
                "timestamp": start.isoformat(),
                "dry_run": dry_run,
                "status": "success" if res.get("ok", True) else "error",
                "result": res,
            }
            self.results.append(out)
            logger.success("✅ Guard cycle terminé")
        except Exception as e:
            logger.error(f"❌ Guard cycle error: {e}")

    def start(self):
        if self.running:
            logger.warning("[GuardScheduler] déjà démarré")
            return

        guard_cfg = self.config.get("positions_guard", {})
        if not guard_cfg.get("enabled", True):
            logger.info("[GuardScheduler] désactivé par configuration")
            return

        self.scheduler.configure(timezone=self.tz)

        cron_expr = guard_cfg.get("cron_trigger", "*")  # toutes les minutes
        trigger = CronTrigger(minute=cron_expr, second=0, timezone=self.tz)

        self.scheduler.add_job(
            self._run_guard_cycle,
            trigger=trigger,
            id="positions_guard_cycle",
            name="Positions Guard (*/1)",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
        )

        if guard_cfg.get("run_on_start", True):
            self.scheduler.add_job(
                self._run_guard_cycle,
                trigger=DateTrigger(run_date=datetime.now(self.tz)),
                id="positions_guard_boot",
                name="Positions Guard (boot)",
                replace_existing=True,
            )

        self.scheduler.start()
        self.running = True
        logger.info("🚀 Guard Scheduler démarré (*/1min)")

    def stop(self):
        if not self.running:
            return
        self.scheduler.shutdown(wait=True)
        self.running = False
        logger.info("🛑 Guard Scheduler arrêté")


async def main():
    sched = PositionsGuardScheduler()
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--once":
            await sched._run_guard_cycle()
        else:
            sched.start()
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sched.stop()


if __name__ == "__main__":
    asyncio.run(main())

