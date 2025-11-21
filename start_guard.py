#!/usr/bin/env python3
"""
Script de démarrage pour le Positions Guard (scheduler 1 minute)
"""
import sys
import asyncio
from pathlib import Path

# Ajouter le répertoire src au path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from guard_scheduler import PositionsGuardScheduler


def main():
    print("🛡️ Positions Guard Scheduler")
    print("=" * 50)

    if len(sys.argv) > 1:
        if sys.argv[1] in ("--help", "-h"):
            print("Usage:")
            print("  python start_guard.py          # Mode scheduler (chaque minute)")
            print("  python start_guard.py --once   # Exécution unique")
            print("  python start_guard.py --help   # Afficher cette aide")
            return
        elif sys.argv[1] == "--once":
            print("🧪 Mode exécution unique (guard)")
            asyncio.run(run_once())
            return
        else:
            print(f"❌ Argument inconnu: {sys.argv[1]}")
            print("Utilisez --help pour voir les options disponibles")
            return

    print("⏰ Mode scheduler - Exécution chaque minute")
    asyncio.run(run_scheduler())


async def run_scheduler():
    scheduler = PositionsGuardScheduler()
    try:
        scheduler.start()
        print("✅ Guard Scheduler démarré")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n⌨️ Arrêt demandé par l'utilisateur")
    finally:
        scheduler.stop()
        print("🛑 Guard Scheduler arrêté")


async def run_once():
    scheduler = PositionsGuardScheduler()
    await scheduler._run_guard_cycle()


if __name__ == "__main__":
    main()

