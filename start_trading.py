#!/usr/bin/env python3
"""
Script de démarrage pour le trading agent avec scheduler
"""
import sys
import os
import asyncio
from pathlib import Path

# Ajouter le répertoire src au path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from scheduler import TradingScheduler

def main():
    """Fonction principale"""
    print("🚀 Trading Agent Scheduler")
    print("=" * 50)
    
    # Vérification des arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python start_trading.py          # Mode scheduler (toutes les 15 minutes)")
            print("  python start_trading.py --once   # Exécution unique")
            print("  python start_trading.py --help   # Afficher cette aide")
            return
        elif sys.argv[1] == "--once":
            # Exécution unique
            print("🧪 Mode exécution unique")
            asyncio.run(run_once())
        else:
            print(f"❌ Argument inconnu: {sys.argv[1]}")
            print("Utilisez --help pour voir les options disponibles")
            return
    else:
        # Mode scheduler
        print("⏰ Mode scheduler - Exécution toutes les 15 minutes")
        asyncio.run(run_scheduler())

async def run_scheduler():
    """Lance le scheduler"""
    scheduler = TradingScheduler()
    try:
        scheduler.start()
        print("✅ Scheduler démarré avec succès")
        print("📊 Appuyez sur Ctrl+C pour arrêter")
        
        # Boucle infinie
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⌨️ Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
    finally:
        scheduler.stop()
        print("🛑 Scheduler arrêté")

async def run_once():
    """Exécute un cycle unique"""
    scheduler = TradingScheduler()
    try:
        await scheduler.run_once()
        print("✅ Exécution unique terminée")
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
