#!/usr/bin/env python3
"""
Script de monitoring pour le trading agent
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import argparse

def load_results_files() -> List[Dict[str, Any]]:
    """Charge tous les fichiers de résultats"""
    results = []
    current_dir = Path(".")
    
    # Chercher les fichiers results_YYYYMMDD.json
    for file_path in current_dir.glob("results_*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {file_path}: {e}")
    
    return results

def filter_recent_results(results: List[Dict[str, Any]], hours: int = 24) -> List[Dict[str, Any]]:
    """Filtre les résultats des dernières N heures"""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    recent_results = []
    
    for result in results:
        try:
            timestamp_str = result.get("timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp >= cutoff_time:
                    recent_results.append(result)
        except Exception:
            # Si on ne peut pas parser la date, on inclut le résultat
            recent_results.append(result)
    
    return recent_results

def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcule les statistiques des résultats"""
    if not results:
        return {
            "total_executions": 0,
            "success_rate": 0.0,
            "error_rate": 0.0,
            "avg_duration": 0.0,
            "symbols_traded": [],
            "decisions": {"BUY": 0, "SELL": 0, "HOLD": 0},
            "errors_by_type": {}
        }
    
    total = len(results)
    successful = sum(1 for r in results if r.get("status") == "success")
    errors = total - successful
    
    # Calcul de la durée moyenne
    durations = [r.get("duration_seconds", 0) for r in results if r.get("duration_seconds")]
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    # Symboles tradés
    symbols = list(set(r.get("symbol", "UNKNOWN") for r in results))
    
    # Décisions
    decisions = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for result in results:
        if result.get("status") == "success":
            decision = result.get("technical_decision", {}).get("decision", {})
            action = decision.get("action", "HOLD")
            if action in decisions:
                decisions[action] += 1
    
    # Erreurs par type
    error_types = {}
    for result in results:
        if result.get("status") == "error":
            error_type = result.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    return {
        "total_executions": total,
        "successful_executions": successful,
        "failed_executions": errors,
        "success_rate": (successful / total * 100) if total > 0 else 0,
        "error_rate": (errors / total * 100) if total > 0 else 0,
        "avg_duration": avg_duration,
        "symbols_traded": symbols,
        "decisions": decisions,
        "errors_by_type": error_types
    }

def print_dashboard(stats: Dict[str, Any], results: List[Dict[str, Any]]):
    """Affiche le tableau de bord"""
    print("📊 TRADING AGENT DASHBOARD")
    print("=" * 50)
    
    print(f"📈 Exécutions totales: {stats['total_executions']}")
    print(f"✅ Succès: {stats['successful_executions']} ({stats['success_rate']:.1f}%)")
    print(f"❌ Erreurs: {stats['failed_executions']} ({stats['error_rate']:.1f}%)")
    print(f"⏱️  Durée moyenne: {stats['avg_duration']:.2f}s")
    
    print(f"\n🎯 Symboles tradés: {', '.join(stats['symbols_traded'])}")
    
    print(f"\n📋 Décisions:")
    for action, count in stats['decisions'].items():
        print(f"  {action}: {count}")
    
    if stats['errors_by_type']:
        print(f"\n🚨 Erreurs par type:")
        for error_type, count in stats['errors_by_type'].items():
            print(f"  {error_type}: {count}")
    
    # Dernières exécutions
    print(f"\n🕐 Dernières exécutions:")
    recent_results = sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)[:5]
    for result in recent_results:
        symbol = result.get("symbol", "UNKNOWN")
        status = result.get("status", "unknown")
        timestamp = result.get("timestamp", "unknown")
        duration = result.get("duration_seconds", 0)
        
        status_emoji = "✅" if status == "success" else "❌"
        print(f"  {status_emoji} {symbol} - {timestamp} ({duration:.1f}s)")

def print_detailed_results(results: List[Dict[str, Any]]):
    """Affiche les résultats détaillés"""
    print("\n📋 RÉSULTATS DÉTAILLÉS")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        symbol = result.get("symbol", "UNKNOWN")
        status = result.get("status", "unknown")
        timestamp = result.get("timestamp", "unknown")
        
        print(f"\n{i}. {symbol} - {timestamp}")
        print(f"   Status: {status}")
        
        if status == "success":
            decision = result.get("technical_decision", {}).get("decision", {})
            action = decision.get("action", "HOLD")
            confidence = decision.get("confidence", 0)
            print(f"   Action: {action} (confiance: {confidence}%)")
            
            news = result.get("news_sentiment", {})
            bias = news.get("global_bias", "neutral")
            print(f"   Sentiment news: {bias}")
        else:
            error_msg = result.get("error_message", "Erreur inconnue")
            print(f"   Erreur: {error_msg}")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Monitor du Trading Agent")
    parser.add_argument("--hours", type=int, default=24, help="Nombre d'heures à analyser (défaut: 24)")
    parser.add_argument("--detailed", action="store_true", help="Afficher les résultats détaillés")
    parser.add_argument("--symbol", type=str, help="Filtrer par symbole")
    
    args = parser.parse_args()
    
    print("🔍 Chargement des résultats...")
    all_results = load_results_files()
    
    if not all_results:
        print("❌ Aucun fichier de résultats trouvé")
        return
    
    # Filtrage par temps
    results = filter_recent_results(all_results, args.hours)
    
    if not results:
        print(f"❌ Aucun résultat trouvé pour les dernières {args.hours} heures")
        return
    
    # Filtrage par symbole si spécifié
    if args.symbol:
        results = [r for r in results if r.get("symbol", "").upper() == args.symbol.upper()]
        if not results:
            print(f"❌ Aucun résultat trouvé pour le symbole {args.symbol}")
            return
    
    # Calcul des statistiques
    stats = calculate_statistics(results)
    
    # Affichage
    print_dashboard(stats, results)
    
    if args.detailed:
        print_detailed_results(results)

if __name__ == "__main__":
    main()
