# Trading Agent - Version Améliorée

## 🚀 Nouvelles fonctionnalités

### ✅ Liste de symboles configurables
- Support de plusieurs symboles simultanés (EURUSD, GBPUSD, USDJPY, BTCUSD, ETHUSD)
- Configuration personnalisable par symbole
- Exécution parallèle pour optimiser les performances

### ⏰ Scheduler automatique
- Exécution automatique toutes les 15 minutes
- Heures de trading configurables (9h-17h par défaut)
- Gestion des fuseaux horaires
- Mode exécution unique pour les tests

### 🛡️ Gestion d'erreurs robuste
- Validation des paramètres d'entrée
- Gestion des erreurs par module
- Logging détaillé avec rotation des fichiers
- Retry automatique en cas d'échec

### 📊 Monitoring et statistiques
- Dashboard en temps réel
- Statistiques de performance
- Historique des exécutions
- Alertes d'erreur

## 📁 Structure du projet

```
tradingAgent/
├── src/
│   ├── agents.py              # Agent principal amélioré
│   ├── scheduler.py           # Scheduler automatique
│   └── mcp/                   # Modules MCP existants
├── config.json                # Configuration des symboles
├── start_trading.py           # Script de démarrage
├── monitor.py                 # Script de monitoring
├── requirements.txt           # Dépendances mises à jour
└── README_IMPROVED.md         # Cette documentation
```

## 🚀 Installation et utilisation

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Configuration
Copiez `env.exemple` vers `.env` et configurez vos clés API :
```bash
cp env.exemple .env
# Éditez .env avec vos clés
```

### 3. Configuration des symboles
Éditez `config.json` pour personnaliser :
- Liste des symboles à trader
- Paramètres par symbole (volume, confiance, etc.)
- Heures de trading
- Paramètres globaux

### 4. Exécution

#### Mode Scheduler (recommandé)
```bash
python start_trading.py
```

#### Mode exécution unique (test)
```bash
python start_trading.py --once
```

#### Mode manuel (un seul symbole)
```bash
cd src
python agents.py
```

### 5. Monitoring
```bash
# Dashboard des dernières 24h
python monitor.py

# Dashboard des dernières 6h
python monitor.py --hours 6

# Résultats détaillés
python monitor.py --detailed

# Filtrer par symbole
python monitor.py --symbol EURUSD
```

## ⚙️ Configuration

### Fichier config.json
```json
{
  "symbols": [
    {
      "symbol": "EURUSD",
      "period": "5d",
      "interval": "15m",
      "horizon": "scalping",
      "risk_level": "high",
      "default_volume": 0.01,
      "min_confidence": 40
    }
  ],
  "global_settings": {
    "honor_hold": true,
    "dry_run": true,
    "max_concurrent_symbols": 3
  },
  "scheduler": {
    "enabled": true,
    "interval_minutes": 15,
    "start_time": "09:00",
    "end_time": "17:00"
  }
}
```

### Variables d'environnement (.env)
```bash
ENDPOINT_URL=your_azure_endpoint
DEPLOYMENT_NAME=your_deployment
AZURE_OPENAI_API_KEY=your_api_key
API_VERSION=2025-01-01-preview
ACCOUNT_ID=your_account_id
API_TOKEN=your_api_token
```

## 📊 Monitoring

### Fichiers de logs
- `trading_agent.log` : Log principal avec rotation automatique
- `results_YYYYMMDD.json` : Résultats quotidiens

### Métriques disponibles
- Taux de succès/échec
- Durée moyenne d'exécution
- Décisions par type (BUY/SELL/HOLD)
- Erreurs par catégorie
- Performance par symbole

## 🔧 Personnalisation

### Ajouter un nouveau symbole
1. Éditez `config.json`
2. Ajoutez une entrée dans la section `symbols`
3. Redémarrez le scheduler

### Modifier l'intervalle d'exécution
1. Éditez `config.json`
2. Modifiez `scheduler.interval_minutes`
3. Redémarrez le scheduler

### Changer les heures de trading
1. Éditez `config.json`
2. Modifiez `scheduler.start_time` et `scheduler.end_time`
3. Redémarrez le scheduler

## 🚨 Sécurité

### Mode dry-run par défaut
- Tous les ordres sont en mode simulation par défaut
- Changez `dry_run: false` dans la config pour l'exécution réelle
- Testez toujours en mode dry-run avant la production

### Gestion des erreurs
- Validation stricte des paramètres
- Gestion des erreurs par module
- Logs détaillés pour le debugging

## 📈 Performance

### Optimisations
- Exécution parallèle des symboles
- Traitement par lots pour éviter la surcharge
- Cache des résultats pour éviter les recalculs
- Logging asynchrone

### Surveillance
- Monitoring en temps réel
- Alertes automatiques en cas d'erreur
- Statistiques de performance

## 🐛 Dépannage

### Problèmes courants
1. **Erreur de configuration** : Vérifiez `config.json` et `.env`
2. **Erreur de connexion** : Vérifiez vos clés API
3. **Erreur de symbole** : Vérifiez que le symbole existe sur votre broker

### Logs
- Consultez `trading_agent.log` pour les détails
- Utilisez `python monitor.py --detailed` pour l'historique

## 🔄 Mise à jour

Pour mettre à jour le système :
1. Sauvegardez votre `config.json`
2. Mettez à jour le code
3. Restaurez votre configuration
4. Redémarrez le scheduler

## 📞 Support

En cas de problème :
1. Consultez les logs
2. Utilisez le mode `--once` pour tester
3. Vérifiez la configuration
4. Consultez la documentation des modules MCP
