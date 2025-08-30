# MCP Trading Agents – Analyse & Exécution

Système d’agents **MCP (Model Context Protocol)** pour analyser des actifs (news + technique) et **exécuter** des ordres via **MetaApi**, orchestré par un agent (LangGraph + Azure OpenAI).

> ⚠️ **Disclaimer** — Projet à but éducatif. **Pas de conseil financier.** Testez d’abord en **dry-run**, comprenez les risques (SL/TP, lot, volatilité).

---

## 🌐 Architecture

```mermaid
flowchart LR
    A[Input: symbol (ex: EURUSD, AAPL)] --> B(Orchestrateur agents.py)
    B --> C[News MCP\nanalyze_news_mcp.py]
    B --> D[Tech MCP\nanalyze_tec_mcp.py]
    C --> E{Synthèse news\nscore/tonalité}
    D --> F{Indicateurs + Plan brut\n(ATR/EMA/RSI/MACD/BB)}
    E --> B
    F --> B
    B --> G{Règles: min_confidence, honor_hold}
    G -->|OK| H[Execution MCP\nexecution_mcp.py]
    H --> I[(MetaApi)]
    G -->|Sinon| J[HOLD / Dry-run]
```

**Composants**

* **`analyze_news_mcp.py`** : récupère des news (ex. via `yfinance`), nettoie/summarise et fait un **sentiment** (ex: FinBERT). Retourne score + résumé.
* **`analyze_tec_mcp.py`** : calcule **EMA/RSI/MACD/ATR/Bollinger** et propose un **plan brut** (entry/SL/TP, sens, sizing minimal).
* **`execution_mcp.py`** : valide le contexte (risques, `min_confidence`, `honor_hold`) et **envoie l’ordre** via `MetaApi` (ou **dry-run**).
* **`meta_api.py`** : wrappers REST **MetaApi** (compte, positions, prix, bougies, ordres).
* **`agents.py`** : **orchestrateur** (LangGraph + AzureChatOpenAI) qui appelle les MCP puis l’exécution.

---

## 📁 Structure du projet

```
.
├── agents.py                 # Orchestrateur
├── analyze_news_mcp.py       # MCP News
├── analyze_tec_mcp.py        # MCP Analyse technique
├── execution_mcp.py          # MCP Exécution
├── meta_api.py               # Client MetaApi
└── README.md                 # Ce fichier
```

> 💡 Si vous ajoutez un module `yf_api.py`, placez‑le à la racine (cf. section Dépendances & intégrations).

---

## 🚀 Démarrage rapide

### 1) Prérequis

* **Python 3.10+** (3.11 recommandé)
* Accès Internet pour télécharger les modèles/paquets
* Un compte **MetaApi** (pour l’exécution réelle)
* **Azure OpenAI** ou compatibilité OpenAI pour l’orchestrateur

### 2) Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt  # ou voir l’exemple ci-dessous
```

**Exemple minimal de `requirements.txt` (à adapter à votre code)**

```
python-dotenv
requests
pydantic>=2
pandas
numpy
yfinance
beautifulsoup4
transformers
torch           # requis par transformers/FinBERT
mcp             # serveur MCP en Python
langgraph
langchain-core
langchain-openai
```

> ℹ️ Si vos indicateurs utilisent une lib externe (ex. `pandas_ta` ou `ta`), ajoutez‑la.

### 3) Configuration (.env)

Créez un fichier `.env` à la racine :

```dotenv
# ---- MetaApi ----
API_TOKEN=xxxxxxxxxxxxxxxx
ACCOUNT_ID=xxxxxxxxxxxxxxxx

# ---- Azure OpenAI (orchestrateur) ----
AZURE_OPENAI_API_KEY=xxxxxxxxxxxxxxxx
AZURE_OPENAI_ENDPOINT=https://votre-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini  # ou votre déploiement
AZURE_OPENAI_API_VERSION=2024-06-01

# ---- Options (facultatives) ----
HTTP_PROXY=
HTTPS_PROXY=
```

### 4) Lancer les MCP (en local)

Dans des terminaux séparés :

```bash
python analyze_news_mcp.py
python analyze_tec_mcp.py
python execution_mcp.py
```

> Les serveurs MCP parlent en **STDIO**. Assurez‑vous que l’orchestrateur pointe vers le bon chemin (voir **Conseils & pièges**).

### 5) Lancer l’orchestrateur

```bash
python agents.py
```

Selon votre implémentation, l’agent peut vous demander un **symbol** (ex: `EURUSD`, `AAPL`) et des **options** (dry-run, min\_confidence, etc.).

---

## ⚙️ Paramètres clés

* **`dry_run`** : `True` pour simuler, `False` pour exécuter réellement via MetaApi.
* **`honor_hold`** : si le Tech MCP retourne HOLD (risque/volatilité), forcer l’abstention.
* **`min_confidence`** : seuil minimal de confiance agrégée news + technique.
* **`default_volume`** : taille par défaut si le sizing dynamique est indisponible.

> 🔧 Assurez‑vous que ces valeurs sont de **vrais booléens/entiers** dans le code (pas des chaînes).

---

## 🔌 Dépendances & intégrations

### News

* **yfinance** pour récupérer des news basiques :

  * Vous pouvez créer un petit wrapper `yf_api.py` :

    ```python
    # yf_api.py
    import yfinance as yf
    def get_ticker_news(symbol: str):
        try:
            return yf.Ticker(symbol).news or []
        except Exception:
            return []
    ```
  * Le MCP **News** peut ensuite appeler `get_ticker_news(symbol)`.
* **NLP / Sentiment** : modèle type **FinBERT** via `transformers`. Préchargez le pipeline une seule fois au démarrage du MCP pour des perfs stables.

### Technique

* Indicateurs calculés sur OHLCV (EMA/RSI/MACD/ATR/Bollinger). Si l’ATR est **très faible** ou `NaN`, le plan devrait **retourner HOLD**.

### Exécution (MetaApi)

* `meta_api.py` expose : infos de compte, positions, prix, bougies, **envoi d’ordre** (market/SL/TP…).
* Vérifiez la présence de `API_TOKEN` et `ACCOUNT_ID` **avant** tout appel réseau.

---

## 🧪 Tests rapides (smoke tests)

1. **News MCP**

   ```bash
   python analyze_news_mcp.py  # lance le serveur MCP
   # puis via l’orchestrateur ou un client MCP, appelez analyze_news(symbol="AAPL")
   ```
2. **Tech MCP**

   ```bash
   python analyze_tec_mcp.py   # lance le serveur MCP
   # appelez compute_indicators(...) ou plan_raw(...) avec un petit OHLCV factice
   ```
3. **Execution MCP**

   ```bash
   python execution_mcp.py     # lance le serveur MCP
   # appelez trade_execute(..., dry_run=True) et vérifiez le payload renvoyé
   ```

---

## 🧭 Conseils & pièges fréquents

* **Chemins des MCP dans `agents.py`** : utilisez des chemins **absolus** basés sur `__file__` pour pointer vers `analyze_news_mcp.py`, `analyze_tec_mcp.py`, `execution_mcp.py`.
* **Imports `meta_api`** : si `meta_api.py` est à la racine, ajoutez `sys.path.append(os.path.dirname(__file__))` avant `import meta_api` dans vos MCP.
* **`yf_api` manquant** : ajoutez le wrapper minimal ci‑dessus si vous l’appelez.
* **Types natifs** : `"FALSE"` (chaîne) est considéré **True** en Python. Utilisez `False` (booléen) pour `dry_run` & co.
* **Timeouts & taille réponses** : fixez un `timeout` HTTP (ex. 10s) et refusez les réponses trop volumineuses avant parsing.
* **Arrondis & marchés** : respectez `tick_size`, et idéalement `stopLevel/freezeLevel` du broker pour SL/TP.

---

## 🧩 Extension du système

* **Ajouter un nouvel agent MCP** (ex. *Risk MCP*) :

  1. Créez `risk_mcp.py` (serveur MCP stdio)
  2. Déclarez ses tools (ex. `risk_check(context) -> verdict`)
  3. Connectez‑le dans `agents.py` (nouvelle étape du graphe)
  4. Mettez à jour l’agrégation de confiance / règles d’acceptation

* **Remplacer l’LLM** : adaptez le client (OpenAI, Azure, autre) dans l’orchestrateur ; gardez des prompts précis qui appellent **uniquement** les tools attendus.

---

## 🐞 Dépannage

* **401/403 MetaApi** : vérifiez `API_TOKEN`, `ACCOUNT_ID`, droits du compte et latence réseau.
* **Modèle introuvable (transformers)** : assurez‑vous que la machine a accès internet au 1er lancement (cache local ensuite).
* **Aucun trade envoyé** : `dry_run=True`, `honor_hold=True`, ou `min_confidence` non atteint. Inspectez les logs d’agrégation.
* **Unicode/HTML dans les news** : nettoyez/strippez avant le LLM pour réduire les hallucinations.

---

## 📚 Roadmap suggérée

* Pondération **récence/source** dans le scoring des news
* Gestion **multitimeframe** pour le Tech MCP
* **Backtesting** simple intégré (Pandas) pour valider le plan brut
* Vérifs broker : `stopLevel/freezeLevel`, taille minimale et pas de lot
* Journaling structuré des décisions (JSONL) + métriques (Prometheus)

---

## 📜 Licence

Proposé sous **MIT** par défaut (à adapter selon vos besoins).

---

## 🙌 Contributions

Issues et PR bienvenues. Merci de décrire votre environnement (OS, Python, versions libs) et de fournir un log minimal reproductible.
