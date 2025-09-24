# ---------------------------------------------------------------
# =============================
# execution_mcp.py
# =============================
import json, os, sys
from string import Template
from mcp.server.fastmcp import FastMCP
from loguru import logger

# --- Import meta_api (same path) ---
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_fetch/"))
)
try:
    import meta_api as meta_api  # type: ignore
except Exception as e:  # pragma: no cover
    meta_api = None  # type: ignore
    logger.error(f"[INIT] meta_api import failed (tools): {e}")

# --- Import core ---
from execution_core import (
    _ok, _err,
    build_and_execute_trade,
)

mcp = FastMCP("Trading Execution MCP Server", log_level="WARNING")
logger.remove()
logger.add(sys.stderr, level="WARNING")


# ============ TOOLS (signatures simples) ============
@mcp.tool()
def get_account_info(id:str) -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_account_info()  # type: ignore
    except Exception as e:
        return _err("account_info_failed", exc=str(e))


@mcp.tool()
def get_positions(id:str) -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_positions()  # type: ignore
    except Exception as e:
        return _err("positions_failed", exc=str(e))


@mcp.tool()
def get_orders(id:str) -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_orders()  # type: ignore
    except Exception as e:
        return _err("orders_failed", exc=str(e))


@mcp.tool()
def get_symbol_spec(symbol: str) -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_symbol_spec(symbol)  # type: ignore
    except Exception as e:
        return _err("symbol_spec_failed", symbol=symbol, exc=str(e))


@mcp.tool()
def get_current_price(symbol: str) -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_current_price(symbol)  # type: ignore
    except Exception as e:
        return _err("current_price_failed", symbol=symbol, exc=str(e))


@mcp.tool()
def execute_trade(
    symbol: str,
    action: str,
    entry: float,
    sl: float,
    tp: float,
    volume: float = 0.01,
    comment: str = "",
    client_id: str = "",
    dry_run: bool = True,
) -> str:
    """Exécute ou simule un trade (JSON)."""
    try:
        return build_and_execute_trade(symbol, action, entry, sl, tp, volume, comment, client_id, dry_run)
    except Exception as e:  # pragma: no cover
        logger.exception("[MCP:EXECUTE] unexpected")
        return _err("trade_failed", exc=str(e), symbol=symbol, action=action)


# =========================
#  PROMPT (inchangé)
# =========================
@mcp.prompt()
def execution_agent(
    context: str,
    default_volume: str = "0.01",
    min_confidence: str = "55",
    honor_hold: str = "True",
    dry_run: str = "True",
    HEDGE_MODE: str = "true",
    HEDGE_MAX_PAIRS: str = "2",
    HEDGE_MAX_NET_USD_MULT: str = "2.0",
    HEDGE_RISK_SPLIT: str = "0.6",
) -> str:
    ctx_str = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)

    tmpl = Template(r"""
Tu es un trader pro. Tu reçois un CONTEXTE JSON MINIMAL pour UN SEUL symbole.
Décide d’ENVOYER, AJUSTER ou ANNULER l’ordre.

PARAMS:
- default_volume = $DEFAULT_VOLUME
- min_confidence = $MIN_CONFIDENCE
- honor_hold = $HONOR_HOLD
- dry_run = $DRY_RUN

HEDGE FLAGS:
- HEDGE_MODE = $HEDGE_MODE
- HEDGE_MAX_PAIRS = $HEDGE_MAX_PAIRS
- HEDGE_MAX_NET_USD_MULT = $HEDGE_MAX_NET_USD_MULT
- HEDGE_RISK_SPLIT = $HEDGE_RISK_SPLIT
- REQUIRE_HTF_ON_EDGES = true

HEDGE_PAIRS (admissibles, sym sans suffixe .pro):
{
  "EURUSD": ["USDCHF","GBPUSD"],
  "GBPUSD": ["USDCHF","EURUSD"],
  "AUDUSD": ["USDCHF","NZDUSD"],
  "NZDUSD": ["USDCHF","AUDUSD"],
  "USDJPY": ["CHFJPY","EURJPY"],
  "USDCHF": ["EURUSD","GBPUSD","AUDUSD","NZDUSD"],
  "EURJPY": ["USDJPY"],
  "CHFJPY": ["USDJPY"]
}

CONTEXTE FOURNI (minifié):
$CTX

OUTILS DISPONIBLES (appelle-les si nécessaire):
- get_account_info(), get_positions(), get_orders(), get_symbol_spec(symbol), get_current_price(symbol)
- execute_trade(symbol, action, entry, sl, tp, volume, comment, client_id, dry_run=$DRY_RUN)

RÈGLES OBLIGATOIRES

1) Respect HOLD/Confiance/No-trade
- Si action == "HOLD" OU confidence < $MIN_CONFIDENCE OU regime == "no-trade" → ANNULER.

2) Edge vs SPREAD
- Obtiens bid/ask via get_current_price(symbol). Si indisponible → ANNULER (prudence).
- spread = ask - bid. entry_ref = ask pour BUY*, bid pour SELL* (mid si besoin).
- tp_dist = (tp - entry_ref) si BUY*, sinon (entry_ref - tp). Si <= 0 → ANNULER.
- Si volatility.band == "HIGH" → exiger `tp_dist ≥ 6 × spread`, sinon `tp_dist ≥ 4 × spread`.

3) RR minimal
- RR (si calculable) ≥ 1.2 sinon ANNULER.
- Tu peux **pousser le TP** si déplacer ≤ 5× spread permet d’atteindre RR=1.2 (ré-autorise ensuite).

4) Spécifications symbole
- Ajuste niveaux au tick/digits. Si incohérents (SL/TP côté opposé, stopsLevel), ANNULER.

5) Marges
- Vérifie account.freeMargin; si indisponible/insuffisante → ANNULER.

6) Gestion exposition (NON-HEDGE)
- Si HEDGE_MODE == "false" :
  - ANNULER si positions ouvertes ≥ 4.
  - ANNULER si positions sur le même symbole ≥ 1.
  - ANNULER si nombre de positions contenant "USD" ≥ 3.

7) Mode HEDGE (HEDGE_MODE == "true")
- Autoriser une position “opposée” seulement si:
  a) Le symbole courant a au moins UNE position ouverte dont le symbole est dans HEDGE_PAIRS[symbol] (ignore .pro).
  b) Caps respectés :
     - Total USD legs < 6.
     - |USD net notional| ≤ (HEDGE_MAX_NET_USD_MULT × equity).
     - Nombre de paires hedgées simultanées ≤ HEDGE_MAX_PAIRS.
  c) Si volatility.band == "HIGH" → exiger confluence HTF (en cas de doute ANNULER).
- Si hedge OK:
  - Réduire le volume de la jambe hedge à max(0.01, (1 - HEDGE_RISK_SPLIT) × default_volume).
  - Ajouter un champ "hedge": {"enabled": true, "paired_with": "<symbole existant>"}.
- Sinon, appliquer les règles NON-HEDGE ci-dessus.

8) Sécurité
- Si un outil échoue → ANNULER (raison explicite).
- Éviter d’ouvrir dans les 2 minutes suivant un changement de bougie H1/M30 si l’heure est disponible.

SORTIE JSON STRICTE (AUCUN TEXTE HORS JSON)
{
  "symbol": "<string>",
  "source": {"news_bias": "<neutral|positive|negative|null>", "confidence": <number>, "risk_level": "<low|medium|high|null>"},
  "portfolio": {"floating_pnl_total": <number|null>, "symbol_floating_pnl": <number|null>, "symbol_net_exposure": <number|null>, "symbol_direction": "<long|short|flat|null>"},
  "account_checks": {"equity": <number|null>, "free_margin": <number|null>, "margin_required_est": <number|null>, "margin_ok": <true|false>, "margin_reason": "<string>"},
  "tools_used": {"get_account_info": <true|false>, "get_positions": <true|false>, "get_orders": <true|false>, "get_symbol_spec": <true|false>, "get_current_price": <true|false>},
  "checks": {"values_ok": <true|false>, "direction_ok": <true|false>, "spec_ok": <true|false>, "rr": <number|null>, "tp_vs_spread_ratio": <number|null>, "hedge_ok": <true|false|null>},
  "order_policy": {"conflict_with_existing": "<none|duplicate|opposite|overexposed>", "action": "<proceed|adjust|cancel>", "adjust_reason": "<string|null>", "cancel_reason": "<string|null>"},
  "normalized_order": null,
  "decision": {"send_order": <true|false>, "reason": "<detail la en francais règle> ", "hedge": {"enabled": <true|false>, "paired_with": "<string|null>", "volume": <number|null>}},
  "execution_result": "<execute_trade result>"
}

IMPORTANT: Réponds UNIQUEMENT avec l’objet JSON. Pas de prose. Pas de code fences.
Taille maximale de sortie : 700 tokens.
""")

    prompt = tmpl.substitute(
        CTX=ctx_str,
        DEFAULT_VOLUME=default_volume,
        MIN_CONFIDENCE=min_confidence,
        HONOR_HOLD=honor_hold,
        DRY_RUN=dry_run,
        HEDGE_MODE=HEDGE_MODE,
        HEDGE_MAX_PAIRS=HEDGE_MAX_PAIRS,
        HEDGE_MAX_NET_USD_MULT=HEDGE_MAX_NET_USD_MULT,
        HEDGE_RISK_SPLIT=HEDGE_RISK_SPLIT,
    )
    return prompt


if __name__ == "__main__":
    mcp.run(transport="stdio")
