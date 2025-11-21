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

mcp = FastMCP("Trading Execution MCP Server", log_level="INFO")
logger.remove()
logger.add(sys.stderr, level="INFO")


# ============ TOOLS (signatures simples) ============
@mcp.tool()
def get_account_info(symbol:str) -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_account_info()  # type: ignore
    except Exception as e:
        return _err("account_info_failed", exc=str(e))


@mcp.tool()
def get_positions(symbol:str) -> str:
    logger.info(f"[MCP:Execution] get_positions {symbol}")
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_positions()  # type: ignore
    except Exception as e:
        return _err("positions_failed", exc=str(e))


@mcp.tool()
def get_orders(symbol:str) -> str:
    logger.info(f"[MCP:Execution] get_current_price {symbol}")
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
    logger.info(f"[MCP:Execution] get_current_price {symbol}")
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
    dry_run: bool = False,
) -> str:
    """Exécute ou simule un trade (JSON)."""
    logger.info(f"[MCP:Execution] execute_trade {symbol}:{dry_run}")
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
    dry_run: str = "False",
    HEDGE_MODE: str = "true",
    HEDGE_MAX_PAIRS: str = "2",
    HEDGE_MAX_NET_USD_MULT: str = "2.0",
    HEDGE_RISK_SPLIT: str = "0.6",
    REQUIRE_HTF_ON_EDGES: str = "true",
) -> str:
    ctx_str = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)

    tmpl = Template(r"""
Tu es trader pro. Contexte minimal (un symbole) ci-dessous.
Objectif : décider d’envoyer, ajuster ou annuler l’ordre via les tools MCP.

Paramètres clefs : default_volume=$DEFAULT_VOLUME, min_confidence=$MIN_CONFIDENCE, honor_hold=$HONOR_HOLD, dry_run=$DRY_RUN.
Hedge flags : mode=$HEDGE_MODE, max_pairs=$HEDGE_MAX_PAIRS, max_net_usd=$HEDGE_MAX_NET_USD_MULT, risk_split=$HEDGE_RISK_SPLIT, REQUIRE_HTF_ON_EDGES=$REQUIRE_HTF_ON_EDGES.
Table d'exemples de paires hedge (sans suffixe .pro) — non exhaustive, ne pas annuler si paire absente :
{"EURUSD":["USDCHF","GBPUSD"],"GBPUSD":["USDCHF","EURUSD"],"AUDUSD":["USDCHF","NZDUSD"],"NZDUSD":["USDCHF","AUDUSD"],"USDJPY":["CHFJPY","EURJPY"],"USDCHF":["EURUSD","GBPUSD","AUDUSD","NZDUSD"],"EURJPY":["USDJPY"],"CHFJPY":["USDJPY"],"EURGBP":["GBPUSD","EURUSD"],"EURCHF":["USDCHF","EURUSD"],"EURCAD":["USDCAD","EURUSD"],"EURNZD":["NZDUSD","EURUSD"],"EURAUD":["AUDUSD","EURUSD"]}.

Contexte (json minifié) :
$CTX

Tools utilisables : get_account_info, get_positions, get_orders, get_symbol_spec, get_current_price, execute_trade(symbol, action, entry, sl, tp, volume, comment, client_id, dry_run=$DRY_RUN).

Style d'ordres (important) :
- Scalping (15m/5m) ⇒ privilégie les ordres au marché (BUY/SELL). Autorise LIMIT/STOP si l'écart entrée/prix ou le spread le justifie; documente la justification. Ajuste le volume selon la volatilité.

Procédure d'exécution (OBLIGATOIRE) :
- Si tu décides `decision.send_order = true` :
  1) Tu DOIS appeler `execute_trade(...)` avec les niveaux normalisés.
  2) Tu copies la RÉPONSE OUTIL (JSON brut, sans paraphrase) dans `execution_result`.
- Si `decision.send_order = false` :
  - Mets `execution_result = null`.

Règles obligatoires :
1. HOLD/confiance/no-trade : si action == "HOLD" ou confidence < $MIN_CONFIDENCE ou regime == "no-trade" ⇒ annuler.
2. Spread edge : récupère bid/ask; sans quote ⇒ annuler. entry_ref = ask (BUY) ou bid (SELL). tp_dist = distance TP. Exige tp_dist > 0. En scalping: band HIGH ⇒ ≥ 5×spread, sinon ≥ 2×spread. Hors scalping: HIGH ⇒ ≥ 6×spread, sinon ≥ 4×spread.
3. RR min : RR ≥ 1.2. Tu peux pousser TP (≤ 5×spread) pour atteindre 1.2 ensuite.
4. Specs : respecte tick/digits/stopsLevel. Si incohérence SL/TP ⇒ annuler.
5. Marges : vérifie freeMargin; si inconnue ou insuffisante ⇒ annuler.
6. Exposition (mode non hedge uniquement) : si HEDGE_MODE=false ⇒ annuler si positions ≥4, ou ≥1 sur le même symbole, ou ≥3 paires contenant "USD". Calculer via get_positions (données réelles), ne pas inférer d'après le nom du symbole. Si l'info d'exposition est indisponible/ambiguë ⇒ ne pas annuler sur ce seul motif.
7. Mode hedge : autorise jambe opposée si possible (voir table). Si la paire n'est pas dans la table, TRAITE COMME NON-HEDGE (ne pas annuler pour ce seul motif). Respecter limites (USD legs <6, |USD net| ≤ max_net_usd×equity, paires hedgées ≤ max_pairs, band HIGH ⇒ confluence HTF si REQUIRE_HTF_ON_EDGES=true). Si ok, réduire volume hedge à max(0.01, (1-HEDGE_RISK_SPLIT)*default_volume) et renseigner `hedge`. Sinon, appliquer règles non-hedge.
8. Sizing volatilité : si `volatility.size_factor` < 1, réduis `volume` = max(0.01, default_volume × size_factor). En band NORMAL sans size_factor ⇒ volume = default_volume.
9. Sécurité : échec tool ⇒ annuler. Pas d’ouverture dans les 2 minutes suivant un changement de bougie H1/M30 si info dispo.

Réponds UNIQUEMENT avec le JSON suivant (700 tokens max) :
{
  "symbol": "<string>",
  "source": {"news_bias": "<neutral|positive|negative|null>", "confidence": <number>, "risk_level": "<low|medium|high|null>"},
  "portfolio": {"floating_pnl_total": <number|null>, "symbol_floating_pnl": <number|null>, "symbol_net_exposure": <number|null>, "symbol_direction": "<long|short|flat|null>"},
  "account_checks": {"equity": <number|null>, "free_margin": <number|null>, "margin_required_est": <number|null>, "margin_ok": <true|false>, "margin_reason": "<string>"},
  "tools_used": {"get_account_info": <true|false>, "get_positions": <true|false>, "get_orders": <true|false>, "get_symbol_spec": <true|false>, "get_current_price": <true|false>},
  "checks": {"values_ok": <true|false>, "direction_ok": <true|false>, "spec_ok": <true|false>, "rr": <number|null>, "tp_vs_spread_ratio": <number|null>, "hedge_ok": <true|false|null>},
  "order_policy": {"conflict_with_existing": "<none|duplicate|opposite|overexposed>", "action": "<proceed|adjust|cancel>", "adjust_reason": "<string|null>", "cancel_reason": "<string|null>"},
  "normalized_order": null,
  "decision": {"send_order": <true|false>, "reason": "<règle en français>", "hedge": {"enabled": <true|false>, "paired_with": "<string|null>", "volume": <number|null>}},
  "execution_result": <object|null>
}
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
        REQUIRE_HTF_ON_EDGES=REQUIRE_HTF_ON_EDGES,
    )
    return prompt


if __name__ == "__main__":
    mcp.run(transport="stdio")
