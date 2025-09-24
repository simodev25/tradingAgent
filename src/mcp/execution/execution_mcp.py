import json
from typing import Any, Optional, Tuple, List
from datetime import datetime
import sys
import os
import math
from loguru import logger

from mcp.server.fastmcp import FastMCP

# --- Import meta_api (relative to this file) ---
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_fetch/"))
)
try:
    import meta_api as meta_api  # type: ignore
except Exception as e:  # pragma: no cover
    meta_api = None  # type: ignore
    logger.error(f"[INIT] meta_api import failed: {e}")

mcp = FastMCP("Trading Execution MCP Server", log_level="DEBUG")

logger.remove()
logger.add(sys.stderr, level="DEBUG")


# ---------- Helpers ----------
def _ok(payload: Any) -> str:
    return json.dumps({"ok": True, "data": payload}, ensure_ascii=False)


def _err(msg: str, **extra: Any) -> str:
    return json.dumps({"ok": False, "error": msg, **extra}, ensure_ascii=False)


def _get(d: dict, path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_json_like(x: Any) -> Optional[dict]:
    """Best effort: return dict if x is JSON string or dict; else None."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def _safe_isfinite(*nums: float) -> bool:
    try:
        return all(isinstance(n, (int, float)) and math.isfinite(float(n)) for n in nums)
    except Exception:
        return False


def _fetch_symbol_spec(symbol: str) -> Optional[dict]:
    if meta_api is None:
        return None
    try:
        raw = meta_api.get_symbol_spec(symbol)  # type: ignore
        return _parse_json_like(raw)
    except Exception:
        return None


# --------- PRICE helpers (nouvelle implémentation) ---------
def _unwrap_data(obj: Any) -> Any:
    """Déballe des enveloppes type {'ok': True, 'data': X} ou {'result': X}."""
    cur = obj
    while isinstance(cur, dict):
        for k in ("data", "result", "Data"):
            if k in cur:
                cur = cur[k]
                break
        else:
            break
    return cur


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _symbol_aliases(symbol: str) -> List[str]:
    """Retourne une petite liste d’alias (avec/sans suffixe .pro, case-insensitive)."""
    s = symbol
    cand = {s, s.upper()}
    for suf in (".pro", ".PRO"):
        if s.endswith(suf):
            base = s[: -len(suf)]
            cand |= {base, base.upper()}
    return list(cand)


def _fetch_current_quote(symbol: str) -> Optional[dict]:
    """
    Renvoie un dict 'quote' normalisé:
    {
      'symbol': str, 'bid': float|None, 'ask': float|None, 'price': float|None,
      'time': str|None, 'brokerTime': str|None, 'raw': <payload brut>
    }
    Compatible avec l’exemple BTCUSD:
    {'symbol':'BTCUSD','bid':112847,'ask':112909,'time':...}
    et quelques variantes (imbriquées dans 'quote' ou 'tick', ou scalaire).
    """
    if meta_api is None:
        return None
    for sym in _symbol_aliases(symbol):
        try:
            raw = meta_api.get_current_price(sym)  # type: ignore
            data_or_raw = _parse_json_like(raw)
            data = _unwrap_data(data_or_raw if data_or_raw is not None else raw)

            if isinstance(data, dict):
                bid = _first_not_none(data.get("bid"), data.get("Bid"))
                ask = _first_not_none(data.get("ask"), data.get("Ask"))
                px = _first_not_none(data.get("price"), data.get("Price"), data.get("last"), data.get("Last"))

                # formats imbriqués: {"quote":{...}} ou {"tick":{...}}
                if (bid is None and ask is None and px is None) and any(k in data for k in ("quote", "tick")):
                    inner = _unwrap_data(data.get("quote") or data.get("tick"))
                    if isinstance(inner, dict):
                        bid = _first_not_none(inner.get("bid"), inner.get("Bid"))
                        ask = _first_not_none(inner.get("ask"), inner.get("Ask"))
                        px = _first_not_none(inner.get("price"), inner.get("Price"), inner.get("last"), inner.get("Last"))

                return {
                    "symbol": str(_first_not_none(data.get("symbol"), sym)),
                    "bid": float(bid) if bid is not None else None,
                    "ask": float(ask) if ask is not None else None,
                    "price": float(px) if px is not None else None,
                    "time": data.get("time"),
                    "brokerTime": data.get("brokerTime"),
                    "raw": data,
                }

            # scalaire: juste un prix
            if isinstance(data, (int, float)):
                return {
                    "symbol": sym,
                    "bid": None,
                    "ask": None,
                    "price": float(data),
                    "time": None,
                    "brokerTime": None,
                    "raw": data,
                }
        except Exception:
            continue
    return None


def _fetch_current_price(symbol: str, action: Optional[str] = None) -> Optional[float]:
    """
    Préfère ASK pour BUY, BID pour SELL ; sinon MID=(bid+ask)/2 ; sinon 'price/last', à défaut bid/ask isolés.
    """
    q = _fetch_current_quote(symbol)
    if not q:
        return None

    bid, ask, px = q["bid"], q["ask"], q["price"]

    if action:
        a = action.upper()
        if a.startswith("BUY") and ask is not None:
            return float(ask)
        if a.startswith("SELL") and bid is not None:
            return float(bid)

    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if px is not None:
        return float(px)
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return None
# -----------------------------------------------------------


def _extract_tick_and_digits(spec: Optional[dict]) -> Tuple[Optional[float], Optional[int]]:
    if not spec:
        return None, None
    tick = (
        spec.get("tickSize")
        or spec.get("tick_size")
        or spec.get("step")
        or spec.get("point")
        or spec.get("minTick")
        or spec.get("points")
    )
    digits = spec.get("digits") or spec.get("precision") or spec.get("pricePrecision")
    try:
        return (float(tick) if tick is not None else None, int(digits) if digits is not None else None)
    except Exception:
        return None, None


def _round_price(v: float, tick: Optional[float], digits: Optional[int]) -> float:
    if tick and tick > 0:
        v = round(v / tick) * tick
    if digits is not None and digits >= 0:
        v = round(v, digits)
    return float(v)


# ---------- Tools facultatifs (info) ----------
@mcp.tool()
def get_account_info() -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_account_info()  # type: ignore
    except Exception as e:
        return _err("account_info_failed", exc=str(e))


@mcp.tool()
def get_positions() -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_positions()  # type: ignore
    except Exception as e:
        return _err("positions_failed", exc=str(e))


@mcp.tool()
def get_orders() -> str:
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


_ALLOWED_ACTIONS = {"BUY", "SELL", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"}
_MARKET_ACTIONS = {"BUY", "SELL"}
_PENDING_ACTIONS = _ALLOWED_ACTIONS - _MARKET_ACTIONS


def _validate_directional_levels(
    action: str, entry: Optional[float], sl: float, tp: float, symbol: str
) -> Tuple[bool, Optional[str]]:
    """Best-effort coherence checks for SL/TP vs. direction et cohérence LIMIT/STOP vs marché."""
    if not _safe_isfinite(sl, tp) or (entry is not None and not _safe_isfinite(entry)):
        return False, "invalid_number"

    # prix de référence: entry si fourni, sinon prix courant cohérent avec action (bid/ask)
    price_ref = entry if entry is not None else _fetch_current_price(symbol, action=action)

    if price_ref is None:
        # impossible de vérifier sans prix
        return True, None

    # Cohérence SL/TP vs direction
    if action in ("BUY", "BUY_LIMIT", "BUY_STOP"):
        if sl >= price_ref:
            return False, "sl_must_be_below_price_for_buy"
        if tp <= price_ref:
            return False, "tp_must_be_above_price_for_buy"
    elif action in ("SELL", "SELL_LIMIT", "SELL_STOP"):
        if sl <= price_ref:
            return False, "sl_must_be_above_price_for_sell"
        if tp >= price_ref:
            return False, "tp_must_be_below_price_for_sell"

    # Prix des ordres en attente vs marché
    if action in _PENDING_ACTIONS:
        cur = _fetch_current_price(symbol, action=action)
        if cur is not None and entry is not None:
            if action == "BUY_LIMIT" and not (entry < cur):
                return False, "buy_limit_entry_must_be_below_market"
            if action == "SELL_LIMIT" and not (entry > cur):
                return False, "sell_limit_entry_must_be_above_market"
            if action == "BUY_STOP" and not (entry > cur):
                return False, "buy_stop_entry_must_be_above_market"
            if action == "SELL_STOP" and not (entry < cur):
                return False, "sell_stop_entry_must_be_below_market"

    return True, None


@mcp.tool()
def execute_trade(
    symbol: str,
    action: str,
    entry: float,
    sl: float,
    tp: float,
    volume: float = 0.01,
    comment: Optional[str] = None,
    client_id: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """
    Exécute une transaction via l’API MetaApi.
    - action: "BUY"/"SELL" (market) ou "BUY_LIMIT"/"SELL_LIMIT"/"BUY_STOP"/"SELL_STOP"
    - entry: utilisé pour LIMIT/STOP (ignoré pour BUY/SELL)
    - dry_run: si True, ne PAS envoyer l’ordre (retourne payload validé ou résultat de validation SDK si dispo)
    """
    try:
        if meta_api is None:
            return _err("meta_api_unavailable")

        normalized_action = action.upper()
        if normalized_action not in _ALLOWED_ACTIONS:
            return _err("invalid_action", got=action, allowed=sorted(_ALLOWED_ACTIONS))

        if not _safe_isfinite(sl, tp, volume):
            return _err("invalid_number", reason="non_finite_sl_tp_or_volume")
        if volume <= 0:
            return _err("invalid_volume", volume=volume)

        spec = _fetch_symbol_spec(symbol)
        tick, digits = _extract_tick_and_digits(spec)

        # Arrondi des niveaux aux contraintes du symbole
        sl_r = _round_price(float(sl), tick, digits)
        tp_r = _round_price(float(tp), tick, digits)

        entry_for_payload: Optional[float] = None
        entry_for_call: Optional[float] = None

        if normalized_action in _PENDING_ACTIONS:
            if not _safe_isfinite(entry):
                return _err("invalid_number", reason="non_finite_entry_for_pending")
            entry_for_payload = _round_price(float(entry), tick, digits)
            entry_for_call = entry_for_payload

        # Vérifs de cohérence (utilisent bid/ask selon action)
        ok, why = _validate_directional_levels(normalized_action, entry_for_call, sl_r, tp_r, symbol)
        if not ok:
            return _err("values_incoherent", reason=why)

        trade = {
            "actionType": normalized_action,
            "symbol": symbol,
            "volume": float(volume),
            "openPrice": entry_for_payload,
            "stopLoss": sl_r,
            "takeProfit": tp_r,
            "comment": comment,
            "client_id": client_id,
        }

        # Log quote complète pour debug
        try:
            q = _fetch_current_quote(symbol)
            if q:
                logger.debug(f"[PRICE] {q['symbol']} bid={q['bid']} ask={q['ask']} time={q.get('time')} brokerTime={q.get('brokerTime')}")
            else:
                logger.debug(f"[PRICE] no quote for {symbol}")
        except Exception:
            pass

        logger.debug(f"[MCP:EXECUTE] dry_run={dry_run} trade={trade}")

        if dry_run:
            # Si le SDK propose une validation locale, l'utiliser
            if hasattr(meta_api, "trade_execute"):
                try:
                    return meta_api.trade_execute(trade, client_id=client_id, dry_run=True)  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning(f"[MCP:EXECUTE] meta_api.trade_execute failed in dry_run: {e}")
            return _ok(trade)

        # Exécution live
        try:
            result = meta_api.execute_trade(  # type: ignore[call-arg]
                symbol,
                normalized_action,
                entry_for_call,
                sl_r,
                tp_r,
                volume=float(volume),
                comment=comment,
                client_id=client_id,
            )
        except TypeError:
            # Compat: certains SDK exigent 'entry' même pour market (on passe un None/valeur sûre)
            result = meta_api.execute_trade(  # type: ignore[call-arg]
                symbol,
                normalized_action,
                entry if entry_for_call is None else entry_for_call,
                sl_r,
                tp_r,
                volume=float(volume),
                comment=comment,
                client_id=client_id,
            )
        except Exception as e:
            logger.exception("[MCP:EXECUTE] trade_failed")
            return _err("trade_failed", exc=str(e), symbol=symbol, action=action)

        return result

    except Exception as e:  # pragma: no cover
        logger.exception("[MCP:EXECUTE] trade_failed_unexpected")
        return _err("trade_failed", exc=str(e), symbol=symbol, action=action)


# ---------- Prompt d’exécution (seul execute_trade garanti) ----------
@mcp.prompt()
def execution_agent(
    context: Any,
    default_volume: str = "0.01",
    min_confidence: str = "60",
    honor_hold: str = "True",
    dry_run: str = "True",
) -> str:
    """
    Construit un prompt riche à partir du JSON global (news + décision technique).
    NB: Seul l’outil execute_trade est garanti disponible. Les données de compte/positions/spec
        doivent venir du CONTEXTE si nécessaires.
    """
    if isinstance(context, str):
        ctx_str = context
    else:
        ctx_str = json.dumps(context, ensure_ascii=False)

    client_ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    from string import Template

    tmpl = Template(
        r"""
Tu es un **trader professionnel expert** (15 ans d'expérience). Tu reçois un CONTEXTE JSON complet (news + décision technique).
Ta mission : décider si l'ordre doit être envoyé, ajusté (arrondi tick/digits, distances), ou refusé.

# PARAMÈTRES
- default_volume = $DEFAULT_VOLUME
- min_confidence = $MIN_CONFIDENCE
- honor_hold = $HONOR_HOLD
- dry_run = $DRY_RUN

# CONTEXTE
$CTX

# Outils MCP disponibles :

- get_account_info()
- get_positions()
- get_orders()
- get_symbol_spec(symbol)
- get_current_price(symbol)
- execute_trade(symbol, action, entry, sl, tp, volume, comment, client_id, dry_run=$DRY_RUN)
# Plan d'appel **obligatoire**
1) **get_account_info** → lire equity/free_margin/margin level.
2) **get_positions** → exposition nette par symbol, sens (long/short), PnL flottant.
3) **get_orders** → ordres en attente sur le même symbol (doublons/conflits).
4) **get_symbol_spec(symbol)** → digits, tick, stopsLevel, step volume, contract size.
5) **get_current_price(symbol)** → référence pour cohérence SL/TP & type LIMIT/STOP.
6) **Décision** : ajuster/annuler la proposition si marge insuffisante, sur-exposition, doublon d'ordre, conflit directionnel, distances mini non respectées.

# Procédure & Règles (résumé)
- **Tu DOIS appeler les 5 outils ci-dessus** avant toute exécution. **N'invente jamais** leurs sorties.
- Si un outil échoue → renseigne `tools_used` et `snapshots` fidèlement et **annule** (approche conservatrice).
- Gating dur :
  - Si `technical_decision.action == HOLD` **ou** `confidence < min_confidence` **ou** `regime == "no-trade"` → **annule**.
  - Si `free_margin` indisponible **ou** insuffisante → **annule**.
  - Si ordre en attente **duplicatif** (mêmes niveaux ± 1 tick) → **annule**.
  - Si position **opposée** déjà ouverte et pas de logique de hedge définie → **annule**.
  - Si SL/TP non cohérents avec la direction **ou** distances mini (`stopsLevel`) non respectées → **ajuste** ou **annule**.
- RR ≥ 1.2 si calculable ; sinon **annule** ou propose un ajustement justifié.
- `normalized_order` **ne doit exister que si** `decision.send_order == true`. Sinon, mets `normalized_order: null`.

# Sortie attendue — JSON strict
{
  "symbol": "<string>",
  "source": {"news_bias": "<neutral|positive|negative|null>", "confidence": <number>, "risk_level": "<low|medium|high|null>"},
  "portfolio": {"floating_pnl_total": <number|null>, "symbol_floating_pnl": <number|null>, "symbol_net_exposure": <number|null>, "symbol_direction": "<long|short|flat|null>"},
  "account_checks": {"equity": <number|null>, "free_margin": <number|null>, "margin_required_est": <number|null>, "margin_ok": <true|false>, "margin_reason": "<string>"},
  "tools_used": {"get_account_info": <true|false>, "get_positions": <true|false>, "get_orders": <true|false>, "get_symbol_spec": <true|false>, "get_current_price": <true|false>},
  "snapshots": {"account": <object|null>, "positions": <array|null>, "orders": <array|null>, "symbol_spec": <object|null>, "price": <number|null>},
  "checks": {"values_ok": <true|false>, "direction_ok": <true|false>, "spec_ok": <true|false>, "rr": <number|null>},
  "order_policy": {"conflict_with_existing": "<none|duplicate|opposite|overexposed>", "action": "<proceed|adjust|cancel>", "adjust_reason": "<string|null>", "cancel_reason": "<string|null>"},
  "normalized_order": null,
  "decision": {"send_order": <true|false>, "reason": "<bref résumé en français>"},
  "execution_result": "<execute_trade result>"
}
"""
    )

    prompt = tmpl.substitute(
        CTX=ctx_str,
        DEFAULT_VOLUME=default_volume,
        MIN_CONFIDENCE=min_confidence,
        HONOR_HOLD=honor_hold,
        DRY_RUN=dry_run,
        CLIENT_TS=client_ts,
    )
    return prompt


if __name__ == "__main__":
    mcp.run(transport="stdio")
