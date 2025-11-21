# =============================
# execution_core.py
# =============================
import json, math, os, sys
from typing import Any, Optional, Tuple, List
from loguru import logger

# --- Import meta_api (relative to this file) ---
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_fetch/"))
)
try:
    import meta_api as meta_api  # type: ignore
except Exception as e:  # pragma: no cover
    meta_api = None  # type: ignore
    logger.error(f"[INIT] meta_api import failed (core): {e}")


# ---------- Generic helpers ----------
def _ok(payload: Any) -> str:
    return json.dumps({"ok": True, "data": payload}, ensure_ascii=False)


def _err(msg: str, **extra: Any) -> str:
    return json.dumps({"ok": False, "error": msg, **extra}, ensure_ascii=False)


def _parse_json_like(x: Any) -> Optional[dict]:
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


def _unwrap_data(obj: Any) -> Any:
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


def _safe_isfinite(*nums: float) -> bool:
    try:
        return all(isinstance(n, (int, float)) and math.isfinite(float(n)) for n in nums)
    except Exception:
        return False


def _sanitize_text(value: str) -> str:
    if not value:
        return ""
    return " ".join(str(value).split())


def _sanitize_comment_and_client_id(comment: str, client_id: str) -> Tuple[str, str]:
    comment = _sanitize_text(comment)
    client_id = _sanitize_text(client_id)
    max_total = 30 if comment and client_id else 31
    if len(comment) + len(client_id) <= max_total:
        return comment, client_id
    # Trim from comment first, keeping client_id intact if possible.
    while len(comment) + len(client_id) > max_total and comment:
        comment = comment[:-1]
    # If still too long, trim client_id.
    while len(comment) + len(client_id) > max_total and client_id:
        client_id = client_id[:-1]
    if len(comment) + len(client_id) > max_total:
        # Fallback: blank client_id, trim comment.
        client_id = ""
        max_total = 31
        comment = comment[:max_total]
    return comment, client_id


# --------- Specs / price helpers ---------

def _symbol_aliases(symbol: str) -> List[str]:
    s = symbol
    cand = {s, s.upper()}
    for suf in (".pro", ".PRO"):
        if s.endswith(suf):
            base = s[: -len(suf)]
            cand |= {base, base.upper()}
    return list(cand)


def fetch_symbol_spec(symbol: str) -> Optional[dict]:
    if meta_api is None:
        return None
    try:
        raw = meta_api.get_symbol_spec(symbol)  # type: ignore
        return _parse_json_like(raw)
    except Exception:
        return None


def fetch_current_quote(symbol: str) -> Optional[dict]:
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

            if isinstance(data, (int, float)):
                return {"symbol": sym, "bid": None, "ask": None, "price": float(data), "time": None, "brokerTime": None, "raw": data}
        except Exception:
            continue
    return None


def fetch_current_price(symbol: str, action: str) -> Optional[float]:
    q = fetch_current_quote(symbol)
    if not q:
        return None
    bid, ask, px = q.get("bid"), q.get("ask"), q.get("price")
    a = (action or "").upper()
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


def extract_tick_and_digits(spec: Optional[dict]) -> Tuple[Optional[float], Optional[int]]:
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


def round_price(v: float, tick: Optional[float], digits: Optional[int]) -> float:
    if tick and tick > 0:
        v = round(v / tick) * tick
    if digits is not None and digits >= 0:
        v = round(v, digits)
    return float(v)


ALLOWED_ACTIONS = {"BUY", "SELL", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"}
MARKET_ACTIONS = {"BUY", "SELL"}
PENDING_ACTIONS = ALLOWED_ACTIONS - MARKET_ACTIONS


def validate_directional_levels(
    action: str,
    entry: float,
    sl: float,
    tp: float,
    symbol: str,
    tick: Optional[float],
    stops_level_points: int,
    epsilon_ticks: int,
) -> Tuple[bool, Optional[str]]:
    if not _safe_isfinite(sl, tp) or (entry != 0 and not _safe_isfinite(entry)):
        return False, "invalid_number"

    price_ref = (entry if entry != 0 else fetch_current_price(symbol, action))
    if price_ref is None:
        return True, None

    t = float(tick) if (tick is not None) else None
    eps = (epsilon_ticks * t) if (t and t > 0) else 0.0
    min_dist = max(float(stops_level_points or 0) * (t or 0.0), eps)

    a = action.upper()
    if a in ("BUY", "BUY_LIMIT", "BUY_STOP"):
        if tp <= price_ref + min_dist:
            return False, "tp_must_be_above_price_for_buy"
        if sl >= price_ref - min_dist:
            return False, "sl_must_be_below_price_for_buy"
    elif a in ("SELL", "SELL_LIMIT", "SELL_STOP"):
        if tp >= price_ref - min_dist:
            return False, "tp_must_be_below_price_for_sell"
        if sl <= price_ref + min_dist:
            return False, "sl_must_be_above_price_for_sell"
    else:
        return False, "invalid_action"

    if a in PENDING_ACTIONS and entry != 0:
        cur = fetch_current_price(symbol, action)
        if cur is not None:
            if a == "BUY_LIMIT" and not (entry < cur - eps):
                return False, "buy_limit_entry_must_be_below_market"
            if a == "SELL_LIMIT" and not (entry > cur + eps):
                return False, "sell_limit_entry_must_be_above_market"
            if a == "BUY_STOP" and not (entry > cur + eps):
                return False, "buy_stop_entry_must_be_above_market"
            if a == "SELL_STOP" and not (entry < cur - eps):
                return False, "sell_stop_entry_must_be_below_market"

    return True, None


def build_and_execute_trade(
    symbol: str,
    action: str,
    entry: float,
    sl: float,
    tp: float,
    volume: float,
    comment: str,
    client_id: str,
    dry_run: bool,
) -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")

    normalized_action = action.upper()
    if normalized_action not in ALLOWED_ACTIONS:
        return _err("invalid_action", got=action, allowed=sorted(ALLOWED_ACTIONS))

    if not _safe_isfinite(sl, tp, volume):
        return _err("invalid_number", reason="non_finite_sl_tp_or_volume")
    if volume <= 0:
        return _err("invalid_volume", volume=volume)

    spec = fetch_symbol_spec(symbol)
    tick, digits = extract_tick_and_digits(spec)

    try:
        stops_level_points = int((spec or {}).get("stopsLevel") or (spec or {}).get("stops_level") or 0)
    except Exception:
        stops_level_points = 0

    quote = fetch_current_quote(symbol)
    bid = (quote or {}).get("bid")
    ask = (quote or {}).get("ask")
    price_mid = (None if (bid is None or ask is None) else (bid + ask) / 2.0)
    spread = (None if (bid is None or ask is None) else (ask - bid))

    action_upper = normalized_action
    if action_upper in MARKET_ACTIONS:
        entry_ref = ask if action_upper == "BUY" else bid
    else:
        entry_ref = float(entry) if _safe_isfinite(entry) else None

    RR_MIN = 1.2
    MAX_TP_BUMP_SPREADS = 5
    sl0, tp0 = float(sl), float(tp)

    if entry_ref is not None and spread is not None and spread > 0 and _safe_isfinite(sl0, tp0):
        if action_upper.startswith("BUY"):
            risk = max(entry_ref - sl0, 0.0)
            reward = max(tp0 - entry_ref, 0.0)
        else:
            risk = max(sl0 - entry_ref, 0.0)
            reward = max(entry_ref - tp0, 0.0)
        rr_now = (reward / risk) if (risk and risk > 0) else None
        if rr_now is not None and 1.0 <= rr_now < RR_MIN:
            needed_reward = RR_MIN * risk
            bump = needed_reward - reward
            max_bump = MAX_TP_BUMP_SPREADS * spread
            if bump > 0 and bump <= max_bump:
                if action_upper.startswith("BUY"):
                    tp0 = entry_ref + needed_reward
                else:
                    tp0 = entry_ref - needed_reward
                tp0 = round_price(tp0, tick, digits)

    sl_r = round_price(sl0, tick, digits)
    tp_r = round_price(tp0, tick, digits)

    entry_for_payload = None
    entry_for_call = None
    if action_upper in PENDING_ACTIONS:
        if not _safe_isfinite(entry):
            return _err("invalid_number", reason="non_finite_entry_for_pending")
        entry_for_payload = round_price(float(entry), tick, digits)
        entry_for_call = entry_for_payload

    ok, why = validate_directional_levels(
        action_upper,
        entry_for_call if entry_for_call is not None else 0.0,
        sl_r,
        tp_r,
        symbol,
        tick,
        stops_level_points,
        2,
    )
    if not ok:
        return _err("values_incoherent", reason=why)

    tp_vs_spread_ratio = None
    rr_out = None
    if entry_ref is not None and spread is not None and spread > 0 and _safe_isfinite(sl_r, tp_r):
        if action_upper.startswith("BUY"):
            risk = max(entry_ref - sl_r, 0.0)
            reward = max(tp_r - entry_ref, 0.0)
        else:
            risk = max(sl_r - entry_ref, 0.0)
            reward = max(entry_ref - tp_r, 0.0)
        rr_out = (reward / risk) if (risk and risk > 0) else None
        tp_vs_spread_ratio = (reward / spread) if reward is not None else None

    raw_comment = comment or ""
    raw_client_id = client_id or ""
    comment_sanitized, client_id_sanitized = _sanitize_comment_and_client_id(raw_comment, raw_client_id)
    if comment_sanitized != raw_comment or client_id_sanitized != raw_client_id:
        logger.debug(
            "[EXEC_CORE] comment/client_id trimmed",
            comment=comment_sanitized,
            client_id=client_id_sanitized,
            original_comment=raw_comment,
            original_client_id=raw_client_id,
        )
    trade = {
        "actionType": action_upper,
        "symbol": symbol,
        "volume": float(volume),
        "openPrice": entry_for_payload,
        "stopLoss": sl_r,
        "takeProfit": tp_r,
        "comment": comment_sanitized,
        "client_id": client_id_sanitized,
    }

    try:
        if quote:
            logger.debug(
                f"[PRICE] {quote['symbol']} bid={quote['bid']} ask={quote['ask']} time={quote.get('time')} brokerTime={quote.get('brokerTime')}"
            )
        else:
            logger.debug(f"[PRICE] no quote for {symbol}")
    except Exception:
        pass

    logger.debug(f"[EXEC_CORE] dry_run={dry_run} trade={trade} rr={rr_out} tp_vs_spread={tp_vs_spread_ratio}")

    if dry_run:
        if hasattr(meta_api, "trade_execute"):
            try:
                return meta_api.trade_execute(trade, client_id=client_id, dry_run=True)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"[EXEC_CORE] meta_api.trade_execute dry_run failed: {e}")
        return _ok({"trade": trade, "rr": rr_out, "tp_vs_spread_ratio": tp_vs_spread_ratio})

    try:
        result = meta_api.execute_trade(  # type: ignore[call-arg]
            symbol,
            action_upper,
            entry_for_call,
            sl_r,
            tp_r,
            volume=float(volume),
            comment=comment_sanitized,
            client_id=client_id_sanitized,
        )
    except TypeError:
        result = meta_api.execute_trade(  # type: ignore[call-arg]
            symbol,
            action_upper,
            entry if entry_for_call is None else entry_for_call,
            sl_r,
            tp_r,
            volume=float(volume),
            comment=comment_sanitized,
            client_id=client_id_sanitized,
        )
    except Exception as e:
        logger.exception("[EXEC_CORE] trade_failed")
        return _err("trade_failed", exc=str(e), symbol=symbol, action=action)

    return result
