import os, sys, json, math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from mcp.server.fastmcp import FastMCP
from string import Template
from dotenv import load_dotenv

# ---- env
load_dotenv()

# ---- meta_api import (garde ton chemin)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_fetch/')))
import meta_api as meta_api  # noqa

# ---- core import
from analysis_core import (
    CONFIG, _ok, _err, _round_to_tick,
    _df_from_ohlcv, _compute_indicators_df, _last_row,
    _infer_digits_from_prices, _infer_tick_from_prices,
    _zigzag_swings, _fib_levels,
    _atr_pct_bands_from_df, _classify_vol_band, _volatility_gate,
    _directional_score, _confidence, _regime_from_df, _position_size,
)

# ===================== MCP bootstrap =====================

mcp = FastMCP("Trading Analysis MCP Server", log_level="INFO")
logger.remove()
logger.add(sys.stderr, level="INFO")

# ===================== Cache OHLCV =====================

CANDLE_CACHE: Dict[str, List[Dict[str, Any]]] = {}
IndicatorKey = Tuple[str, int, int, int, int, int, int, float]
INDICATOR_CACHE: Dict[IndicatorKey, pd.DataFrame] = {}


def _make_cache_key(symbol: str, period: str, interval: str) -> str:
    return f"{symbol}|{period}|{interval}".lower()


def _resolve_indicator_params(
    rsi_len: int = 0,
    ema_fast: int = 0,
    ema_slow: int = 0,
    macd_signal: int = 0,
    atr_len: int = 0,
    bb_len: int = 0,
    bb_mult: float = 0.0,
) -> Tuple[int, int, int, int, int, int, float]:
    return (
        rsi_len or CONFIG["RSI_LEN"],
        ema_fast or CONFIG["EMA_FAST"],
        ema_slow or CONFIG["EMA_SLOW"],
        macd_signal or CONFIG["MACD_SIGNAL"],
        atr_len or CONFIG["ATR_LEN"],
        bb_len or CONFIG["BB_LEN"],
        bb_mult or CONFIG["BB_MULT"],
    )


def _indicator_key(cache_key: str, params: Tuple[int, int, int, int, int, int, float]) -> IndicatorKey:
    rsi_len, ema_fast, ema_slow, macd_signal, atr_len, bb_len, bb_mult = params
    return (cache_key, rsi_len, ema_fast, ema_slow, macd_signal, atr_len, bb_len, float(bb_mult))


def _invalidate_indicator_cache(cache_key: str) -> None:
    if not INDICATOR_CACHE:
        return
    stale = [key for key in INDICATOR_CACHE if key[0] == cache_key]
    for key in stale:
        INDICATOR_CACHE.pop(key, None)


def _df_with_indicators(
    ohlcv: List[Dict[str, Any]],
    params: Tuple[int, int, int, int, int, int, float],
) -> Optional[pd.DataFrame]:
    df = _df_from_ohlcv(ohlcv)
    if df is None or df.empty:
        return None
    rsi_len, ema_fast, ema_slow, macd_signal, atr_len, bb_len, bb_mult = params
    return _compute_indicators_df(df, rsi_len, ema_fast, ema_slow, macd_signal, atr_len, bb_len, bb_mult)


def _df_from_cache(cache_key: str, params: Tuple[int, int, int, int, int, int, float]) -> Optional[pd.DataFrame]:
    src = CANDLE_CACHE.get(cache_key)
    if src is None:
        return None
    key = _indicator_key(cache_key, params)
    cached = INDICATOR_CACHE.get(key)
    if cached is not None and not cached.empty:
        return cached
    df = _df_with_indicators(src, params)
    if df is None:
        return None
    INDICATOR_CACHE[key] = df
    return df


def _ensure_indicator_df(
    df: Optional[pd.DataFrame],
    params: Tuple[int, int, int, int, int, int, float],
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    required = {
        "EMA_Fast", "EMA_Slow", "RSI", "MACD_Line", "MACD_Signal",
        "ATR", "BB_Mid", "BB_Upper", "BB_Lower",
    }
    if required.issubset(df.columns):
        return df
    rsi_len, ema_fast, ema_slow, macd_signal, atr_len, bb_len, bb_mult = params
    return _compute_indicators_df(df, rsi_len, ema_fast, ema_slow, macd_signal, atr_len, bb_len, bb_mult)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _ensure_cached_ohlcv(symbol: str, period: str, interval: str) -> str:
    cache_key = _make_cache_key(symbol, period, interval)
    if cache_key in CANDLE_CACHE:
        return cache_key
    data = meta_api.get_historical_candles(symbol, period, interval)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = []
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    # Fallback: try Yahoo Finance if MetaApi returned empty/error
    if not isinstance(data, list) or not data:
        try:
            # lazy import to avoid hard dependency on import
            import yfinance as yf  # type: ignore

            def _to_yf_symbol(sym: str) -> str:
                base = (sym or "").split(".")[0].upper()
                # Map common FX pairs like EURUSD.pro -> EURUSD=X
                if len(base) == 6 and base.isalpha():
                    return f"{base}=X"
                return base

            yf_symbol = _to_yf_symbol(symbol)
            tkr = yf.Ticker(yf_symbol)
            df_yf = tkr.history(period=period, interval=interval, rounding=True)
            if df_yf is not None and not df_yf.empty:
                df_yf = df_yf.reset_index()
                # Normalize to records compatible with _df_from_ohlcv
                data = json.loads(df_yf.to_json(orient="records", date_format="iso"))
                logger.info(f"[MCP:ANALYSIS] Used Yahoo fallback for {symbol} -> {yf_symbol} ({period}/{interval}), rows={len(data)})")
            else:
                data = []
        except Exception as _e:
            # keep data as empty list so we raise below, but log for diagnostics
            logger.warning(f"[MCP:ANALYSIS] Yahoo fallback failed for {symbol} ({period}/{interval}): {_e}")
            data = []
    if not isinstance(data, list) or not data:
        raise ValueError(f"Empty OHLCV for {symbol}:{period}:{interval}")
    CANDLE_CACHE[cache_key] = data
    _invalidate_indicator_cache(cache_key)
    return cache_key


def _fetch_and_features(
    symbol: str,
    period_ltf: str,
    interval_ltf: str,
    period_htf: str,
    interval_htf: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    params = _resolve_indicator_params()
    cache_ltf = _ensure_cached_ohlcv(symbol, period_ltf, interval_ltf)
    cache_htf = _ensure_cached_ohlcv(symbol, period_htf, interval_htf)
    ltf = _df_from_cache(cache_ltf, params)
    htf = _df_from_cache(cache_htf, params)
    if ltf is None or htf is None:
        raise ValueError("Failed to compute indicator dataframes for LTF/HTF")
    return ltf, htf

# ===================== Helpers =====================

def _parse_ohlcv_json(ohlcv_json: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(ohlcv_json)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list) or not data:
            return []
        return data
    except Exception:
        return []

# ===================== Tools (PRIMITIFS UNIQUEMENT) =====================

@mcp.tool()
def ping(x: str) -> str:
    """ok"""
    return _ok({"pong": x})

# ---- Fetch ----

@mcp.tool()
def get_historical_candles(symbol: str, period: str = "1mo", interval: str = "1d", compact: bool = True) -> str:
    """fetch"""
    logger.info(f"[MCP:ANALYSIS] get_historical_candles {symbol}:{period}:{interval}")
    try:
        data = meta_api.get_historical_candles(symbol, period, interval)
        if isinstance(data, str):
            data = json.loads(data)
        ohlcv = data.get("data") if isinstance(data, dict) and "data" in data else data
        if not isinstance(ohlcv, list) or not ohlcv:
            return _err("fetch returned empty data", symbol=symbol, period=period, interval=interval)
        cache_key = _make_cache_key(symbol, period, interval)
        CANDLE_CACHE[cache_key] = ohlcv
        _invalidate_indicator_cache(cache_key)
        payload = {"cache_key": cache_key, "count": len(ohlcv)}
        if not compact:
            payload["data"] = ohlcv
        return _ok(payload)
    except Exception as e:
        return _err("fetch failed", exc=str(e), symbol=symbol, period=period, interval=interval)

# ---- Indicators (deux tools séparés) ----

@mcp.tool()
def compute_indicators_from_cache(cache_key: str, tail: int = 200, rsi_len: int = 0, ema_fast: int = 0, ema_slow: int = 0, macd_signal: int = 0, atr_len: int = 0, bb_len: int = 0, bb_mult: float = 0.0, last_only: bool = True) -> str:
    """ind_from_cache"""
    try:
        if cache_key not in CANDLE_CACHE:
            return _err("cache_key not found", cache_key=cache_key)
        if tail <= 0:
            tail = 200
        params = _resolve_indicator_params(rsi_len, ema_fast, ema_slow, macd_signal, atr_len, bb_len, bb_mult)
        df = _df_from_cache(cache_key, params)
        if df is None:
            return _err("Invalid OHLCV")
        tail = min(tail, len(df))
        if last_only:
            payload = _last_row(df)
        else:
            payload = json.loads(df.tail(tail).replace({np.nan: None}).to_json(orient="records", date_format="iso"))
        return _ok(payload)
    except Exception as e:
        return _err("compute_indicators_from_cache failed", exc=str(e))

@mcp.tool()
def compute_indicators_from_json(ohlcv_json: str, tail: int = 200, rsi_len: int = 0, ema_fast: int = 0, ema_slow: int = 0, macd_signal: int = 0, atr_len: int = 0, bb_len: int = 0, bb_mult: float = 0.0, last_only: bool = True) -> str:
    """ind_from_json"""
    try:
        src = _parse_ohlcv_json(ohlcv_json)
        if not src:
            return _err("Invalid OHLCV json")
        if tail <= 0:
            tail = 200
        params = _resolve_indicator_params(rsi_len, ema_fast, ema_slow, macd_signal, atr_len, bb_len, bb_mult)
        df = _df_with_indicators(src, params)
        if df is None:
            return _err("Invalid OHLCV")
        tail = min(tail, len(df))
        if last_only:
            payload = _last_row(df)
        else:
            payload = json.loads(df.tail(tail).replace({np.nan: None}).to_json(orient="records", date_format="iso"))
        return _ok(payload)
    except Exception as e:
        return _err("compute_indicators_from_json failed", exc=str(e))

# ---- Volatility bands (deux tools) ----

@mcp.tool()
def volatility_bands_from_cache(cache_key: str, lookback_days: int = 0, low_pct: int = 0, high_pct: int = 0, extreme_pct: int = 0, size_high: float = 0.0) -> str:
    """vol_from_cache"""
    try:
        if cache_key not in CANDLE_CACHE:
            return _err("cache_key not found", cache_key=cache_key)
        lb = lookback_days or CONFIG["VOL_LOOKBACK_DAYS"]
        lp = low_pct or CONFIG["VOL_LOW_PCT"]
        hp = high_pct or CONFIG["VOL_HIGH_PCT"]
        ep = extreme_pct or CONFIG["VOL_EXTREME_PCT"]
        sh = size_high or CONFIG["VOL_SIZE_HIGH"]
        params = _resolve_indicator_params()
        df = _df_from_cache(cache_key, params)
        if df is None or df.empty:
            return _err("Invalid OHLCV")
        bands = _atr_pct_bands_from_df(df, lb, lp, hp, ep)
        if bands is None:
            return _err("Not enough data to compute percentiles", rows=len(df))
        cls = _classify_vol_band(bands["atr_now"], bands["p10"], bands["p90"], bands.get("p95"), size_high=sh)
        return _ok({
            "atr_pct_now": bands["atr_now"],
            "p10": bands["p10"], "p90": bands["p90"], "p95": bands.get("p95"),
            "band": cls["band"], "size_factor": cls["size_factor"], "reason": cls["reason"],
        })
    except Exception as e:
        return _err("volatility_bands_from_cache failed", exc=str(e))

@mcp.tool()
def volatility_bands_from_json(ohlcv_json: str, lookback_days: int = 0, low_pct: int = 0, high_pct: int = 0, extreme_pct: int = 0, size_high: float = 0.0) -> str:
    """vol_from_json"""
    try:
        src = _parse_ohlcv_json(ohlcv_json)
        if not src:
            return _err("Invalid OHLCV json")
        lb = lookback_days or CONFIG["VOL_LOOKBACK_DAYS"]
        lp = low_pct or CONFIG["VOL_LOW_PCT"]
        hp = high_pct or CONFIG["VOL_HIGH_PCT"]
        ep = extreme_pct or CONFIG["VOL_EXTREME_PCT"]
        sh = size_high or CONFIG["VOL_SIZE_HIGH"]
        params = _resolve_indicator_params()
        df = _df_with_indicators(src, params)
        if df is None or df.empty:
            return _err("Invalid OHLCV")
        bands = _atr_pct_bands_from_df(df, lb, lp, hp, ep)
        if bands is None:
            return _err("Not enough data to compute percentiles", rows=len(df))
        cls = _classify_vol_band(bands["atr_now"], bands["p10"], bands["p90"], bands.get("p95"), size_high=sh)
        return _ok({
            "atr_pct_now": bands["atr_now"],
            "p10": bands["p10"], "p90": bands["p90"], "p95": bands.get("p95"),
            "band": cls["band"], "size_factor": cls["size_factor"], "reason": cls["reason"],
        })
    except Exception as e:
        return _err("volatility_bands_from_json failed", exc=str(e))

# ---- Levels helpers ----

def _prepare_htf_df(htf_df: Optional[pd.DataFrame], params: Tuple[int, int, int, int, int, int, float]) -> Optional[pd.DataFrame]:
    if htf_df is None or htf_df.empty or len(htf_df) < 20:
        return None
    return _ensure_indicator_df(htf_df, params)


def _levels_autonomous_from_df(
    df: Optional[pd.DataFrame],
    action: str,
    horizon: str,
    risk_level: str,
    use_fib: bool = True,
    fib_atr_mult: float = 2.0,
    anchor_tf: str = "auto",
    htf_df: Optional[pd.DataFrame] = None,
) -> str:
    try:
        params = _resolve_indicator_params()
        df = _ensure_indicator_df(df, params)
        if df is None or len(df) < 20:
            return _err("Invalid or too short OHLCV")
        df = df.copy()
        action_up = (action or "").upper()
        if action_up not in {"BUY", "SELL"}:
            return _err("action must be BUY or SELL")
        last_close = float(df.iloc[-1]["Close"])
        atr_ltf = float(df.iloc[-1]["ATR"]) if pd.notna(df.iloc[-1]["ATR"]) else 0.0
        prices = pd.concat([df["Open"], df["High"], df["Low"], df["Close"]], ignore_index=True)
        digits_guess = _infer_digits_from_prices(prices)
        tick = _infer_tick_from_prices(prices, digits_guess) or (10 ** (-max(digits_guess or 3, 3)))
        tick = float(tick)
        eps = max(tick * 0.5, 1e-12)
        prof = CONFIG["PROFILES"].get(horizon.lower(), CONFIG["PROFILES"]["swing"])
        atr_basis = atr_ltf
        htf_prepared = _prepare_htf_df(htf_df, params)
        if (anchor_tf or "").lower() in {"htf", "auto"} and htf_prepared is not None:
            atr_htf_last = float(htf_prepared.iloc[-1]["ATR"]) if pd.notna(htf_prepared.iloc[-1]["ATR"]) else None
            if atr_htf_last is not None and np.isfinite(atr_htf_last):
                atr_basis = atr_htf_last
        min_stop_ticks = int(prof.get("min_ticks", 100))
        if atr_basis and tick > 0:
            min_stop_ticks = max(
                min_stop_ticks,
                int(math.ceil(CONFIG["LEVELS_MIN_ATR_FRACTION"] * atr_basis / tick)),
            )
        min_stop_price = float(min_stop_ticks * tick)
        spread_buffer = max(2.0 * tick, float(prof.get("spread_frac", 0.08)) * min_stop_price)
        struct_floor = 0.0
        try:
            hz = (horizon or "").lower()
            if hz == "scalping":
                # Pour le scalping, utiliser la structure LTF plus proche du prix et un seuil ATR plus bas
                piv_ltf = _zigzag_swings(df, atr_mult=max(1.2, float(fib_atr_mult) * 0.8), min_bars=3)
                if action_up == "BUY":
                    for i in range(len(piv_ltf) - 1, -1, -1):
                        if piv_ltf[i][2] == "L":
                            struct_floor = abs(last_close - float(piv_ltf[i][1])) + 6 * tick
                            break
                else:  # SELL
                    for i in range(len(piv_ltf) - 1, -1, -1):
                        if piv_ltf[i][2] == "H":
                            struct_floor = abs(float(piv_ltf[i][1]) - last_close) + 6 * tick
                            break
            else:
                # Hors scalping, garder la structure HTF si disponible
                if htf_prepared is not None and len(htf_prepared) >= 30:
                    piv = _zigzag_swings(htf_prepared, atr_mult=max(1.8, float(fib_atr_mult)), min_bars=3)
                    if action_up == "BUY":
                        for i in range(len(piv) - 1, -1, -1):
                            if piv[i][2] == "L":
                                struct_floor = abs(last_close - float(piv[i][1])) + 10 * tick
                                break
                    elif action_up == "SELL":
                        for i in range(len(piv) - 1, -1, -1):
                            if piv[i][2] == "H":
                                struct_floor = abs(float(piv[i][1]) - last_close) + 10 * tick
                                break
        except Exception:
            struct_floor = 0.0
        rr_target = float(prof.get("rr_target", 1.8))
        sl_dist0 = float(prof.get("sl_atr_mult", 1.1)) * atr_ltf
        tp_dist0 = float(prof.get("tp_atr_mult", 2.2)) * atr_ltf
        req_dist = max(min_stop_price + spread_buffer, struct_floor)
        sl_dist = max(sl_dist0, req_dist)
        tp_dist = max(tp_dist0, rr_target * sl_dist, req_dist)
        entry_ref = last_close
        if action_up == "BUY":
            sl = _round_to_tick(entry_ref - sl_dist, tick)
            tp = _round_to_tick(entry_ref + tp_dist, tick)
        else:
            sl = _round_to_tick(entry_ref + sl_dist, tick)
            tp = _round_to_tick(entry_ref - tp_dist, tick)
        rr = float(tp_dist / sl_dist) if sl_dist > 0 else 0.0
        fib_used = False
        fib_tp_raw = None
        if _as_bool(use_fib) and len(df) >= 30:
            try:
                piv = _zigzag_swings(df, atr_mult=float(fib_atr_mult))
                if len(piv) >= 2:
                    i2, p2, t2 = piv[-1]
                    i1, p1, t1 = piv[-2]
                    up_segment = (t1 == "L" and t2 == "H" and p2 > p1)
                    down_segment = (t1 == "H" and t2 == "L" and p2 < p1)
                    fib = _fib_levels(high=max(p1, p2), low=min(p1, p2))
                    if action_up == "BUY" and up_segment:
                        for tp_cand in sorted(list(fib["ext_up"].values())):
                            tpr = _round_to_tick(tp_cand, tick)
                            if tpr and (tpr >= entry_ref + req_dist - eps) and ((tpr - entry_ref) / sl_dist) >= rr_target:
                                tp = tpr
                                tp_dist = tp - entry_ref
                                rr = tp_dist / sl_dist
                                fib_used = True
                                fib_tp_raw = tp_cand
                                break
                    elif action_up == "SELL" and down_segment:
                        for tp_cand in sorted(list(fib["ext_down"].values()), reverse=True):
                            tpr = _round_to_tick(tp_cand, tick)
                            if tpr and (tpr <= entry_ref - req_dist + eps) and ((entry_ref - tpr) / sl_dist) >= rr_target:
                                tp = tpr
                                tp_dist = entry_ref - tp
                                rr = tp_dist / sl_dist
                                fib_used = True
                                fib_tp_raw = tp_cand
                                break
            except Exception:
                pass
        if action_up == "BUY":
            ok_side = (sl <= entry_ref - req_dist + eps) and (tp >= entry_ref + req_dist - eps)
            if not ok_side:
                tp = _round_to_tick(entry_ref + max(tp_dist0, rr_target * sl_dist, req_dist), tick)
                ok_side = (sl <= entry_ref - req_dist + eps) and (tp >= entry_ref + req_dist - eps)
                if not ok_side:
                    return _err("Constraints not satisfied (BUY)")
        else:
            ok_side = (sl >= entry_ref + req_dist - eps) and (tp <= entry_ref - req_dist + eps)
            if not ok_side:
                tp = _round_to_tick(entry_ref - max(tp_dist0, rr_target * sl_dist, req_dist), tick)
                ok_side = (sl >= entry_ref + req_dist - eps) and (tp <= entry_ref - req_dist + eps)
                if not ok_side:
                    return _err("Constraints not satisfied (SELL)")
        return _ok({
            "entry_ref": _round_to_tick(entry_ref, tick),
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "meta": {
                "tick": float(tick),
                "digits_guess": int(digits_guess),
                "atr_ltf": float(atr_ltf),
                "atr_basis": float(atr_basis),
                "min_stop_price": float(min_stop_price),
                "spread_buffer": float(spread_buffer),
                "struct_floor": float(struct_floor),
                "sl_dist_final": float(sl_dist),
                "tp_dist_final": float(tp_dist),
                "rr_target": float(rr_target),
                "horizon": horizon,
                "risk_level": risk_level,
                "action": action_up,
                "fib_used": fib_used,
                "fib_tp_raw": (float(fib_tp_raw) if fib_tp_raw is not None else None),
                "anchor_tf": anchor_tf,
                "eps": float(eps),
            },
        })
    except Exception as e:
        return _err("levels_autonomous_from_df failed", exc=str(e))


# ---- Plan / Levels (JSON only) ----

@mcp.tool()
def plan_raw_from_json(ohlcv_json: str, risk_level: str = "medium", direction: str = "auto", tick_size: float = 0.0, horizon: str = "scalping") -> str:
    """plan_from_json"""
    try:
        src = _parse_ohlcv_json(ohlcv_json)
        if not src:
            return _err("Invalid OHLCV json")
        df = _df_from_ohlcv(src)
        if df is None or df.empty:
            return _err("Invalid OHLCV")
        df = _compute_indicators_df(df)
        last = _last_row(df)
        atr_val, close = last.get("ATR"), last.get("Close")
        if atr_val is None or close is None or not np.isfinite(atr_val) or not np.isfinite(close):
            return _err("ATR/Close missing")
        side = (direction or "auto").lower()
        if side not in {"long", "short"}:
            ef = float(last.get("EMA_Fast") or 0.0)
            es = float(last.get("EMA_Slow") or 0.0)
            rsi_v = float(last.get("RSI") or 50.0)
            ml = float(last.get("MACD_Line") or 0.0)
            ms = float(last.get("MACD_Signal") or 0.0)
            bull = (ef >= es) and (rsi_v >= CONFIG["DEC_RSI_POS"] or ml >= ms)
            bear = (ef < es) and (rsi_v <= CONFIG["DEC_RSI_NEG"] or ml <= ms)
            side = "long" if (bull and not bear) else ("short" if (bear and not bull) else ("long" if ef >= es else "short"))
        prices = pd.concat([df["Open"], df["High"], df["Low"], df["Close"]], ignore_index=True)
        digits_guess = _infer_digits_from_prices(prices)
        tick = tick_size if tick_size > 0 else (_infer_tick_from_prices(prices, digits_guess) or (10 ** (-max(digits_guess or 3, 3))))
        point = float(tick)
        prof = CONFIG["PROFILES"].get(horizon.lower(), CONFIG["PROFILES"]["swing"])
        min_stop_ticks = int(prof.get("min_ticks", CONFIG["IND_MIN_STOP_TICKS_BASE"]))
        if atr_val and point > 0:
            dyn_ticks = int(math.ceil(CONFIG["IND_MINSTOP_ATR_FRAC"] * float(atr_val) / point))
            min_stop_ticks = max(min_stop_ticks, dyn_ticks)
        min_stop_price = float(min_stop_ticks * point)
        spread_frac = float(prof.get("spread_frac", 0.08))
        spread_buffer = max(2.0 * point, spread_frac * min_stop_price)
        req_dist = float(min_stop_price + spread_buffer)
        tp_mult_map = {"low": 1.5, "medium": 2.0, "high": 3.0}
        tp_mult = float(tp_mult_map.get(risk_level.lower(), 2.0))
        sl_dist = max(1.0 * float(atr_val), req_dist)
        tp_dist = max(tp_mult * float(atr_val), 1.5 * sl_dist, req_dist)
        if side == "long":
            sl = _round_to_tick(close - sl_dist, tick); tp = _round_to_tick(close + tp_dist, tick)
        else:
            sl = _round_to_tick(close + sl_dist, tick); tp = _round_to_tick(close - tp_dist, tick)
        entry = _round_to_tick(close, tick)
        if side == "long":
            if not (sl is not None and tp is not None and sl <= entry - req_dist and tp >= entry + req_dist):
                tp = _round_to_tick(entry + max(tp_dist, req_dist, 1.5 * sl_dist), tick)
                if not (sl is not None and tp is not None and sl <= entry - req_dist and tp >= entry + req_dist):
                    return _err("Constraints not satisfied after rounding (LONG)", entry=entry, sl=sl, tp=tp, req=req_dist, tick=tick)
        else:
            if not (sl is not None and tp is not None and sl >= entry + req_dist and tp <= entry - req_dist):
                tp = _round_to_tick(entry - max(tp_dist, req_dist, 1.5 * sl_dist), tick)
                if not (sl is not None and tp is not None and sl >= entry + req_dist and tp <= entry - req_dist):
                    return _err("Constraints not satisfied after rounding (SHORT)", entry=entry, sl=sl, tp=tp, req=req_dist, tick=tick)
        return _ok({
            "entry": entry, "sl": sl, "tp": tp, "atr": float(atr_val),
            "risk_level": risk_level, "side": side,
            "meta": {
                "tick": float(tick), "digits_guess": int(digits_guess),
                "min_stop_price": float(min_stop_price), "spread_buffer": float(spread_buffer),
                "req_dist": float(req_dist), "horizon": horizon,
            }
        })
    except Exception as e:
        return _err("plan_raw_from_json failed", exc=str(e))

@mcp.tool()
def levels_autonomous_from_json(ohlcv_json: str, action: str, horizon: str, risk_level: str, use_fib: bool = True, fib_atr_mult: float = 2.0, anchor_tf: str = "auto", htf_ohlcv_json: str = "") -> str:
    """levels_from_json"""
    try:
        src = _parse_ohlcv_json(ohlcv_json)
        if not src:
            return _err("Invalid OHLCV json")
        raw_df = _df_from_ohlcv(src)
        params = _resolve_indicator_params()
        df = _ensure_indicator_df(raw_df, params)
        if df is None or len(df) < 20:
            return _err("Invalid or too short OHLCV")
        htf_df = None
        if htf_ohlcv_json:
            htf_src = _parse_ohlcv_json(htf_ohlcv_json)
            if htf_src:
                htf_raw = _df_from_ohlcv(htf_src)
                htf_df = _ensure_indicator_df(htf_raw, params)
        return _levels_autonomous_from_df(df, action, horizon, risk_level, use_fib, fib_atr_mult, anchor_tf, htf_df)
    except Exception as e:
        return _err("levels_autonomous_from_json failed", exc=str(e))


@mcp.tool()
def levels_autonomous(cache_key: str, action: str, horizon: str, risk_level: str, use_fib: bool = True, fib_atr_mult: float = 2.0, anchor_tf: str = "auto", htf_cache_key: str = "") -> str:
    """levels_from_cache"""
    try:
        params = _resolve_indicator_params()
        df = _df_from_cache(cache_key, params)
        if df is None or len(df) < 20:
            return _err("Invalid or too short OHLCV", cache_key=cache_key)
        htf_df = None
        if htf_cache_key:
            htf_df = _df_from_cache(htf_cache_key, params)
        return _levels_autonomous_from_df(df, action, horizon, risk_level, use_fib, fib_atr_mult, anchor_tf, htf_df)
    except Exception as e:
        return _err("levels_autonomous failed", exc=str(e), cache_key=cache_key)

# ---- Price-Action fallback (Box + M1 sweep + M5 confirm) ----
def _price_action_fallback(
    symbol: str,
    inter: str,
    ltf: pd.DataFrame,
    ltf_last: Dict[str, Any],
    equity: float,
    risk_pct: float,
    cap_leverage: float,
    risk_level: str,
    vol_meta: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    try:
        if not _as_bool(os.getenv("PRICE_ACTION_FALLBACK", "false")):
            return None
        allow_on_vgate = _as_bool(os.getenv("PRICE_ACTION_ON_VGATE", "false"))

        # Box parameters
        is_15m = (inter == "15m")
        lookback_m15 = int(os.getenv("BOX_LOOKBACK_M15", "6"))
        lookback_m5 = int(os.getenv("BOX_LOOKBACK_M5", "18"))
        n_box = lookback_m15 if is_15m else lookback_m5
        if len(ltf) < max(40, n_box + 5):
            return None

        # Tick and box
        tick = float(ltf_last.get("TickSize_Guess") or 0.0)
        if not (tick and np.isfinite(tick) and tick > 0):
            prices = pd.concat([ltf["Open"], ltf["High"], ltf["Low"], ltf["Close"]], ignore_index=True)
            from analysis_core import _infer_digits_from_prices as _idp, _infer_tick_from_prices as _itp
            d = _idp(prices)
            tick = float(_itp(prices, d) or (10 ** (-max(d or 3, 3))))
        box_min_ticks = int(os.getenv("BOX_MIN_TICKS", "40"))
        box_max_ticks = int(os.getenv("BOX_MAX_TICKS", "400"))
        sweep_buf_ticks = int(os.getenv("SWEEP_BUFFER_TICKS", "5"))
        setup_window_min = int(os.getenv("SETUP_WINDOW_MIN", "30"))
        retest_bars_5m = int(os.getenv("RETEST_BARS_5M", "2"))
        rr_target = float(os.getenv("RR_TARGET", "1.6"))
        tp_mult = float(os.getenv("BOX_TP_MULT", "1.0"))

        tail = ltf.tail(max(n_box, 6)).copy()
        box_hi = float(tail["High"].max())
        box_lo = float(tail["Low"].min())
        box_h = float(max(0.0, box_hi - box_lo))
        if box_h <= 0:
            return None
        box_ticks = int(round(box_h / tick)) if tick else 0
        if box_ticks < box_min_ticks or box_ticks > box_max_ticks:
            return None

        # Ultra LTF data
        params = _resolve_indicator_params()
        ck5 = _ensure_cached_ohlcv(symbol, f"{CONFIG['LTF_PERIOD_5M']}d", "5m")
        df5 = _df_from_cache(ck5, params)
        ck1 = _ensure_cached_ohlcv(symbol, "1d", "1m")
        df1 = _df_from_cache(ck1, params)
        if df5 is None or df5.empty or df1 is None or df1.empty:
            return None
        last5 = _last_row(df5)
        ema_up_5m = (last5.get("EMA_Fast") or 0) >= (last5.get("EMA_Slow") or 0)

        # Sweeps within window on M1
        now_ts = pd.Timestamp.utcnow().tz_localize(None)
        cutoff = now_ts - pd.Timedelta(minutes=setup_window_min)
        d1w = df1[df1["Date"] >= cutoff].copy()
        buf = sweep_buf_ticks * tick
        def _swept_down(row) -> bool:
            try:
                return (float(row["Low"]) <= box_lo - buf) and (float(row["Close"]) >= box_lo)
            except Exception:
                return False
        def _swept_up(row) -> bool:
            try:
                return (float(row["High"]) >= box_hi + buf) and (float(row["Close"]) <= box_hi)
            except Exception:
                return False
        has_sweep_down = any(_swept_down(r) for _, r in d1w.iterrows())
        has_sweep_up = any(_swept_up(r) for _, r in d1w.iterrows())

        # M5 retest of boundary
        d5n = df5.tail(max(2, retest_bars_5m)).copy()
        retest_lo = (d5n["Low"].min() <= (box_lo + buf))
        retest_hi = (d5n["High"].max() >= (box_hi - buf))

        ltf_up = (ltf_last.get("EMA_Fast") or 0) >= (ltf_last.get("EMA_Slow") or 0)
        want_buy = has_sweep_down and ema_up_5m and retest_lo and ltf_up
        want_sell = has_sweep_up and (not ema_up_5m) and retest_hi and (not ltf_up)
        if not (want_buy or want_sell):
            return None

        entry_ref = float(ltf_last.get("Close") or 0.0)
        if want_buy:
            sl = _round_to_tick(box_lo - buf, tick)
            sl_dist = float(entry_ref - (sl or entry_ref))
            tp_box = entry_ref + tp_mult * box_h
            tp_rr  = entry_ref + rr_target * max(sl_dist, tick)
            tp = _round_to_tick(max(tp_box, tp_rr), tick)
            action = "BUY"
        else:
            sl = _round_to_tick(box_hi + buf, tick)
            sl_dist = float((sl or entry_ref) - entry_ref)
            tp_box = entry_ref - tp_mult * box_h
            tp_rr  = entry_ref - rr_target * max(sl_dist, tick)
            tp = _round_to_tick(min(tp_box, tp_rr), tick)
            action = "SELL"

        # Spread gating similar to scalping
        tp_spread_ratio = None
        try:
            raw = meta_api.get_current_price(symbol)
            data_or_raw = json.loads(raw) if isinstance(raw, str) else raw
            q = data_or_raw.get("data") if isinstance(data_or_raw, dict) and "data" in data_or_raw else data_or_raw
            if isinstance(q, dict):
                bid = q.get("bid") or q.get("Bid")
                ask = q.get("ask") or q.get("Ask")
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask > bid:
                    spread = float(ask - bid)
                    if action == "BUY" and isinstance(tp, (int, float)):
                        reward = max(0.0, float(tp) - entry_ref)
                    elif action == "SELL" and isinstance(tp, (int, float)):
                        reward = max(0.0, entry_ref - float(tp))
                    else:
                        reward = 0.0
                    if spread > 0 and reward > 0:
                        tp_spread_ratio = reward / spread
                        band = (vol_meta or {}).get("band")
                        try:
                            min_norm = float(os.getenv("TP_SPREAD_MIN_SCALP", "3.0"))
                            min_high = float(os.getenv("TP_SPREAD_MIN_HIGH_SCALP", "5.0"))
                        except Exception:
                            min_norm, min_high = 3.0, 5.0
                        need = min_high if band == "HIGH" else min_norm
                        if tp_spread_ratio < need and not allow_on_vgate:
                            return None
        except Exception:
            pass

        size = _position_size(entry=entry_ref, sl=sl, equity=equity, risk_pct=risk_pct, cap_leverage=cap_leverage)
        size_factor = (vol_meta or {}).get("size_factor", 1.0)
        if size_factor < 1.0 and size.get("units", 0) > 0:
            size["units"] = float(size["units"]) * float(size_factor)
            size["size_factor_vol"] = float(size_factor)

        levels = {
            "entry_ref": _round_to_tick(entry_ref, tick),
            "sl": sl,
            "tp": tp,
            "rr": float(abs((tp - entry_ref) / max(abs(entry_ref - sl), tick))) if (tp is not None and sl is not None) else None,
            "meta": {
                "tick": float(tick),
                "box_high": box_hi,
                "box_low": box_lo,
                "box_height": box_h,
                "box_ticks": box_ticks,
                "sweep_down": bool(has_sweep_down),
                "sweep_up": bool(has_sweep_up),
                "retest_lo": bool(retest_lo),
                "retest_hi": bool(retest_hi),
            },
            "source": "price_action_fallback",
        }

        return _ok({
            "symbol": symbol, "interval": inter, "regime": "trend",
            "decision": {"action": action, "entry": levels["entry_ref"], "sl": sl, "tp": tp, "confidence": 60, "risk_level": risk_level},
            "levels": levels,
            "position": size,
            "reason": f"Price-action fallback: box({n_box}) sweep + M5 confirm; band={(vol_meta or {}).get('band','NA')}",
            "volatility": vol_meta,
            "tp_vs_spread_ratio": tp_spread_ratio,
        })
    except Exception as _e:
        logger.warning(f"[MCP:ANALYSIS] price-action fallback error: {_e}")
        return None

# ---- Intraday decision (primitifs only) ----

@mcp.tool()
def intraday_decision(symbol: str, interval: str = "15m", equity: float = 10000.0, risk_pct: float = 0.005, cap_leverage: float = 5.0, risk_level: str = "medium", vol_enabled: bool = True, lookback_days: int = 0, vol_low_pct: int = 0, vol_high_pct: int = 0, vol_extreme_pct: int = 0, vol_size_high: float = 0.0, require_htf_on_edges: bool = False, trend_only: bool = False) -> str:
    """intraday"""
    try:
        inter = (interval or "15m").lower()
        if inter not in {"5m", "15m"}:
            inter = "15m"
        # Default to CONFIG value if not explicitly requested
        try:
            if not require_htf_on_edges:
                require_htf_on_edges = bool(CONFIG.get("REQUIRE_HTF_ON_EDGES", False))
        except Exception:
            pass
        period_ltf = f"{CONFIG['LTF_PERIOD_5M']}d" if inter == "5m" else f"{CONFIG['LTF_PERIOD_15M']}d"
        period_htf = f"{CONFIG['HTF_PERIOD_DAYS']}d"
        interval_htf = CONFIG["HTF_INTERVAL"]
        lb = lookback_days or CONFIG["VOL_LOOKBACK_DAYS"]
        lp = vol_low_pct or CONFIG["VOL_LOW_PCT"]
        hp = vol_high_pct or CONFIG["VOL_HIGH_PCT"]
        ep = vol_extreme_pct or CONFIG["VOL_EXTREME_PCT"]
        sh = vol_size_high or CONFIG["VOL_SIZE_HIGH"]
        ltf, htf = _fetch_and_features(symbol, period_ltf, inter, period_htf, interval_htf)
        last_ltf = _last_row(ltf)
        last_htf = _last_row(htf)
        vol_meta = None
        vgate_denied = False
        if vol_enabled:
            bands = _atr_pct_bands_from_df(ltf, lookback_days=lb, low_pct=lp, high_pct=hp, extreme_pct=ep)
            if bands is not None:
                allowed, band, size_factor, reason_code = _volatility_gate(bands["atr_now"], bands["p10"], bands["p90"], bands.get("p95"), size_high=sh)
                # Option: autoriser des entrées en bande LOW avec réduction de taille
                try:
                    allow_low = str(os.getenv("VOL_ALLOW_LOW_ENTRIES", "false")).strip().lower() in {"1","true","yes","on"}
                except Exception:
                    allow_low = False
                if (band == "LOW") and allow_low:
                    allowed = True
                    size_factor = min(size_factor or 1.0, 0.5)
                    reason_code = "ATR_LOW_SIZE_DOWN"
                vol_meta = {"atr_pct_now": bands["atr_now"], "p10": bands["p10"], "p90": bands["p90"], "p95": bands.get("p95"), "band": band, "size_factor": size_factor, "reason": reason_code}
                if not allowed:
                    vgate_denied = True
                    if not _as_bool(os.getenv("PRICE_ACTION_ON_VGATE", "false")):
                        return _ok({
                            "symbol": symbol, "interval": inter, "regime": "no-trade",
                            "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": 0, "risk_level": risk_level},
                            "reason": f"Volatility gate {band} ({reason_code})", "volatility": vol_meta,
                        })
        regime = _regime_from_df(ltf, htf)
        if trend_only and regime != "trend":
            return _ok({
                "symbol": symbol, "interval": inter, "regime": regime,
                "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": 0, "risk_level": risk_level},
                "reason": "Filtre trend-only", "volatility": vol_meta,
            })
        if regime == "no-trade":
            return _ok({
                "symbol": symbol, "interval": inter, "regime": regime,
                "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": 0, "risk_level": risk_level},
                "reason": "Régime no-trade", "volatility": vol_meta,
            })
        score = _directional_score(last_ltf)
        conf = _confidence(last_ltf, score)
        ltf_up = (last_ltf.get("EMA_Fast") or 0) >= (last_ltf.get("EMA_Slow") or 0)
        htf_up = (last_htf.get("EMA_Fast") or 0) >= (last_htf.get("EMA_Slow") or 0)
        action = "HOLD"
        if regime == "trend":
            if score >= CONFIG["DEC_TREND_BUY_SCORE"] and ltf_up and htf_up:
                action = "BUY"
            elif score <= CONFIG["DEC_TREND_SELL_SCORE"] and (not ltf_up) and (not htf_up):
                action = "SELL"
        else:
            if score <= CONFIG["DEC_RANGE_SELL_SCORE"] and (not htf_up):
                action = "SELL"
            elif score >= CONFIG["DEC_RANGE_BUY_SCORE"] and htf_up:
                action = "BUY"
        if vol_meta and vol_meta["band"] == "HIGH" and require_htf_on_edges:
            if (action == "BUY" and not (ltf_up and htf_up)) or (action == "SELL" and not ((not ltf_up) and (not htf_up))):
                action = "HOLD"
        horizon = "scalping" if inter in {"5m", "15m"} else "swing"
        # Early-bar gating: éviter les 1ères secondes de la nouvelle bougie (plus de faux signaux)
        try:
            early_block = str(os.getenv("EARLY_BAR_BLOCK", "true")).strip().lower() in {"1","true","yes","on"}
        except Exception:
            early_block = True
        if early_block and (action != "HOLD") and horizon == "scalping":
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            step_min = 5 if inter == "5m" else 15
            secs_in_bar = (now.minute % step_min) * 60 + now.second
            try:
                block_secs = int(os.getenv("EARLY_BLOCK_SECS_5M", "60")) if step_min == 5 else int(os.getenv("EARLY_BLOCK_SECS_15M", "120"))
            except Exception:
                block_secs = 60 if step_min == 5 else 120
            if secs_in_bar <= max(10, block_secs):
                return _ok({
                    "symbol": symbol, "interval": inter, "regime": regime,
                    "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": conf, "risk_level": risk_level},
                    "reason": f"Early-bar block: {secs_in_bar}s into {step_min}m bar",
                    "volatility": vol_meta,
                })
        min_conf = CONFIG["PROFILES"].get(horizon, {}).get("min_confidence", None)
        if action != "HOLD" and isinstance(min_conf, (int, float)) and conf < float(min_conf):
            action = "HOLD"
        if action == "HOLD":
            pa = _price_action_fallback(
                symbol=symbol, inter=inter, ltf=ltf, ltf_last=last_ltf,
                equity=equity, risk_pct=risk_pct, cap_leverage=cap_leverage,
                risk_level=risk_level, vol_meta=vol_meta,
            )
            if pa:
                return pa
            return _ok({
                "symbol": symbol, "interval": inter, "regime": regime,
                "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": conf, "risk_level": risk_level},
                "reason": f"Score={score:.2f} / Confluence insuffisante.", "volatility": vol_meta,
            })
        # Micro-trend gating for scalping: confirm 15m with 5m and 1m (optional via env MICRO_TREND_CHECK)
        # BUY requires both 5m and 1m EMAs aligned up; SELL requires both aligned down.
        if (action != "HOLD") and (inter in {"15m", "5m"}):
            try:
                micro_check = str(os.getenv("MICRO_TREND_CHECK", "true")).strip().lower() in {"1","true","yes","on"}
            except Exception:
                micro_check = True
            if micro_check:
                params_ind = _resolve_indicator_params()
                micro = {"checked": [], "ema_up": {}}
                try:
                    # Always try 5m if main is 15m
                    if inter == "15m":
                        ck5 = _ensure_cached_ohlcv(symbol, f"{CONFIG['LTF_PERIOD_5M']}d", "5m")
                        df5 = _df_from_cache(ck5, params_ind)
                        if df5 is not None and not df5.empty:
                            last5 = _last_row(df5)
                            up5 = (last5.get("EMA_Fast") or 0) >= (last5.get("EMA_Slow") or 0)
                            micro["checked"].append("5m")
                            micro["ema_up"]["5m"] = bool(up5)
                        else:
                            micro["ema_up"]["5m"] = None
                    # Always try 1m as ultra-LTF confirmation
                    try:
                        ck1 = _ensure_cached_ohlcv(symbol, "1d", "1m")
                        df1 = _df_from_cache(ck1, params_ind)
                        if df1 is not None and not df1.empty:
                            last1 = _last_row(df1)
                            up1 = (last1.get("EMA_Fast") or 0) >= (last1.get("EMA_Slow") or 0)
                            micro["checked"].append("1m")
                            micro["ema_up"]["1m"] = bool(up1)
                        else:
                            micro["ema_up"]["1m"] = None
                    except Exception:
                        micro["ema_up"]["1m"] = None
                except Exception:
                    micro = None

                def _micro_all_up(mi: dict) -> Optional[bool]:
                    if not mi: return None
                    v5 = mi["ema_up"].get("5m")
                    v1 = mi["ema_up"].get("1m")
                    require_both = _as_bool(os.getenv("MICRO_REQUIRE_BOTH", "true"))
                    if inter == "15m":
                        if require_both:
                            return (v5 is True) and (v1 is True)
                        votes = [ltf_up, v5 is True, v1 is True]
                        return sum(1 for v in votes if v) >= 2
                    # inter == "5m": require only 1m
                    return (v1 is True)

                def _micro_all_down(mi: dict) -> Optional[bool]:
                    if not mi: return None
                    v5 = mi["ema_up"].get("5m")
                    v1 = mi["ema_up"].get("1m")
                    require_both = _as_bool(os.getenv("MICRO_REQUIRE_BOTH", "true"))
                    if inter == "15m":
                        if require_both:
                            return (v5 is False) and (v1 is False)
                        votes = [not ltf_up, v5 is False, v1 is False]
                        return sum(1 for v in votes if v) >= 2
                    return (v1 is False)

                misaligned = False
                if action == "BUY":
                    ok_micro = _micro_all_up(micro)
                    if ok_micro is False:
                        misaligned = True
                elif action == "SELL":
                    ok_micro = _micro_all_down(micro)
                    if ok_micro is False:
                        misaligned = True

                if misaligned:
                    return _ok({
                        "symbol": symbol, "interval": inter, "regime": regime,
                        "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": conf, "risk_level": risk_level},
                        "reason": f"Micro-trend misaligned on 5m/1m for action={action}",
                        "volatility": vol_meta,
                        "micro_trend": micro,
                    })

        # If volatility gate denied but allowed to use price-action fallback, try it now
        if vgate_denied and _as_bool(os.getenv("PRICE_ACTION_ON_VGATE", "false")):
            pa = _price_action_fallback(
                symbol=symbol, inter=inter, ltf=ltf, ltf_last=last_ltf,
                equity=equity, risk_pct=risk_pct, cap_leverage=cap_leverage,
                risk_level=risk_level, vol_meta=vol_meta,
            )
            if pa:
                return pa

        d1_df = None
        # Skip daily context fetch for scalping to reduce overhead; keep for higher horizons
        if not (inter in {"5m", "15m"}):
            try:
                d1_raw = meta_api.get_historical_candles(symbol, f"{CONFIG['D1_PERIOD_MONTHS']}mo", "1d")
                if isinstance(d1_raw, str):
                    d1_payload = json.loads(d1_raw)
                    d1_data = d1_payload["data"] if isinstance(d1_payload, dict) and "data" in d1_payload else d1_payload
                else:
                    d1_data = d1_raw["data"] if isinstance(d1_raw, dict) and "data" in d1_raw else d1_raw
                d1_df = _ensure_indicator_df(_df_from_ohlcv(d1_data), _resolve_indicator_params())
                if d1_df is not None and len(d1_df) > 200:
                    d1_df = d1_df.tail(200).copy()
            except Exception:
                d1_df = None
        # Paramètres de niveaux adaptés au scalping 15m: fibs désactivés et ancrage LTF
        use_fib_flag = False if horizon == "scalping" else True
        fib_mult = 1.5 if horizon == "scalping" else 2.0
        anchor = "ltf" if horizon == "scalping" else "htf"
        levels_json = _levels_autonomous_from_df(
            ltf.tail(300).copy(),
            action=action,
            horizon=horizon,
            risk_level=risk_level,
            use_fib=use_fib_flag,
            fib_atr_mult=fib_mult,
            anchor_tf=anchor,
            htf_df=d1_df,
        )
        levels = json.loads(levels_json)
        if not levels.get("ok"):
            return _err("levels_autonomous_from_json failed inside intraday_decision", inner=levels)
        lv = levels["data"]
        entry = lv.get("entry_ref") or last_ltf.get("Close")
        sl = lv.get("sl")
        tp = lv.get("tp")
        # Spread gating (scalping): exiger un TP à >= k×spread (k=5 en band HIGH, sinon 3 par défaut)
        tp_spread_ratio = None
        if horizon == "scalping":
            try:
                raw = meta_api.get_current_price(symbol)
                data_or_raw = json.loads(raw) if isinstance(raw, str) else raw
                q = data_or_raw.get("data") if isinstance(data_or_raw, dict) and "data" in data_or_raw else data_or_raw
                if isinstance(q, dict):
                    bid = q.get("bid") or q.get("Bid")
                    ask = q.get("ask") or q.get("Ask")
                    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask > bid:
                        spread = float(ask - bid)
                        entry_ref = float(lv.get("entry_ref") or last_ltf.get("Close") or 0.0)
                        if action == "BUY" and isinstance(tp, (int, float)):
                            reward = max(0.0, float(tp) - entry_ref)
                        elif action == "SELL" and isinstance(tp, (int, float)):
                            reward = max(0.0, entry_ref - float(tp))
                        else:
                            reward = 0.0
                        if spread > 0 and reward > 0:
                            tp_spread_ratio = reward / spread
                            band = (vol_meta or {}).get("band")
                            try:
                                min_norm = float(os.getenv("TP_SPREAD_MIN_SCALP", "3.0"))
                                min_high = float(os.getenv("TP_SPREAD_MIN_HIGH_SCALP", "5.0"))
                            except Exception:
                                min_norm, min_high = 3.0, 5.0
                            need = min_high if band == "HIGH" else min_norm
                            if tp_spread_ratio < need:
                                return _ok({
                                    "symbol": symbol, "interval": inter, "regime": regime,
                                    "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": conf, "risk_level": risk_level},
                                    "reason": f"tp_vs_spread too small: {tp_spread_ratio:.2f} < {need}",
                                    "volatility": vol_meta,
                                })
            except Exception:
                pass
        size = _position_size(entry=lv.get("entry_ref") or last_ltf.get("Close"), sl=sl, equity=equity, risk_pct=risk_pct, cap_leverage=cap_leverage)
        size_factor = (vol_meta or {}).get("size_factor", 1.0)
        if size_factor < 1.0 and size.get("units", 0) > 0:
            size["units"] = float(size["units"]) * float(size_factor)
            size["size_factor_vol"] = float(size_factor)
        reason = (
            f"Regime={regime}, Score={score:.2f}, EMA_LTF={'up' if ltf_up else 'down'}, EMA_HTF={'up' if htf_up else 'down'}; "
            f"ATR%={round(100.0*(last_ltf.get('ATR_PCT') or 0.0), 4)}; "
            f"Tick={round(lv['meta'].get('tick') or 0, 6)}; "
            f"MinStop≈{round(lv['meta'].get('min_stop_price') or 0, 6)}; "
            f"Buffer≈{round(lv['meta'].get('spread_buffer') or 0, 6)}; "
            f"Struct≈{round(lv['meta'].get('struct_floor') or 0, 6)}; FibUsed={lv['meta'].get('fib_used')}; "
            f"VolBand={(vol_meta or {}).get('band', 'NA')}"
        )
        return _ok({
            "symbol": symbol, "interval": inter, "regime": regime,
            "decision": {"action": action, "entry": entry, "sl": sl, "tp": tp, "confidence": conf, "risk_level": risk_level},
            "levels": lv, "position": size,
            "management": {
                "move_be_at_R": CONFIG["MANAGE_MOVE_BE_AT_R"],
                "partial_exit_at_R": CONFIG["MANAGE_PARTIAL_AT_R"],
                "partial_fraction": CONFIG["MANAGE_PARTIAL_FRAC"],
                "trail_at_R": CONFIG["MANAGE_TRAIL_AT_R"],
                "trail_type": CONFIG["MANAGE_TRAIL_TYPE"],
                "trail_len": CONFIG["MANAGE_TRAIL_LEN"],
                "time_stop_bars": CONFIG["MANAGE_TIME_STOP_BARS"],
            },
            "reason": reason, "volatility": vol_meta,
            "tp_vs_spread_ratio": tp_spread_ratio,
        })
    except Exception as e:
        return _err("intraday_decision failed", exc=str(e))

# ===================== Prompt =====================
# ================================================================
# --------------------- LLM Prompt (updated) ---------------------
# ================================================================

PROMPT_TMPL = Template(r"""
Tu es quant trader senior. Produis un seul JSON final pour $symbol ($horizon).

Pipeline (ordre strict) :
1. get_historical_candles("$symbol", "$period", "$interval", compact=true) → cache_key_ltf (base 15m).
2. get_historical_candles("$symbol", f"{CONFIG['MICRO_PERIOD_5M_DAYS']}d", "5m", compact=true) → cache_key_5m.
3. get_historical_candles("$symbol", f"{CONFIG['MICRO_PERIOD_1M_DAYS']}d", "1m", compact=true) → cache_key_1m.
4. compute_indicators(cache_key=cache_key_ltf) → dernière ligne (Close, EMA, RSI, MACD, BB, ATR, ATR_PCT, BBW_PCT, TickSize, Digits, MinStopPrice_Fallback).
5. compute_indicators(cache_key=cache_key_5m) et compute_indicators(cache_key=cache_key_1m) → au minimum EMA_Fast/EMA_Slow pour direction micro.
6. volatility_bands(cache_key=cache_key_ltf, lookback_days=${lookback}, low_pct=${lowp}, high_pct=${highp}, extreme_pct=${extp}, size_high=${sizeh}) → atr_pct_now, band, size_factor, reason.
7. Décision :
   - Volatilité : band ∈ {LOW, EXTREME} ⇒ action="HOLD". band == HIGH ⇒ action possible mais mentionne size_factor dans la sortie.
   - Régime : no-trade si ATR_PCT > ${atr_high} ou (ATR_PCT < ${atr_low} et BBW_PCT < ${bbw_squeeze}); sinon trend si BBW_PCT ≥ ${bbw_trend} et EMAs alignées LTF/HTF (si HTF présent), sinon range.
   - Score = ${w_ema} (EMA), ${w_macd} (MACD), ${w_rsi} (RSI seuils ${rsi_pos}/${rsi_neg}), ${w_bb} (BB mid). BUY si score ≥ ${trend_buy} avec confluence HTF (si présent), SELL si score ≤ ${trend_sell} avec confluence, sinon HOLD.
   - Micro-confluence (scalping 15m) : exiger 5m **et** 1m alignés avec la direction LTF; sinon action="HOLD". Option : si CONFIG["MICRO_REQUIRE_BOTH"]=False, appliquer une règle 2/3 (15m/5m/1m).

8. Niveaux :
   - Si HOLD ⇒ entry/sl/tp = null.
   - Si action ≠ HOLD :
       • Scalping (15m) : levels_autonomous(cache_key=cache_key_ltf, action, horizon="$horizon", risk_level="$risk_level", use_fib=false, fib_atr_mult=1.5, anchor_tf="ltf", htf_cache_key="") — **ignorer totalement le 1D**.
       • Autres horizons : levels_autonomous(cache_key=cache_key_ltf, action, horizon="$horizon", risk_level="$risk_level", use_fib=true, fib_atr_mult=2.0, anchor_tf="htf", htf_cache_key="<htf_cache>" si disponible).
   - Si <240 barres ou indicateurs NaN : rallonge la période et recommence. Si malgré tout impossible, fournis quand même sl/tp avec confidence=10.

Confiance : `round((abs(score_total)/4)*(1-clamp((ATR_PCT-${conf_start})/${conf_range},0,${conf_cap}))*100)`.

Réponds UNIQUEMENT par l’objet JSON demandé (pas de texte hors JSON).
{
  "symbol": "$symbol",
  "horizon": "$horizon",
  "decision": {"action": "<BUY|SELL|HOLD>", "entry": <number|null>, "sl": <number|null>, "tp": <number|null>, "confidence": <number>, "risk_level": "$risk_level"},
  "reason": "<résumé concis en FR>",
  "regime": "<trend|range|no-trade>",
  "volatility": {"atr_pct_now": <number>, "p10": <number>, "p90": <number>, "p95": <number|null>, "band": "<LOW|NORMAL|HIGH|EXTREME>", "size_factor": <number>, "reason": "<ATR_OK|ATR_HIGH_SIZE_DOWN|ATR_GATE_LOW|ATR_GATE_HIGH>"}
}
""")

@mcp.prompt()
def analysis_agent(
    symbol: str,
    period: str = "1mo",
    interval: str = "15m",
    horizon: str = "swing",
    risk_level: str = "medium",
) -> str:
    return PROMPT_TMPL.substitute(
        symbol=symbol,
        period=period,
        interval=interval,
        horizon=horizon,
        risk_level=risk_level,
        lookback=CONFIG["VOL_LOOKBACK_DAYS"],
        lowp=CONFIG["VOL_LOW_PCT"],
        highp=CONFIG["VOL_HIGH_PCT"],
        extp=CONFIG["VOL_EXTREME_PCT"],
        sizeh=CONFIG["VOL_SIZE_HIGH"],
        atr_high=CONFIG["REGIME_ATR_HIGH"],
        atr_low=CONFIG["REGIME_ATR_LOW"],
        bbw_squeeze=CONFIG["REGIME_SQUEEZE_BBW"],
        bbw_trend=CONFIG["REGIME_BBW_TREND"],
        w_ema=CONFIG["SCORE_W_EMA"],
        w_macd=CONFIG["SCORE_W_MACD"],
        w_rsi=CONFIG["SCORE_W_RSI"],
        rsi_pos=CONFIG["DEC_RSI_POS"],
        rsi_neg=CONFIG["DEC_RSI_NEG"],
        w_bb=CONFIG["SCORE_W_BBPOS"],
        trend_buy=CONFIG["DEC_TREND_BUY_SCORE"],
        trend_sell=CONFIG["DEC_TREND_SELL_SCORE"],
        conf_start=CONFIG["CONF_DAMP_START"],
        conf_range=CONFIG["CONF_DAMP_RANGE"],
        conf_cap=CONFIG["CONF_DAMP_CAP"],
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")
