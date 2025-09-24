import os, sys, json, math
from typing import Any, Dict, List, Tuple

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

mcp = FastMCP("Trading Analysis MCP Server", log_level="WARNING")
logger.remove()
logger.add(sys.stderr, level="WARNING")

# ===================== Cache OHLCV =====================

CANDLE_CACHE: Dict[str, List[Dict[str, Any]]] = {}

def _make_cache_key(symbol: str, period: str, interval: str) -> str:
    return f"{symbol}|{period}|{interval}".lower()

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
    try:
        data = meta_api.get_historical_candles(symbol, period, interval)
        if isinstance(data, str):
            data = json.loads(data)
        ohlcv = data.get("data") if isinstance(data, dict) and "data" in data else data
        if not isinstance(ohlcv, list) or not ohlcv:
            return _err("fetch returned empty data", symbol=symbol, period=period, interval=interval)
        cache_key = _make_cache_key(symbol, period, interval)
        CANDLE_CACHE[cache_key] = ohlcv
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
        src = CANDLE_CACHE.get(cache_key)
        if src is None:
            return _err("cache_key not found", cache_key=cache_key)
        if tail <= 0:
            tail = 200
        df = _df_from_ohlcv(src)
        if df is None:
            return _err("Invalid OHLCV")
        # Remplacement des 0 par valeurs CONFIG
        rsi = rsi_len or CONFIG["RSI_LEN"]
        ef = ema_fast or CONFIG["EMA_FAST"]
        es = ema_slow or CONFIG["EMA_SLOW"]
        ms = macd_signal or CONFIG["MACD_SIGNAL"]
        al = atr_len or CONFIG["ATR_LEN"]
        bl = bb_len or CONFIG["BB_LEN"]
        bm = bb_mult or CONFIG["BB_MULT"]
        df = _compute_indicators_df(df, rsi, ef, es, ms, al, bl, bm)
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
        df = _df_from_ohlcv(src)
        if df is None:
            return _err("Invalid OHLCV")
        rsi = rsi_len or CONFIG["RSI_LEN"]
        ef = ema_fast or CONFIG["EMA_FAST"]
        es = ema_slow or CONFIG["EMA_SLOW"]
        ms = macd_signal or CONFIG["MACD_SIGNAL"]
        al = atr_len or CONFIG["ATR_LEN"]
        bl = bb_len or CONFIG["BB_LEN"]
        bm = bb_mult or CONFIG["BB_MULT"]
        df = _compute_indicators_df(df, rsi, ef, es, ms, al, bl, bm)
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
        src = CANDLE_CACHE.get(cache_key)
        if src is None:
            return _err("cache_key not found", cache_key=cache_key)
        lb = lookback_days or CONFIG["VOL_LOOKBACK_DAYS"]
        lp = low_pct or CONFIG["VOL_LOW_PCT"]
        hp = high_pct or CONFIG["VOL_HIGH_PCT"]
        ep = extreme_pct or CONFIG["VOL_EXTREME_PCT"]
        sh = size_high or CONFIG["VOL_SIZE_HIGH"]
        df = _df_from_ohlcv(src)
        if df is None or df.empty:
            return _err("Invalid OHLCV")
        df = _compute_indicators_df(df)
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
        df = _df_from_ohlcv(src)
        if df is None or df.empty:
            return _err("Invalid OHLCV")
        df = _compute_indicators_df(df)
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
        df = _df_from_ohlcv(src)
        if df is None or len(df) < 20:
            return _err("Invalid or too short OHLCV")
        df = _compute_indicators_df(df)
        last_close = float(df.iloc[-1]["Close"])
        atr_ltf = float(df.iloc[-1]["ATR"]) if pd.notna(df.iloc[-1]["ATR"]) else 0.0
        prices = pd.concat([df["Open"], df["High"], df["Low"], df["Close"]], ignore_index=True)
        digits_guess = _infer_digits_from_prices(prices)
        tick = _infer_tick_from_prices(prices, digits_guess) or (10 ** (-max(digits_guess or 3, 3)))
        eps = max(tick * 0.5, 1e-12)
        prof = CONFIG["PROFILES"].get(horizon.lower(), CONFIG["PROFILES"]["swing"])
        # Base ATR
        atr_basis = atr_ltf
        # HTF optionnelle
        htf_src = _parse_ohlcv_json(htf_ohlcv_json) if htf_ohlcv_json else []
        htf_df = None
        if (anchor_tf in ("htf", "auto")) and htf_src:
            try:
                htf_df = _df_from_ohlcv(htf_src)
                if htf_df is not None and len(htf_df) >= 20:
                    htf_df = _compute_indicators_df(htf_df)
                    atr_htf_last = float(htf_df["ATR"].iloc[-1])
                    if np.isfinite(atr_htf_last):
                        atr_basis = atr_htf_last
            except Exception:
                htf_df = None
        # Planchers & buffers
        min_stop_ticks = int(prof.get("min_ticks", 100))
        if atr_basis and tick > 0:
            min_stop_ticks = max(min_stop_ticks, int(math.ceil(CONFIG["LEVELS_MIN_ATR_FRACTION"] * atr_basis / tick)))
        min_stop_price = float(min_stop_ticks * tick)
        spread_buffer = max(2 * tick, float(prof.get("spread_frac", 0.08)) * min_stop_price)
        # Pivot HTF structurel
        struct_floor = 0.0
        if htf_df is not None and len(htf_df) >= 30:
            try:
                piv = _zigzag_swings(htf_df, atr_mult=max(1.8, float(fib_atr_mult)), min_bars=3)
                if action.upper() == "BUY":
                    for i in range(len(piv)-1, -1, -1):
                        if piv[i][2] == 'L':
                            struct_floor = abs(last_close - float(piv[i][1])) + 10 * tick; break
                elif action.upper() == "SELL":
                    for i in range(len(piv)-1, -1, -1):
                        if piv[i][2] == 'H':
                            struct_floor = abs(float(piv[i][1]) - last_close) + 10 * tick; break
            except Exception:
                struct_floor = 0.0
        rr_target = float(prof.get("rr_target", 1.8))
        sl_dist0 = float(prof.get("sl_atr_mult", 1.1)) * atr_ltf
        tp_dist0 = float(prof.get("tp_atr_mult", 2.2)) * atr_ltf
        req_dist = max(min_stop_price + spread_buffer, struct_floor)
        sl_dist = max(sl_dist0, req_dist)
        tp_dist = max(tp_dist0, rr_target * sl_dist, req_dist)
        entry_ref = last_close
        if action.upper() == "BUY":
            sl = _round_to_tick(entry_ref - sl_dist, tick)
            tp = _round_to_tick(entry_ref + tp_dist, tick)
        elif action.upper() == "SELL":
            sl = _round_to_tick(entry_ref + sl_dist, tick)
            tp = _round_to_tick(entry_ref - tp_dist, tick)
        else:
            return _err("action must be BUY or SELL")
        rr = float(tp_dist / sl_dist) if sl_dist > 0 else 0.0
        # Fibonacci (optionnel)
        fib_used = False
        fib_tp_raw = None
        if use_fib and len(df) >= 30:
            try:
                piv = _zigzag_swings(df, atr_mult=float(fib_atr_mult))
                if len(piv) >= 2:
                    i2, p2, t2 = piv[-1]
                    i1, p1, t1 = piv[-2]
                    up_segment = (t1 == 'L' and t2 == 'H' and p2 > p1)
                    down_segment = (t1 == 'H' and t2 == 'L' and p2 < p1)
                    fib = _fib_levels(high=max(p1, p2), low=min(p1, p2))
                    if action.upper() == "BUY" and up_segment:
                        for tp_cand in sorted(list(fib["ext_up"].values())):
                            tpr = _round_to_tick(tp_cand, tick)
                            if tpr and (tpr >= entry_ref + req_dist - eps) and ((tpr - entry_ref)/sl_dist) >= rr_target:
                                tp = tpr; tp_dist = tp - entry_ref; rr = tp_dist / sl_dist; fib_used = True; fib_tp_raw = tp_cand; break
                    elif action.upper() == "SELL" and down_segment:
                        for tp_cand in sorted(list(fib["ext_down"].values()), reverse=True):
                            tpr = _round_to_tick(tp_cand, tick)
                            if tpr and (tpr <= entry_ref - req_dist + eps) and ((entry_ref - tpr)/sl_dist) >= rr_target:
                                tp = tpr; tp_dist = entry_ref - tp; rr = tp_dist / sl_dist; fib_used = True; fib_tp_raw = tp_cand; break
            except Exception:
                pass
        # Contraintes finales
        if action.upper() == "BUY":
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
            "sl": sl, "tp": tp, "rr": rr,
            "meta": {
                "tick": float(tick), "digits_guess": int(digits_guess),
                "atr_ltf": float(atr_ltf), "atr_basis": float(atr_basis),
                "min_stop_price": float(min_stop_price), "spread_buffer": float(spread_buffer),
                "struct_floor": float(struct_floor), "sl_dist_final": float(sl_dist), "tp_dist_final": float(tp_dist),
                "rr_target": float(rr_target), "horizon": horizon, "risk_level": risk_level, "action": action.upper(),
                "fib_used": fib_used, "fib_tp_raw": (float(fib_tp_raw) if fib_tp_raw is not None else None),
                "anchor_tf": anchor_tf, "eps": float(eps),
            }
        })
    except Exception as e:
        return _err("levels_autonomous_from_json failed", exc=str(e))

# ---- Intraday decision (primitifs only) ----

@mcp.tool()
def intraday_decision(symbol: str, interval: str = "15m", equity: float = 10000.0, risk_pct: float = 0.005, cap_leverage: float = 5.0, risk_level: str = "medium", vol_enabled: bool = True, lookback_days: int = 0, vol_low_pct: int = 0, vol_high_pct: int = 0, vol_extreme_pct: int = 0, vol_size_high: float = 0.0, require_htf_on_edges: bool = False, trend_only: bool = False) -> str:
    """intraday"""
    try:
        inter = (interval or "15m").lower()
        if inter not in {"5m", "15m"}:
            inter = "15m"
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
        if vol_enabled:
            bands = _atr_pct_bands_from_df(ltf, lookback_days=lb, low_pct=lp, high_pct=hp, extreme_pct=ep)
            if bands is not None:
                allowed, band, size_factor, reason_code = _volatility_gate(bands["atr_now"], bands["p10"], bands["p90"], bands.get("p95"), size_high=sh)
                vol_meta = {"atr_pct_now": bands["atr_now"], "p10": bands["p10"], "p90": bands["p90"], "p95": bands.get("p95"), "band": band, "size_factor": size_factor, "reason": reason_code}
                if not allowed:
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
        min_conf = CONFIG["PROFILES"].get(horizon, {}).get("min_confidence", None)
        if action != "HOLD" and isinstance(min_conf, (int, float)) and conf < float(min_conf):
            action = "HOLD"
        if action == "HOLD":
            return _ok({
                "symbol": symbol, "interval": inter, "regime": regime,
                "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": conf, "risk_level": risk_level},
                "reason": f"Score={score:.2f} / Confluence insuffisante.", "volatility": vol_meta,
            })
        d1_raw = meta_api.get_historical_candles(symbol, f"{CONFIG['D1_PERIOD_MONTHS']}mo", "1d")
        d1 = _df_from_ohlcv(json.loads(d1_raw)["data"] if isinstance(d1_raw, str) else d1_raw["data"])
        levels_json = levels_autonomous_from_json.__wrapped__(
            ohlcv_json=ltf.tail(300).to_json(orient="records", date_format="iso"),
            action=action, horizon=horizon, risk_level=risk_level,
            use_fib=True, fib_atr_mult=2.0, anchor_tf="htf",
            htf_ohlcv_json=(d1.tail(200).to_json(orient="records", date_format="iso") if d1 is not None else ""),
        )
        levels = json.loads(levels_json)
        if not levels.get("ok"):
            return _err("levels_autonomous_from_json failed inside intraday_decision", inner=levels)
        lv = levels["data"]
        entry = 0
        sl = lv.get("sl")
        tp = lv.get("tp")
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
        })
    except Exception as e:
        return _err("intraday_decision failed", exc=str(e))

# ===================== Prompt =====================
# ================================================================
# --------------------- LLM Prompt (updated) ---------------------
# ================================================================

PROMPT_TMPL = Template(r"""
Tu es **Senior Quant-Trader**.
But : produire **une décision exploitable** (BUY/SELL/HOLD) et des **niveaux robustes** (entry/sl/tp) pour $symbol en te basant **uniquement** sur les outils listés.

### OUTILS (dans l'ordre)
1) get_historical_candles("$symbol", "$period", "$interval", compact=True) → {cache_key, count}
2) compute_indicators(cache_key="<cache_key>") → dernière ligne (Close, EMA_Fast/Slow, RSI, MACD_Line/Signal, BB_Mid, BB_Upper/Lower, ATR, ATR_PCT (fraction 0–1), BBW_PCT, TickSize_Guess, Digits_Guess, MinStopPrice_Fallback).
3) volatility_bands(cache_key="<cache_key>", lookback_days=${lookback}, low_pct=${lowp}, high_pct=${highp}, extreme_pct=${extp}, size_high=${sizeh})
   → **OBLIGATOIRE** : renvoie { atr_pct_now, p10, p90, p95, band, size_factor, reason }.
4) levels_autonomous(cache_key="<cache_key>", action, horizon="$horizon", risk_level="$risk_level", use_fib=True, fib_atr_mult=2.0) **uniquement si action ≠ HOLD**.

### RÈGLES VOLATILITÉ
- Si band ∈ {LOW, EXTREME} → action="HOLD" (raison = reason du tool).
- Si band == HIGH → tu peux conserver l'action mais **note** size_factor (réduction de taille) dans la sortie.

### RÉGIME
- **no-trade** si ATR_PCT > ${atr_high},
  ou si ATR_PCT < ${atr_low} **et** BBW_PCT < ${bbw_squeeze}.
  Sinon: *trend* si BBW_PCT ≥ ${bbw_trend} **et** EMAs alignées LTF=HTF, *range* sinon.

### DÉCISION
Score directionnel (pondéré) :
- EMA : +${w_ema} si EMA_Fast ≥ EMA_Slow, sinon −${w_ema}.
- MACD : +${w_macd} si MACD_Line ≥ MACD_Signal, sinon −${w_macd}.
- RSI : +${w_rsi} si RSI ≥ ${rsi_pos}, −${w_rsi} si RSI ≤ ${rsi_neg}, sinon 0.
- Position Bollinger : +${w_bb} si Close ≥ BB_Mid, sinon −${w_bb}.

BUY si score ≥ ${trend_buy} **et** confluence avec la TF supérieure; SELL si score ≤ ${trend_sell} **et** confluence; sinon HOLD.

**Confiance** : `confidence = round((abs(score_total)/4) * (1 - clamp((ATR_PCT - ${conf_start})/${conf_range}, 0, ${conf_cap})) * 100)`.

### NIVEAUX
- Si **HOLD** → `entry/sl/tp = null`.
- Sinon, **appelle levels_autonomous** avec `action` décidée.
- Si ohlcv a < MIN_BARS (ex 240) ou des indicateurs manquent/NaN :
  1) rappelle get_historical_candles avec une période plus grande (double les jours),
  2) réessaie compute_indicators.
- Si après tentative(s) tu ne peux pas décider, PRODUIS QUAND MÊME la SORTIE JSON stricte.

-  RÉPONDS **UNIQUEMENT** par l’objet JSON final, **sans** texte autour, **sans** code fences, **sans** commentaires.
### SORTIE (JSON strict)
{
  "symbol": "$symbol",
  "horizon": "$horizon",
  "decision": {
    "action": "<BUY|SELL|HOLD>",
    "entry": <number|null>,
    "sl": <number|null>,
    "tp": <number|null>,
    "confidence": <number>,
    "risk_level": "$risk_level"
  },
  "reason": "<résumé concis en FR>",
  "regime": "<trend|range|no-trade>",
  "volatility": {
    "atr_pct_now": <number>,
    "p10": <number>,
    "p90": <number>,
    "p95": <number|null>,
    "band": "<LOW|NORMAL|HIGH|EXTREME>",
    "size_factor": <number>,
    "reason": "<ATR_OK|ATR_HIGH_SIZE_DOWN|ATR_GATE_LOW|ATR_GATE_HIGH>"
  }
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
