# analysis_core.py
import os, json, math
from decimal import Decimal, ROUND_HALF_UP
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# ===================== ENV & CONFIG =====================

def _env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name, None)
        return int(v) if v is not None else default
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name, None)
        return float(v) if v is not None else default
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in {"1","true","yes","y","on"}

def _env_json(name: str, default: dict) -> dict:
    v = os.getenv(name, None)
    if not v:
        return default
    try:
        parsed = json.loads(v)
        return parsed if isinstance(parsed, dict) else default
    except Exception:
        return default

CONFIG = {
    "RSI_LEN":  _env_int("TA_RSI_LEN", 9),
    "EMA_FAST": _env_int("TA_EMA_FAST", 8),
    "EMA_SLOW": _env_int("TA_EMA_SLOW", 21),
    "MACD_SIGNAL": _env_int("TA_MACD_SIGNAL", 7),
    "ATR_LEN":  _env_int("TA_ATR_LEN", 10),
    "BB_LEN":   _env_int("TA_BB_LEN", 14),
    "BB_MULT":  _env_float("TA_BB_MULT", 1.8),

    "IND_MIN_STOP_TICKS_BASE": _env_int("IND_MIN_STOP_TICKS_BASE", 200),
    "IND_MINSTOP_ATR_FRAC":    _env_float("IND_MINSTOP_ATR_FRAC", 0.15),

    "VOL_ENABLED":        _env_bool("VOL_ENABLED", True),
    "VOL_LOOKBACK_DAYS":  _env_int("VOL_LOOKBACK_DAYS", 40),
    "VOL_LOW_PCT":        _env_int("VOL_LOW_PCT", 5),
    "VOL_HIGH_PCT":       _env_int("VOL_HIGH_PCT", 92),
    "VOL_EXTREME_PCT":    _env_int("VOL_EXTREME_PCT", 98),
    "VOL_SIZE_HIGH":      _env_float("VOL_SIZE_HIGH", 0.9),
    "VOL_MIN_SAMPLES_WINDOW":   _env_int("VOL_MIN_SAMPLES_WINDOW", 50),
    "VOL_MIN_SAMPLES_FALLBACK": _env_int("VOL_MIN_SAMPLES_FALLBACK", 20),
    "VOL_FALLBACK_TAIL":        _env_int("VOL_FALLBACK_TAIL", 2000),

    "REGIME_ATR_HIGH":    _env_float("REGIME_ATR_HIGH", 0.02),
    "REGIME_ATR_LOW":     _env_float("REGIME_ATR_LOW", 0.00025),
    "REGIME_BBW_TREND":   _env_float("REGIME_BBW_TREND", 4.2),
    "REGIME_SQUEEZE_BBW": _env_float("REGIME_SQUEEZE_BBW", 2.6),
    "TREND_ONLY":         _env_bool("TREND_ONLY", False),

    "DEC_RSI_POS":          _env_int("DEC_RSI_POS", 53),
    "DEC_RSI_NEG":          _env_int("DEC_RSI_NEG", 47),
    "SCORE_W_EMA":          _env_float("SCORE_W_EMA", 1.0),
    "SCORE_W_MACD":         _env_float("SCORE_W_MACD", 1.0),
    "SCORE_W_RSI":          _env_float("SCORE_W_RSI", 1.0),
    "SCORE_W_BBPOS":        _env_float("SCORE_W_BBPOS", 1.0),
    "DEC_TREND_BUY_SCORE":  _env_int("DEC_TREND_BUY_SCORE", 1),
    "DEC_TREND_SELL_SCORE": _env_int("DEC_TREND_SELL_SCORE", -1),
    "DEC_RANGE_BUY_SCORE":  _env_int("DEC_RANGE_BUY_SCORE", 1),
    "DEC_RANGE_SELL_SCORE": _env_int("DEC_RANGE_SELL_SCORE", -1),
    "REQUIRE_HTF_ON_EDGES": _env_bool("REQUIRE_HTF_ON_EDGES", False),

    "CONF_DAMP_START":  _env_float("CONF_DAMP_START", 0.02),
    "CONF_DAMP_RANGE":  _env_float("CONF_DAMP_RANGE", 0.05),
    "CONF_DAMP_CAP":    _env_float("CONF_DAMP_CAP", 0.4),

    "LTF_PERIOD_5M":   _env_int("LTF_PERIOD_5M", 5),
    "LTF_PERIOD_15M":  _env_int("LTF_PERIOD_15M", 30),
    "HTF_PERIOD_DAYS": _env_int("HTF_PERIOD_DAYS", 30),
    "HTF_INTERVAL":    os.getenv("HTF_INTERVAL", "1h"),
    "D1_PERIOD_MONTHS": _env_int("D1_PERIOD_MONTHS", 6),

    "LEVELS_MIN_ATR_FRACTION": _env_float("LEVELS_MIN_ATR_FRACTION", 0.06),

    "PROFILES": {
        "scalping": {"rr_target": 1.45, "sl_atr_mult": 0.9, "tp_atr_mult": 1.7, "min_ticks": 60,  "spread_frac": 0.05},
        "daytrade": {"rr_target": 1.9,  "sl_atr_mult": 1.1, "tp_atr_mult": 2.3, "min_ticks": 140, "spread_frac": 0.05},
        "swing":    {"rr_target": 2.1,  "sl_atr_mult": 1.2, "tp_atr_mult": 2.5, "min_ticks": 200, "spread_frac": 0.10},
    },
    "PROFILE_OVERRIDES": _env_json("PROFILE_OVERRIDES", {}),

    "MANAGE_MOVE_BE_AT_R":   _env_float("MANAGE_MOVE_BE_AT_R", 1.0),
    "MANAGE_PARTIAL_AT_R":   _env_float("MANAGE_PARTIAL_AT_R", 1.5),
    "MANAGE_PARTIAL_FRAC":   _env_float("MANAGE_PARTIAL_FRAC", 0.5),
    "MANAGE_TRAIL_AT_R":     _env_float("MANAGE_TRAIL_AT_R", 2.0),
    "MANAGE_TRAIL_TYPE":     os.getenv("MANAGE_TRAIL_TYPE", "ATR"),
    "MANAGE_TRAIL_LEN":      _env_int("MANAGE_TRAIL_LEN", 14),
    "MANAGE_TIME_STOP_BARS": _env_int("MANAGE_TIME_STOP_BARS", 6),
}

for k, v in CONFIG["PROFILE_OVERRIDES"].items():
    if k in CONFIG["PROFILES"] and isinstance(v, dict):
        CONFIG["PROFILES"][k].update(v)

# ===================== Helpers génériques =====================

def _ok(payload: Any) -> str:
    return json.dumps({"ok": True, "data": payload}, ensure_ascii=False)

def _err(msg: str, **extra) -> str:
    logger.error(f"[CORE] {msg} | extra={extra}")
    return json.dumps({"ok": False, "error": msg, "extra": extra}, ensure_ascii=False)

def _assert_pos_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")

def _assert_pos_float(name: str, value: float) -> None:
    try:
        v = float(value)
    except Exception:
        raise ValueError(f"{name} must be a positive float")
    if v <= 0:
        raise ValueError(f"{name} must be > 0")

def _round_to_tick(value: float, tick_size: Optional[float], ndigits: int = 6) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    if tick_size and tick_size > 0:
        q = Decimal(str(tick_size))
        if q.adjusted() < -28:
            q = Decimal('1e-28')
        v = (Decimal(str(value)) / q).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * q
        return float(v)
    return float(Decimal(str(value)).quantize(Decimal(f"1e-{ndigits}"), rounding=ROUND_HALF_UP))

# ===================== OHLCV parsing =====================

_DEF_DATE_CAND_COLS = (
    "datetime","date","timestamp","time","timestamp_ms","ts",
    "t","open_time","bar_time","bar_ts","bar_index","index"
)

def _infer_epoch_unit(series: pd.Series) -> Optional[str]:
    s = series.dropna()
    s_num = pd.to_numeric(s, errors="coerce")
    frac_num = s_num.notna().mean() if len(s) else 0.0
    if frac_num < 0.7:
        return None
    s_str = s_num.dropna().astype("int64").astype(str)
    if s_str.empty:
        return None
    L = int(s_str.str.len().median())
    if L == 10: return "s"
    if L == 13: return "ms"
    if L == 16: return "us"
    if 17 <= L <= 19: return "ns"
    med = float(s_num.dropna().median()) if s_num.notna().any() else 0
    if med > 1e17: return "ns"
    if med > 1e14: return "us"
    if med > 1e11: return "ms"
    if med > 1e8:  return "s"
    return None

def _normalize_naive_utc(dt: pd.Series) -> pd.Series:
    try:
        if pd.api.types.is_datetime64_any_dtype(dt):
            if getattr(dt.dtype, "tz", None) is not None:
                return dt.dt.tz_convert("UTC").dt.tz_localize(None)
            return dt
        parsed = pd.to_datetime(dt, errors="coerce", utc=True)
        return parsed.dt.tz_localize(None)
    except Exception:
        return pd.to_datetime(dt, errors="coerce", utc=True).dt.tz_localize(None)

def _df_from_ohlcv(ohlcv: Any) -> Optional[pd.DataFrame]:
    if isinstance(ohlcv, str):
        try:
            ohlcv = json.loads(ohlcv)
            if isinstance(ohlcv, dict) and "data" in ohlcv:
                ohlcv = ohlcv["data"]
        except Exception:
            return None
    if not isinstance(ohlcv, list) or not ohlcv:
        return None
    df = pd.DataFrame(ohlcv)
    if df.empty:
        return None

    cols_map = {c.lower(): c for c in df.columns}
    def pick(*names) -> Optional[str]:
        for n in names:
            if n.lower() in cols_map:
                return cols_map[n.lower()]
        return None

    date_col = pick(*_DEF_DATE_CAND_COLS)
    o_col = pick("open","o","open_price")
    h_col = pick("high","h","high_price")
    l_col = pick("low","l","low_price")
    c_col = pick("close","c","close_price")
    v_col = pick("volume","vol","v")
    if not all([o_col, h_col, l_col, c_col]):
        return None

    rename = {o_col:"Open", h_col:"High", l_col:"Low", c_col:"Close"}
    if v_col: rename[v_col] = "Volume"
    df = df.rename(columns=rename)

    synthetic_date = False
    if date_col:
        df = df.rename(columns={date_col: "Date"})
        is_numeric_like = pd.api.types.is_numeric_dtype(df["Date"]) or df["Date"].astype(str).str.fullmatch(r"\d+").all()
        date_unit: Optional[str] = None
        if is_numeric_like:
            date_unit = _infer_epoch_unit(df["Date"]) or ("ms" if "ms" in date_col.lower() else None)
        try:
            if is_numeric_like:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce", unit=date_unit, utc=True)
            else:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        except Exception:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df["Date"] = _normalize_naive_utc(df["Date"])
    else:
        n = len(df)
        df.insert(0, "Date", pd.to_datetime(np.arange(n), unit="s", origin="unix", utc=True).tz_localize(None))
        synthetic_date = True  # not used, but kept for parity

    for c in ["Open","High","Low","Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    before = len(df)
    df = (df.dropna(subset=["Date","Open","High","Low","Close"])
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True))
    logger.debug(f"[CORE] OHLCV parsed: {before} -> {len(df)} rows")
    return df

# ===================== Indicators =====================

def ema(series: pd.Series, length: int) -> pd.Series:
    _assert_pos_int("ema length", length)
    return series.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    _assert_pos_int("rsi length", length)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / float(length)
    avg_gain = pd.Series(gain, index=close.index).ewm(alpha=alpha, adjust=False).mean()
    avg_loss = pd.Series(loss, index=close.index).ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    rsi_val = rsi_val.fillna(50.0)
    return rsi_val.clip(lower=0.0, upper=100.0).astype(float)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    _assert_pos_int("macd fast", fast); _assert_pos_int("macd slow", slow); _assert_pos_int("macd signal", signal)
    if not fast < slow:
        raise ValueError("macd fast must be < slow")
    fast_ema, slow_ema = ema(close, fast), ema(close, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    return line, sig, line - sig

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    _assert_pos_int("atr length", length)
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    alpha = 1.0 / float(length)
    return tr.ewm(alpha=alpha, adjust=False).mean()

def bbands(close: pd.Series, length: int = 20, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    _assert_pos_int("bb length", length); _assert_pos_float("bb mult", mult)
    ma = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    return ma - mult * sd, ma, ma + mult * sd

# ===================== Market helpers =====================

def _infer_digits_from_prices(prices: pd.Series) -> int:
    s = prices.dropna().astype(str)
    dec_lens = s[s.str.contains(r"\.")].str.split(".", n=1).str[1].str.len()
    return int(dec_lens.max()) if not dec_lens.empty else 0

def _infer_tick_from_prices(prices: pd.Series, max_digits: int) -> float:
    vals = (prices.dropna().round(max_digits).astype(float) * (10 ** max_digits)).round().astype("Int64").dropna().astype(int)
    if vals.empty:
        return float(Decimal(1).scaleb(-max_digits))
    uniq = np.unique(vals.values)
    if uniq.size > 50000:
        uniq = np.random.choice(uniq, 50000, replace=False)
    if uniq.size < 2:
        return float(Decimal(1).scaleb(-max_digits))
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float(Decimal(1).scaleb(-max_digits))
    tick_int = reduce(math.gcd, diffs.tolist())
    tick_size = tick_int / (10 ** max_digits)
    if tick_size <= 0:
        tick_size = float(Decimal(1).scaleb(-max_digits))
    return float(tick_size)

# ===================== Swings / Fib =====================

def _zigzag_swings(df: pd.DataFrame, atr_mult: float = 2.0, min_bars: int = 5) -> List[Tuple[int, float, str]]:
    h, l, c, atr_v = df["High"].values, df["Low"].values, df["Close"].values, df["ATR"].values
    pivots: List[Tuple[int, float, str]] = []
    mode = None
    start = int(df["ATR"].first_valid_index() or 1)
    start = max(1, start)
    last_pivot_i = start
    last_pivot_p = c[start]
    for i in range(start + 1, len(df)):
        if mode in (None, 'down'):
            if h[i] >= last_pivot_p + atr_mult * atr_v[i] and (i - last_pivot_i) >= min_bars:
                pivots.append((i, h[i], 'H')); last_pivot_i, last_pivot_p, mode = i, h[i], 'up'
        if mode in (None, 'up'):
            if l[i] <= last_pivot_p - atr_mult * atr_v[i] and (i - last_pivot_i) >= min_bars:
                pivots.append((i, l[i], 'L')); last_pivot_i, last_pivot_p, mode = i, l[i], 'down'
    filt: List[Tuple[int, float, str]] = []
    for idx, p, t in pivots:
        if not filt or filt[-1][2] != t:
            filt.append((idx, p, t))
        else:
            if (t == 'H' and p > filt[-1][1]) or (t == 'L' and p < filt[-1][1]):
                filt[-1] = (idx, p, t)
    return filt

def _fib_levels(high: float, low: float) -> Dict[str, Dict[str, float]]:
    rng = high - low
    retr = {"23.6%": high - 0.236*rng, "38.2%": high - 0.382*rng, "50.0%": high - 0.500*rng,
            "61.8%": high - 0.618*rng, "78.6%": high - 0.786*rng}
    ext_up = {"127.2%": high + 0.272*rng, "161.8%": high + 0.618*rng, "200%": high + 1.000*rng}
    ext_down = {"127.2%": low - 0.272*rng, "161.8%": low - 0.618*rng, "200%": low - 1.000*rng}
    return {"retr": retr, "ext_up": ext_up, "ext_down": ext_down}

# ===================== Feature engineering & scores =====================

def _compute_indicators_df(
    df: pd.DataFrame,
    rsi_len: Optional[int] = None,
    ema_fast: Optional[int] = None,
    ema_slow: Optional[int] = None,
    macd_signal: Optional[int] = None,
    atr_len: Optional[int] = None,
    bb_len: Optional[int] = None,
    bb_mult: Optional[float] = None,
) -> pd.DataFrame:
    rsi_len     = CONFIG["RSI_LEN"]     if rsi_len     is None else rsi_len
    ema_fast    = CONFIG["EMA_FAST"]    if ema_fast    is None else ema_fast
    ema_slow    = CONFIG["EMA_SLOW"]    if ema_slow    is None else ema_slow
    macd_signal = CONFIG["MACD_SIGNAL"] if macd_signal is None else macd_signal
    atr_len     = CONFIG["ATR_LEN"]     if atr_len     is None else atr_len
    bb_len      = CONFIG["BB_LEN"]      if bb_len      is None else bb_len
    bb_mult     = CONFIG["BB_MULT"]     if bb_mult     is None else bb_mult

    df = df.copy()
    df["EMA_Fast"], df["EMA_Slow"] = ema(df["Close"], ema_fast), ema(df["Close"], ema_slow)
    df["RSI"] = rsi(df["Close"], rsi_len)
    macd_line, macd_sig, macd_hist = macd(df["Close"], ema_fast, ema_slow, macd_signal)
    df["MACD_Line"], df["MACD_Signal"], df["MACD_Hist"] = macd_line, macd_sig, macd_hist
    df["ATR"] = atr(df, atr_len)
    bb_l, bb_m, bb_u = bbands(df["Close"], bb_len, bb_mult)
    df["BB_Lower"], df["BB_Mid"], df["BB_Upper"] = bb_l, bb_m, bb_u

    prices = pd.concat([df["Open"], df["High"], df["Low"], df["Close"]], ignore_index=True)
    digits_guess = _infer_digits_from_prices(prices)
    tick_guess = _infer_tick_from_prices(prices, digits_guess)
    df["Digits_Guess"] = digits_guess
    df["TickSize_Guess"] = float(tick_guess)
    df["ATR_PCT"] = np.where(df["Close"] > 0, (df["ATR"] / df["Close"]), np.nan)
    df["BBW_PCT"] = np.where(df["BB_Mid"].abs() > 0, (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"] * 100.0, np.nan)

    last_atr = float(df["ATR"].iloc[-1]) if len(df) else None
    point = float(tick_guess if tick_guess else (10 ** (-(digits_guess if digits_guess else 3))))
    min_stop_ticks = CONFIG["IND_MIN_STOP_TICKS_BASE"]
    if last_atr and point > 0:
        min_stop_ticks = max(min_stop_ticks, int(math.ceil(CONFIG["IND_MINSTOP_ATR_FRAC"] * last_atr / point)))
    df["MinStopPrice_Fallback"] = float(min_stop_ticks * point)
    return df

def _last_row(df: pd.DataFrame) -> Dict[str, Any]:
    return json.loads(df.tail(1).replace({np.nan: None}).to_json(orient="records", date_format="iso"))[0]

def _atr_pct_bands_from_df(
    df: pd.DataFrame,
    lookback_days: int = 30,
    low_pct: int = 10,
    high_pct: int = 90,
    extreme_pct: int = 95,
) -> Optional[Dict[str, float]]:
    if df is None or "Date" not in df.columns or "ATR_PCT" not in df.columns or df.empty:
        return None
    try:
        cutoff = df["Date"].max() - pd.Timedelta(days=int(lookback_days))
        s = df.loc[df["Date"] >= cutoff, "ATR_PCT"].dropna()
        if s.size < CONFIG["VOL_MIN_SAMPLES_WINDOW"]:
            s = df["ATR_PCT"].dropna().tail(CONFIG["VOL_FALLBACK_TAIL"])
        if s.size < CONFIG["VOL_MIN_SAMPLES_FALLBACK"]:
            return None
        p10 = float(np.percentile(s, low_pct))
        p90 = float(np.percentile(s, high_pct))
        p95 = float(np.percentile(s, extreme_pct)) if extreme_pct else None
        atr_now = float(df["ATR_PCT"].iloc[-1])
        return {"atr_now": atr_now, "p10": p10, "p90": p90, "p95": p95}
    except Exception:
        return None

def _classify_vol_band(atr_now: float, p10: float, p90: float, p95: Optional[float] = None, size_high: float = 0.5) -> Dict[str, Any]:
    if atr_now is None or not np.isfinite(atr_now):
        return {"band": "NORMAL", "size_factor": 1.0, "reason": "ATR_NA"}
    if p10 is not None and atr_now < p10:
        return {"band": "LOW", "size_factor": 0.0, "reason": "ATR_GATE_LOW"}
    if p95 is not None and atr_now > p95:
        return {"band": "EXTREME", "size_factor": 0.0, "reason": "ATR_GATE_HIGH"}
    if p90 is not None and atr_now > p90:
        return {"band": "HIGH", "size_factor": float(size_high), "reason": "ATR_HIGH_SIZE_DOWN"}
    return {"band": "NORMAL", "size_factor": 1.0, "reason": "ATR_OK"}

def _volatility_gate(atr_now: float, p10: float, p90: float, p95: Optional[float] = None, size_high: float = 0.5) -> Tuple[bool, str, float, str]:
    cls = _classify_vol_band(atr_now, p10, p90, p95, size_high)
    band = cls["band"]
    if band in ("LOW", "EXTREME"):
        return False, band, cls["size_factor"], cls["reason"]
    return True, band, cls["size_factor"], cls["reason"]

def _directional_score(last: Dict[str, Any]) -> float:
    s = 0.0
    s += CONFIG["SCORE_W_EMA"]   if (last.get("EMA_Fast") or 0) >= (last.get("EMA_Slow") or 0) else -CONFIG["SCORE_W_EMA"]
    s += CONFIG["SCORE_W_MACD"]  if (last.get("MACD_Line") or 0) >= (last.get("MACD_Signal") or 0) else -CONFIG["SCORE_W_MACD"]
    rsi_v = last.get("RSI") or 50
    if rsi_v >= CONFIG["DEC_RSI_POS"]: s += CONFIG["SCORE_W_RSI"]
    elif rsi_v <= CONFIG["DEC_RSI_NEG"]: s -= CONFIG["SCORE_W_RSI"]
    s += CONFIG["SCORE_W_BBPOS"] if (last.get("Close") or 0) >= (last.get("BB_Mid") or 0) else -CONFIG["SCORE_W_BBPOS"]
    return float(s)

def _confidence(last: Dict[str, Any], score_total: float) -> int:
    atr_pct = (last.get("ATR_PCT") or 0.0)
    damp = max(0.0, min(CONFIG["CONF_DAMP_CAP"], (atr_pct - CONFIG["CONF_DAMP_START"]) / max(CONFIG["CONF_DAMP_RANGE"], 1e-9)))
    conf = round((abs(score_total) / 4.0) * (1 - damp) * 100)
    return int(max(0, min(100, conf)))

def _regime_from_df(df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> str:
    last_ltf = _last_row(df_ltf); last_htf = _last_row(df_htf)
    atr_pct = last_ltf.get("ATR_PCT") or 0.0
    bbw     = last_ltf.get("BBW_PCT") or 0.0
    if atr_pct > CONFIG["REGIME_ATR_HIGH"]:
        return "no-trade"
    if atr_pct < CONFIG["REGIME_ATR_LOW"] and bbw < CONFIG["REGIME_SQUEEZE_BBW"]:
        return "no-trade"
    ema_align_ltf = (last_ltf.get("EMA_Fast") or 0) >= (last_ltf.get("EMA_Slow") or 0)
    ema_align_htf = (last_htf.get("EMA_Fast") or 0) >= (last_htf.get("EMA_Slow") or 0)
    trend_like = (bbw >= CONFIG["REGIME_BBW_TREND"]) and (ema_align_ltf == ema_align_htf)
    return "trend" if trend_like else "range"

def _position_size(entry: float, sl: float, equity: float, risk_pct: float, cap_leverage: float, price_mult: float = 1.0) -> Dict[str, Any]:
    if entry is None or sl is None:
        return {"units": 0, "risk_eur": 0.0}
    dist = abs(entry - sl)
    if dist <= 0:
        return {"units": 0, "risk_eur": 0.0}
    risk_eur = float(equity) * float(risk_pct)
    units_risk = risk_eur / dist
    notional_cap = float(equity) * float(cap_leverage) * price_mult
    units = float(min(units_risk, notional_cap))
    return {"units": units, "risk_eur": risk_eur, "distance": dist}

__all__ = [
    "CONFIG","_ok","_err","_round_to_tick",
    "_df_from_ohlcv","_compute_indicators_df","_last_row",
    "_infer_digits_from_prices","_infer_tick_from_prices",
    "_zigzag_swings","_fib_levels",
    "_atr_pct_bands_from_df","_classify_vol_band","_volatility_gate",
    "_directional_score","_confidence","_regime_from_df","_position_size",
]
