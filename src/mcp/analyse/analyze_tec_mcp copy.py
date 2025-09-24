
import json
from typing import Annotated, Any, Dict, List, Optional, Tuple
from datetime import datetime
import sys, os, math
from decimal import Decimal, ROUND_HALF_UP
from string import Template
from functools import reduce
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()

# ================================================================
# --------------------- ENV helpers & CONFIG ---------------------
# ================================================================

def _env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name, None)
        return int(v) if v is not None else default
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name, None)
       #  logger.info(f"[MCP:_env_float] name: {name}:{v}")
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
    # ===== Indicateurs =====
    "RSI_LEN":              _env_int("TA_RSI_LEN", 9),
    "EMA_FAST":             _env_int("TA_EMA_FAST", 8),
    "EMA_SLOW":             _env_int("TA_EMA_SLOW", 21),
    "MACD_SIGNAL":          _env_int("TA_MACD_SIGNAL", 7),
    "ATR_LEN":              _env_int("TA_ATR_LEN", 10),
    "BB_LEN":               _env_int("TA_BB_LEN", 14),
    "BB_MULT":              _env_float("TA_BB_MULT", 1.8),

    # Fallback min stop (compute_indicators_df / plan_raw)
    "IND_MIN_STOP_TICKS_BASE": _env_int("IND_MIN_STOP_TICKS_BASE", 200),
    "IND_MINSTOP_ATR_FRAC":    _env_float("IND_MINSTOP_ATR_FRAC", 0.15),

    # ===== Volatility bands (percentiles) =====
    "VOL_ENABLED":          _env_bool("VOL_ENABLED", True),
    "VOL_LOOKBACK_DAYS":    _env_int("VOL_LOOKBACK_DAYS", 40),
    "VOL_LOW_PCT":          _env_int("VOL_LOW_PCT", 5),
    "VOL_HIGH_PCT":         _env_int("VOL_HIGH_PCT", 92),
    "VOL_EXTREME_PCT":      _env_int("VOL_EXTREME_PCT", 98),
    "VOL_SIZE_HIGH":        _env_float("VOL_SIZE_HIGH", 0.9),
    "VOL_MIN_SAMPLES_WINDOW":   _env_int("VOL_MIN_SAMPLES_WINDOW", 50),
    "VOL_MIN_SAMPLES_FALLBACK": _env_int("VOL_MIN_SAMPLES_FALLBACK", 20),
    "VOL_FALLBACK_TAIL":        _env_int("VOL_FALLBACK_TAIL", 2000),

    # ===== Régime =====
    "REGIME_ATR_HIGH":      _env_float("REGIME_ATR_HIGH", 0.02),    # 2%
    "REGIME_ATR_LOW":       _env_float("REGIME_ATR_LOW", 0.00025),  # 0,025%
    "REGIME_BBW_TREND":     _env_float("REGIME_BBW_TREND", 4.2),
    "REGIME_SQUEEZE_BBW":   _env_float("REGIME_SQUEEZE_BBW", 2.6),
    "TREND_ONLY":           _env_bool("TREND_ONLY", False),

    # ===== Scoring / décision =====
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

    # Confiance
    "CONF_DAMP_START":      _env_float("CONF_DAMP_START", 0.02),  # à partir de 2% ATR_PCT
    "CONF_DAMP_RANGE":      _env_float("CONF_DAMP_RANGE", 0.05),  # pendant 5% d'amplitude
    "CONF_DAMP_CAP":        _env_float("CONF_DAMP_CAP", 0.4),     # clamp à 40%

    # ===== Periods / TF (data fetch) =====
    "LTF_PERIOD_5M":        _env_int("LTF_PERIOD_5M", 5),   # jours
    "LTF_PERIOD_15M":       _env_int("LTF_PERIOD_15M", 30), # jours (≈1mo)
    "HTF_PERIOD_DAYS":      _env_int("HTF_PERIOD_DAYS", 30),
    "HTF_INTERVAL":         os.getenv("HTF_INTERVAL", "1h"),
    "D1_PERIOD_MONTHS":     _env_int("D1_PERIOD_MONTHS", 6),

    # ===== Levels / stops =====
    "LEVELS_MIN_ATR_FRACTION": _env_float("LEVELS_MIN_ATR_FRACTION", 0.06),

    # ===== Profils =====
    "PROFILES": {
        "scalping": {"rr_target": 1.45, "sl_atr_mult": 0.9, "tp_atr_mult": 1.7, "min_ticks": 60,  "spread_frac": 0.05},
        "daytrade": {"rr_target": 1.9,  "sl_atr_mult": 1.1, "tp_atr_mult": 2.3, "min_ticks": 140, "spread_frac": 0.05},
        "swing":    {"rr_target": 2.1,  "sl_atr_mult": 1.2, "tp_atr_mult": 2.5, "min_ticks": 200, "spread_frac": 0.10},
    },

    # Overrides optionnels (JSON) pour PROFILES
    "PROFILE_OVERRIDES": _env_json("PROFILE_OVERRIDES", {}),

    # ===== Management par défaut =====
    "MANAGE_MOVE_BE_AT_R":   _env_float("MANAGE_MOVE_BE_AT_R", 1.0),
    "MANAGE_PARTIAL_AT_R":   _env_float("MANAGE_PARTIAL_AT_R", 1.5),
    "MANAGE_PARTIAL_FRAC":   _env_float("MANAGE_PARTIAL_FRAC", 0.5),
    "MANAGE_TRAIL_AT_R":     _env_float("MANAGE_TRAIL_AT_R", 2.0),
    "MANAGE_TRAIL_TYPE":     os.getenv("MANAGE_TRAIL_TYPE", "ATR"),
    "MANAGE_TRAIL_LEN":      _env_int("MANAGE_TRAIL_LEN", 14),
    "MANAGE_TIME_STOP_BARS": _env_int("MANAGE_TIME_STOP_BARS", 6),
}

# Merge overrides propres sur les profils
for k, v in CONFIG["PROFILE_OVERRIDES"].items():
    if k in CONFIG["PROFILES"] and isinstance(v, dict):
        CONFIG["PROFILES"][k].update(v)

# ================================================================
# --------------------- MCP bootstrap / logging ------------------
# ================================================================

# Import meta_api
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../data_fetch/')))
import meta_api as meta_api

mcp = FastMCP("Trading Analysis MCP Server", log_level="WARNING")

logger.remove()
logger.add(sys.stderr, level="WARNING")

# ================================================================
# --------------------- In-memory candles cache ------------------
# ================================================================

CANDLE_CACHE: Dict[str, List[Dict[str, Any]]] = {}

def _make_cache_key(symbol: str, period: str, interval: str) -> str:
    return f"{symbol}|{period}|{interval}".lower()

# ================================================================
# --------------------- Helpers / Core Utils ---------------------
# ================================================================

def _ok(payload: Any) -> str:
    return json.dumps({"ok": True, "data": payload}, ensure_ascii=False)

def _err(msg: str, **extra) -> str:
    logger.error(f"[MCP:ANALYSIS] {msg} | extra={extra}")
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
    """Arrondit à la taille de tick si fournie; sinon arrondi décimal standard (Decimal)."""
    if value is None or not np.isfinite(value):
        return None
    if tick_size and tick_size > 0:
        q = Decimal(str(tick_size))
        # Guard contre des ticks ultra-petits (évite Overflow)
        if q.adjusted() < -28:
            q = Decimal('1e-28')
        v = (Decimal(str(value)) / q).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * q
        return float(v)
    return float(Decimal(str(value)).quantize(Decimal(f"1e-{ndigits}"), rounding=ROUND_HALF_UP))

# ================================================================
# ------------------------ OHLCV ingestion -----------------------
# ================================================================

_DEF_DATE_CAND_COLS = (
    "datetime", "date", "timestamp", "time", "timestamp_ms", "ts",
    "t", "open_time", "bar_time", "bar_ts", "bar_index", "index"
)

def _infer_epoch_unit(series: pd.Series) -> Optional[str]:
    """Détecte s/ms/us/ns pour un epoch numérique par longueur; fallback par ordre de grandeur."""
    s = series.dropna()
    s_num = pd.to_numeric(s, errors="coerce")
    frac_num = s_num.notna().mean() if len(s) else 0.0
    if frac_num < 0.7:
        return None
    s_str = s_num.dropna().astype("int64").astype(str)
    if s_str.empty:
        return None
    L = int(s_str.str.len().median())
    if L == 10:
        return "s"
    if L == 13:
        return "ms"
    if L == 16:
        return "us"
    if 17 <= L <= 19:
        return "ns"
    # ordre de grandeur en fallback
    med = float(s_num.dropna().median()) if s_num.notna().any() else 0
    if med > 1e17:
        return "ns"
    if med > 1e14:
        return "us"
    if med > 1e11:
        return "ms"
    if med > 1e8:
        return "s"
    return None

def _normalize_naive_utc(dt: pd.Series) -> pd.Series:
    """Toujours retourner des datetimes naïves supposées UTC."""
    try:
        if pd.api.types.is_datetime64_any_dtype(dt):
            # tz-aware -> converti UTC puis drop tz
            if getattr(dt.dtype, "tz", None) is not None:
                return dt.dt.tz_convert("UTC").dt.tz_localize(None)
            return dt
        parsed = pd.to_datetime(dt, errors="coerce", utc=True)
        return parsed.dt.tz_localize(None)
    except Exception:
        return pd.to_datetime(dt, errors="coerce", utc=True).dt.tz_localize(None)

def _df_from_ohlcv(ohlcv: Any) -> Optional[pd.DataFrame]:
    """Construit un DataFrame propre depuis une liste d'objets OHLCV.
       Accepte l'absence de colonne date et en synthétise une (naïve UTC).
    """
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

    logger.debug(f"[MCP:ANALYSIS] First OHLCV row: {df.iloc[0].to_dict()}")

    # Column picking (case-insensitive)
    cols_map = {c.lower(): c for c in df.columns}

    def pick(*names) -> Optional[str]:
        for n in names:
            if n.lower() in cols_map:
                return cols_map[n.lower()]
        return None

    date_col = pick(*_DEF_DATE_CAND_COLS)
    o_col = pick("open", "o", "open_price")
    h_col = pick("high", "h", "high_price")
    l_col = pick("low", "l", "low_price")
    c_col = pick("close", "c", "close_price")
    v_col = pick("volume", "vol", "v")

    # Il faut au minimum OHLC
    if not all([o_col, h_col, l_col, c_col]):
        return None

    rename = {o_col: "Open", h_col: "High", l_col: "Low", c_col: "Close"}
    if v_col:
        rename[v_col] = "Volume"

    # 1) Renommage des colonnes connues (hors date pour l'instant)
    df = df.rename(columns=rename)

    # 2) Gestion de la date : vraie colonne ou synthétique
    synthetic_date = False
    if date_col:
        df = df.rename(columns={date_col: "Date"})
        # Parse Date robuste (ns/ms/us/s/ISO) ➜ naïf UTC
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
        df["Date"] = _normalize_naive_utc(df["Date"])  # drop tz (supposé UTC)
    else:
        # Pas de date fournie : on synthétise une séquence croissante (1s d'écart), naïve UTC
        logger.warning("[MCP:ANALYSIS] No date column found; synthesizing Date from row order.")
        n = len(df)
        df.insert(0, "Date", pd.to_datetime(np.arange(n), unit="s", origin="unix", utc=True).tz_localize(None))
        synthetic_date = True

    # 3) Casting numériques
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # 4) Nettoyage & tri
    before = len(df)
    df = (
        df.dropna(subset=["Date", "Open", "High", "Low", "Close"])\
          .drop_duplicates(subset=["Date"])\
          .sort_values("Date")\
          .reset_index(drop=True)
    )
    logger.debug(f"[MCP:ANALYSIS] OHLCV parsed: {before} ➜ {len(df)} rows | synthetic_date={synthetic_date}")
    return df

# ================================================================
# -------------------- Indicator primitives ----------------------
# ================================================================

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
    _assert_pos_int("macd fast", fast)
    _assert_pos_int("macd slow", slow)
    _assert_pos_int("macd signal", signal)
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
    _assert_pos_int("bb length", length)
    _assert_pos_float("bb mult", mult)
    ma = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    return ma - mult * sd, ma, ma + mult * sd

# ---------------------- Market helpers --------------------------

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

# ---------------------- Fibonacci utils -------------------------

def _zigzag_swings(df: pd.DataFrame, atr_mult: float = 2.0, min_bars: int = 5) -> List[Tuple[int, float, str]]:
    """Retourne [(idx, prix, 'H'/'L')] pour des swings confirmés."""
    h, l, c, atr_v = df["High"].values, df["Low"].values, df["Close"].values, df["ATR"].values
    pivots: List[Tuple[int, float, str]] = []
    mode = None  # 'up' ou 'down'
    start = int(df["ATR"].first_valid_index() or 1)
    start = max(1, start)
    last_pivot_i = start
    last_pivot_p = c[start]
    for i in range(start + 1, len(df)):
        if mode in (None, 'down'):
            if h[i] >= last_pivot_p + atr_mult * atr_v[i] and (i - last_pivot_i) >= min_bars:
                pivots.append((i, h[i], 'H'))
                last_pivot_i, last_pivot_p, mode = i, h[i], 'up'
        if mode in (None, 'up'):
            if l[i] <= last_pivot_p - atr_mult * atr_v[i] and (i - last_pivot_i) >= min_bars:
                pivots.append((i, l[i], 'L'))
                last_pivot_i, last_pivot_p, mode = i, l[i], 'down'
    # assurer alternance H/L et garder l'extrême le plus significatif
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
    retr = {
        "23.6%": high - 0.236 * rng,
        "38.2%": high - 0.382 * rng,
        "50.0%": high - 0.500 * rng,
        "61.8%": high - 0.618 * rng,
        "78.6%": high - 0.786 * rng,
    }
    ext_up = {
        "127.2%": high + 0.272 * rng,
        "161.8%": high + 0.618 * rng,
        "200%":   high + 1.000 * rng,
    }
    ext_down = {
        "127.2%": low - 0.272 * rng,
        "161.8%": low - 0.618 * rng,
        "200%":   low - 1.000 * rng,
    }
    return {"retr": retr, "ext_up": ext_up, "ext_down": ext_down}

# ================================================================
# ---------------- Indicators & market info DF -------------------
# ================================================================

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
    # Fallback sur CONFIG si None
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

    # Extras utiles
    prices = pd.concat([df["Open"], df["High"], df["Low"], df["Close"]], ignore_index=True)

    digits_guess = _infer_digits_from_prices(prices)
    tick_guess = _infer_tick_from_prices(prices, digits_guess)
    df["Digits_Guess"] = digits_guess
    df["TickSize_Guess"] = float(tick_guess)

    # ATR_PCT en fraction (0.05 = 5 %)
    df["ATR_PCT"] = np.where(df["Close"] > 0, (df["ATR"] / df["Close"]), np.nan)

    last_atr = float(df["ATR"].iloc[-1]) if len(df) else None
    point = float(tick_guess if tick_guess else (10 ** (-(digits_guess if digits_guess else 3))))
    min_stop_ticks = CONFIG["IND_MIN_STOP_TICKS_BASE"]
    if last_atr and point > 0:
        min_stop_ticks = max(min_stop_ticks, int(math.ceil(CONFIG["IND_MINSTOP_ATR_FRAC"] * last_atr / point)))
    min_stop_price_fallback = float(min_stop_ticks * point)
    df["MinStopPrice_Fallback"] = float(min_stop_price_fallback)

    # Bollinger Band Width %
    df["BBW_PCT"] = np.where(df["BB_Mid"].abs() > 0, (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"] * 100.0, np.nan)

    return df

def _last_row(df: pd.DataFrame) -> Dict[str, Any]:
    return json.loads(df.tail(1).replace({np.nan: None}).to_json(orient="records", date_format="iso"))[0]

# ================================================================
# ----------------------- MCP Tools (base) -----------------------
# ================================================================

@mcp.tool()
def compute_indicators(
    ohlcv: Annotated[Optional[List[Dict[str, Any]]], Field(description="OHLCV records (optional si cache_key)")] = None,
    cache_key: Annotated[Optional[str], Field(description="Clé de cache renvoyée par get_historical_candles(compact=True)")] = None,
    tail: int = 200,
    rsi_len: Optional[int] = None,
    ema_fast: Optional[int] = None,
    ema_slow: Optional[int] = None,
    macd_signal: Optional[int] = None,
    atr_len: Optional[int] = None,
    bb_len: Optional[int] = None,
    bb_mult: Optional[float] = None,
    last_only: bool = True,
) -> str:
    """Calcule indicateurs + infos marché dérivées de l'OHLCV.
       Note: ATR_PCT est une fraction (0.05 = 5 %), BBW_PCT est en pourcentage."""
    try:
        # Récup OHLCV
        src = None
        if cache_key:
            src = CANDLE_CACHE.get(cache_key)
            if src is None:
                return _err("cache_key not found", cache_key=cache_key)
        elif ohlcv is not None:
            src = ohlcv
        else:
            return _err("Missing ohlcv or cache_key")

        if not isinstance(tail, int) or tail <= 0:
            tail = 200
        df = _df_from_ohlcv(src)
        if df is None:
            return _err("Invalid OHLCV")
        df = _compute_indicators_df(df, rsi_len, ema_fast, ema_slow, macd_signal, atr_len, bb_len, bb_mult)
        tail = min(tail, len(df))
        if last_only:
            payload = _last_row(df)
        else:
            payload = json.loads(df.tail(tail).replace({np.nan: None}).to_json(orient="records", date_format="iso"))
        return _ok(payload)
    except Exception as e:
        return _err("compute_indicators failed", exc=str(e))

# ================================================================
# --------------- Volatility (ATR% percentile) helpers -----------
# ================================================================

def _atr_pct_bands_from_df(
    df: pd.DataFrame,
    lookback_days: int = 30,
    low_pct: int = 10,
    high_pct: int = 90,
    extreme_pct: int = 95,
) -> Optional[Dict[str, float]]:
    """Calcule p10/p90/p95 de ATR_PCT (fraction) sur la fenêtre 'lookback_days' (fallback: dernières N barres)."""
    if df is None or "Date" not in df.columns or "ATR_PCT" not in df.columns or df.empty:
        return None
    try:
        cutoff = df["Date"].max() - pd.Timedelta(days=int(lookback_days))
        s = df.loc[df["Date"] >= cutoff, "ATR_PCT"].dropna()
        if s.size < CONFIG["VOL_MIN_SAMPLES_WINDOW"]:  # fallback si peu d'historique dans la fenêtre
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

def _classify_vol_band(
    atr_now: float,
    p10: float,
    p90: float,
    p95: Optional[float] = None,
    size_high: float = 0.5,
) -> Dict[str, Any]:
    """Classe LOW/NORMAL/HIGH/EXTREME + size_factor & reason. (Inputs en fraction)"""
    if atr_now is None or not np.isfinite(atr_now):
        return {"band": "NORMAL", "size_factor": 1.0, "reason": "ATR_NA"}
    if p10 is not None and atr_now < p10:
        return {"band": "LOW", "size_factor": 0.0, "reason": "ATR_GATE_LOW"}
    if p95 is not None and atr_now > p95:
        return {"band": "EXTREME", "size_factor": 0.0, "reason": "ATR_GATE_HIGH"}
    if p90 is not None and atr_now > p90:
        return {"band": "HIGH", "size_factor": float(size_high), "reason": "ATR_HIGH_SIZE_DOWN"}
    return {"band": "NORMAL", "size_factor": 1.0, "reason": "ATR_OK"}

def _volatility_gate(
    atr_now: float,
    p10: float,
    p90: float,
    p95: Optional[float] = None,
    size_high: float = 0.5,
) -> Tuple[bool, str, float, str]:
    """
    Renvoie: allowed, band ('LOW'|'NORMAL'|'HIGH'|'EXTREME'), size_factor, reason_code
    """
    cls = _classify_vol_band(atr_now, p10, p90, p95, size_high)
    band = cls["band"]
    if band in ("LOW", "EXTREME"):
        return False, band, cls["size_factor"], cls["reason"]
    return True, band, cls["size_factor"], cls["reason"]

# ================================================================
# -------------------- Volatility Bands Tool ---------------------
# ================================================================

@mcp.tool()
def volatility_bands(
    ohlcv: Annotated[Optional[List[Dict[str, Any]]], Field(description="OHLCV (optionnel si cache_key)")] = None,
    cache_key: Annotated[Optional[str], Field(description="clé de cache de get_historical_candles")] = None,
    lookback_days: Optional[int] = None,
    low_pct: Optional[int] = None,
    high_pct: Optional[int] = None,
    extreme_pct: Optional[int] = None,
    size_high: Optional[float] = None,
) -> str:
    """Retourne ATR_PCT_now + p10/p90/p95 + band + size_factor + reason (par symbole/TF)."""
    try:
        # Source
        src = None
        if cache_key:
            src = CANDLE_CACHE.get(cache_key)
            if src is None:
                return _err("cache_key not found", cache_key=cache_key)
        elif ohlcv is not None:
            src = ohlcv
        else:
            return _err("Missing ohlcv or cache_key")

        # Defaults via CONFIG
        lookback_days = CONFIG["VOL_LOOKBACK_DAYS"] if lookback_days is None else lookback_days
        low_pct       = CONFIG["VOL_LOW_PCT"]       if low_pct       is None else low_pct
        high_pct      = CONFIG["VOL_HIGH_PCT"]      if high_pct      is None else high_pct
        extreme_pct   = CONFIG["VOL_EXTREME_PCT"]   if extreme_pct   is None else extreme_pct
        size_high     = CONFIG["VOL_SIZE_HIGH"]     if size_high     is None else size_high

        # DataFrame + indicateurs (ATR_PCT)
        df = _df_from_ohlcv(src)
        if df is None or df.empty:
            return _err("Invalid OHLCV")
        df = _compute_indicators_df(df)

        bands = _atr_pct_bands_from_df(df, lookback_days, low_pct, high_pct, extreme_pct)
        if bands is None:
            return _err("Not enough data to compute percentiles", rows=len(df))

        cls = _classify_vol_band(bands["atr_now"], bands["p10"], bands["p90"], bands.get("p95"), size_high=size_high)
        payload = {
            "atr_pct_now": bands["atr_now"],
            "p10": bands["p10"],
            "p90": bands["p90"],
            "p95": bands.get("p95"),
            "band": cls["band"],
            "size_factor": cls["size_factor"],
            "reason": cls["reason"],
        }
        return _ok(payload)
    except Exception as e:
        return _err("volatility_bands failed", exc=str(e))

# ================================================================
# -------------------- Other Base Tools --------------------------
# ================================================================

@mcp.tool()
def plan_raw(
    ohlcv: Annotated[List[Dict[str, Any]], Field(description="OHLCV records")],
    risk_level: str = "medium",
    direction: str = "auto",
    tick_size: Optional[float] = None,
    horizon: str = "scalping",
) -> str:
    """
    Plan ATR simple, harmonisé avec les profils (min_ticks, spread_frac) et les contraintes (min stop, buffer, RR).
    - Utilise le profil choisi via `horizon` ("scalping" | "daytrade" | "swing") pour dimensionner les minima.
    - Garde un plancher dynamique via IND_MINSTOP_ATR_FRAC * ATR.
    - Si `direction="auto"`, déduit long/short à partir d'EMA/RSI/MACD et fallback EMA.
    - Retourne: entry (market), sl, tp, atr, side, meta (tick, min_stop_price, spread_buffer).
    """
    try:
        # 1) Parse OHLCV -> DF + indicateurs
        df = _df_from_ohlcv(ohlcv)
        if df is None or df.empty:
            return _err("Invalid OHLCV")

        df = _compute_indicators_df(df)
        last = _last_row(df)
        atr_val, close = last.get("ATR"), last.get("Close")
        if atr_val is None or close is None or not np.isfinite(atr_val) or not np.isfinite(close):
            return _err("ATR/Close missing")

        # 2) Détermination du sens (si auto)
        side = direction.lower()
        if side not in {"long", "short"}:
            ema_fast = float(last.get("EMA_Fast") or 0.0)
            ema_slow = float(last.get("EMA_Slow") or 0.0)
            rsi_v    = float(last.get("RSI") or 50.0)
            macd_ln  = float(last.get("MACD_Line") or 0.0)
            macd_sig = float(last.get("MACD_Signal") or 0.0)

            bullish_bias = (ema_fast >= ema_slow) and (rsi_v >= CONFIG["DEC_RSI_POS"] or macd_ln >= macd_sig)
            bearish_bias = (ema_fast <  ema_slow) and (rsi_v <= CONFIG["DEC_RSI_NEG"] or macd_ln <= macd_sig)
            if bullish_bias and not bearish_bias:
                side = "long"
            elif bearish_bias and not bullish_bias:
                side = "short"
            else:
                side = "long" if ema_fast >= ema_slow else "short"

        # 3) Tick & contraintes de base (profil-aware)
        prices = pd.concat([df["Open"], df["High"], df["Low"], df["Close"]], ignore_index=True)
        digits_guess = _infer_digits_from_prices(prices)
        tick = tick_size or _infer_tick_from_prices(prices, digits_guess) or float(Decimal(1).scaleb(-max(digits_guess or 3, 3)))
        point = float(tick)

        # Profil (scalping/daytrade/swing)
        prof_conf = CONFIG["PROFILES"].get(horizon.lower(), CONFIG["PROFILES"]["swing"])
        min_stop_ticks = int(prof_conf.get("min_ticks", CONFIG["IND_MIN_STOP_TICKS_BASE"]))

        # Plancher dynamique en fonction de l'ATR
        if atr_val and point > 0:
            dyn_ticks = int(math.ceil(CONFIG["IND_MINSTOP_ATR_FRAC"] * float(atr_val) / point))
            min_stop_ticks = max(min_stop_ticks, dyn_ticks)

        min_stop_price = float(min_stop_ticks * point)

        # Buffer spread cohérent avec le profil (fallback 8%)
        spread_frac = float(prof_conf.get("spread_frac", 0.08))
        spread_buffer = max(2.0 * point, spread_frac * min_stop_price)

        # Distance minimale requise de part et d'autre (plancher + buffer)
        req_dist = float(min_stop_price + spread_buffer)

        # 4) Distances SL/TP à partir de l'ATR + risk_level
        tp_mult_map = {"low": 1.5, "medium": 2.0, "high": 3.0}
        tp_mult = float(tp_mult_map.get(risk_level.lower(), 2.0))

        sl_dist = max(1.0 * float(atr_val), req_dist)
        tp_dist = max(tp_mult * float(atr_val), 1.5 * sl_dist, req_dist)

        # 5) Niveaux (market entry)
        if side == "long":
            sl = _round_to_tick(close - sl_dist, tick)
            tp = _round_to_tick(close + tp_dist, tick)
        else:
            sl = _round_to_tick(close + sl_dist, tick)
            tp = _round_to_tick(close - tp_dist, tick)

        entry = _round_to_tick(close, tick)

        # 6) Validation post-arrondi (garantir req_dist de chaque côté)
        if side == "long":
            if not (sl is not None and tp is not None and sl <= entry - req_dist and tp >= entry + req_dist):
                # Forcer un TP conforme si l'arrondi l'a rendu trop court
                tp = _round_to_tick(entry + max(tp_dist, req_dist, 1.5 * sl_dist), tick)
                if not (sl is not None and tp is not None and sl <= entry - req_dist and tp >= entry + req_dist):
                    return _err("Constraints not satisfied after rounding (LONG)",
                                entry=entry, sl=sl, tp=tp, req=req_dist, tick=tick)
        else:
            if not (sl is not None and tp is not None and sl >= entry + req_dist and tp <= entry - req_dist):
                tp = _round_to_tick(entry - max(tp_dist, req_dist, 1.5 * sl_dist), tick)
                if not (sl is not None and tp is not None and sl >= entry + req_dist and tp <= entry - req_dist):
                    return _err("Constraints not satisfied after rounding (SHORT)",
                                entry=entry, sl=sl, tp=tp, req=req_dist, tick=tick)

        return _ok({
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "atr": float(atr_val),
            "risk_level": risk_level,
            "side": side,
            "meta": {
                "tick": float(tick),
                "digits_guess": int(digits_guess),
                "min_stop_price": float(min_stop_price),
                "spread_buffer": float(spread_buffer),
                "req_dist": float(req_dist),
                "horizon": horizon,
            }
        })

    except Exception as e:
        return _err("plan_raw failed", exc=str(e))


@mcp.tool()
def get_historical_candles(symbol: str, period: str = "1mo", interval: str = "1d", compact: bool = True) -> str:
    """Récupération OHLCV via meta_api."""
    try:
        data = meta_api.get_historical_candles(symbol, period, interval)
        logger.info(f"[MCP:ANALYSIS] get_historical_candles {symbol}:{period}:{interval}")
        if isinstance(data, str):
            data = json.loads(data)
        ohlcv = data.get("data") if isinstance(data, dict) and "data" in data else data
        if not isinstance(ohlcv, list) or not ohlcv:
            logger.info(f"[MCP:ANALYSIS] get_historical_candles {symbol}:{period}:{data}")
            return _err("fetch returned empty data", symbol=symbol, period=period, interval=interval)

        cache_key = _make_cache_key(symbol, period, interval)
        CANDLE_CACHE[cache_key] = ohlcv
        payload = {"cache_key": cache_key, "count": len(ohlcv)}
        if not compact:
            payload["data"] = ohlcv
        return _ok(payload)
    except Exception as e:
        return _err("fetch failed", exc=str(e), symbol=symbol, period=period, interval=interval)

# ================================================================
# ------- Levels engine (avec plancher HTF pour 15m) -------------
# ================================================================
@mcp.tool()
def levels_autonomous(
    ohlcv: Annotated[Optional[List[Dict[str, Any]]], Field(description="OHLCV records (optionnel si cache_key)")] = None,
    action: str = "BUY",               # "BUY" ou "SELL"
    horizon: str = "scalping",         # "scalping" | "daytrade" | "swing"
    risk_level: str = "medium",        # "low" | "medium" | "high"
    use_fib: bool = True,              # Option: extensions Fibonacci pour le TP
    fib_atr_mult: float = 2.0,         # Sensibilité du ZigZag (>= 2.0 conseillé)
    cache_key: Optional[str] = None,   # Clé cache renvoyée par get_historical_candles
    # Nouveaux paramètres pour éviter stop trop court en 15m :
    anchor_tf: str = "auto",           # "ltf" | "htf" | "auto"
    htf_ohlcv: Optional[List[Dict[str, Any]]] = None,  # OHLCV d'une TF supérieure (ex: 1h ou D1)
) -> str:
    """
    Calcule entry_ref, SL, TP à partir de l'OHLCV, avec **plancher HTF** (ATR + pivot) pour le scalping.
    - Déduit TickSize/Digits depuis les prix
    - Plancher stop = max(min_ticks, LEVELS_MIN_ATR_FRACTION * ATR_basis) + buffer spread
    - ATR_basis = ATR(LTF) ou ATR(HTF) si fourni (D1 conseillé pour 15m)
    - Pivot HTF (dernier creux/haut) ajoute un **plancher structurel**
    - R/R cible adaptatif
    - Fibonacci optionnel pour étendre le TP si disponible
    Intègre une **tolérance epsilon = 0.5 * tick** dans les contrôles finaux pour éviter
    les rejets dus aux arrondis flottants (ex. SL == entry - req_dist).
    """
    try:
        # ---------------- Helpers locaux (tolérance) ----------------
        def _eps_from_tick(tick_val: Optional[float]) -> float:
            try:
                if tick_val and tick_val > 0:
                    return max(float(tick_val) * 0.5, 1e-12)  # 1/2 tick
            except Exception:
                pass
            return 1e-9

        def _le(a: float, b: float, eps: float) -> bool:
            # a <= b (avec marge eps)
            return float(a) <= float(b) + eps

        def _ge(a: float, b: float, eps: float) -> bool:
            # a >= b (avec marge eps)
            return float(a) >= float(b) - eps

        # ---------------- Source OHLCV ----------------
        src = None
        if cache_key:
            src = CANDLE_CACHE.get(cache_key)
            if src is None:
                return _err("cache_key not found", cache_key=cache_key)
        elif ohlcv is not None:
            src = ohlcv
        else:
            return _err("Missing ohlcv or cache_key")

        df = _df_from_ohlcv(src)
        if df is None or len(df) < 20:
            return _err("Invalid or too short OHLCV")

        df = _compute_indicators_df(df)
        last = df.iloc[-1]
        close = float(last["Close"])
        atr_ltf = float(last["ATR"])

        # ---------------- Tick / Digits ----------------
        prices = pd.concat([df["Open"], df["High"], df["Low"], df["Close"]], ignore_index=True)
        digits_guess = _infer_digits_from_prices(prices)
        tick = _infer_tick_from_prices(prices, digits_guess) or float(Decimal(1).scaleb(-max(digits_guess, 3)))
        eps = _eps_from_tick(tick)

        # ---------------- Profils (config) ----------------
        PROFILE = CONFIG["PROFILES"]
        prof = PROFILE.get(horizon.lower(), PROFILE["swing"])

        # ---------------- ATR de base (HTF si fournie) ----------------
        atr_basis = atr_ltf
        htf_df = None
        if (anchor_tf in ("htf", "auto")) and htf_ohlcv:
            try:
                htf_df = _df_from_ohlcv(htf_ohlcv)
                if htf_df is not None and len(htf_df) >= 20:
                    htf_df = _compute_indicators_df(htf_df)
                    atr_htf_last = float(htf_df["ATR"].iloc[-1])
                    if atr_htf_last and np.isfinite(atr_htf_last):
                        atr_basis = atr_htf_last
            except Exception:
                htf_df = None  # ignore si échec

        # ---------------- Planchers (ticks + ATR_basis) + buffer ----------------
        min_stop_ticks = int(prof.get("min_ticks", 100))
        if atr_basis and tick > 0:
            min_stop_ticks = max(min_stop_ticks, int(math.ceil(CONFIG["LEVELS_MIN_ATR_FRACTION"] * atr_basis / tick)))
        min_stop_price = float(min_stop_ticks * tick)
        spread_buffer  = max(2 * tick, float(prof.get("spread_frac", 0.08)) * min_stop_price)

        # ---------------- Plancher pivot HTF (structure) ----------------
        struct_floor = 0.0
        if htf_df is not None and len(htf_df) >= 30:
            try:
                piv = _zigzag_swings(htf_df, atr_mult=max(1.8, float(fib_atr_mult)), min_bars=3)
                if action.upper() == "BUY":
                    for i in range(len(piv)-1, -1, -1):
                        if piv[i][2] == 'L':
                            struct_floor = abs(close - float(piv[i][1])) + 10 * tick
                            break
                elif action.upper() == "SELL":
                    for i in range(len(piv)-1, -1, -1):
                        if piv[i][2] == 'H':
                            struct_floor = abs(float(piv[i][1]) - close) + 10 * tick
                            break
            except Exception:
                struct_floor = 0.0

        # ---------------- Distances de base (réactivité LTF) ----------------
        rr_target = float(prof.get("rr_target", 1.8))
        sl_dist0 = float(prof.get("sl_atr_mult", 1.1)) * atr_ltf
        tp_dist0 = float(prof.get("tp_atr_mult", 2.2)) * atr_ltf

        # ---------------- Distance minimale requise ----------------
        req_dist = max(min_stop_price + spread_buffer, struct_floor)
        sl_dist  = max(sl_dist0, req_dist)
        tp_dist  = max(tp_dist0, rr_target * sl_dist, req_dist)

        entry_ref = close

        # ---------------- SL/TP init ----------------
        if action.upper() == "BUY":
            sl = _round_to_tick(entry_ref - sl_dist, tick)
            tp = _round_to_tick(entry_ref + tp_dist, tick)
        elif action.upper() == "SELL":
            sl = _round_to_tick(entry_ref + sl_dist, tick)
            tp = _round_to_tick(entry_ref - tp_dist, tick)
        else:
            return _err("action must be BUY or SELL")

        rr = float(tp_dist / sl_dist) if sl_dist > 0 else None

        # ---------------- Fibonacci (optionnel) ----------------
        fib_used = False
        fib_tp_raw: Optional[float] = None
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
                            tp_cand_round = _round_to_tick(tp_cand, tick)
                            if (
                                tp_cand_round
                                and _ge(tp_cand_round, entry_ref + req_dist, eps)
                                and ((tp_cand_round - entry_ref) / sl_dist) >= rr_target
                            ):
                                tp = tp_cand_round
                                tp_dist = tp - entry_ref
                                rr = tp_dist / sl_dist if sl_dist > 0 else None
                                fib_used = True
                                fib_tp_raw = tp_cand
                                break
                    elif action.upper() == "SELL" and down_segment:
                        for tp_cand in sorted(list(fib["ext_down"].values()), reverse=True):
                            tp_cand_round = _round_to_tick(tp_cand, tick)
                            if (
                                tp_cand_round
                                and _le(tp_cand_round, entry_ref - req_dist, eps)
                                and ((entry_ref - tp_cand_round) / sl_dist) >= rr_target
                            ):
                                tp = tp_cand_round
                                tp_dist = entry_ref - tp
                                rr = tp_dist / sl_dist if sl_dist > 0 else None
                                fib_used = True
                                fib_tp_raw = tp_cand
                                break
            except Exception as e:
                logger.debug(f"[MCP:ANALYSIS] Fibonacci step skipped due to: {e}")

        # ---------------- Contraintes finales (après arrondis) + tolérance ----------------
        if action.upper() == "BUY":
            ok_side = _le(sl, entry_ref - req_dist, eps) and _ge(tp, entry_ref + req_dist, eps)
            if not ok_side:
                # Forcer un TP conforme si l'arrondi l'a raccourci
                tp = _round_to_tick(entry_ref + max(tp_dist0, rr_target * sl_dist, req_dist), tick)
                ok_side = _le(sl, entry_ref - req_dist, eps) and _ge(tp, entry_ref + req_dist, eps)
                if not ok_side:
                    return _err(
                        "Constraints not satisfied after rounding (BUY)",
                        entry_ref=float(entry_ref), sl=float(sl), tp=float(tp),
                        req=float(req_dist), tick=float(tick), eps=float(eps)
                    )
        else:  # SELL
            ok_side = _ge(sl, entry_ref + req_dist, eps) and _le(tp, entry_ref - req_dist, eps)
            if not ok_side:
                tp = _round_to_tick(entry_ref - max(tp_dist0, rr_target * sl_dist, req_dist), tick)
                ok_side = _ge(sl, entry_ref + req_dist, eps) and _le(tp, entry_ref - req_dist, eps)
                if not ok_side:
                    return _err(
                        "Constraints not satisfied after rounding (SELL)",
                        entry_ref=float(entry_ref), sl=float(sl), tp=float(tp),
                        req=float(req_dist), tick=float(tick), eps=float(eps)
                    )

        payload = {
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
                "action": action.upper(),
                "fib_used": fib_used,
                "fib_tp_raw": float(fib_tp_raw) if fib_tp_raw is not None else None,
                "anchor_tf": anchor_tf,
                "eps": float(eps),
            }
        }
        return _ok(payload)
    except Exception as e:
        return _err("levels_autonomous failed", exc=str(e))


# ================================================================
# --------------- Strategy logic: regime & decision --------------
# ================================================================

def _directional_score(last: Dict[str, Any]) -> float:
    score = 0.0
    # EMA
    if (last.get("EMA_Fast") or 0) >= (last.get("EMA_Slow") or 0):
        score += CONFIG["SCORE_W_EMA"]
    else:
        score -= CONFIG["SCORE_W_EMA"]
    # MACD
    if (last.get("MACD_Line") or 0) >= (last.get("MACD_Signal") or 0):
        score += CONFIG["SCORE_W_MACD"]
    else:
        score -= CONFIG["SCORE_W_MACD"]
    # RSI
    rsi_v = last.get("RSI") or 50
    if rsi_v >= CONFIG["DEC_RSI_POS"]:
        score += CONFIG["SCORE_W_RSI"]
    elif rsi_v <= CONFIG["DEC_RSI_NEG"]:
        score -= CONFIG["SCORE_W_RSI"]
    # Position vs BB_Mid
    if (last.get("Close") or 0) >= (last.get("BB_Mid") or 0):
        score += CONFIG["SCORE_W_BBPOS"]
    else:
        score -= CONFIG["SCORE_W_BBPOS"]
    return float(score)

def _confidence(last: Dict[str, Any], score_total: float) -> int:
    atr_pct = (last.get("ATR_PCT") or 0.0)
    damp = max(0.0, min(CONFIG["CONF_DAMP_CAP"], (atr_pct - CONFIG["CONF_DAMP_START"]) / max(CONFIG["CONF_DAMP_RANGE"], 1e-9)))
    conf = round((abs(score_total) / 4.0) * (1 - damp) * 100)
    return int(max(0, min(100, conf)))

def _regime_from_df(df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> str:
    """Trend si BBW% élevé **et** EMAs alignées sur HTF. Range sinon.
       No-trade seulement pour vol extrême haute, ou calme extrême + squeeze."""
    last_ltf = _last_row(df_ltf)
    last_htf = _last_row(df_htf)

    atr_pct = last_ltf.get("ATR_PCT") or 0.0     # fraction (0.0008 = 0.08 %)
    bbw     = last_ltf.get("BBW_PCT") or 0.0

    # Kill-switch haut
    if atr_pct > CONFIG["REGIME_ATR_HIGH"]:
        return "no-trade"

    # Calme extrême seulement si squeeze prononcé
    if atr_pct < CONFIG["REGIME_ATR_LOW"] and bbw < CONFIG["REGIME_SQUEEZE_BBW"]:
        return "no-trade"

    ema_align_ltf = (last_ltf.get("EMA_Fast") or 0) >= (last_ltf.get("EMA_Slow") or 0)
    ema_align_htf = (last_htf.get("EMA_Fast") or 0) >= (last_htf.get("EMA_Slow") or 0)

    # Seuil BBW pour "trend"
    trend_like = (bbw >= CONFIG["REGIME_BBW_TREND"]) and (ema_align_ltf == ema_align_htf)
    return "trend" if trend_like else "range"

def _fetch_and_features(symbol: str, period_ltf: str, interval_ltf: str, period_htf: str, interval_htf: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # LTF
    ltf_raw = meta_api.get_historical_candles(symbol, period_ltf, interval_ltf)
    ltf = _df_from_ohlcv(json.loads(ltf_raw)["data"] if isinstance(ltf_raw, str) else ltf_raw["data"])
    ltf = _compute_indicators_df(ltf)
    # HTF
    htf_raw = meta_api.get_historical_candles(symbol, period_htf, interval_htf)
    htf = _df_from_ohlcv(json.loads(htf_raw)["data"] if isinstance(htf_raw, str) else htf_raw["data"])
    htf = _compute_indicators_df(htf)
    return ltf, htf

def _position_size(entry: float, sl: float, equity: float, risk_pct: float, cap_leverage: float, price_mult: float = 1.0) -> Dict[str, Any]:
    """Calcule une taille générique: units = min(Risk€/|entry-sl|, equity*cap_leverage*price_mult)."""
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

# ================================================================
# -------------------- Intraday Decision Tool --------------------
# ================================================================

@mcp.tool()
def intraday_decision(
    symbol: str,
    interval: str = "15m",          # "5m" ou "15m"
    equity: float = 10000.0,         # capital de référence en EUR
    risk_pct: float = 0.005,         # 0.5% par trade
    cap_leverage: float = 5.0,       # plafond notionnel
    risk_level: str = "medium",

    # ---- Filtre de volatilité (configurable) ----
    vol_enabled: Optional[bool] = None,
    lookback_days: Optional[int] = None,
    vol_low_pct: Optional[int] = None,
    vol_high_pct: Optional[int] = None,
    vol_extreme_pct: Optional[int] = None,
    vol_size_high: Optional[float] = None,
    require_htf_on_edges: Optional[bool] = None,

    # option coupe-chop
    trend_only: Optional[bool] = None,
) -> str:
    """Décision intraday avec filtre de volatilité dynamique (ATR_PCT percentiles) + option trend-only.
       Note: ATR_PCT est une fraction (0.05 = 5 %) dans toute la logique."""
    try:
        interval = interval.lower()
        if interval not in {"5m", "15m"}:
            interval = "15m"
        # périodes adéquates pour avoir assez d'historique
        if interval == "5m":
            period_ltf = f"{CONFIG['LTF_PERIOD_5M']}d"
        else:
            period_ltf = f"{CONFIG['LTF_PERIOD_15M']}d"  # ~ 1mo
        period_htf = f"{CONFIG['HTF_PERIOD_DAYS']}d"
        interval_htf = CONFIG["HTF_INTERVAL"]

        # Defaults via CONFIG
        vol_enabled    = CONFIG["VOL_ENABLED"]     if vol_enabled    is None else vol_enabled
        lookback_days  = CONFIG["VOL_LOOKBACK_DAYS"] if lookback_days  is None else lookback_days
        vol_low_pct    = CONFIG["VOL_LOW_PCT"]       if vol_low_pct    is None else vol_low_pct
        vol_high_pct   = CONFIG["VOL_HIGH_PCT"]      if vol_high_pct   is None else vol_high_pct
        vol_extreme_pct= CONFIG["VOL_EXTREME_PCT"]   if vol_extreme_pct is None else vol_extreme_pct
        vol_size_high  = CONFIG["VOL_SIZE_HIGH"]     if vol_size_high  is None else vol_size_high
        trend_only     = CONFIG["TREND_ONLY"]        if trend_only     is None else trend_only
        require_htf_on_edges = CONFIG["REQUIRE_HTF_ON_EDGES"] if require_htf_on_edges is None else require_htf_on_edges

        # Feature engineering multi-TF
        ltf, htf = _fetch_and_features(symbol, period_ltf, interval, period_htf, interval_htf)
        last_ltf = _last_row(ltf)
        last_htf = _last_row(htf)

        # ---------------- Volatility gating (percentiles) ----------------
        vol_meta = None
        if vol_enabled:
            bands = _atr_pct_bands_from_df(
                ltf, lookback_days=lookback_days,
                low_pct=vol_low_pct, high_pct=vol_high_pct, extreme_pct=vol_extreme_pct
            )
            if bands is not None:
                allowed, band, size_factor, reason_code = _volatility_gate(
                    bands["atr_now"], bands["p10"], bands["p90"], bands.get("p95"), size_high=vol_size_high
                )
                vol_meta = {"atr_pct_now": bands["atr_now"], "p10": bands["p10"], "p90": bands["p90"], "p95": bands.get("p95"),
                            "band": band, "size_factor": size_factor, "reason": reason_code}

                # Hard gate si LOW ou EXTREME
                if not allowed:
                    out = {
                        "symbol": symbol,
                        "interval": interval,
                        "regime": "no-trade",
                        "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": 0, "risk_level": risk_level},
                        "reason": (
                            f"Volatility gate {band} ({reason_code}): "
                            f"ATR%={round(100.0*(bands['atr_now'] or 0.0), 4)} "
                            f"vs p10={round(100.0*(bands['p10'] or 0.0), 4)} "
                            f"p95={round(100.0*((bands.get('p95') or 0.0)), 4)}."
                        ),
                        "volatility": vol_meta,
                    }
                    return _ok(out)
            # (si pas de bands disponibles: on continue sans gating)

        # ---------------- Régime (bornes globales) -----------------------
        regime = _regime_from_df(ltf, htf)

        # Coupe-chop (trend_only)
        if trend_only and regime != "trend":
            out = {
                "symbol": symbol,
                "interval": interval,
                "regime": regime,
                "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": 0, "risk_level": risk_level},
                "reason": "Filtre trend-only activé (on évite les ranges).",
                "volatility": vol_meta,
            }
            return _ok(out)

        if regime == "no-trade":
            out = {
                "symbol": symbol,
                "interval": interval,
                "regime": regime,
                "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": 0, "risk_level": risk_level},
                "reason": "Régime no-trade (ATR% extrême ou trop faible).",
                "volatility": vol_meta,
            }
            return _ok(out)

        # ---------------- Score / Confluence -----------------------------
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
        else:  # range
            if score <= CONFIG["DEC_RANGE_SELL_SCORE"] and (not htf_up):
                action = "SELL"
            elif score >= CONFIG["DEC_RANGE_BUY_SCORE"] and htf_up:
                action = "BUY"

        # Si on est “au bord” (band HIGH) et require_htf_on_edges, impose confluence HTF stricte
        if vol_enabled and vol_meta and vol_meta["band"] == "HIGH" and require_htf_on_edges:
            if (action == "BUY" and not (ltf_up and htf_up)) or (action == "SELL" and not ((not ltf_up) and (not htf_up))):
                action = "HOLD"

        # Min confidence optionnel via profil
        horizon = "scalping" if interval in {"5m", "15m"} else "swing"
        prof = CONFIG["PROFILES"].get(horizon, {})
        min_conf = prof.get("min_confidence", None)
        if action != "HOLD" and isinstance(min_conf, (int, float)) and conf < float(min_conf):
            action = "HOLD"

        if action == "HOLD":
            out = {
                "symbol": symbol,
                "interval": interval,
                "regime": regime,
                "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": conf, "risk_level": risk_level},
                "reason": f"Score={score:.2f}, confluence HTF insuffisante ou confiance<{min_conf}.",
                "volatility": vol_meta,
            }
            return _ok(out)

        # ========= Niveaux via levels_autonomous (scalping 15m + ancrage D1) =========
        d1_raw = meta_api.get_historical_candles(symbol, f"{CONFIG['D1_PERIOD_MONTHS']}mo", "1d")
        d1 = _df_from_ohlcv(json.loads(d1_raw)["data"] if isinstance(d1_raw, str) else d1_raw["data"])

        levels_json = levels_autonomous.__wrapped__(
            ohlcv=json.loads(ltf.tail(300).to_json(orient="records", date_format="iso")),
            action=action,
            horizon=horizon,
            risk_level=risk_level,
            use_fib=True,
            fib_atr_mult=2.0,
            anchor_tf="htf",
            htf_ohlcv=(json.loads(d1.tail(200).to_json(orient="records", date_format="iso")) if d1 is not None else None),
        )
        levels = json.loads(levels_json)
        if not levels.get("ok"):
            return _err("levels_autonomous failed inside intraday_decision", inner=levels)
        lv = levels["data"]
        entry = 0  # market
        sl = lv.get("sl")
        tp = lv.get("tp")

        # Position sizing (simple) + éventuelle réduction en band HIGH
        size = _position_size(entry=lv.get("entry_ref") or last_ltf.get("Close"), sl=sl, equity=equity, risk_pct=risk_pct, cap_leverage=cap_leverage)
        size_factor = vol_meta["size_factor"] if (vol_enabled and vol_meta) else 1.0
        if size_factor < 1.0 and size.get("units", 0) > 0:
            size["units"] = float(size["units"]) * float(size_factor)
            size["size_factor_vol"] = size_factor

        plan_mgmt = {
            "move_be_at_R": CONFIG["MANAGE_MOVE_BE_AT_R"],
            "partial_exit_at_R": CONFIG["MANAGE_PARTIAL_AT_R"],
            "partial_fraction": CONFIG["MANAGE_PARTIAL_FRAC"],
            "trail_at_R": CONFIG["MANAGE_TRAIL_AT_R"],
            "trail_type": CONFIG["MANAGE_TRAIL_TYPE"],
            "trail_len": CONFIG["MANAGE_TRAIL_LEN"],
            "time_stop_bars": CONFIG["MANAGE_TIME_STOP_BARS"],
        }

        reason = (
            f"Regime={regime}, Score={score:.2f}, EMA_LTF={'up' if ltf_up else 'down'}, EMA_HTF={'up' if htf_up else 'down'}, "
            f"ATR%={round(100.0*(last_ltf.get('ATR_PCT') or 0.0), 4)}, BBW%={round(last_ltf.get('BBW_PCT') or 0, 2)}. "
            f"Tick={round(lv['meta'].get('tick') or 0, 6)}, MinStop≈{round(lv['meta'].get('min_stop_price') or 0, 6)}, "
            f"Buffer≈{round(lv['meta'].get('spread_buffer') or 0, 6)}, StructFloor≈{round(lv['meta'].get('struct_floor') or 0, 6)}, "
            f"FibUsed={lv['meta'].get('fib_used')}, VolBand={(vol_meta or {}).get('band', 'NA')}"
        )

        out = {
            "symbol": symbol,
            "interval": interval,
            "regime": regime,
            "decision": {
                "action": action,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "confidence": conf,
                "risk_level": risk_level,
            },
            "levels": lv,
            "position": size,
            "management": plan_mgmt,
            "reason": reason,
            "volatility": vol_meta,
        }
        return _ok(out)
    except Exception as e:
        return _err("intraday_decision failed", exc=str(e))

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
