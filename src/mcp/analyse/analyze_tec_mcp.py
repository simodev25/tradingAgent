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

# Import meta_api
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../data_fetch/')))
import meta_api as meta_api

mcp = FastMCP("Trading Analysis MCP Server", log_level="DEBUG")

logger.remove()
logger.add(sys.stderr, level="DEBUG")

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
        df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
          .drop_duplicates(subset=["Date"])
          .sort_values("Date")
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
    rsi_len: int = 14,
    ema_fast: int = 12,
    ema_slow: int = 26,
    macd_signal: int = 9,
    atr_len: int = 14,
    bb_len: int = 20,
    bb_mult: float = 2.0,
) -> pd.DataFrame:
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

    df["ATR_PCT"] = np.where(df["Close"] > 0, (df["ATR"] / df["Close"]) * 100.0, np.nan)

    last_atr = float(df["ATR"].iloc[-1]) if len(df) else None
    point = float(tick_guess if tick_guess else (10 ** (-(digits_guess if digits_guess else 3))))
    min_stop_ticks = 200
    if last_atr and point > 0:
        min_stop_ticks = max(min_stop_ticks, int(math.ceil(0.15 * last_atr / point)))
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
    rsi_len: int = 14,
    ema_fast: int = 12,
    ema_slow: int = 26,
    macd_signal: int = 9,
    atr_len: int = 14,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    last_only: bool = True,
) -> str:
    """Calcule indicateurs + infos marché dérivées de l'OHLCV (TickSize/Digits/MinStop/ATR%/BBW%)."""
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
    """Calcule p10/p90/p95 de ATR_PCT sur la fenêtre 'lookback_days' (fallback: dernières 2000 barres)."""
    if df is None or "Date" not in df.columns or "ATR_PCT" not in df.columns or df.empty:
        return None
    try:
        cutoff = df["Date"].max() - pd.Timedelta(days=int(lookback_days))
        s = df.loc[df["Date"] >= cutoff, "ATR_PCT"].dropna()
        if s.size < 50:  # fallback si peu d'historique dans la fenêtre
            s = df["ATR_PCT"].dropna().tail(2000)
        if s.size < 20:
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
    """Classe LOW/NORMAL/HIGH/EXTREME + size_factor & reason."""
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
    lookback_days: int = 30,
    low_pct: int = 10,
    high_pct: int = 90,
    extreme_pct: int = 95,
    size_high: float = 0.5,
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
) -> str:
    """Plan ATR simple, harmonisé avec contraintes (min stop, buffer, RR)."""
    try:
        df = _df_from_ohlcv(ohlcv)
        if df is None:
            return _err("Invalid OHLCV")
        df = _compute_indicators_df(df)
        last = _last_row(df)
        atr_val, close = last.get("ATR"), last.get("Close")
        if atr_val is None or close is None:
            return _err("ATR/Close missing")
        side = direction.lower()
        if side not in {"long", "short"}:
            ema_fast = last.get("EMA_Fast") or 0.0
            ema_slow = last.get("EMA_Slow") or 0.0
            rsi_v = last.get("RSI") or 50.0
            macd_line = last.get("MACD_Line") or 0.0
            macd_sig  = last.get("MACD_Signal") or 0.0
            bullish_bias = (ema_fast >= ema_slow) and (rsi_v >= 50 or macd_line >= macd_sig)
            bearish_bias = (ema_fast <  ema_slow) and (rsi_v <= 50 or macd_line <= macd_sig)
            side = "long" if bullish_bias and not bearish_bias else ("short" if bearish_bias and not bullish_bias else ("long" if ema_fast >= ema_slow else "short"))

        # Tick & contraintes
        prices = pd.concat([df["Open"], df["High"], df["Low"], df["Close"]], ignore_index=True)
        digits_guess = _infer_digits_from_prices(prices)
        tick = tick_size or _infer_tick_from_prices(prices, digits_guess) or float(Decimal(1).scaleb(-max(digits_guess, 3)))
        point = tick
        min_stop_ticks = 200
        if atr_val and point > 0:
            min_stop_ticks = max(min_stop_ticks, int(math.ceil(0.15 * atr_val / point)))
        min_stop_price = float(min_stop_ticks * point)
        spread_buffer = max(5 * point, 0.10 * min_stop_price)
        req_dist = min_stop_price + spread_buffer

        tp_mult = {"low": 1.5, "medium": 2.0, "high": 3.0}.get(risk_level.lower(), 2.0)
        sl_dist = max(1.0 * atr_val, req_dist)
        tp_dist = max(tp_mult * atr_val, 1.5 * sl_dist, req_dist)

        if side == "long":
            sl = _round_to_tick(close - sl_dist, tick)
            tp = _round_to_tick(close + tp_dist, tick)
        else:
            sl = _round_to_tick(close + sl_dist, tick)
            tp = _round_to_tick(close - tp_dist, tick)

        return _ok({
            "entry": _round_to_tick(close, tick),
            "sl": sl, "tp": tp,
            "atr": atr_val,
            "risk_level": risk_level, "side": side,
            "meta": {"tick": tick, "min_stop_price": min_stop_price, "spread_buffer": spread_buffer}
        })
    except Exception as e:
        return _err("plan_raw failed", exc=str(e))


@mcp.tool()
def get_historical_candles(symbol: str, period: str = "1mo", interval: str = "1d", compact: bool = True) -> str:
    """Récupération OHLCV via meta_api."""
    try:
        data = meta_api.get_historical_candles(symbol, period, interval)
        logger.debug(f"[MCP:ANALYSIS] get_historical_candles {symbol}:{period}:{interval}")
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
    - Plancher stop = max(min_ticks, 12% ATR_basis) + buffer spread
    - ATR_basis = ATR(LTF) ou ATR(HTF) si fourni (D1 conseillé pour 15m)
    - Pivot HTF (dernier creux/haut) ajoute un **plancher structurel**
    - R/R cible adaptatif: scalping 1.5, daytrade 1.8, swing 2.0
    - Fibonacci optionnel pour étendre le TP si disponible
    """
    try:
        # Source OHLCV
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

        # Tick / Digits
        prices = pd.concat([df["Open"], df["High"], df["Low"], df["Close"]], ignore_index=True)
        digits_guess = _infer_digits_from_prices(prices)
        tick = _infer_tick_from_prices(prices, digits_guess) or float(Decimal(1).scaleb(-max(digits_guess, 3)))

        # Profils par horizon
        PROFILE = {
            "scalping": {"rr_target": 1.5, "sl_atr_mult": 0.9, "tp_atr_mult": 1.8, "min_ticks": 60,  "spread_frac": 0.06},
            "daytrade": {"rr_target": 1.8, "sl_atr_mult": 1.0, "tp_atr_mult": 2.1, "min_ticks": 120, "spread_frac": 0.08},
            "swing":    {"rr_target": 2.0, "sl_atr_mult": 1.2, "tp_atr_mult": 2.4, "min_ticks": 200, "spread_frac": 0.10},
        }
        prof = PROFILE.get(horizon.lower(), PROFILE["swing"])

        # ATR de base (HTF si fournie)
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

        # Planchers (ticks + ATR_basis) + buffer
        min_stop_ticks = prof["min_ticks"]
        if atr_basis and tick > 0:
            min_stop_ticks = max(min_stop_ticks, int(math.ceil(0.12 * atr_basis / tick)))
        min_stop_price = float(min_stop_ticks * tick)
        spread_buffer  = max(2 * tick, prof["spread_frac"] * min_stop_price)

        # Plancher pivot HTF (structure)
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

        # Distances de base (réactivité LTF)
        rr_target = prof["rr_target"]
        sl_dist0 = prof["sl_atr_mult"] * atr_ltf
        tp_dist0 = prof["tp_atr_mult"] * atr_ltf

        # Distance minimale requise
        req_dist = max(min_stop_price + spread_buffer, struct_floor)
        sl_dist  = max(sl_dist0, req_dist)
        tp_dist  = max(tp_dist0, rr_target * sl_dist, req_dist)

        entry_ref = close

        # SL/TP init
        if action.upper() == "BUY":
            sl = _round_to_tick(entry_ref - sl_dist, tick)
            tp = _round_to_tick(entry_ref + tp_dist, tick)
        elif action.upper() == "SELL":
            sl = _round_to_tick(entry_ref + sl_dist, tick)
            tp = _round_to_tick(entry_ref - tp_dist, tick)
        else:
            return _err("action must be BUY or SELL")

        rr = float(tp_dist / sl_dist) if sl_dist > 0 else None

        # Fibonacci (optionnel)
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
                            if tp_cand_round and tp_cand_round >= entry_ref + req_dist and (tp_cand_round - entry_ref) / sl_dist >= rr_target:
                                tp = tp_cand_round
                                tp_dist = tp - entry_ref
                                rr = tp_dist / sl_dist if sl_dist > 0 else None
                                fib_used = True
                                fib_tp_raw = tp_cand
                                break
                    elif action.upper() == "SELL" and down_segment:
                        for tp_cand in sorted(list(fib["ext_down"].values()), reverse=True):
                            tp_cand_round = _round_to_tick(tp_cand, tick)
                            if tp_cand_round and tp_cand_round <= entry_ref - req_dist and (entry_ref - tp_cand_round) / sl_dist >= rr_target:
                                tp = tp_cand_round
                                tp_dist = entry_ref - tp
                                rr = tp_dist / sl_dist if sl_dist > 0 else None
                                fib_used = True
                                fib_tp_raw = tp_cand
                                break
            except Exception as e:
                logger.debug(f"[MCP:ANALYSIS] Fibonacci step skipped due to: {e}")

        # Contraintes finales (après arrondis)
        if action.upper() == "BUY":
            if not (sl <= entry_ref - req_dist and tp >= entry_ref + req_dist):
                tp = _round_to_tick(entry_ref + max(tp_dist0, rr_target * sl_dist, req_dist), tick)
                if not (sl <= entry_ref - req_dist and tp >= entry_ref + req_dist):
                    return _err("Constraints not satisfied after rounding (BUY)",
                                entry_ref=entry_ref, sl=sl, tp=tp, req=req_dist, tick=tick)
        else:
            if not (sl >= entry_ref + req_dist and tp <= entry_ref - req_dist):
                tp = _round_to_tick(entry_ref - max(tp_dist0, rr_target * sl_dist, req_dist), tick)
                if not (sl >= entry_ref + req_dist and tp <= entry_ref - req_dist):
                    return _err("Constraints not satisfied after rounding (SELL)",
                                entry_ref=entry_ref, sl=sl, tp=tp, req=req_dist, tick=tick)

        payload = {
            "entry_ref": _round_to_tick(entry_ref, tick),
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "meta": {
                "tick": tick,
                "digits_guess": digits_guess,
                "atr_ltf": atr_ltf,
                "atr_basis": atr_basis,
                "min_stop_price": min_stop_price,
                "spread_buffer": spread_buffer,
                "struct_floor": struct_floor,
                "sl_dist_final": sl_dist,
                "tp_dist_final": tp_dist,
                "rr_target": rr_target,
                "horizon": horizon,
                "risk_level": risk_level,
                "action": action.upper(),
                "fib_used": fib_used,
                "fib_tp_raw": fib_tp_raw,
                "anchor_tf": anchor_tf,
            }
        }
        return _ok(payload)
    except Exception as e:
        return _err("levels_autonomous failed", exc=str(e))


# ================================================================
# --------------- Strategy logic: regime & decision --------------
# ================================================================

def _directional_score(last: Dict[str, Any]) -> int:
    score = 0
    if (last.get("EMA_Fast") or 0) >= (last.get("EMA_Slow") or 0):
        score += 1
    else:
        score -= 1
    if (last.get("MACD_Line") or 0) >= (last.get("MACD_Signal") or 0):
        score += 1
    else:
        score -= 1
    rsi_v = last.get("RSI") or 50
    if rsi_v >= 55:
        score += 1
    elif rsi_v <= 45:
        score -= 1
    # Position vs BB_Mid
    if (last.get("Close") or 0) >= (last.get("BB_Mid") or 0):
        score += 1
    else:
        score -= 1
    return int(score)


def _confidence(last: Dict[str, Any], score_total: int) -> int:
    atr_pct = (last.get("ATR_PCT") or 0.0)
    damp = max(0.0, min(0.4, (atr_pct - 2.0) / 5.0))
    conf = round((abs(score_total) / 4.0) * (1 - damp) * 100)
    return int(max(0, min(100, conf)))


def _regime_from_df(df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> str:
    """Trend si BBW% élevé **et** EMAs alignées sur HTF. Range sinon. No-trade si ATR% extrêmes."""
    last_ltf = _last_row(df_ltf)
    last_htf = _last_row(df_htf)
    atr_pct = last_ltf.get("ATR_PCT") or 0.0
    if atr_pct > 5.0 or atr_pct < 0.3:
        return "no-trade"
    bbw = last_ltf.get("BBW_PCT") or 0.0
    ema_align = (last_ltf.get("EMA_Fast") or 0) >= (last_ltf.get("EMA_Slow") or 0)
    ema_align_htf = (last_htf.get("EMA_Fast") or 0) >= (last_htf.get("EMA_Slow") or 0)
    trend_like = bbw >= 6.0 and (ema_align == ema_align_htf)
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
    vol_enabled: bool = True,
    lookback_days: int = 30,
    vol_low_pct: int = 10,
    vol_high_pct: int = 90,
    vol_extreme_pct: int = 95,
    vol_size_high: float = 0.5,      # taille réduite si p90 < ATR% ≤ p95
    require_htf_on_edges: bool = True,
) -> str:
    """Décision intraday avec filtre de volatilité dynamique (ATR% percentiles)."""
    try:
        interval = interval.lower()
        if interval not in {"5m", "15m"}:
            interval = "15m"
        # périodes adéquates pour avoir assez d'historique
        period_ltf = "5d" if interval == "5m" else "1mo"
        period_htf = "1mo"
        interval_htf = "1h"

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
                        "reason": f"Volatility gate {band} ({reason_code}): ATR%={round(bands['atr_now'], 3)} "
                                  f"vs p10={round(bands['p10'],3)} p95={round(bands.get('p95') or 0,3)}.",
                        "volatility": vol_meta,
                    }
                    return _ok(out)
            # (si pas de bands disponibles: on continue sans gating)

        # ---------------- Régime (conserve tes bornes globales) ----------
        regime = _regime_from_df(ltf, htf)
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
            if score >= 2 and ltf_up and htf_up:
                action = "BUY"
            elif score <= -2 and (not ltf_up) and (not htf_up):
                action = "SELL"
        else:  # range
            if score <= -2 and (not htf_up):
                action = "SELL"
            elif score >= 2 and htf_up:
                action = "BUY"

        # Si on est “au bord” (band HIGH) et require_htf_on_edges, impose confluence HTF stricte
        if vol_enabled and vol_meta and vol_meta["band"] == "HIGH" and require_htf_on_edges:
            if (action == "BUY" and not (ltf_up and htf_up)) or (action == "SELL" and not ((not ltf_up) and (not htf_up))):
                action = "HOLD"

        if action == "HOLD":
            out = {
                "symbol": symbol,
                "interval": interval,
                "regime": regime,
                "decision": {"action": "HOLD", "entry": None, "sl": None, "tp": None, "confidence": conf, "risk_level": risk_level},
                "reason": f"Score={score}, confluence HTF insuffisante pour {regime}.",
                "volatility": vol_meta,
            }
            return _ok(out)

        # ========= Niveaux via levels_autonomous (scalping 15m + ancrage D1) =========
        horizon = "scalping" if interval in {"5m", "15m"} else "swing"

        d1_raw = meta_api.get_historical_candles(symbol, "6mo", "1d")
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
            "move_be_at_R": 1.0,
            "partial_exit_at_R": 1.5,
            "partial_fraction": 0.5,
            "trail_at_R": 2.0,
            "trail_type": "ATR",
            "trail_len": 14,
            "time_stop_bars": 8,
        }

        reason = (
            f"Regime={regime}, Score={score}, EMA_LTF={'up' if ltf_up else 'down'}, EMA_HTF={'up' if htf_up else 'down'}, "
            f"ATR%={round(last_ltf.get('ATR_PCT') or 0, 2)}, BBW%={round(last_ltf.get('BBW_PCT') or 0, 2)}. "
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
2) compute_indicators(cache_key="<cache_key>") → dernière ligne (Close, EMA_Fast/Slow, RSI, MACD_Line/Signal, BB_Mid, BB_Upper/Lower, ATR, ATR_PCT, BBW_PCT, TickSize_Guess, Digits_Guess, MinStopPrice_Fallback).
3) volatility_bands(cache_key="<cache_key>", lookback_days=30, low_pct=10, high_pct=90, extreme_pct=95, size_high=0.5)
   → **OBLIGATOIRE** : renvoie { atr_pct_now, p10, p90, p95, band, size_factor, reason }.
4) levels_autonomous(cache_key="<cache_key>", action, horizon="$horizon", risk_level="$risk_level", use_fib=True, fib_atr_mult=2.0) **uniquement si action ≠ HOLD**.

### RÈGLES VOLATILITÉ
- Si band ∈ {LOW, EXTREME} → action="HOLD" (raison = reason du tool).
- Si band == HIGH → tu peux conserver l'action mais **note** size_factor (réduction de taille) dans la sortie.

### RÉGIME
- **no-trade** si ATR_PCT > 5% ou < 0.3%.
- **trend** si BBW_PCT ≥ 6% et EMAs alignées avec la TF au-dessus (ex: 1H ou D1).
- sinon **range**.

### DÉCISION
Score directionnel (−4..+4) :
- EMA : +1 si EMA_Fast ≥ EMA_Slow, sinon −1.
- MACD : +1 si MACD_Line ≥ MACD_Signal, sinon −1.
- RSI : +1 si RSI ≥ 55, −1 si RSI ≤ 45, sinon 0.
- Position Bollinger : +1 si Close ≥ BB_Mid, sinon −1.

BUY si score ≥ +2 **et** confluence avec la TF supérieure; SELL si score ≤ −2 **et** confluence; sinon HOLD.

**Confiance** : `confidence = round((abs(score_total)/4) * (1 - clamp((ATR_PCT - 2.0)/5.0, 0, 0.4)) * 100)`.

### NIVEAUX
- Si **HOLD** → `entry/sl/tp = null`.
- Sinon, **appelle levels_autonomous** avec `action` décidée.
- Si ohlcv a < MIN_BARS (ex 240) ou des indicateurs manquent/NaN :
  1) rappelle get_historical_candles avec une période plus grande (double les jours),
  2) réessaie compute_indicators.
- Si après tentative(s) tu ne peux pas décider, PRODUIS QUAND MÊME la SORTIE JSON
  avec action="HOLD", entry/sl/tp=null, confidence=0, et reason explicite
  (ex: "insufficient bars: have=100, need>=240").
- Tu ne t’arrêtes JAMAIS après un tool_call. Tu dois toujours terminer par la SORTIE JSON stricte.

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
        symbol=symbol, period=period, interval=interval,
        horizon=horizon, risk_level=risk_level,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
