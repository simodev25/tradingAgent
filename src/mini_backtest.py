# mini_backtest.py
import os, json, math, sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
# ====== Meta API import (même chemin que ton projet) ======
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp/data_fetch")))
try:
    import meta_api  # doit exposer get_historical_candles(symbol, period, interval)
except Exception as e:
    print("[ERR] meta_api introuvable:", e)
    meta_api = None

# =========================================================
# --------------------- Helpers ENV -----------------------
# =========================================================
def _env_int(k: str, d: int) -> int:
    try:
        v = os.getenv(k)
        return int(v) if v is not None else d
    except Exception:
        return d

def _env_float(k: str, d: float) -> float:
    try:
        v = os.getenv(k)
        return float(v) if v is not None else d
    except Exception:
        return d

def _env_bool(k: str, d: bool) -> bool:
    v = os.getenv(k)
    if v is None: return d
    return str(v).strip().lower() in {"1","true","yes","y","on"}

# =========================================================
# -------------------- CONFIG (defaults) ------------------
# =========================================================
CFG = {
    "TA_RSI_LEN": _env_int("TA_RSI_LEN", 9),
    "TA_EMA_FAST": _env_int("TA_EMA_FAST", 8),
    "TA_EMA_SLOW": _env_int("TA_EMA_SLOW", 21),
    "TA_MACD_SIGNAL": _env_int("TA_MACD_SIGNAL", 7),
    "TA_ATR_LEN": _env_int("TA_ATR_LEN", 10),
    "TA_BB_LEN": _env_int("TA_BB_LEN", 14),
    "TA_BB_MULT": _env_float("TA_BB_MULT", 1.8),

    "VOL_ENABLED": _env_bool("VOL_ENABLED", True),
    "VOL_LOOKBACK_DAYS": _env_int("VOL_LOOKBACK_DAYS", 40),
    "VOL_LOW_PCT": _env_int("VOL_LOW_PCT", 5),
    "VOL_HIGH_PCT": _env_int("VOL_HIGH_PCT", 92),
    "VOL_EXTREME_PCT": _env_int("VOL_EXTREME_PCT", 98),
    "VOL_SIZE_HIGH": _env_float("VOL_SIZE_HIGH", 0.7),

    "REGIME_ATR_HIGH": _env_float("REGIME_ATR_HIGH", 0.006),     # 0.6%
    "REGIME_ATR_LOW":  _env_float("REGIME_ATR_LOW", 0.00025),    # 0.025%
    "REGIME_BBW_TREND": _env_float("REGIME_BBW_TREND", 4.2),
    "REGIME_SQUEEZE_BBW": _env_float("REGIME_SQUEEZE_BBW", 2.6),

    "DEC_RSI_POS": _env_int("DEC_RSI_POS", 53),
    "DEC_RSI_NEG": _env_int("DEC_RSI_NEG", 47),
    "SCORE_W_EMA": _env_float("SCORE_W_EMA", 1.0),
    "SCORE_W_MACD": _env_float("SCORE_W_MACD", 1.0),
    "SCORE_W_RSI": _env_float("SCORE_W_RSI", 1.0),
    "SCORE_W_BBPOS": _env_float("SCORE_W_BBPOS", 1.0),
    "DEC_TREND_BUY_SCORE": _env_int("DEC_TREND_BUY_SCORE", 1),
    "DEC_TREND_SELL_SCORE": _env_int("DEC_TREND_SELL_SCORE", -1),
    "DEC_RANGE_BUY_SCORE": _env_int("DEC_RANGE_BUY_SCORE", 1),
    "DEC_RANGE_SELL_SCORE": _env_int("DEC_RANGE_SELL_SCORE", -1),

    "CONF_DAMP_START": _env_float("CONF_DAMP_START", 0.004),
    "CONF_DAMP_RANGE": _env_float("CONF_DAMP_RANGE", 0.01),
    "CONF_DAMP_CAP":   _env_float("CONF_DAMP_CAP", 0.4),

    # Profil scalping (R en ATR)
    "SCALP_SL_ATR": _env_float("SCALP_SL_ATR", 0.9),
    "SCALP_TP_ATR": _env_float("SCALP_TP_ATR", 1.7),
    "SCALP_MIN_CONF": _env_int("SCALP_MIN_CONF", 55),

    # Exécution backtest
    "RR_MIN": _env_float("RR_MIN", 1.2),
    "MAX_ONE_AT_A_TIME": _env_bool("MAX_ONE_AT_A_TIME", True),  # 1 trade à la fois par symbole
}

# =========================================================
# -------------------- TA primitives ----------------------
# =========================================================
def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / float(length)
    avg_gain = pd.Series(gain, index=close.index).ewm(alpha=alpha, adjust=False).mean()
    avg_loss = pd.Series(loss, index=close.index).ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    r = 100 - (100 / (1 + rs))
    return r.fillna(50.0).clip(0,100)

def macd(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema, slow_ema = ema(close, fast), ema(close, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    alpha = 1.0 / float(length)
    return tr.ewm(alpha=alpha, adjust=False).mean()

def bbands(close: pd.Series, length: int, mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    return ma - mult*sd, ma, ma + mult*sd

def build_df(ohlcv: List[Dict[str,Any]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv).copy()
    if df.empty:
        return df

    # Auto-map colonnes (Open/High/Low/Close/Time)
    cmap = {c.lower(): c for c in df.columns}
    o = cmap.get("open") or cmap.get("o")
    h = cmap.get("high") or cmap.get("h")
    l = cmap.get("low") or cmap.get("l")
    c = cmap.get("close") or cmap.get("c")
    t = cmap.get("time") or cmap.get("datetime") or cmap.get("timestamp") or cmap.get("date")

    df = df.rename(columns={o:"Open", h:"High", l:"Low", c:"Close", t:"Date"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None)

    # Force numérique (au cas où ce soit des strings)
    for col in ("Open","High","Low","Close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date","Open","High","Low","Close"]).sort_values("Date").reset_index(drop=True)

    # TA…
    cfg=CFG
    df["EMA_Fast"] = ema(df["Close"], cfg["TA_EMA_FAST"])
    df["EMA_Slow"] = ema(df["Close"], cfg["TA_EMA_SLOW"])
    df["RSI"] = rsi(df["Close"], cfg["TA_RSI_LEN"])
    macd_line, macd_sig, macd_hist = macd(df["Close"], cfg["TA_EMA_FAST"], cfg["TA_EMA_SLOW"], cfg["TA_MACD_SIGNAL"])
    df["MACD_Line"], df["MACD_Signal"], df["MACD_Hist"] = macd_line, macd_sig, macd_hist
    df["ATR"] = atr(df, cfg["TA_ATR_LEN"])
    bb_l, bb_m, bb_u = bbands(df["Close"], cfg["TA_BB_LEN"], cfg["TA_BB_MULT"])
    df["BB_Lower"], df["BB_Mid"], df["BB_Upper"] = bb_l, bb_m, bb_u
    df["ATR_PCT"] = np.where(df["Close"]>0, df["ATR"]/df["Close"], np.nan)
    df["BBW_PCT"] = np.where(df["BB_Mid"].abs()>0, (df["BB_Upper"]-df["BB_Lower"])/df["BB_Mid"]*100.0, np.nan)
    return df


# =========================================================
# ------------- Volatility gating (percentiles) -----------
# =========================================================
def atr_pct_bands(df: pd.DataFrame, lookback_days: int, p_low:int, p_high:int, p_ext:int) -> Optional[Dict[str,float]]:
    if df.empty: return None
    cutoff = df["Date"].max() - pd.Timedelta(days=lookback_days)
    s = df.loc[df["Date"]>=cutoff, "ATR_PCT"].dropna()
    if s.size < 50:
        s = df["ATR_PCT"].dropna().tail(2000)
    if s.size < 20:
        return None
    return {
        "atr_now": float(df["ATR_PCT"].iloc[-1]),
        "p10": float(np.percentile(s, p_low)),
        "p90": float(np.percentile(s, p_high)),
        "p95": float(np.percentile(s, p_ext)),
    }

def vol_gate(atr_now, p10, p90, p95) -> Tuple[bool,str,float,str]:
    # returns allowed, band, size_factor, reason
    if atr_now is None or not np.isfinite(atr_now):
        return True, "NORMAL", 1.0, "ATR_OK"
    if atr_now < p10:
        return False, "LOW", 0.0, "ATR_GATE_LOW"
    if atr_now > p95:
        return False, "EXTREME", 0.0, "ATR_GATE_HIGH"
    if atr_now > p90:
        return True, "HIGH", CFG["VOL_SIZE_HIGH"], "ATR_HIGH_SIZE_DOWN"
    return True, "NORMAL", 1.0, "ATR_OK"

# =========================================================
# --------------- Regime & score & confidence -------------
# =========================================================
def regime_from(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    atr_pct = float(last["ATR_PCT"] or 0.0)
    bbw = float(last["BBW_PCT"] or 0.0)
    if atr_pct > CFG["REGIME_ATR_HIGH"]:
        return "no-trade"
    if atr_pct < CFG["REGIME_ATR_LOW"] and bbw < CFG["REGIME_SQUEEZE_BBW"]:
        return "no-trade"
    ema_up = last["EMA_Fast"] >= last["EMA_Slow"]
    trend_like = (bbw >= CFG["REGIME_BBW_TREND"]) and ema_up
    return "trend" if trend_like else "range"

def directional_score(last: pd.Series) -> float:
    s=0.0
    s += CFG["SCORE_W_EMA"] if last["EMA_Fast"]>=last["EMA_Slow"] else -CFG["SCORE_W_EMA"]
    s += CFG["SCORE_W_MACD"] if last["MACD_Line"]>=last["MACD_Signal"] else -CFG["SCORE_W_MACD"]
    rsi_v = float(last["RSI"] or 50.0)
    if rsi_v >= CFG["DEC_RSI_POS"]: s += CFG["SCORE_W_RSI"]
    elif rsi_v <= CFG["DEC_RSI_NEG"]: s -= CFG["SCORE_W_RSI"]
    s += CFG["SCORE_W_BBPOS"] if last["Close"] >= (last["BB_Mid"] or 0.0) else -CFG["SCORE_W_BBPOS"]
    return float(s)

def confidence_from(last: pd.Series, score_total: float) -> int:
    atr_pct = float(last["ATR_PCT"] or 0.0)
    damp = max(0.0, min(CFG["CONF_DAMP_CAP"], (atr_pct - CFG["CONF_DAMP_START"]) / max(CFG["CONF_DAMP_RANGE"], 1e-9)))
    conf = round((abs(score_total)/4.0) * (1 - damp) * 100)
    return int(max(0, min(100, conf)))
from typing import Iterable

def unwrap_candles(raw: Any) -> List[Dict[str, Any]]:
    """
    Accepte:
      - list[dict] directement
      - str JSON
      - dict imbriqué: {"ok":true,"data":[...]}, {"data":{"bars":[...]}}, {"result":{...}}, etc.
    Renvoie toujours list[dict].
    """
    obj: Any = raw
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            # parfois l'API renvoie une string JSONL ou autre → on échoue proprement
            raise TypeError("get_historical_candles returned a non-JSON string")

    # forets d’enveloppes possibles → on creuse jusqu’à tomber sur une liste
    for _ in range(6):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # clés fréquentes qui contiennent la liste de barres
            for k in ("data", "Data", "result", "Result", "bars", "candles", "items", "quotes"):
                if k in obj:
                    obj = obj[k]
                    break
            else:
                # si c’est un dict avec un seul champ, on descend dedans
                vals = list(obj.values())
                if len(vals) == 1:
                    obj = vals[0]
                else:
                    break
        else:
            break

    if isinstance(obj, list):
        return obj
    raise TypeError(f"Unsupported get_historical_candles payload: {type(raw)} -> {type(obj)}")
# =========================================================
# ---------------- Backtest core (M15) --------------------
# =========================================================
@dataclass
class Trade:
    open_time: pd.Timestamp
    direction: str          # "long" / "short"
    entry: float
    sl: float
    tp: float
    risk_R: float           # distance SL en R = 1.0
    close_time: Optional[pd.Timestamp]=None
    result_R: Optional[float]=None
import numpy as np
import pandas as pd
import math
from datetime import datetime, date
from typing import Mapping, Sequence, Any

def _to_jsonable(x: Any):
    # Dates / temps
    if isinstance(x, (pd.Timestamp, datetime, date)):
        return x.isoformat()
    if isinstance(x, pd.Timedelta):
        return x.total_seconds()

    # NumPy scalaires
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)

    # Séquences NumPy/Pandas
    if isinstance(x, (np.ndarray, pd.Series)):
        return [_to_jsonable(v) for v in x.tolist()]

    # Containers
    if isinstance(x, Mapping):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return [_to_jsonable(v) for v in x]

    # Floats non finis
    if isinstance(x, float) and not math.isfinite(x):
        return None

    return x
def simulate_symbol(symbol: str, period: str="60d", interval: str="15m") -> Dict[str,Any]:
    # ------- fetch -------
    if meta_api is None:
        raise RuntimeError("meta_api introuvable. Assure-toi que src/data_fetch/meta_api.py est accessible.")
    raw = meta_api.get_historical_candles(symbol, period, interval)
    data = unwrap_candles(raw)
    df = build_df(data)
    if df.shape[0] < 400:
        return {"symbol": symbol, "trades": [], "summary": {"error":"not_enough_bars"}}

    # ------- percentiles de vol -------
    bands = atr_pct_bands(df, CFG["VOL_LOOKBACK_DAYS"], CFG["VOL_LOW_PCT"], CFG["VOL_HIGH_PCT"], CFG["VOL_EXTREME_PCT"])
    # iteration
    trades: List[Trade] = []
    in_pos = False
    cur_trade: Optional[Trade] = None

    # paramètres R (profil scalping)
    sl_mult = CFG["SCALP_SL_ATR"]
    tp_mult = CFG["SCALP_TP_ATR"]

    for i in range(max(CFG["TA_BB_LEN"], CFG["TA_EMA_SLOW"], CFG["TA_ATR_LEN"]) + 2, len(df)):
        row = df.iloc[i]
        last = df.iloc[i-1]  # décision à la clôture précédente, entrée au bar i (open)

        # gating vol (sur la série entière) – statique simple
        allowed, band, size_factor, reason = (True, "NORMAL", 1.0, "ATR_OK")
        if bands:
            allowed, band, size_factor, reason = vol_gate(bands["atr_now"], bands["p10"], bands["p90"], bands["p95"])
        if not allowed:
            # si en position, continue à gérer le trade
            pass

        # régime + score + confiance
        reg = regime_from(df.iloc[:i])  # jusqu'au bar i
        sc = directional_score(last)
        conf = confidence_from(last, sc)

        action = "HOLD"
        if reg == "trend":
            if sc >= CFG["DEC_TREND_BUY_SCORE"] and last["EMA_Fast"]>=last["EMA_Slow"]:
                action="BUY"
            elif sc <= CFG["DEC_TREND_SELL_SCORE"] and last["EMA_Fast"]<last["EMA_Slow"]:
                action="SELL"
        elif reg == "range":
            if sc >= CFG["DEC_RANGE_BUY_SCORE"]:
                action="BUY"
            elif sc <= CFG["DEC_RANGE_SELL_SCORE"]:
                action="SELL"

        # ouverture éventuelle
        if (not in_pos) and action!="HOLD" and conf>=CFG["SCALP_MIN_CONF"] and allowed and reg!="no-trade":
            entry = float(row["Open"])  # entre à l'open de la bougie suivante
            atrv  = float(last["ATR"])
            if not (np.isfinite(entry) and np.isfinite(atrv) and atrv>0): 
                pass
            else:
                if action=="BUY":
                    sl = entry - sl_mult*atrv
                    tp = entry + tp_mult*atrv
                    direction="long"
                else:
                    sl = entry + sl_mult*atrv
                    tp = entry - tp_mult*atrv
                    direction="short"

                # RR check
                rr = (tp - entry)/(entry - sl) if direction=="long" else (entry - tp)/(sl - entry)
                if rr >= CFG["RR_MIN"]:
                    cur_trade = Trade(open_time=row["Date"], direction=direction, entry=entry, sl=sl, tp=tp, risk_R=1.0)
                    in_pos=True

        # gestion position
        if in_pos and cur_trade is not None:
            hi = float(row["High"]); lo=float(row["Low"])
            hit_tp=False; hit_sl=False
            if cur_trade.direction=="long":
                if lo <= cur_trade.sl and hi >= cur_trade.tp:
                    # conservateur: SL d'abord
                    hit_sl=True
                elif hi >= cur_trade.tp:
                    hit_tp=True
                elif lo <= cur_trade.sl:
                    hit_sl=True
            else:
                if hi >= cur_trade.sl and lo <= cur_trade.tp:
                    hit_sl=True
                elif lo <= cur_trade.tp:
                    hit_tp=True
                elif hi >= cur_trade.sl:
                    hit_sl=True

            if hit_tp or hit_sl:
                cur_trade.close_time = row["Date"]
                cur_trade.result_R = (CFG["SCALP_TP_ATR"]/CFG["SCALP_SL_ATR"]) if hit_tp else -1.0
                trades.append(cur_trade)
                in_pos=False
                cur_trade=None

    # ---------- Metrics ----------
    if not trades:
        return {"symbol": symbol, "trades": [], "summary": {"n":0,"win%":0,"avgR":0,"expectancy":0,"maxDD_R":0,"PF":0}}

    results = pd.Series([t.result_R for t in trades], dtype=float)
    wins = (results>0).sum()
    n = len(trades)
    win_rate = 100.0*wins/n
    avgR = results.mean()
    # expectancy en R: moyenne simple (déjà)
    expectancy = avgR
    # profit factor
    gross_win = results[results>0].sum()
    gross_loss = -results[results<0].sum()
    PF = (gross_win / gross_loss) if gross_loss>0 else np.inf
    # max drawdown (sur equity en R)
    eq = results.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    maxDD_R = dd.min()

    # trades / jour (approx)
    days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days + 1
    tpd = n / max(1, days)

    summary = {
        "n": n,
        "win%": round(win_rate,2),
        "avgR": round(float(avgR),4),
        "expectancy_R": round(float(expectancy),4),
        "profit_factor": (float(PF) if np.isfinite(PF) else None),
        "max_drawdown_R": round(float(maxDD_R),4),
        "trades_per_day": round(float(tpd),3),
        "settings": {
            "interval": interval, "period": period,
            "sl_atr": CFG["SCALP_SL_ATR"], "tp_atr": CFG["SCALP_TP_ATR"],
            "min_conf": CFG["SCALP_MIN_CONF"], "rr_min_rule": CFG["RR_MIN"],
            "vol_enabled": CFG["VOL_ENABLED"]
        }
    }
    return {
        "symbol": symbol,
        "trades": [t.__dict__ for t in trades],
        "summary": summary
    }

# ================ Runner =================
if __name__=="__main__":
    universe = [
        ("EURUSD.pro","60d","15m"),
        ("GBPUSD.pro","60d","15m"),

    ]
    all_res=[]
    for sym,period,tf in universe:
        try:
            res = simulate_symbol(sym, period, tf)
            all_res.append(res)
            s = res["summary"]
            print(f"{sym:10s} | n={s.get('n')} win%={s.get('win%')} avgR={s.get('avgR')} expR={s.get('expectancy_R')} PF={s.get('profit_factor')} maxDD_R={s.get('max_drawdown_R')} t/day={s.get('trades_per_day')}")
        except Exception as e:
            print(f"[ERR] {sym}: {e}")

    # Export JSON (facultatif)
    out = {
        "generated_at": datetime.utcnow().isoformat()+"Z",
        "config": CFG,
        "results": all_res
    }
    with open("mini_backtest_results.json","w",encoding="utf-8") as f:
        json.dump(_to_jsonable(out), f, ensure_ascii=False, indent=2)
    print("\nRésultats sauvegardés → mini_backtest_results.json")
