import json
from typing import Annotated, Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Trading Analysis MCP Server", log_level="DEBUG")

# ---------- Helpers JSON ----------
def _ok(payload: Any) -> str:
    return json.dumps({"ok": True, "data": payload}, ensure_ascii=False)

def _err(msg: str, **extra) -> str:
    logger.error(f"[MCP:ANALYSIS] {msg} | extra={extra}")
    return json.dumps({"ok": False, "error": msg, "extra": extra}, ensure_ascii=False)

def _df_from_ohlcv(ohlcv: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not isinstance(ohlcv, list) or not ohlcv:
        return None
    df = pd.DataFrame(ohlcv)
    cols_map = {c.lower(): c for c in df.columns}
    def pick(name): 
        return cols_map.get(name.lower(), name)
    needed = ["date","open","high","low","close","volume"]
    for n in needed:
        if pick(n) not in df.columns:
            return None
    df = df.rename(columns={
        pick("date"): "Date",
        pick("open"): "Open",
        pick("high"): "High",
        pick("low"): "Low",
        pick("close"): "Close",
        pick("volume"): "Volume",
    })
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        pass
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"]).reset_index(drop=True)
    return df

# ---------- Indicators ----------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).ewm(span=length, adjust=False).mean()
    roll_down = pd.Series(down).ewm(span=length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return pd.Series(out, index=close.index)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()

def bbands(close: pd.Series, length: int = 20, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = ma + mult * sd
    lower = ma - mult * sd
    return lower, ma, upper

# ---------- Tools ----------
@mcp.tool()
def compute_indicators(
    ohlcv: Annotated[List[Dict[str, Any]], Field(description="List of OHLCV records (Date,Open,High,Low,Close,Volume)")],
    rsi_len: int = 14,
    ema_fast: int = 12,
    ema_slow: int = 26,
    macd_signal: int = 9,
    atr_len: int = 14,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    tail: int = 200,
) -> str:
    """
    Calcule RSI, EMA fast/slow, MACD(line,signal,hist), ATR, Bollinger Bands.
    """
    df = _df_from_ohlcv(ohlcv)
    if df is None or df.empty:
        return _err("Invalid or empty OHLCV")

    try:
        df["EMA_Fast"] = ema(df["Close"], ema_fast)
        df["EMA_Slow"] = ema(df["Close"], ema_slow)
        df["RSI"] = rsi(df["Close"], rsi_len)
        macd_line, macd_sig, macd_hist = macd(df["Close"], ema_fast, ema_slow, macd_signal)
        df["MACD_Line"] = macd_line
        df["MACD_Signal"] = macd_sig
        df["MACD_Hist"] = macd_hist
        df["ATR"] = atr(df, atr_len)
        bb_lower, bb_mid, bb_upper = bbands(df["Close"], bb_len, bb_mult)
        df["BB_Lower"] = bb_lower
        df["BB_Mid"] = bb_mid
        df["BB_Upper"] = bb_upper

        out = df.tail(max(1, int(tail))).copy()
        out = out.replace({np.nan: None})
        payload = json.loads(out.to_json(orient="records", date_format="iso"))
        return _ok(payload)
    except Exception as e:
        return _err("Indicator computation failed", exc=str(e))

@mcp.tool()
def detect_signals(
    ohlcv: Annotated[List[Dict[str, Any]], Field(description="OHLCV records")],
    rsi_buy: int = 35,
    rsi_sell: int = 65,
    ema_cross_confirm: bool = True,
    bb_touch_confirm: bool = False,
) -> str:
    """
    Détection améliorée : labels forts/faibles.
    """
    ind = json.loads(compute_indicators(ohlcv))
    if not ind.get("ok"):
        return _err("Indicators unavailable", reason=ind.get("error"))
    rows = ind["data"]
    if not rows:
        return _err("No rows after indicators")

    last = rows[-1]
    def _get(k, default=None): return last.get(k, default)

    rsi_val = _get("RSI")
    ema_fast = _get("EMA_Fast")
    ema_slow = _get("EMA_Slow")
    close = _get("Close")
    bb_u = _get("BB_Upper")
    bb_l = _get("BB_Lower")

    reasons, score, raw_bias = [], 0.0, None

    # RSI
    if rsi_val is not None:
        if rsi_val <= rsi_buy:
            raw_bias = "bullish"; score += 0.4; reasons.append(f"RSI={rsi_val:.2f} <= {rsi_buy}")
        elif rsi_val >= rsi_sell:
            raw_bias = "bearish"; score += 0.4; reasons.append(f"RSI={rsi_val:.2f} >= {rsi_sell}")
        else:
            reasons.append(f"RSI={rsi_val:.2f} neutre")

    # EMA cross
    if ema_cross_confirm and ema_fast is not None and ema_slow is not None:
        if ema_fast > ema_slow:
            if raw_bias == "bullish": score += 0.3
            else: score += 0.15
            reasons.append("EMA Fast > EMA Slow")
            if raw_bias is None: raw_bias = "bullish"
        elif ema_fast < ema_slow:
            if raw_bias == "bearish": score += 0.3
            else: score += 0.15
            reasons.append("EMA Fast < EMA Slow")
            if raw_bias is None: raw_bias = "bearish"

    # Bollinger
    if bb_touch_confirm and close is not None and bb_u is not None and bb_l is not None:
        width = (bb_u - bb_l) if (bb_u and bb_l) else None
        if width and width > 0:
            dist_low = abs(close - bb_l) / width
            dist_up = abs(bb_u - close) / width
            if dist_low < 0.15:
                raw_bias = "bullish"; score += 0.2; reasons.append("Close near BB Lower")
            if dist_up < 0.15:
                raw_bias = "bearish"; score += 0.2; reasons.append("Close near BB Upper")

    # Label final
    label = "neutral"
    if raw_bias == "bullish":
        label = "bullish_strong" if score >= 0.6 else "bullish_weak"
    elif raw_bias == "bearish":
        label = "bearish_strong" if score >= 0.6 else "bearish_weak"

    score = float(max(0.0, min(1.0, score)))

    return _ok({
        "bias": label,
        "confidence": score,
        "last_row": last,
        "reasons": reasons
    })

@mcp.tool()
def trade_plan(
    ohlcv: Annotated[List[Dict[str, Any]], Field(description="OHLCV records")],
    risk_level: str = "medium",
    rr_min: float = 1.5,
    atr_mult_sl: float = 1.2,
    atr_mult_tp: float = 2.0,
) -> str:
    """
    Génère un plan de trade basé sur detect_signals + ATR.
    """
    ind = json.loads(compute_indicators(ohlcv))
    if not ind.get("ok"):
        return _err("Indicators unavailable", reason=ind.get("error"))
    rows = ind["data"]
    if not rows:
        return _err("No rows after indicators")

    sig = json.loads(detect_signals(ohlcv))
    if not sig.get("ok"):
        return _err("Signals unavailable", reason=sig.get("error"))

    last = rows[-1]
    atr_val = last.get("ATR")
    close = last.get("Close")
    bias = sig["data"]["bias"]
    conf = sig["data"]["confidence"]

    if atr_val is None or close is None:
        return _err("ATR or Close missing")

    # Ajuste multiplicateurs selon risque
    rl = (risk_level or "medium").lower()
    if rl == "low":
        sl_mult = atr_mult_sl * 1.2
        tp_mult = atr_mult_tp * 0.9
    elif rl == "high":
        sl_mult = atr_mult_sl * 0.9
        tp_mult = atr_mult_tp * 1.2
    else:
        sl_mult = atr_mult_sl
        tp_mult = atr_mult_tp

    action, entry, sl, tp = "HOLD", close, None, None

    if bias.startswith("bullish"):
        entry = close
        sl = round(entry - sl_mult * atr_val, 6)
        tp = round(entry + tp_mult * atr_val, 6)
        action = "BUY"
    elif bias.startswith("bearish"):
        entry = close
        sl = round(entry + sl_mult * atr_val, 6)
        tp = round(entry - tp_mult * atr_val, 6)
        action = "SELL"

    # Ajuste fermeté selon strong/weak
    if "weak" in bias and action != "HOLD":
        conf *= 0.8  # réduit confiance
    elif "strong" in bias:
        conf *= 1.1  # accentue confiance

    # Risk/Reward
    rr = None
    if action in ("BUY","SELL") and sl is not None and tp is not None:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = round(reward / risk, 3) if risk else None

    decision = {
        "action": action,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk_reward": rr,
        "confidence": round(conf, 3),
        "risk_level": rl,
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "bias": bias,
            "rules": "RSI/EMA/MACD/BB/ATR",
            "requirements": {"rr_min": rr_min}
        }
    }

    if decision["action"] in ("BUY","SELL") and rr is not None and rr < rr_min:
        decision["note"] = f"RR {rr} < min {rr_min}, suggestion: HOLD"
        decision["action"] = "HOLD"

    return _ok(decision)

# ---------- Prompt Agent ----------
@mcp.prompt()
def analysis_agent(
    symbol: str,
    horizon: str = "swing",
    risk_level: str = "medium",
) -> str:
    return f"""
Tu es l'agent d'analyse. Tu reçois des OHLCV déjà fournis par le **Data MCP** pour {symbol}.
N'essaie PAS de récupérer des données toi-même. Utilise UNIQUEMENT les tools:

1) compute_indicators(ohlcv) → RSI/EMA/MACD/ATR/Bollinger
2) detect_signals(ohlcv) → biais enrichi + confiance
3) trade_plan(ohlcv, risk_level="{risk_level}") → décision exploitable

Contexte:
- Horizon: {horizon}
- Risque: {risk_level}

Sortie JSON finale attendue:
{{
  "symbol": "{symbol}",
  "horizon": "{horizon}",
  "decision": {{ "action": "...", "entry": ..., "sl": ..., "tp": ..., "risk_reward": ..., "confidence": ... }},
  "reason": "Résumé clair des raisons techniques"
}}

Si le biais est neutre ou la confiance trop faible, propose HOLD.
"""

if __name__ == "__main__":
    mcp.run(transport="stdio")
