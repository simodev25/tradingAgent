# metaapi_mcp.py (lecture + news via Yahoo)
import os, requests, json, uuid  # <-- ajout: uuid
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from typing import Annotated
from mcp.server.fastmcp import FastMCP
import math, re


# --- Config ---
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

BASE_URL = "https://mt-client-api-v1.london.agiliumtrade.ai"
BASE_MARKET_URL = "https://mt-market-data-client-api-v1.london.agiliumtrade.ai"
HEADERS = {"auth-token": API_TOKEN, "Accept": "application/json"}

# =========================
# HELPERS
# =========================
def _request(method: str, url: str, *, params=None):
    try:
        resp = requests.request(method, url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code >= 400:
            return {"ok": False, "status": resp.status_code, "error": resp.text}
        return {"ok": True, "data": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# <-- ajout: POST helper utilisé par trade_execute
def _post_json(url: str, payload: dict, *, timeout: int = 20, headers: dict | None = None):
    try:
        resp = requests.post(
            url,
            headers=(headers or {**HEADERS, "Content-Type": "application/json"}),
            json=payload,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            return {"ok": False, "status": resp.status_code, "error": resp.text}
        return {"ok": True, "data": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _tf_map(interval: str) -> str:
    s = (interval or "").strip().lower()
    # Allowed directly by MetaApi
    allowed = {
        "1m","2m","3m","4m","5m","6m","10m","12m","15m","20m","30m",
        "1h","2h","3h","4h","6h","8h","12h","1d","1w","1mn"
    }
    if s in allowed:
        return s
    # Common aliases -> MetaApi format
    aliases = {
        "m1":"1m","m2":"2m","m3":"3m","m4":"4m","m5":"5m","m6":"6m","m10":"10m","m12":"12m",
        "m15":"15m","m20":"20m","m30":"30m",
        "h1":"1h","h2":"2h","h3":"3h","h4":"4h","h6":"6h","h8":"8h","h12":"12h",
        "d1":"1d","w1":"1w","mn1":"1mn",
        "1mo":"1mn","1month":"1mn","1wk":"1w"
    }
    return aliases.get(s, "1h")

def _period_to_days(period: str) -> int:
    m = {"1d":1,"5d":5,"1mo":30,"3mo":90,"6mo":180,"1y":365}
    return m.get(period.lower(), 30)

# =========================

def get_account_info() -> str:
    """Infos du compte (balance, equity, margin, etc.)."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/account-information"
    return json.dumps(_request("GET", url), ensure_ascii=False)

def get_positions() -> str:
    """Positions ouvertes."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/positions"
    return json.dumps(_request("GET", url), ensure_ascii=False)

def get_orders() -> str:
    """Ordres en attente."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/orders"
    return json.dumps(_request("GET", url), ensure_ascii=False)

def get_symbols() -> str:
    """Liste des symboles disponibles."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols"
    return json.dumps(_request("GET", url), ensure_ascii=False)

def get_symbol_spec(symbol: str) -> str:
    """Spécifications d’un symbole (contractSize, tickSize, margin, etc.)."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/specification"
    return json.dumps(_request("GET", url), ensure_ascii=False)

def get_current_price(symbol: str) -> str:
    """Prix actuel bid/ask d’un symbole."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/current-price"
    return json.dumps(_request("GET", url), ensure_ascii=False)

def _period_to_days(period: str) -> int:
    """
    Supporte: 1d, 5d, 7d, 10d, 1w, 2w, 3w, 1mo, 3mo, 6mo, 1y, 2y, etc.
    Par défaut: 30 jours.
    """
    if not period:
        return 30
    s = period.strip().lower()
    preset = {"1d":1, "5d":5, "1mo":30, "3mo":90, "6mo":180, "1y":365}
    if s in preset:
        return preset[s]
    m = re.fullmatch(r"(\d+)\s*(d|w|mo|y)", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "d":
            return n
        if unit == "w":
            return n * 7
        if unit == "mo":
            return n * 30
        if unit == "y":
            return n * 365
    return 30


def _tf_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("mn"):   # 1mn = 1 month chez MetaApi (oui oui)
        return 30 * 24 * 60
    if tf.endswith("w"):
        return int(tf[:-1]) * 7 * 24 * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 24 * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("m"):
        return int(tf[:-1])
    raise ValueError(f"Unsupported timeframe: {tf}")

def _period_to_minutes(period: str) -> int:
    """
    Supporte: Xm, Xh, Xd, Xw, Xmo, Xy + quelques presets.
    Par défaut: 30 jours.
    """
    if not period:
        return 30 * 24 * 60
    s = period.strip().lower()
    presets = {
        "1h": 60, "4h": 240, "24h": 1440,
        "1d": 1440, "5d": 5*1440, "1w": 7*1440,
        "1mo": 30*1440, "3mo": 90*1440, "6mo": 180*1440,
        "1y": 365*1440
    }
    if s in presets:
        return presets[s]
    m = re.fullmatch(r"(\d+)\s*(m|h|d|w|mo|y)", s)
    if m:
        n, u = int(m.group(1)), m.group(2)
        return (
            n if u=="m" else
            n*60 if u=="h" else
            n*1440 if u=="d" else
            n*7*1440 if u=="w" else
            n*30*1440 if u=="mo" else
            n*365*1440
        )
    return 30 * 24 * 60
# --- update get_historical_candles ---
def get_historical_candles(symbol: str, period="1mo", interval="1h", limit=800) -> str:
    """
    Bougies OHLCV historiques (MetaApi charge vers l'arrière).
    - Pas de endTime ; on borne avec un limit calculé d'après `period`.
    """
    tf = _tf_map(interval)
    tf_min = _tf_minutes(tf)
    period_min = _period_to_minutes(period)

    bars_needed = max(1, math.ceil(period_min / tf_min))
    # si l'appelant passe un limit plus petit, on respecte ; sinon on prend bars_needed (capé à 1000)
    limit_final = min(int(limit or bars_needed), bars_needed, 1000)

    url = (
        f"{BASE_MARKET_URL}/users/current/accounts/{ACCOUNT_ID}"
        f"/historical-market-data/symbols/{symbol}/timeframes/{tf}/candles"
    )
    params = {"limit": limit_final}  # <-- pas d'endTime

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code >= 400:
            return json.dumps({"ok": False, "status": resp.status_code, "error": resp.text}, ensure_ascii=False)
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            return json.dumps(data, ensure_ascii=False)

        df = pd.DataFrame(data)
        if df.empty:
            return json.dumps([], ensure_ascii=False)

        if "time" in df.columns:
            df["Datetime"] = pd.to_datetime(df["time"], utc=True)

        df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"})
        if "tickVolume" in df.columns and "Volume" not in df.columns:
            df = df.rename(columns={"tickVolume": "Volume"})
        elif "volume" in df.columns and "Volume" not in df.columns:
            df = df.rename(columns={"volume": "Volume"})

        cols = [c for c in ["Datetime","Open","High","Low","Close","Volume"] if c in df.columns]
        df = df[cols]
        return df.to_json(orient="records", date_format="iso")
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

def fetch_market(symbol: str, period="1mo", interval="1h", news_top_n=20) -> str:
    """
    Récupère et assemble toutes les données de marché (account, positions, orders,
    spécifications, prix, bougies historiques, news) en un seul JSON complet.
    """
    try:
        account = json.loads(get_account_info())
        positions = json.loads(get_positions())
        orders = json.loads(get_orders())
        symbol_spec = json.loads(get_symbol_spec(symbol))
        price = json.loads(get_current_price(symbol))
        history = json.loads(get_historical_candles(symbol, period=period, interval=interval, limit=500))

        result = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "account": account,
            "positions": positions,
            "orders": orders,
            "symbol_spec": symbol_spec,
            "price": price,
            "history": history
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

# =========================
# TRADE (POST /trade) + WRAPPERS
# =========================

_ACTION_TYPES = {
    'ORDER_TYPE_SELL', 'ORDER_TYPE_BUY',
    'ORDER_TYPE_BUY_LIMIT', 'ORDER_TYPE_SELL_LIMIT',
    'ORDER_TYPE_BUY_STOP', 'ORDER_TYPE_SELL_STOP',
    'ORDER_TYPE_BUY_STOP_LIMIT', 'ORDER_TYPE_SELL_STOP_LIMIT',
    'POSITION_MODIFY', 'POSITION_PARTIAL', 'POSITION_CLOSE_ID',
    'POSITIONS_CLOSE_SYMBOL', 'ORDER_MODIFY', 'ORDER_CANCEL',
    'POSITION_CLOSE_BY'
}
_UNITS = {
    'ABSOLUTE_PRICE', 'RELATIVE_PRICE', 'RELATIVE_POINTS', 'RELATIVE_PIPS',
    'RELATIVE_CURRENCY', 'RELATIVE_BALANCE_PERCENTAGE'
}
_STOP_PRICE_BASE = {'CURRENT_PRICE', 'OPEN_PRICE', 'STOP_PRICE'}
_OPEN_PRICE_BASE = {'CURRENT_PRICE', 'OPEN_PRICE', 'STOP_LIMIT_PRICE'}

def _map_action_string(action: str) -> str:
    a = (action or "").strip().upper()
    mapping = {
        "BUY": "ORDER_TYPE_BUY",
        "SELL": "ORDER_TYPE_SELL",
        "BUY_LIMIT": "ORDER_TYPE_BUY_LIMIT",
        "SELL_LIMIT": "ORDER_TYPE_SELL_LIMIT",
        "BUY_STOP": "ORDER_TYPE_BUY_STOP",
        "SELL_STOP": "ORDER_TYPE_SELL_STOP",
        "BUY_STOP_LIMIT": "ORDER_TYPE_BUY_STOP_LIMIT",
        "SELL_STOP_LIMIT": "ORDER_TYPE_SELL_STOP_LIMIT",
    }
    return mapping.get(a, a)

def trade_execute(trade: dict, *, client_id: str | None = None, dry_run: bool = False) -> str:
    """
    POST officiel /users/current/accounts/{accountId}/trade
    - `trade`: dict MetaTraderTrade (actionType, symbol, volume, openPrice, stopLoss, takeProfit, etc.)
    - `client_id`: optionnel pour idempotence (sinon auto)
    - `dry_run`: ne poste pas, renvoie l’URL et le payload
    """
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/trade"
    payload = dict(trade)
    
    if dry_run:
        return json.dumps({"ok": True, "dry_run": True, "url": url, "payload": payload}, ensure_ascii=False)

    res = _post_json(url, payload)  # <-- maintenant défini
    return json.dumps(res, ensure_ascii=False)

# Wrappers pratiques (retour str(JSON))
def buy_market(symbol: str, volume: float, *, sl: float | None = None, tp: float | None = None,
               sl_units: str = "ABSOLUTE_PRICE", tp_units: str = "ABSOLUTE_PRICE",
               slippage: float | None = None, comment: str | None = None,
               client_id: str | None = None, magic: int | None = None) -> str:
    trade = {
        "actionType": "ORDER_TYPE_BUY",
        "symbol": symbol,
        "volume": volume,
        "stopLoss": sl, "takeProfit": tp,
        "stopLossUnits": sl_units, "takeProfitUnits": tp_units,
        "slippage": slippage, "comment": comment, "magic": magic,
    }
    return trade_execute(trade, client_id=client_id)

def sell_market(symbol: str, volume: float, *, sl: float | None = None, tp: float | None = None,
                sl_units: str = "ABSOLUTE_PRICE", tp_units: str = "ABSOLUTE_PRICE",
                slippage: float | None = None, comment: str | None = None,
                client_id: str | None = None, magic: int | None = None) -> str:
    trade = {
        "actionType": "ORDER_TYPE_SELL",
        "symbol": symbol,
        "volume": volume,
        "stopLoss": sl, "takeProfit": tp,
        "stopLossUnits": sl_units, "takeProfitUnits": tp_units,
        "slippage": slippage, "comment": comment, "magic": magic,
    }
    return trade_execute(trade, client_id=client_id)

def place_limit(symbol: str, side: str, volume: float, price: float, *,
                sl: float | None = None, tp: float | None = None,
                sl_units: str = "ABSOLUTE_PRICE", tp_units: str = "ABSOLUTE_PRICE",
                expiration: dict | None = None, comment: str | None = None,
                client_id: str | None = None, magic: int | None = None) -> str:
    action = "ORDER_TYPE_BUY_LIMIT" if (side or "").lower() == "buy" else "ORDER_TYPE_SELL_LIMIT"
    trade = {
        "actionType": action, "symbol": symbol, "volume": volume,
        "openPrice": price, "stopLoss": sl, "takeProfit": tp,
        "stopLossUnits": sl_units, "takeProfitUnits": tp_units,
        "expiration": expiration, "comment": comment, "magic": magic,
    }
    return trade_execute(trade, client_id=client_id)

def place_stop(symbol: str, side: str, volume: float, stop_price: float, *,
               sl: float | None = None, tp: float | None = None,
               sl_units: str = "ABSOLUTE_PRICE", tp_units: str = "ABSOLUTE_PRICE",
               expiration: dict | None = None, comment: str | None = None,
               client_id: str | None = None, magic: int | None = None) -> str:
    action = "ORDER_TYPE_BUY_STOP" if (side or "").lower() == "buy" else "ORDER_TYPE_SELL_STOP"
    trade = {
        "actionType": action, "symbol": symbol, "volume": volume,
        "openPrice": stop_price, "stopLoss": sl, "takeProfit": tp,
        "stopLossUnits": sl_units, "takeProfitUnits": tp_units,
        "expiration": expiration, "comment": comment, "magic": magic,
    }
    return trade_execute(trade, client_id=client_id)

def place_stop_limit(symbol: str, side: str, volume: float, *,
                     stop_price: float, stop_limit_price: float,
                     sl: float | None = None, tp: float | None = None,
                     sl_units: str = "ABSOLUTE_PRICE", tp_units: str = "ABSOLUTE_PRICE",
                     expiration: dict | None = None, comment: str | None = None,
                     client_id: str | None = None, magic: int | None = None) -> str:
    action = "ORDER_TYPE_BUY_STOP_LIMIT" if (side or "").lower() == "buy" else "ORDER_TYPE_SELL_STOP_LIMIT"
    trade = {
        "actionType": action, "symbol": symbol, "volume": volume,
        "openPrice": stop_price, "stopLimitPrice": stop_limit_price,
        "stopLoss": sl, "takeProfit": tp,
        "stopLossUnits": sl_units, "takeProfitUnits": tp_units,
        "expiration": expiration, "comment": comment, "magic": magic,
    }
    return trade_execute(trade, client_id=client_id)

def position_modify(position_id: str, *,
                    sl: float | None = None, tp: float | None = None,
                    sl_units: str = "ABSOLUTE_PRICE", tp_units: str = "ABSOLUTE_PRICE",
                    stop_price_base: str = "OPEN_PRICE",
                    trailing_stop_loss: dict | None = None,
                    comment: str | None = None, client_id: str | None = None) -> str:
    trade = {
        "actionType": "POSITION_MODIFY", "positionId": position_id,
        "stopLoss": sl, "takeProfit": tp,
        "stopLossUnits": sl_units, "takeProfitUnits": tp_units,
        "stopPriceBase": stop_price_base,
        "trailingStopLoss": trailing_stop_loss,
        "comment": comment,
    }
    return trade_execute(trade, client_id=client_id)

def position_partial(position_id: str, volume: float, *,
                     comment: str | None = None, client_id: str | None = None) -> str:
    trade = {
        "actionType": "POSITION_PARTIAL", "positionId": position_id,
        "volume": volume, "comment": comment,
    }
    return trade_execute(trade, client_id=client_id)

def position_close_id(position_id: str, *,
                      comment: str | None = None, client_id: str | None = None) -> str:
    trade = {"actionType": "POSITION_CLOSE_ID", "positionId": position_id, "comment": comment}
    return trade_execute(trade, client_id=client_id)

def positions_close_symbol(symbol: str, *,
                           comment: str | None = None, client_id: str | None = None) -> str:
    trade = {"actionType": "POSITIONS_CLOSE_SYMBOL", "symbol": symbol, "comment": comment}
    return trade_execute(trade, client_id=client_id)

def order_modify(order_id: str, *,
                 open_price: float | None = None,
                 sl: float | None = None, tp: float | None = None,
                 sl_units: str = "ABSOLUTE_PRICE", tp_units: str = "ABSOLUTE_PRICE",
                 stop_price_base: str = "OPEN_PRICE",
                 expiration: dict | None = None,
                 comment: str | None = None, client_id: str | None = None) -> str:
    trade = {
        "actionType": "ORDER_MODIFY",
        "orderId": order_id,
        "openPrice": open_price,
        "stopLoss": sl, "takeProfit": tp,
        "stopLossUnits": sl_units, "takeProfitUnits": tp_units,
        "stopPriceBase": stop_price_base,
        "expiration": expiration,
        "comment": comment,
    }
    return trade_execute(trade, client_id=client_id)

def order_cancel(order_id: str, *, comment: str | None = None, client_id: str | None = None) -> str:
    trade = {"actionType": "ORDER_CANCEL", "orderId": order_id, "comment": comment}
    return trade_execute(trade, client_id=client_id)

def position_close_by(position_id: str, close_by_position_id: str, *,
                      comment: str | None = None, client_id: str | None = None) -> str:
    trade = {
        "actionType": "POSITION_CLOSE_BY",
        "positionId": position_id,
        "closeByPositionId": close_by_position_id,
        "comment": comment,
    }
    return trade_execute(trade, client_id=client_id)

# =========================
# Compat: ancienne execute_trade simple (BUY/SELL/..)
# =========================
def execute_trade(symbol: str, action: str, entry: float, sl: float, tp: float,
                  volume: float = 0.01, comment: str | None = None, client_id: str | None = None) -> str:
    """
    Compatibilité ascendante avec ta signature initiale.
    - action: "BUY"/"SELL" (market) ou "BUY_LIMIT"/"SELL_LIMIT" / "BUY_STOP"/"SELL_STOP"
    - entry: utilisé pour LIMIT/STOP comme openPrice. Ignoré pour market.
    - volume: ajouté (défaut 0.01) pour exécuter un ordre valide.
    """
    action_type = _map_action_string(action)
    trade = {
        "actionType": action_type,
        "symbol": symbol,
        "volume": volume,
        "stopLoss": sl,
        "takeProfit": tp,
        "comment": comment,
    }
    # openPrice requis pour LIMIT/STOP
    if action_type in {"ORDER_TYPE_BUY_LIMIT","ORDER_TYPE_SELL_LIMIT","ORDER_TYPE_BUY_STOP","ORDER_TYPE_SELL_STOP"}:
        trade["openPrice"] = entry

    return trade_execute(trade, client_id=client_id)

if __name__ == "__main__":
    print("\n🚀 Running Trading Data Agent...")

    print("\n=== Historical 15m Candles BTCUSD (last 7 days) ===")
    try:
        # 15 minutes sur 7 jours (cohérent avec le libellé)
        js = get_historical_candles("EURUSD.pro", period="7H", interval="15m", limit=500)
        # `js` est une chaîne JSON (liste). On affiche juste les ~200 premiers caractères pour sanity check
        print(js)
    except Exception as e:
        print("historical_candles error:", e)
