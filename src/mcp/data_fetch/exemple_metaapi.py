# trading_client.py
# -*- coding: utf-8 -*-
"""
Client REST MetaApi (AgiliumTrade London) : compte, marché, bougies & TRADES.

- Charge le token depuis .env (API_TOKEN).
- Fournit des helpers GET/POST robustes (retries sur GET, timeouts).
- Normalise les bougies -> DataFrame avec colonnes: Datetime, Open, High, Low, Close, Volume (UTC naïf).
- Implémente POST /trade + wrappers pour types d'ordres courants.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# ENV & CONSTANTES
# ---------------------------------------------------------------------

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
ACCOUNT_ID = os.getenv("ACCOUNT_ID", "6960ffda-de14-4708-a922-eef2f97f4c4b")

if not API_TOKEN:
    raise RuntimeError("API_TOKEN manquant dans .env (exporte API_TOKEN=...)")

BASE_URL = "https://mt-client-api-v1.london.agiliumtrade.ai"
BASE_MARKET_URL = "https://mt-market-data-client-api-v1.london.agiliumtrade.ai"

HEADERS = {
    "auth-token": API_TOKEN,
    "Accept": "application/json",
}

# Mapping des timeframes (REST vs WebSocket)
TF_REST = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}
TF_WS = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30", "1h": "H1", "4h": "H4", "1d": "D1"}

# ---------------------------------------------------------------------
# SESSION HTTP ROBUSTE
# ---------------------------------------------------------------------

def _make_session(timeout: tuple[int, int] = (10, 30)) -> requests.Session:
    """Crée une session requests avec retries pour GET (jamais pour POST trade)."""
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.headers.update(HEADERS)
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    # attribut interne simple pour propager le timeout
    s._default_timeout = timeout  # type: ignore[attr-defined]
    return s

SESSION = _make_session()

# ---------------------------------------------------------------------
# HELPERS HTTP
# ---------------------------------------------------------------------

def _get(url: str, params: dict | None = None, base: str = "json") -> dict | str:
    """GET avec retries. Lève en cas d'erreur HTTP. Tente JSON sinon renvoie texte."""
    resp = SESSION.get(url, params=params, timeout=SESSION._default_timeout)  # type: ignore[attr-defined]
    resp.raise_for_status()
    if base == "json":
        try:
            return resp.json()
        except ValueError:
            return {"error": "invalid_json", "text": resp.text[:500]}
    return resp.text

def _post(url: str, json_body: dict) -> dict:
    """POST sans retry (évite les doublons d’ordres). Lève si HTTP != 2xx."""
    resp = SESSION.post(url, json=json_body, timeout=SESSION._default_timeout)  # type: ignore[attr-defined]
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return {"error": "invalid_json", "text": resp.text[:500]}

def _clean_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    """Retire clés None/listes vides/dicts vides pour payload propre."""
    return {k: v for k, v in d.items() if v is not None and v != [] and v != {}}

def _validate_enum(name: str, value: Optional[str], allowed: set[str]):
    if value is None:
        return
    if value not in allowed:
        raise ValueError(f"{name} invalide: {value}. Autorisés: {sorted(allowed)}")

# ---------------------------------------------------------------------
# API: COMPTE / POSITIONS / ORDRES
# ---------------------------------------------------------------------

def get_account_information() -> dict:
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/account-information"
    return _get(url)

def get_positions() -> dict:
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/positions"
    return _get(url)

def get_orders() -> dict:
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/orders"
    return _get(url)

# ---------------------------------------------------------------------
# API: SYMBOLS & PRIX
# ---------------------------------------------------------------------

def get_symbols(limit_preview: int | None = 5) -> list | dict:
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols"
    data = _get(url)
   
    return data

def get_symbol_spec(symbol: str) -> dict:
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/specification"
    return _get(url)

def get_current_price(symbol: str) -> dict:
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/current-price"
    return _get(url)

# ---------------------------------------------------------------------
# API: CANDLES
# ---------------------------------------------------------------------

def get_current_candles(symbol: str, tf_key: str = "15m", limit: int = 5) -> dict | str:
    """
    ⚠️ Officiellement WS-only pour MetaApi. REST peut renvoyer NotFound.
    On garde pour cohérence.
    """
    tf_ws = TF_WS.get(tf_key, "M15")
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/current-candles/{tf_ws}"
    return _get(url, params={"limit": limit})

def get_historical_candles(symbol: str, tf_key: str = "15m", days: int = 7, limit: int = 500) -> pd.DataFrame | dict:
    """
    Bougies historiques via REST.
    Retour: DataFrame avec colonnes [Datetime, Open, High, Low, Close, Volume] (Datetime UTC naïf).
    En cas d'erreur API -> dict {'error': ...}
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    timeframe = TF_REST.get(tf_key, tf_key)  # support direct "15m"
    url = (
        f"{BASE_MARKET_URL}/users/current/accounts/{ACCOUNT_ID}"
        f"/historical-market-data/symbols/{symbol}/timeframes/{timeframe}/candles"
    )
    params = {
        "startTime": start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "limit": limit,
    }
    r = _get(url, params=params)
    if isinstance(r, dict) and "error" in r:
        return r

    df = pd.DataFrame(r)
    if df.empty:
        return df

    # Normalisation
    time_col = "time" if "time" in df.columns else ("Datetime" if "Datetime" in df.columns else None)
    if time_col is None:
        return {"error": "missing_time_column", "columns": list(df.columns)}

    df["Datetime"] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(None)
    rename = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tickVolume": "Volume",
        "volume": "Volume",
    }
    df = df.rename(columns=rename)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df[[c for c in ["Datetime", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
        .dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
        .drop_duplicates(subset=["Datetime"])
        .sort_values("Datetime")
        .reset_index(drop=True)
    )
    return df

# ---------------------------------------------------------------------
# API: TRADE (POST /trade)
# ---------------------------------------------------------------------

# Enums autorisées (selon spécification)
ACTION_TYPES = {
    'ORDER_TYPE_SELL', 'ORDER_TYPE_BUY',
    'ORDER_TYPE_BUY_LIMIT', 'ORDER_TYPE_SELL_LIMIT',
    'ORDER_TYPE_BUY_STOP', 'ORDER_TYPE_SELL_STOP',
    'ORDER_TYPE_BUY_STOP_LIMIT', 'ORDER_TYPE_SELL_STOP_LIMIT',
    'POSITION_MODIFY', 'POSITION_PARTIAL', 'POSITION_CLOSE_ID',
    'POSITIONS_CLOSE_SYMBOL', 'ORDER_MODIFY', 'ORDER_CANCEL',
    'POSITION_CLOSE_BY'
}
UNITS = {
    'ABSOLUTE_PRICE', 'RELATIVE_PRICE', 'RELATIVE_POINTS', 'RELATIVE_PIPS',
    'RELATIVE_CURRENCY', 'RELATIVE_BALANCE_PERCENTAGE'
}
STOP_PRICE_BASE = {'CURRENT_PRICE', 'OPEN_PRICE', 'STOP_PRICE'}
OPEN_PRICE_BASE = {'CURRENT_PRICE', 'OPEN_PRICE', 'STOP_LIMIT_PRICE'}
FILLING_MODES = {'ORDER_FILLING_FOK', 'ORDER_FILLING_IOC'}  # si tu veux forcer un mode
EXPIRATION_TYPES = {'ORDER_TIME_GTC', 'ORDER_TIME_DAY', 'ORDER_TIME_SPECIFIED', 'ORDER_TIME_SPECIFIED_DAY'}

def execute_trade(trade: Dict[str, Any], *, client_id: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    Exécute une commande trade.
    - client_id: idempotence côté MetaApi; généré auto si omis.
    - dry_run: True -> ne POST pas; renvoie url+payload (debug/tests).
    """
    if 'actionType' not in trade:
        raise ValueError("actionType est requis")

    _validate_enum("actionType", trade.get('actionType'), ACTION_TYPES)
    _validate_enum("stopLossUnits", trade.get('stopLossUnits'), UNITS)
    _validate_enum("takeProfitUnits", trade.get('takeProfitUnits'), UNITS)
    _validate_enum("openPriceUnits", trade.get('openPriceUnits'), UNITS)
    _validate_enum("stopLimitPriceUnits", trade.get('stopLimitPriceUnits'), UNITS)
    _validate_enum("stopPriceBase", trade.get('stopPriceBase'), STOP_PRICE_BASE)
    _validate_enum("openPriceBase", trade.get('openPriceBase'), OPEN_PRICE_BASE)

    payload = dict(trade)
    if client_id:
        payload['clientId'] = client_id
    elif 'clientId' not in payload:
        payload['clientId'] = f"ta-{uuid.uuid4().hex}"

    payload = _clean_payload(payload)

    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/trade"
    if dry_run:
        return {"dry_run": True, "url": url, "payload": payload}

    return _post(url, payload)

# ----------------------- WRAPPERS PRATIQUES -------------------------

def buy_market(symbol: str, volume: float, *,
               sl: float | None = None, tp: float | None = None,
               sl_units: str = 'ABSOLUTE_PRICE', tp_units: str = 'ABSOLUTE_PRICE',
               slippage: Optional[float] = None, comment: Optional[str] = None,
               client_id: Optional[str] = None, magic: Optional[int] = None) -> dict:
    trade = {
        "actionType": "ORDER_TYPE_BUY",
        "symbol": symbol,
        "volume": volume,
        "stopLoss": sl,
        "takeProfit": tp,
        "stopLossUnits": sl_units,
        "takeProfitUnits": tp_units,
        "slippage": slippage,
        "comment": comment,
        "magic": magic,
    }
    return execute_trade(trade, client_id=client_id)

def sell_market(symbol: str, volume: float, *,
                sl: float | None = None, tp: float | None = None,
                sl_units: str = 'ABSOLUTE_PRICE', tp_units: str = 'ABSOLUTE_PRICE',
                slippage: Optional[float] = None, comment: Optional[str] = None,
                client_id: Optional[str] = None, magic: Optional[int] = None) -> dict:
    trade = {
        "actionType": "ORDER_TYPE_SELL",
        "symbol": symbol,
        "volume": volume,
        "stopLoss": sl,
        "takeProfit": tp,
        "stopLossUnits": sl_units,
        "takeProfitUnits": tp_units,
        "slippage": slippage,
        "comment": comment,
        "magic": magic,
    }
    return execute_trade(trade, client_id=client_id)

def place_limit(symbol: str, side: str, volume: float, price: float, *,
                sl: float | None = None, tp: float | None = None,
                sl_units: str = 'ABSOLUTE_PRICE', tp_units: str = 'ABSOLUTE_PRICE',
                open_price_units: str = 'ABSOLUTE_PRICE',
                expiration: Optional[dict] = None, comment: Optional[str] = None,
                client_id: Optional[str] = None, magic: Optional[int] = None) -> dict:
    action = "ORDER_TYPE_BUY_LIMIT" if side.lower() == "buy" else "ORDER_TYPE_SELL_LIMIT"
    trade = {
        "actionType": action,
        "symbol": symbol,
        "volume": volume,
        "openPrice": price,
        "openPriceUnits": open_price_units,
        "stopLoss": sl,
        "takeProfit": tp,
        "stopLossUnits": sl_units,
        "takeProfitUnits": tp_units,
        "expiration": expiration,  # {"type": "...", "time": "...Z"}
        "comment": comment,
        "magic": magic,
    }
    return execute_trade(trade, client_id=client_id)

def place_stop(symbol: str, side: str, volume: float, stop_price: float, *,
               sl: float | None = None, tp: float | None = None,
               sl_units: str = 'ABSOLUTE_PRICE', tp_units: str = 'ABSOLUTE_PRICE',
               open_price_units: str = 'ABSOLUTE_PRICE',
               expiration: Optional[dict] = None, comment: Optional[str] = None,
               client_id: Optional[str] = None, magic: Optional[int] = None) -> dict:
    action = "ORDER_TYPE_BUY_STOP" if side.lower() == "buy" else "ORDER_TYPE_SELL_STOP"
    trade = {
        "actionType": action,
        "symbol": symbol,
        "volume": volume,
        "openPrice": stop_price,
        "openPriceUnits": open_price_units,
        "stopLoss": sl,
        "takeProfit": tp,
        "stopLossUnits": sl_units,
        "takeProfitUnits": tp_units,
        "expiration": expiration,
        "comment": comment,
        "magic": magic,
    }
    return execute_trade(trade, client_id=client_id)

def place_stop_limit(symbol: str, side: str, volume: float, *,
                     stop_price: float, stop_limit_price: float,
                     sl: float | None = None, tp: float | None = None,
                     sl_units: str = 'ABSOLUTE_PRICE', tp_units: str = 'ABSOLUTE_PRICE',
                     open_price_units: str = 'ABSOLUTE_PRICE',
                     stop_limit_units: str = 'ABSOLUTE_PRICE',
                     expiration: Optional[dict] = None, comment: Optional[str] = None,
                     client_id: Optional[str] = None, magic: Optional[int] = None) -> dict:
    action = "ORDER_TYPE_BUY_STOP_LIMIT" if side.lower() == "buy" else "ORDER_TYPE_SELL_STOP_LIMIT"
    trade = {
        "actionType": action,
        "symbol": symbol,
        "volume": volume,
        "openPrice": stop_price,
        "openPriceUnits": open_price_units,
        "stopLimitPrice": stop_limit_price,
        "stopLimitPriceUnits": stop_limit_units,
        "stopLoss": sl,
        "takeProfit": tp,
        "stopLossUnits": sl_units,
        "takeProfitUnits": tp_units,
        "expiration": expiration,
        "comment": comment,
        "magic": magic,
    }
    return execute_trade(trade, client_id=client_id)

def position_modify(position_id: str, *,
                    sl: float | None = None, tp: float | None = None,
                    sl_units: str = 'ABSOLUTE_PRICE', tp_units: str = 'ABSOLUTE_PRICE',
                    stop_price_base: str = 'OPEN_PRICE',
                    trailing_stop_loss: Optional[dict] = None,
                    comment: Optional[str] = None, client_id: Optional[str] = None) -> dict:
    _validate_enum("stopPriceBase", stop_price_base, STOP_PRICE_BASE)
    trade = {
        "actionType": "POSITION_MODIFY",
        "positionId": position_id,
        "stopLoss": sl,
        "takeProfit": tp,
        "stopLossUnits": sl_units,
        "takeProfitUnits": tp_units,
        "stopPriceBase": stop_price_base,
        "trailingStopLoss": trailing_stop_loss,  # ex: {"distance": {"distance": 100, "units": "RELATIVE_POINTS"}}
        "comment": comment,
    }
    return execute_trade(trade, client_id=client_id)

def position_partial(position_id: str, volume: float, *,
                     client_id: Optional[str] = None, comment: Optional[str] = None) -> dict:
    trade = {
        "actionType": "POSITION_PARTIAL",
        "positionId": position_id,
        "volume": volume,
        "comment": comment,
    }
    return execute_trade(trade, client_id=client_id)

def position_close_id(position_id: str, *, client_id: Optional[str] = None, comment: Optional[str] = None) -> dict:
    trade = {
        "actionType": "POSITION_CLOSE_ID",
        "positionId": position_id,
        "comment": comment,
    }
    return execute_trade(trade, client_id=client_id)

def positions_close_symbol(symbol: str, *, client_id: Optional[str] = None, comment: Optional[str] = None) -> dict:
    trade = {
        "actionType": "POSITIONS_CLOSE_SYMBOL",
        "symbol": symbol,
        "comment": comment,
    }
    return execute_trade(trade, client_id=client_id)

def order_modify(order_id: str, *, open_price: Optional[float] = None, open_price_units: str = 'ABSOLUTE_PRICE',
                 sl: float | None = None, tp: float | None = None,
                 sl_units: str = 'ABSOLUTE_PRICE', tp_units: str = 'ABSOLUTE_PRICE',
                 stop_price_base: str = 'OPEN_PRICE', comment: Optional[str] = None,
                 expiration: Optional[dict] = None, client_id: Optional[str] = None) -> dict:
    trade = {
        "actionType": "ORDER_MODIFY",
        "orderId": order_id,
        "openPrice": open_price,
        "openPriceUnits": open_price_units,
        "stopLoss": sl,
        "takeProfit": tp,
        "stopLossUnits": sl_units,
        "takeProfitUnits": tp_units,
        "stopPriceBase": stop_price_base,
        "expiration": expiration,
        "comment": comment,
    }
    return execute_trade(trade, client_id=client_id)

def order_cancel(order_id: str, *, client_id: Optional[str] = None, comment: Optional[str] = None) -> dict:
    trade = {
        "actionType": "ORDER_CANCEL",
        "orderId": order_id,
        "comment": comment,
    }
    return execute_trade(trade, client_id=client_id)

def position_close_by(position_id: str, close_by_position_id: str, *,
                      client_id: Optional[str] = None, comment: Optional[str] = None) -> dict:
    trade = {
        "actionType": "POSITION_CLOSE_BY",
        "positionId": position_id,
        "closeByPositionId": close_by_position_id,
        "comment": comment,
    }
    return execute_trade(trade, client_id=client_id)

# ---------------------------------------------------------------------
# DEMO RAPIDE
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Account Information ===")
    try:
        print(get_account_information())
    except Exception as e:
        print("account_information error:", e)

    print("\n=== Positions ===")
    try:
        print(get_positions())
    except Exception as e:
        print("positions error:", e)

    print("\n=== Orders ===")
    try:
        print(get_orders())
    except Exception as e:
        print("orders error:", e)

    print("\n=== Symbols (preview) ===")
    try:
        print(get_symbols())
    except Exception as e:
        print("symbols error:", e)

    print("\n=== Specification BTCUSD ===")
    try:
        print(get_symbol_spec("BTCUSD"))
    except Exception as e:
        print("spec error:", e)

    print("\n=== Current Price BTCUSD ===")
    try:
        print(get_current_price("BTCUSD"))
    except Exception as e:
        print("price error:", e)

    print("\n=== Current 15m Candles BTCUSD (WS-only endpoint) ===")
    try:
        print(get_current_candles("BTCUSD", tf_key="15m", limit=5))
    except Exception as e:
        print("current_candles error:", e)

    print("\n=== Historical 15m Candles BTCUSD (last 7 days) ===")
    try:
        df = get_historical_candles("BTCUSD", tf_key="15m", days=7, limit=100)
        if isinstance(df, dict) and "error" in df:
            print("API error:", df)
        else:
            print(df.head())
    except Exception as e:
        print("historical_candles error:", e)

    # --- Exemple trade en DRY RUN (ne passe pas d'ordre réel) ---
    print("\n=== DRY RUN: BUY MARKET 0.1 BTCUSD with SL/TP ===")
    try:
        payload_preview = execute_trade(
            {
                "actionType": "ORDER_TYPE_BUY",
                "symbol": "BTCUSD",
                "volume": 0.1,
                "stopLoss": 108500,
                "takeProfit": 109500,
                "stopLossUnits": "ABSOLUTE_PRICE",
                "takeProfitUnits": "ABSOLUTE_PRICE",
                "comment": "TA BUY demo",
            },
            dry_run=True,
        )
        print(payload_preview)
    except Exception as e:
        print("dry_run trade error:", e)
