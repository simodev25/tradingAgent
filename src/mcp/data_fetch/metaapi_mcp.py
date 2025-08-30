# metaapi_mcp.py (lecture + news via Yahoo)
import os, requests, json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from typing import Annotated
from mcp.server.fastmcp import FastMCP

# --- Config ---
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

BASE_URL = "https://mt-client-api-v1.london.agiliumtrade.ai"
BASE_MARKET_URL = "https://mt-market-data-client-api-v1.london.agiliumtrade.ai"
HEADERS = {"auth-token": API_TOKEN, "Accept": "application/json"}

mcp = FastMCP("MetaApi MCP Server")

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
# TOOLS MCP (Lecture)
# =========================
@mcp.tool()
def get_account_info() -> str:
    """Infos du compte (balance, equity, margin, etc.)."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/account-information"
    return json.dumps(_request("GET", url), ensure_ascii=False)

@mcp.tool()
def get_positions() -> str:
    """Positions ouvertes."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/positions"
    return json.dumps(_request("GET", url), ensure_ascii=False)

@mcp.tool()
def get_orders() -> str:
    """Ordres en attente."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/orders"
    return json.dumps(_request("GET", url), ensure_ascii=False)

@mcp.tool()
def get_symbols() -> str:
    """Liste des symboles disponibles."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols"
    return json.dumps(_request("GET", url), ensure_ascii=False)

@mcp.tool()
def get_symbol_spec(symbol: str) -> str:
    """Spécifications d’un symbole (contractSize, tickSize, margin, etc.)."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/specification"
    return json.dumps(_request("GET", url), ensure_ascii=False)

@mcp.tool()
def get_current_price(symbol: str) -> str:
    """Prix actuel bid/ask d’un symbole."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/current-price"
    return json.dumps(_request("GET", url), ensure_ascii=False)

@mcp.tool()
def get_historical_candles(symbol: str, period="1mo", interval="1h", limit=100) -> str:
    """
    Bougies OHLCV historiques via REST officiel.
    Retourne une LISTE de records: [{Datetime, Open, High, Low, Close, Volume}, ...]
    """
    tf = _tf_map(interval)
    days = _period_to_days(period)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    url = (
        f"{BASE_MARKET_URL}/users/current/accounts/{ACCOUNT_ID}"
        f"/historical-market-data/symbols/{symbol}/timeframes/{tf}/candles"
    )
    params = {"startTime": start.isoformat(), "limit": limit}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)

    if resp.status_code >= 400:
        return json.dumps({"ok": False, "status": resp.status_code, "error": resp.text}, ensure_ascii=False)

    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        return json.dumps(data, ensure_ascii=False)

    # Standardize OHLCV
    df = pd.DataFrame(data)
    if df.empty:
        return json.dumps([], ensure_ascii=False)

    df["Datetime"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tickVolume": "Volume"
    })
    # Keep only standard columns
    cols = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]
    # Return raw list (not wrapper) — matches your Yahoo MCP contract
    return df.to_json(orient="records", date_format="iso")

@mcp.tool()
def get_ticker_news(symbol: str) -> str:
    """News financières via Yahoo Finance (fallback car MetaApi ne gère pas les news)."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.get_news()
        return json.dumps(news, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

# =========================
# PROMPT
# =========================
@mcp.prompt()
def fetch_market_context(symbol: str, period="1mo", interval="1h", news_top_n="20") -> str:
    return f"""
Tu es un **agent Data**. Ta mission est **exclusivement** de collecter et assembler les données de marché pour {symbol}.
**NE FOURNIS AUCUNE RECOMMANDATION**.

Utilise uniquement ces tools :
- get_account_info()
- get_positions()
- get_orders()
- get_symbol_spec("{symbol}")
- get_current_price("{symbol}")
- get_historical_candles("{symbol}", period="{period}", interval="{interval}", limit=500)
- get_ticker_news("{symbol}")

Assemble un JSON unique :

{{
  "symbol": "{symbol}",
  "period": "{period}",
  "interval": "{interval}",
  "account": <résultat get_account_info>,
  "positions": <résultat get_positions>,
  "orders": <résultat get_orders>,
  "symbol_spec": <résultat get_symbol_spec>,
  "price": <résultat get_current_price>,
  "history": <résultat get_historical_candles>,
  "news": <résultat get_ticker_news>
}}
"""

if __name__ == "__main__":
    mcp.run(transport="stdio")
