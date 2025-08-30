import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd

# --- Load token from .env ---
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
ACCOUNT_ID = "6960ffda-de14-4708-a922-eef2f97f4c4b"

# ⚠️ Attention : deux bases URL différentes
BASE_URL = "https://mt-client-api-v1.london.agiliumtrade.ai"
BASE_MARKET_URL = "https://mt-market-data-client-api-v1.london.agiliumtrade.ai"

headers = {
    "auth-token": API_TOKEN,
    "Accept": "application/json"
}

# --- Account & Positions ---
def get_account_information():
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/account-information"
    return requests.get(url, headers=headers).json()

def get_positions():
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/positions"
    return requests.get(url, headers=headers).json()

def get_orders():
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/orders"
    return requests.get(url, headers=headers).json()

# --- Symbols & Prices ---
def get_symbols():
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols"
    return requests.get(url, headers=headers).json()

def get_symbol_spec(symbol):
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/specification"
    return requests.get(url, headers=headers).json()

def get_current_price(symbol):
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/current-price"
    return requests.get(url, headers=headers).json()

# --- Candles ---
def get_current_candles(symbol, timeframe="M15", limit=5):
    """⚠️ Cet endpoint ne marche qu’en Websocket. En REST, il retournera probablement NotFoundError."""
    url = f"{BASE_URL}/users/current/accounts/{ACCOUNT_ID}/symbols/{symbol}/current-candles/{timeframe}?limit={limit}"
    return requests.get(url, headers=headers).json()

def get_historical_candles(symbol, timeframe="15m", days=7, limit=500):
    """Bougies historiques via REST officiel"""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    url = f"{BASE_MARKET_URL}/users/current/accounts/{ACCOUNT_ID}/historical-market-data/symbols/{symbol}/timeframes/{timeframe}/candles"
    params = {
        "startTime": start.isoformat() + "Z",
        "limit": limit
    }
    r = requests.get(url, headers=headers, params=params).json()

    if isinstance(r, dict) and "error" in r:
        return r  # renvoie l'erreur brute

    df = pd.DataFrame(r)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tickVolume": "Volume"
        })
    return df

# --- Main ---
if __name__ == "__main__":
    print("=== Account Information ===")
    print(get_account_information())

    print("\n=== Positions ===")
    print(get_positions())

    print("\n=== Orders ===")
    print(get_orders())

    print("\n=== Symbols (first 5) ===")
    print(get_symbols())

    print("\n=== Specification BTCUSD ===")
    print(get_symbol_spec("BTCUSD"))

    print("\n=== Current Price BTCUSD ===")
    print(get_current_price("BTCUSD"))

    print("\n=== Current 15m Candles BTCUSD (REST/Websocket only) ===")
    print(get_current_candles("BTCUSD", timeframe="M15", limit=5))

    print("\n=== Historical 15m Candles BTCUSD (last 7 days) ===")
    df = get_historical_candles("BTCUSD", timeframe="15m", days=7, limit=100)
    print(df.head())
