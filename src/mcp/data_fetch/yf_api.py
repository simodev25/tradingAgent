import json
from datetime import datetime
from typing import Annotated

import yfinance as yf
from loguru import logger
from mcp.server.fastmcp import FastMCP
# from pydantic import Field  # Commenté pour tester
try:
    from yfinance.const import SECTOR_INDUSTRY_MAPPING as SECTOR_INDUSTY_MAPPING
except ImportError:
    SECTOR_INDUSTY_MAPPING = {}

import sys
import os
# Add the current directory to sys.path to find yftypes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yftypes import Interval
from yftypes import Period
from yftypes import SearchType
from yftypes import Sector
from yftypes import TopType


mcp = FastMCP("Yahoo Finance MCP Server", log_level="DEBUG")


def get_ticker_info(symbol: str) -> str:
    """Retrieve stock data including company info, financials, trading metrics and governance data."""
    ticker = yf.Ticker(symbol)

    # Convert timestamps to human-readable format
    info = ticker.info
    for key, value in info.items():
        if not isinstance(key, str):
            continue

        if key.lower().endswith(("date", "start", "end", "timestamp", "time", "quarter")):
            try:
                info[key] = datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.error("Unable to convert {}: {} to datetime, got error: {}", key, value, e)
                continue

    return json.dumps(info, ensure_ascii=False)



def get_ticker_news(symbol: str) -> str:
    """Fetches recent news articles related to a specific stock symbol with title, content, and source details."""
    ticker = yf.Ticker(symbol)
    news = ticker.get_news()
    return str(news)



def search(
    query: str,
    search_type: SearchType,
) -> str:
    """Fetches and organizes search results from Yahoo Finance, including stock quotes and news articles."""
    s = yf.Search(query)
    match search_type.lower():
        case "all":
            return json.dumps(s.all, ensure_ascii=False)
        case "quotes":
            return json.dumps(s.quotes, ensure_ascii=False)
        case "news":
            return json.dumps(s.news, ensure_ascii=False)
        case _:
            return "Invalid output_type. Use 'all', 'quotes', or 'news'."


def get_top_etfs(
    sector: Sector,
    top_n: int,
) -> str:
    """Retrieve popular ETFs for a sector, returned as a list in 'SYMBOL: ETF Name' format."""
    if top_n < 1:
        return "top_n must be greater than 0"

    s = yf.Sector(sector)

    result = [f"{symbol}: {name}" for symbol, name in s.top_etfs.items()]

    return "\n".join(result[:top_n])


def get_top_mutual_funds(
    sector: Sector,
    top_n: int,
) -> str:
    """Retrieve popular mutual funds for a sector, returned as a list in 'SYMBOL: Fund Name' format."""
    if top_n < 1:
        return "top_n must be greater than 0"

    s = yf.Sector(sector)
    return "\n".join(f"{symbol}: {name}" for symbol, name in s.top_mutual_funds.items())


def get_top_companies(
    sector: Sector,
    top_n: int,
) -> str:
    """Get top companies in a sector with name, analyst rating, and market weight as JSON array."""
    if top_n < 1:
        return "top_n must be greater than 0"

    s = yf.Sector(sector)
    df = s.top_companies
    if df is None:
        return f"No top companies available for {sector} sector."

    return df.iloc[:top_n].to_json(orient="records")


def get_top_growth_companies(
    sector: Sector,
    top_n: int,
) -> str:
    """Get top growth companies grouped by industry within a sector as JSON array with growth metrics."""
    if top_n < 1:
        return "top_n must be greater than 0"

    results = []

    for industry_name in SECTOR_INDUSTY_MAPPING[sector]:
        industry = yf.Industry(industry_name)

        df = industry.top_growth_companies
        if df is None:
            continue

        results.append(
            {
                "industry": industry_name,
                "top_growth_companies": df.iloc[:top_n].to_json(orient="records"),
            }
        )
    return json.dumps(results, ensure_ascii=False)


def get_top_performing_companies(
    sector: Sector,
    top_n: int,
) -> str:
    """Get top performing companies grouped by industry within a sector as JSON array with performance metrics."""
    if top_n < 1:
        return "top_n must be greater than 0"

    results = []

    for industry_name in SECTOR_INDUSTY_MAPPING[sector]:
        industry = yf.Industry(industry_name)

        df = industry.top_performing_companies
        if df is None:
            continue

        results.append(
            {
                "industry": industry_name,
                "top_performing_companies": df.iloc[:top_n].to_json(orient="records"),
            }
        )
    return json.dumps(results, ensure_ascii=False)



def get_top(
    sector: Sector,
    top_type: TopType,
    top_n: int = 10,
) -> str:
    """Get top entities (ETFs, mutual funds, companies, growth companies, or performing companies) in a sector."""
    match top_type:
        case "top_etfs":
            return get_top_etfs(sector, top_n)
        case "top_mutual_funds":
            return get_top_mutual_funds(sector, top_n)
        case "top_companies":
            return get_top_companies(sector, top_n)
        case "top_growth_companies":
            return get_top_growth_companies(sector, top_n)
        case "top_performing_companies":
            return get_top_performing_companies(sector, top_n)
        case _:
            return "Invalid top_type"



def get_price_history(
    symbol: str,
    period: Period = "1mo",
    interval: Interval = "1d",
    output_format: str = "json",
    limit_rows: int = 1000,
    include_columns: str = "Open,High,Low,Close,Volume"
) -> str:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, rounding=True)

    if include_columns:
        cols = [c.strip() for c in include_columns.split(",")]
        df = df[[c for c in cols if c in df.columns]]

    df = df.tail(limit_rows).reset_index()

    if output_format == "json":
        return df.to_json(orient="records", date_format="iso")
    elif output_format == "csv":
        return df.to_csv(index=False)
    elif output_format == "markdown":
        return df.to_markdown(index=False)
    else:
        return "Invalid output_format"



if __name__ == "__main__":
    print("\n🚀 Running Yahoo Finance MCP Server...")
