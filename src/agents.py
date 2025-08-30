from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

import sys
import asyncio
import os
import re
import json
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# -------------------------
# Setup Azure OpenAI
# -------------------------
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("API_VERSION")

model = AzureChatOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version=api_version,
    deployment_name=deployment,
)

# -------------------------
# Helpers
# -------------------------
def strip_json_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(^|\s)//.*?$", "", s, flags=re.M)
    return s.strip()

# -------------------------
# Data MCP
# -------------------------
async def run_data_mcp(params: dict) -> dict:
    data_mcp_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "mcp/data_fetch/metaapi_mcp.py")
    )
    print("data MCP server path:", data_mcp_path)

    server_params = StdioServerParameters(
        command="python",
        args=["-u", data_mcp_path],
        cwd=os.path.dirname(data_mcp_path),
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            prompt = await load_mcp_prompt(
                session,
                "fetch_market_context",
                arguments={
                    "symbol": params.get("symbol", "AAPL"),
                    "period": params.get("period", "3mo"),
                    "interval": params.get("interval", "1h"),
                },
            )

            agent = create_react_agent(model, tools)
            resp = await agent.ainvoke({"messages": prompt})
            raw = resp["messages"][-1].content

            clean = strip_json_comments(raw)
            bundle = json.loads(clean)
            return bundle

# -------------------------
# News MCP
# -------------------------
async def run_news_mcp(news_items: list, symbol: str) -> dict:
    news_mcp_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "mcp/analyse/analyze_news_mcp.py")
    )
    print("news MCP server path:", news_mcp_path)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", news_mcp_path],
        cwd=os.path.dirname(news_mcp_path),
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            base_prompt = await load_mcp_prompt(
                session,
                "news_agent",
                arguments={"symbol": symbol, "news_items": json.dumps(news_items, ensure_ascii=False)},
            )

            agent = create_react_agent(model, tools)
            resp = await agent.ainvoke({"messages": base_prompt})
            raw = resp["messages"][-1].content
            clean = strip_json_comments(raw)
            return json.loads(clean)

# -------------------------
# Analysis Technique MCP
# -------------------------
async def run_analysis_tec_mcp(bundle: dict, risk_level="medium", horizon="swing") -> dict:
    analysis_mcp_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "mcp/analyse/analyze_tec_mcp.py")
    )
    print("analysis MCP server path:", analysis_mcp_path)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", analysis_mcp_path],
        cwd=os.path.dirname(analysis_mcp_path),
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    symbol = bundle.get("symbol", "AAPL")
    ohlcv = bundle.get("history", [])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            base_prompt = await load_mcp_prompt(
                session,
                "analysis_agent",
                arguments={"symbol": symbol, "horizon": horizon, "risk_level": risk_level},
            )

            final_prompt = f"""{base_prompt}

Voici la variable `ohlcv` (JSON):
ohlcv = {json.dumps(ohlcv, ensure_ascii=False)}

Réponds UNIQUEMENT par l'objet JSON retourné par trade_plan.
"""
            agent = create_react_agent(model, tools)
            resp = await agent.ainvoke({"messages": [HumanMessage(content=final_prompt)]})
            raw = resp["messages"][-1].content
            clean = strip_json_comments(raw)
            return json.loads(clean)

# -------------------------
# Main Trading Agent
# -------------------------
async def trading_agent(params: dict) -> dict:
    bundle = await run_data_mcp(params)
    print("\n✅ Data bundle récupéré")

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    news_items = bundle.get("news", [])
    news_analysis = {}
    if news_items:
        news_analysis = await run_news_mcp(news_items, bundle["symbol"])
        print("\n✅ News sentiment analysé")

    decision = await run_analysis_tec_mcp(
        bundle,
        risk_level=params.get("risk_level", "medium"),
        horizon=params.get("horizon", "swing")
    )
    print("\n✅ Analyse technique faite")

    return {
        "symbol": bundle["symbol"],
        "news_sentiment": news_analysis,
        "technical_decision": decision,
    }

# -------------------------
if __name__ == "__main__":
    print("\n🚀 Running Trading Agent...")
    response = asyncio.run(
        trading_agent({
            "symbol": "BTCUSD",
            "period": "5d",                        # 5 jours d’historique
            "interval": "15m",                     # bougie 15 minutes
            "news_top_n": 10,                      # pas besoin de trop de news
            "include_columns": "Open,High,Low,Close,Volume",
            "horizon": "intraday",                 # scalping/intraday
            "risk_level": "high"                   # agressif
        })
        )
    print("\n📊 Final Response:\n", json.dumps(response, indent=2, ensure_ascii=False))
