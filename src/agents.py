# agents.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent

import sys
import asyncio
import os
import re
import json
import time
from asyncio import TimeoutError
from contextlib import suppress
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, Any, Tuple
from datetime import datetime

# --- pour détecter GraphRecursionError si dispo ---
try:
    from langchain_core.runnables.graph import GraphRecursionError  # v0.2+
except Exception:  # fallback
    class GraphRecursionError(Exception):
        pass

load_dotenv()
# -------------------------
# Setup OpenAI (ChatOpenAI) — timeouts propres
# -------------------------
from langchain_ollama import ChatOllama

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

# Important: on passe le header au client httpx via client_kwargs
model = ChatOllama(
    model=ollama_model,
    base_url=ollama_base_url,
    temperature=0,
    client_kwargs={"headers": {"Authorization": ollama_api_key}},
)

# Limite de récursion LangGraph (configurable)
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "600"))

# -------------------------
# Configuration et validation
# -------------------------
def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on", "oui"}


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("Les paramètres doivent être un dictionnaire")
    if "symbol" not in params:
        raise ValueError("Le paramètre 'symbol' est obligatoire")

    normalized = {
        "symbol": str(params["symbol"]),
        "period": str(params.get("period", "1mo")),
        "interval": str(params.get("interval", "1d")),
        "horizon": str(params.get("horizon", "swing")),
        "risk_level": str(params.get("risk_level", "medium")),
        "default_volume": float(params.get("default_volume", 0.01)),
        "min_confidence": int(params.get("min_confidence", 60)),
        "honor_hold": _to_bool(params.get("honor_hold", True)),
        "dry_run": _to_bool(params.get("dry_run", True)),
        "news_top_n": int(params.get("news_top_n", 10)),
        "include_columns": str(params.get("include_columns", "Open,High,Low,Close,Volume")),
    }

    if normalized["default_volume"] <= 0:
        raise ValueError("default_volume doit être positif")
    if not 0 <= normalized["min_confidence"] <= 100:
        raise ValueError("min_confidence doit être entre 0 et 100")
    if normalized["risk_level"] not in ["low", "medium", "high"]:
        raise ValueError("risk_level doit être 'low', 'medium' ou 'high'")
    if normalized["horizon"] not in ["intraday", "scalping", "swing", "long_term"]:
        raise ValueError("horizon doit être 'intraday', 'scalping', 'swing' ou 'long_term'")
    return normalized


# -------------------------
# Helpers
# -------------------------
def strip_json_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(^|\s)//.*?$", "", s, flags=re.M)
    return s.strip()


def extract_last_message(resp) -> str:
    try:
        if isinstance(resp, dict) and "messages" in resp:
            return resp["messages"][-1].content
        if hasattr(resp, "content"):
            return resp.content
        if hasattr(resp, "generations"):
            return resp.generations[0][0].text
        return str(resp)
    except Exception as e:
        logger.error(f"❌ Impossible d'extraire le texte de la réponse: {e}")
        return ""


def _format_exception_chain(e: BaseException) -> str:
    parts = [f"{type(e).__name__}: {e}"]
    cur = e
    if hasattr(e, "exceptions") and isinstance(getattr(e, "exceptions"), (list, tuple)):
        for i, sub in enumerate(e.exceptions):
            parts.append(f" └─[sub {i}] {type(sub).__name__}: {sub}")
    while True:
        nxt = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        if not nxt:
            break
        parts.append(f" -> {type(nxt).__name__}: {nxt}")
        cur = nxt
    return " | ".join(parts)


# --- Extracteur JSON strict (gère ```json ... ``` + équilibrage) ---
def _extract_json_block(text: str) -> str:
    """
    1) si un bloc code-fencé ```json ... ``` existe, on retourne son contenu
    2) sinon, on lit un JSON équilibré à partir du premier '{' ou '['
    """
    if not isinstance(text, str):
        return ""

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        candidate = re.sub(r"/\*.*?\*/", "", candidate, flags=re.S)
        candidate = re.sub(r"(^|\s)//.*?$", "", candidate, flags=re.M)
        return candidate.strip()

    start = None
    for ch in ("{", "["):
        i = text.find(ch)
        if i != -1 and (start is None or i < start):
            start = i
    if start is None:
        return text.strip()

    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    i = start
    in_string = False
    string_char = ""
    escape = False

    while i < len(text):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == string_char:
                in_string = False
        else:
            if c in ('"', "'"):
                in_string = True
                string_char = c
            elif c == opening:
                depth += 1
            elif c == closing:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    candidate = re.sub(r"/\*.*?\*/", "", candidate, flags=re.S)
                    candidate = re.sub(r"(^|\s)//.*?$", "", candidate, flags=re.M)
                    return candidate.strip()
        i += 1

    candidate = text[start:]
    candidate = re.sub(r"/\*.*?\*/", "", candidate, flags=re.S)
    candidate = re.sub(r"(^|\s)//.*?$", "", candidate, flags=re.M)
    return candidate.strip()


def _validate_levels(entry, sl, tp) -> bool:
    try:
        entry, sl, tp = float(entry), float(sl), float(tp)
        if not (entry > 0 and sl > 0 and tp > 0):
            return False
        return (sl < entry < tp) or (tp < entry < sl)
    except Exception:
        return False


def _should_send_order(context: dict, min_confidence: int, honor_hold: bool) -> Tuple[bool, str]:
    tec = (context or {}).get("technical_decision", {})
    dec = tec.get("decision", {}) if isinstance(tec, dict) else {}
    action = str(dec.get("action", "HOLD")).upper()
    conf = int(dec.get("confidence", 0) or 0)

    if honor_hold and action == "HOLD":
        return False, "Honor HOLD"
    if conf < min_confidence:
        return False, f"Confidence {conf}% < min {min_confidence}%"
    entry, sl, tp = dec.get("entry"), dec.get("sl"), dec.get("tp")
    if not _validate_levels(entry, sl, tp):
        return False, f"Invalid levels (entry={entry}, sl={sl}, tp={tp})"
    return True, "OK"


# -------------------------
# News MCP
# -------------------------
async def run_news_mcp(symbol: str) -> dict:
    news_mcp_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "mcp/analyse/analyze_news_mcp.py")
    )
    logger.debug(f"News MCP server path: {news_mcp_path}")

    try:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-u", news_mcp_path],
            cwd=os.path.dirname(news_mcp_path),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)

                base_prompt = await load_mcp_prompt(
                    session, "news_agent", arguments={"symbol": symbol}
                )

                agent = create_react_agent(model, tools)

                resp = await agent.ainvoke(
                    {"messages": base_prompt},
                    config={"recursion_limit": AGENT_RECURSION_LIMIT},
                )

                raw = extract_last_message(resp)
                logger.info(f"🚀 run_news_mcp {raw}")

                # Safeguard: réponse non JSON => fallback sans lever d'exception
                if isinstance(raw, str):
                    lw = raw.lower()
                    if ("need more steps" in lw) or ("sorry" in lw and "step" in lw):
                        logger.warning("[news] modèle a répondu 'need more steps' → fallback HOLD")
                        return {
                            "ok": False,
                            "global_bias": "neutral",
                            "reason": "model returned incomplete non-JSON response",
                            "summary": {"positive": 0, "negative": 0, "neutral": 1},
                            "global_score": 0.0,
                            "top_influential_titles": [],
                            "raw": raw,
                        }

                clean = strip_json_comments(raw)
                json_txt = _extract_json_block(clean)

                # Si pas d’accolade → fallback propre
                if not (json_txt.strip().startswith("{") or json_txt.strip().startswith("[")):
                    logger.warning("[news] non-JSON content, fallback HOLD")
                    return {
                        "ok": False,
                        "global_bias": "neutral",
                        "reason": "non-json content from news agent",
                        "summary": {"positive": 0, "negative": 0, "neutral": 1},
                        "global_score": 0.0,
                        "top_influential_titles": [],
                        "raw": raw,
                    }

                try:
                    res = json.loads(json_txt)
                except Exception:
                    logger.warning("[news] JSON parse failed, fallback HOLD")
                    return {
                        "ok": False,
                        "global_bias": "neutral",
                        "reason": "json parse failed (news)",
                        "summary": {"positive": 0, "negative": 0, "neutral": 1},
                        "global_score": 0.0,
                        "top_influential_titles": [],
                        "raw": raw,
                    }

                if isinstance(res, dict):
                    res.setdefault("ok", True)
                else:
                    res = {"ok": True, "data": res}
                return res

    except GraphRecursionError as e:
        logger.error(f"GraphRecursionError (news): {e}")
        return {
            "ok": False,
            "global_bias": "neutral",
            "reason": "news graph recursion limit exceeded",
            "summary": {"positive": 0, "negative": 0, "neutral": 1},
            "global_score": 0.0,
            "top_influential_titles": [],
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Erreur dans run_news_mcp pour {symbol}: {e}")
        return {
            "ok": False,
            "global_bias": "neutral",
            "reason": f"Erreur d'analyse des news: {str(e)}",
            "summary": {"positive": 0, "negative": 0, "neutral": 1},
            "global_score": 0.0,
            "top_influential_titles": [],
            "error": str(e),
        }


# -------------------------
# Analysis Technique MCP (timeout + recursion_limit + non-JSON fallback)
# -------------------------
async def run_analysis_tec_mcp(
    symbol: str, period="1mo", interval="1d", risk_level="medium", horizon="swing"
) -> dict:
    analysis_mcp_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "mcp/analyse/analyze_tec_mcp.py")
    )
    logger.debug(f"Analysis MCP server path: {analysis_mcp_path}")

    def _fallback(reason: str, extra: dict | None = None) -> dict:
        base = {
            "ok": False,
            "symbol": symbol,
            "horizon": horizon,
            "decision": {
                "action": "HOLD",
                "entry": None,
                "sl": None,
                "tp": None,
                "confidence": 0,
                "risk_level": risk_level,
            },
            "reason": reason,
        }
        if extra:
            base.update(extra)
        return base

    try:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-u", analysis_mcp_path],
            cwd=os.path.dirname(analysis_mcp_path),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)

                base_prompt = await load_mcp_prompt(
                    session,
                    "analysis_agent",
                    arguments={
                        "symbol": symbol,
                        "period": period,
                        "interval": interval,
                        "horizon": horizon,
                        "risk_level": risk_level,
                    },
                )

                agent = create_react_agent(model, tools)

                task = asyncio.create_task(
                    agent.ainvoke(
                        {"messages": base_prompt},
                        config={"recursion_limit": AGENT_RECURSION_LIMIT},
                    )
                )
                try:
                    resp = await asyncio.wait_for(task, timeout=180)
                except TimeoutError:
                    logger.error("⏱️ Timeout analysis agent (60s)")
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    return _fallback("analysis agent timeout", {"error": "timeout"})
                except GraphRecursionError as e:
                    logger.error(f"GraphRecursionError (analysis): {e}")
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    return _fallback("analysis graph recursion limit exceeded", {"error": str(e)})
                except Exception as e:
                    msg = _format_exception_chain(e)
                    if "GraphRecursionError" in msg:
                        logger.error(f"GraphRecursionError (wrapped): {msg}")
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                        return _fallback("analysis graph recursion (wrapped) exceeded", {"error": msg})
                    logger.error("Analysis agent invoke failed: " + msg)
                    raise

                raw = extract_last_message(resp)
                logger.info(f"🚀 run_analysis_tec_mcp {raw}")

                # ➜ Si la réponse n’est pas du JSON (ex. “need more steps”), on fallback sans lever d’exception
                if isinstance(raw, str):
                    lw = raw.lower()
                    if ("need more steps" in lw) or ("sorry" in lw and "step" in lw):
                        logger.warning("[analysis] modèle a répondu 'need more steps' → fallback HOLD")
                        return _fallback("model returned incomplete non-JSON response", {"raw": raw})

                clean = strip_json_comments(raw)
                json_txt = _extract_json_block(clean)

                if not (json_txt.strip().startswith("{") or json_txt.strip().startswith("[")):
                    logger.warning("[analysis] non-JSON content → fallback HOLD")
                    return _fallback("non-json content from analysis agent", {"raw": raw})

                try:
                    res = json.loads(json_txt)
                except Exception:
                    logger.warning("[analysis] JSON parse failed → fallback HOLD")
                    return _fallback("json parse failed (analysis)", {"raw": raw})

                if isinstance(res, dict):
                    res.setdefault("ok", True)
                else:
                    res = {"ok": True, "data": res}
                return res

    except Exception as e:
        logger.error(f"Erreur dans run_analysis_tec_mcp pour {symbol}: {e}")
        return {
            "ok": False,
            "symbol": symbol,
            "horizon": horizon,
            "decision": {
                "action": "HOLD",
                "entry": None,
                "sl": None,
                "tp": None,
                "confidence": 0,
                "risk_level": risk_level,
            },
            "reason": f"Erreur d'analyse technique: {str(e)}",
            "error": str(e),
        }


# -------------------------
# Execution MCP (whitelist, timeout, recursion_limit, garde-fous)
# -------------------------
# -------------------------
# Execution MCP — version simple (TOUS les tools)
# -------------------------
async def run_execution_mcp(
    context: dict,
    default_volume: float = 0.01,
    min_confidence: int = 60,
    honor_hold: bool = True,
    dry_run: bool = True,
) -> dict:
    """
    Démarre le serveur MCP d'exécution, charge TOUS les tools,
    envoie le prompt 'execution_agent' et retourne le plan JSON.
    """
    execution_mcp_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "mcp/execution/execution_mcp.py")
    )
    logger.debug(f"Execution MCP server path: {execution_mcp_path}")

    def _safe_plan(reason: str, extra: dict | None = None) -> dict:
        plan = {
            "symbol": (context or {}).get("symbol"),
            "normalized_order": None,
            "checks": {"values_ok": False, "direction_ok": False, "spec_ok": False, "rr": None},
            "decision": {"send_order": False, "reason": reason},
        }
        if extra:
            plan.update(extra)
        return {"plan": plan, "dry_run": dry_run}

    try:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-u", execution_mcp_path],
            cwd=os.path.dirname(execution_mcp_path),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 1) Charger TOUS les tools (aucun filtrage)
                tools = await load_mcp_tools(session)

                # 2) Construire le prompt (style run_analysis_tec_mcp)
                context_str = json.dumps(context or {}, ensure_ascii=False)
                base_prompt = await load_mcp_prompt(
                    session,
                    "execution_agent",
                    arguments={
                        "context": context_str,
                        "default_volume": str(default_volume),
                        "min_confidence": str(min_confidence),
                        "honor_hold": "True" if honor_hold else "False",
                        "dry_run": "True" if dry_run else "False",
                    },
                )

                # 3) Agent ReAct + timeout simple
                agent = create_react_agent(model, tools)
                task = asyncio.create_task(
                    agent.ainvoke(
                        {"messages": base_prompt},
                        config={"recursion_limit": AGENT_RECURSION_LIMIT},
                    )
                )
                try:
                    resp = await asyncio.wait_for(task, timeout=90)
                except TimeoutError:
                    logger.error("⏱️ Timeout execution agent (90s)")
                    with suppress(asyncio.CancelledError):
                        task.cancel()
                        await task
                    return _safe_plan("execution agent timeout")
                except GraphRecursionError as e:
                    logger.error(f"GraphRecursionError (execution): {e}")
                    with suppress(asyncio.CancelledError):
                        task.cancel()
                        await task
                    return _safe_plan("execution graph recursion limit exceeded", {"error": str(e)})
                except Exception as e:
                    msg = _format_exception_chain(e)
                    logger.error("Agent exec invoke failed: " + msg)
                    with suppress(asyncio.CancelledError):
                        task.cancel()
                        await task
                    return _safe_plan("execution agent error", {"error": msg})

                # 4) Extraction + parsage JSON (identique à run_analysis_tec_mcp)
                raw = extract_last_message(resp)
                logger.info(f"🚀 run_execution_mcp {raw}")
                if isinstance(raw, str):
                    lw = raw.lower()
                    if ("need more steps" in lw) or ("sorry" in lw and "step" in lw):
                        logger.warning("[exec] modèle a répondu 'need more steps' → safe cancel")
                        return _safe_plan("model returned incomplete non-JSON response", {"raw": raw})

                clean = strip_json_comments(raw)
                json_txt = _extract_json_block(clean)

                if not (isinstance(json_txt, str) and (json_txt.strip().startswith("{") or json_txt.strip().startswith("["))):
                    logger.warning("[exec] non-JSON content → safe cancel")
                    return _safe_plan("non-json content from execution agent", {"raw": raw})

                try:
                    plan = json.loads(json_txt)
                except Exception as e:
                    logger.warning(f"[exec] JSON parse failed → safe cancel | err={e}")
                    return _safe_plan(f"json parse failed (execution): {e}", {"raw": raw})

                # Normalisation de sortie
                if isinstance(plan, dict):
                    return {"plan": plan, "dry_run": dry_run}
                else:
                    return {
                        "plan": {"raw": plan, "decision": {"send_order": False, "reason": "non-dict plan"}},
                        "dry_run": dry_run,
                    }

    except Exception as e:
        logger.error(f"❌ Erreur dans run_execution_mcp: {_format_exception_chain(e)}")
        return _safe_plan(f"execution mcp error: {e}")
# -------------------------
# Main Trading Agent (gating strict + skip si HOLD)
# -------------------------
async def trading_agent(params: dict) -> dict:
    start_time = datetime.now()
    symbol = params.get("symbol", "UNKNOWN")

    def _halt(reason: str, news: dict | None, tec: dict | None, stopped_at: str):
        duration = (datetime.now() - start_time).total_seconds()
        res = {
            "symbol": symbol,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "news_sentiment": news,
            "technical_decision": tec,
            "execution": {"skipped": True, "reason": f"pipeline halted at {stopped_at}: {reason}"},
            "status": "halted",
            "stopped_at": stopped_at,
        }
        logger.warning(f"⛔ Pipeline halted at {stopped_at}: {reason}")
        return res

    def _skip_hold(news: dict, tec: dict, reason: str):
        duration = (datetime.now() - start_time).total_seconds()
        res = {
            "symbol": symbol,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "news_sentiment": news,
            "technical_decision": tec,
            "execution": {"skipped": True, "reason": reason},
            "status": "skipped",
        }
        logger.info(f"🛑 Execution skipped: {reason}")
        return res

    try:
        params = _normalize_params(params)
        symbol = params["symbol"]
        logger.info(f"🚀 Début de l'analyse pour {symbol}")

        # Étape 1: NEWS (si erreur -> stop)
        news_analysis = await run_news_mcp(symbol)
        if not news_analysis.get("ok", True):
            return _halt(news_analysis.get("reason", news_analysis.get("error", "news error")), news_analysis, None, "news")

        # Étape 2: ANALYSE TECHNIQUE (si erreur -> stop)
        decision = await run_analysis_tec_mcp(
            symbol,
            period=params["period"],
            interval=params["interval"],
            risk_level=params["risk_level"],
            horizon=params["horizon"],
        )
        if not decision.get("ok", True):
            return _halt(decision.get("reason", decision.get("error", "analysis error")), news_analysis, decision, "analysis")

        # *** Gating HOLD : on n'appelle pas l'exécution si HOLD + honor_hold=True ***
        action = str(((decision or {}).get("decision", {}) or {}).get("action", "")).upper()
        if params.get("honor_hold", True) and action == "HOLD":
            return _skip_hold(news_analysis, decision, "Analysis returned HOLD and honor_hold=True; skipping execution.")

        # Étape 3: EXÉCUTION
        context = {"symbol": symbol, "news_sentiment": news_analysis, "technical_decision": decision}
        execution = await run_execution_mcp(
            context=context,
            default_volume=params["default_volume"],
            min_confidence=params["min_confidence"],
            honor_hold=params["honor_hold"],
            dry_run=params["dry_run"],
        )

        duration = (datetime.now() - start_time).total_seconds()
        result = {
            "symbol": symbol,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "news_sentiment": news_analysis,
            "technical_decision": decision,
            "execution": execution,
            "status": "success",
        }

        logger.success(f"✅ Analyse complète pour {symbol} en {duration:.2f}s")
        return result

    except ValueError as e:
        logger.error(f"❌ Erreur de validation pour {symbol}: {e}")
        return {
            "symbol": symbol,
            "timestamp": start_time.isoformat(),
            "status": "error",
            "error_type": "validation",
            "error_message": str(e),
        }
    except Exception as e:
        logger.error(f"❌ Erreur inattendue pour {symbol}: {e}")
        return {
            "symbol": symbol,
            "timestamp": start_time.isoformat(),
            "status": "error",
            "error_type": "unexpected",
            "error_message": str(e),
        }


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    print("\n🚀 Running Trading Agent...")
    try:
        response = asyncio.run(
            trading_agent(
                {
                    "symbol": "BTCUSD",
                    "period": "5d",
                    "interval": "15m",
                    "news_top_n": 10,
                    "include_columns": "Open,High,Low,Close,Volume",
                    "horizon": "scalping",
                    "risk_level": "high",
                    "default_volume": 0.01,
                    "min_confidence": 40,
                    "honor_hold": True,
                    "dry_run": False,
                }
            )
        )
        print("\n📊 Final Response:\n", json.dumps(response, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)
