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
from typing import Dict, Any, Tuple, Optional, Callable
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
import importlib.util
from functools import lru_cache

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
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


ANALYSIS_AGENT_TIMEOUT = int(os.getenv("ANALYSIS_AGENT_TIMEOUT", "120"))
ENABLE_ANALYSIS_DIRECT_FALLBACK = _to_bool(os.getenv("ENABLE_ANALYSIS_DIRECT_FALLBACK", "true"))
DIRECT_ANALYSIS_FIRST = _to_bool(os.getenv("DIRECT_ANALYSIS_FIRST", "false"))
AUTO_EXECUTE_PLAN = _to_bool(os.getenv("AUTO_EXECUTE_PLAN", "true"))

import importlib.util as _imp_util
def _load_exec_core():
    path = os.path.join(os.path.dirname(__file__), "mcp", "execution", "execution_core.py")
    spec = _imp_util.spec_from_file_location("execution_core_direct", path)
    if not spec or not spec.loader:
        return None
    mod = _imp_util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    except Exception as e:
        logger.error(f"[AUTO_EXECUTE] load error: {e}")
        return None


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


@lru_cache(maxsize=1)
def _load_intraday_tool() -> Optional[Callable[..., str]]:
    module_path = os.path.join(os.path.dirname(__file__), "mcp", "analyse", "analyze_tec_mcp.py")
    if not os.path.isfile(module_path):
        logger.error(f"[direct_intraday] module not found at {module_path}")
        return None
    spec = importlib.util.spec_from_file_location("analyze_tec_mcp_direct", module_path)
    if spec is None or spec.loader is None:
        logger.error("[direct_intraday] failed to create spec for analyze_tec_mcp")
        return None
    module = importlib.util.module_from_spec(spec)
    module_dir = os.path.dirname(module_path)
    added_path = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added_path = True
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.error(f"[direct_intraday] import error: {exc}")
        if added_path and module_dir in sys.path:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass
        return None
    tool = getattr(module, "intraday_decision", None)
    if not callable(tool):
        logger.error("[direct_intraday] intraday_decision not found in module")
        if added_path and module_dir in sys.path:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass
        return None
    return tool


def _direct_intraday_analysis(symbol: str, interval: str, horizon: str, risk_level: str) -> Optional[dict]:
    try:
        tool = _load_intraday_tool()
        if tool is None:
            return None
        callable_tool = getattr(tool, "__wrapped__", tool)
        # 1st attempt with requested interval
        raw = callable_tool(symbol=symbol, interval=interval, risk_level=risk_level)
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not payload.get("ok"):
            logger.error(f"[direct_intraday] tool returned error: {payload}")
            return None
        data = payload.get("data") or {}
        decision_raw = data.get("decision") or {}
        decision = {
            "action": decision_raw.get("action", "HOLD"),
            "entry": decision_raw.get("entry"),
            "sl": decision_raw.get("sl"),
            "tp": decision_raw.get("tp"),
            "confidence": decision_raw.get("confidence", 0),
            "risk_level": decision_raw.get("risk_level", risk_level),
        }
        # If HOLD on 15m, try 5m as a second chance to increase entries
        if decision["action"].upper() == "HOLD" and interval.lower() == "15m":
            try:
                raw2 = callable_tool(symbol=symbol, interval="5m", risk_level=risk_level)
                payload2 = json.loads(raw2)
                if isinstance(payload2, dict) and payload2.get("ok"):
                    data2 = payload2.get("data") or {}
                    decision2 = (data2.get("decision") or {})
                    if str(decision2.get("action", "HOLD")).upper() != "HOLD":
                        decision = {
                            "action": decision2.get("action"),
                            "entry": decision2.get("entry"),
                            "sl": decision2.get("sl"),
                            "tp": decision2.get("tp"),
                            "confidence": decision2.get("confidence", 0),
                            "risk_level": decision2.get("risk_level", risk_level),
                        }
                        data = data2
            except Exception as _:
                pass
        # If still HOLD after 5m fallback, try 1m as a last resort
        if decision["action"].upper() == "HOLD" and interval.lower() in ("15m", "5m"):
            try:
                raw3 = callable_tool(symbol=symbol, interval="1m", risk_level=risk_level)
                payload3 = json.loads(raw3)
                if isinstance(payload3, dict) and payload3.get("ok"):
                    data3 = payload3.get("data") or {}
                    decision3 = (data3.get("decision") or {})
                    if str(decision3.get("action", "HOLD")).upper() != "HOLD":
                        decision = {
                            "action": decision3.get("action"),
                            "entry": decision3.get("entry"),
                            "sl": decision3.get("sl"),
                            "tp": decision3.get("tp"),
                            "confidence": decision3.get("confidence", 0),
                            "risk_level": decision3.get("risk_level", risk_level),
                        }
                        data = data3
            except Exception as _:
                pass
        result = {
            "ok": True,
            "symbol": symbol,
            "horizon": horizon,
            "decision": decision,
            "reason": f"{data.get('reason', 'Direct intraday fallback')}",
            "regime": data.get("regime"),
            "volatility": data.get("volatility"),
            "source": "direct_intraday",
        }
        for extra_key in ("levels", "position", "management"):
            if extra_key in data:
                result[extra_key] = data[extra_key]
        return result
    except Exception as exc:
        logger.error(f"[direct_intraday] failed: {exc}")
        return None



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
                    resp = await asyncio.wait_for(task, timeout=ANALYSIS_AGENT_TIMEOUT)
                except TimeoutError:
                    logger.error(f"⏱️ Timeout analysis agent ({ANALYSIS_AGENT_TIMEOUT}s)")
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    direct = None
                    if ENABLE_ANALYSIS_DIRECT_FALLBACK:
                        direct = await asyncio.to_thread(
                            _direct_intraday_analysis,
                            symbol,
                            interval,
                            horizon,
                            risk_level,
                        )
                    if direct:
                        logger.warning("[analysis] direct intraday fallback used after LLM timeout")
                        return direct
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
                # Hedge/expo controls sourced from environment (strings expected by prompt)
                hedge_mode = os.getenv("HEDGE_MODE", "false").strip().lower()
                hedge_max_pairs = os.getenv("HEDGE_MAX_PAIRS", "2").strip()
                hedge_max_net_usd_mult = os.getenv("HEDGE_MAX_NET_USD_MULT", "2.0").strip()
                hedge_risk_split = os.getenv("HEDGE_RISK_SPLIT", "0.6").strip()
                require_htf_on_edges = os.getenv("REQUIRE_HTF_ON_EDGES", "true").strip().lower()

                base_prompt = await load_mcp_prompt(
                    session,
                    "execution_agent",
                    arguments={
                        "context": context_str,
                        "default_volume": str(default_volume),
                        "min_confidence": str(min_confidence),
                        "honor_hold": "True" if honor_hold else "False",
                        "dry_run": "True" if dry_run else "False",
                        "HEDGE_MODE": "true" if hedge_mode in {"1","true","yes","on"} else "false",
                        "HEDGE_MAX_PAIRS": hedge_max_pairs,
                        "HEDGE_MAX_NET_USD_MULT": hedge_max_net_usd_mult,
                        "HEDGE_RISK_SPLIT": hedge_risk_split,
                        "REQUIRE_HTF_ON_EDGES": "true" if require_htf_on_edges in {"1","true","yes","on"} else "false",
                    },
                )

                # 3) Agent ReAct + timeout + simple retry sur erreurs transitoires
                agent = create_react_agent(model, tools)
                last_err = None
                for attempt in range(2):
                    task = asyncio.create_task(
                        agent.ainvoke(
                            {"messages": base_prompt},
                            config={"recursion_limit": AGENT_RECURSION_LIMIT},
                        )
                    )
                    try:
                        resp = await asyncio.wait_for(task, timeout=90)
                        break
                    except TimeoutError:
                        logger.error("⏱️ Timeout execution agent (90s)")
                        last_err = "timeout"
                    except GraphRecursionError as e:
                        logger.error(f"GraphRecursionError (execution): {e}")
                        with suppress(asyncio.CancelledError):
                            task.cancel(); await task
                        return _safe_plan("execution graph recursion limit exceeded", {"error": str(e)})
                    except Exception as e:
                        last_err = _format_exception_chain(e)
                        logger.error("Agent exec invoke failed: " + last_err)
                    finally:
                        with suppress(asyncio.CancelledError):
                            task.cancel(); await task
                    # petit backoff avant 2e tentative
                    await asyncio.sleep(1)
                else:
                    # Après 2 tentatives, abandon sécurisé
                    reason = "execution agent timeout" if last_err == "timeout" else "execution agent error"
                    return _safe_plan(reason, {"error": last_err} if last_err else None)

                # 4) Extraction + parsage JSON (identique à run_analysis_tec_mcp)
                raw = extract_last_message(resp)
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
                result_out = {"plan": plan if isinstance(plan, dict) else {"raw": plan, "decision": {"send_order": False, "reason": "non-dict plan"}}, "dry_run": dry_run}

                # Auto-execution fallback si nécessaire
                try:
                    if AUTO_EXECUTE_PLAN and isinstance(result_out.get("plan"), dict):
                        p = result_out["plan"]
                        dec = (p.get("decision") or {})
                        if dec.get("send_order") is True and (p.get("execution_result") in (None, {})):
                            sym = (p.get("symbol") or (context or {}).get("symbol"))
                            td = ((context or {}).get("technical_decision") or {})
                            td_dec = (td.get("decision") or {})
                            action = str(td_dec.get("action") or dec.get("action") or "").upper()
                            entry = float(td_dec.get("entry") or 0.0)
                            sl = td_dec.get("sl")
                            tp = td_dec.get("tp")
                            if sym and action in ("BUY","SELL") and sl is not None and tp is not None:
                                mod = _load_exec_core()
                                if mod and hasattr(mod, "build_and_execute_trade"):
                                    logger.warning("[AUTO_EXECUTE] calling build_and_execute_trade")
                                    resp2 = mod.build_and_execute_trade(
                                        symbol=sym,
                                        action=action,
                                        entry=entry,
                                        sl=float(sl),
                                        tp=float(tp),
                                        volume=float(default_volume),
                                        comment="AUTO_EXECUTE fallback",
                                        client_id="",
                                        dry_run=dry_run,
                                    )
                                    try:
                                        p["execution_result"] = json.loads(resp2) if isinstance(resp2, str) else resp2
                                    except Exception:
                                        p["execution_result"] = resp2
                except Exception as e:
                    logger.error(f"[AUTO_EXECUTE] failed: {e}")

                return result_out

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
       #  news_analysis = await run_news_mcp(symbol)
       #  if not news_analysis.get("ok", True):
       #      return _halt(news_analysis.get("reason", news_analysis.get("error", "news error")), news_analysis, None, "news")

        # Étape 2: ANALYSE TECHNIQUE (chemin direct ou via LLM MCP)
        decision = None
        if DIRECT_ANALYSIS_FIRST:
            decision = _direct_intraday_analysis(
                symbol,
                interval=params["interval"],
                horizon=params["horizon"],
                risk_level=params["risk_level"],
            )
        if not decision:
            decision = await run_analysis_tec_mcp(
                symbol,
                period=params["period"],
                interval=params["interval"],
                risk_level=params["risk_level"],
                horizon=params["horizon"],
            )
        if not decision.get("ok", True):
            return _halt(decision.get("reason", decision.get("error", "analysis error")), {}, decision, "analysis")

        # *** Gating HOLD : on n'appelle pas l'exécution si HOLD + honor_hold=True ***
        action = str(((decision or {}).get("decision", {}) or {}).get("action", "")).upper()
        if params.get("honor_hold", True) and action == "HOLD":
            return _skip_hold({}, decision, "Analysis returned HOLD and honor_hold=True; skipping execution.")

        # Étape 3: EXÉCUTION
        context = {"symbol": symbol, "news_sentiment": {}, "technical_decision": decision}
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
