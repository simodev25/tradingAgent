import json
import re
import sys
from typing import Annotated, Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from requests.adapters import HTTPAdapter
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from urllib3.util.retry import Retry
import os
sys.path.append(os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../data_fetch/')))
import yf_api as yf_api

mcp = FastMCP("News Analysis MCP Server", log_level="DEBUG")

# -------------------------------
# Logging: STDERR only (safe for stdio transport)
# -------------------------------
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# -------------------------------
# FinBERT sentiment pipeline
# -------------------------------
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# -------------------------------
# HTTP session with retries
# -------------------------------
_session = requests.Session()
_retry = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("http://", HTTPAdapter(max_retries=_retry))
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_DEFAULT_HEADERS = {"User-Agent": "NewsMCP/1.0 (+https://example.local)"}

# -------------------------------
# Helpers
# -------------------------------
def _ok(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)

def _err(msg: str, **extra) -> str:
    logger.error(f"[NewsMCP] {msg} | extra={extra}")
    return json.dumps({"ok": False, "error": msg, "extra": extra}, ensure_ascii=False)

def _is_public_http_url(url: str) -> bool:
    """Basic SSRF guard: allow only http/https + block obvious private/localhost hosts.
    (No DNS resolution here for simplicity.)"""
    try:
        u = urlparse(url.strip())
        if u.scheme not in ("http", "https"):
            return False
        host = u.hostname
        if not host:
            return False
        host_l = host.lower()
        if host_l in {"localhost", "127.0.0.1", "::1"}:
            return False
        # Block typical RFC1918/link-local ranges by prefix match
        if re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[0-1])\.|169\.254\.)", host_l):
            return False
        return True
    except Exception:
        return False

def fetch_article_text(url: str) -> str:
    """Télécharge et extrait le texte principal d’une page article."""
    if not url or not _is_public_http_url(url):
        return ""
    try:
        resp = _session.get(url, headers=_DEFAULT_HEADERS, timeout=10)
        if resp.status_code != 200 or not resp.text:
            return ""
        # Prefer lxml if installed, fallback to html.parser
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        container = soup.find("article") or soup.find("main") or soup
        paras = [p.get_text(" ", strip=True) for p in container.find_all("p")]
        text = " ".join(paras) if paras else soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        logger.error(f"Erreur fetch {url}: {e}")
        return ""

def _as_list(news_items: Any) -> List[Dict[str, Any]]:
    if isinstance(news_items, str):
        try:
            parsed = json.loads(news_items)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
    return news_items if isinstance(news_items, list) else []

def _dedupe_by_link_or_title(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for it in items:
        link = (it.get("link") or "").strip()
        key = link or (it.get("title") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out

def _batch_sentiments(texts: List[str], batch_size: int = 8):
    out: List[Dict[str, Any]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        res = sentiment_analyzer(chunk, truncation=True, max_length=512)
        out.extend(res)
    return out

def _heuristic_is_french(text: str) -> bool:
    # Naive heuristic: count of accented French characters
    accented = "àâäçéèêëîïôöùûüÿÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ"
    count = sum(ch in accented for ch in text)
    return count >= 10

# -------------------------------
# Tool
# -------------------------------
@mcp.tool()
async def analyze_news(symbol:str) -> str:
    """
    Analyse le sentiment des news financières avec FinBERT.
    - Accepte une liste d'objets news ou une chaîne JSON sérialisée.
    - Récupère le texte via 'link' (si public), sinon fallback sur le 'title'.
    - Déduplique par lien (ou titre si pas de lien).
    - Batch l'inférence pour de meilleures perfs.
    """
    try:
        news_items =  yf_api.get_ticker_news(symbol)
        items = _as_list(news_items)
        items = _dedupe_by_link_or_title(items)
        if not items:
            return _ok({"ok": True, "summary": {"positive": 0, "negative": 0, "neutral": 1}, "global_bias": "neutral", "data": []})

        prepared = []
        for it in items:
            title = (it.get("title") or "").strip()
            link = (it.get("link") or "").strip()
            publisher = (it.get("publisher") or "").strip()
            ts = it.get("timestamp")
            content = fetch_article_text(link) if link else ""
            text = content or title
            # If looks strongly French (FinBERT is EN), fallback to title
            if text and _heuristic_is_french(text):
                text = title or text
            snippet = (text or title or "")[:1000]
            prepared.append({"title": title, "publisher": publisher, "timestamp": ts, "link": link, "snippet": snippet})

        sentiments = _batch_sentiments([p["snippet"] for p in prepared])
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        results: List[Dict[str, Any]] = []
        weights: List[Dict[str, float]] = []

        for p, s in zip(prepared, sentiments):
            sd = s[0] if isinstance(s, list) else s  # normalize
            label = str(sd.get("label", "")).lower()
            score = float(sd.get("score", 0.0))

            impact = "low"
            w = score
            if score > 0.7 and label in ("positive", "negative"):
                impact = "high"; w *= 1.3
            elif score > 0.5:
                impact = "medium"; w *= 1.1

            counts[label] = counts.get(label, 0) + 1
            weights.append({"label": label, "w": w})

            results.append({
                "title": p["title"],
                "publisher": p["publisher"],
                "timestamp": p["timestamp"],
                "link": p["link"],
                "text_excerpt": (p["snippet"][:300] + "...") if p["snippet"] else "",
                "sentiment": label,
                "score": round(score, 3),
                "impact": impact,
            })

        total = sum(counts.values()) or 1
        summary = {
            "positive": round(counts.get("positive", 0) / total, 2),
            "negative": round(counts.get("negative", 0) / total, 2),
            "neutral": round(counts.get("neutral", 0) / total, 2),
        }

        # Weighted global bias
        signed = 0.0
        denom = 0.0
        for x in weights:
            if x["label"] == "positive":
                signed += x["w"]
                denom += abs(x["w"])
            elif x["label"] == "negative":
                signed -= x["w"]
                denom += abs(x["w"])
            else:
                denom += 0.5 * abs(x["w"])  # give small weight to neutral as uncertainty

        global_score = signed / max(1.0, denom)  # -1..+1
        if global_score > 0.15:
            bias = "bullish"
        elif global_score < -0.15:
            bias = "bearish"
        else:
            bias = "neutral"

        return _ok({"ok": True, "summary": summary, "global_bias": bias, "data": results})
    except Exception as e:
        logger.exception("analyze_news failed")
        return _err("analyze_news failed", exc=str(e))

# -------------------------------
# Prompt
# -------------------------------
@mcp.prompt()
def news_agent(
    symbol: Annotated[str, Field(description="Ticker symbol associé aux news")]
) -> str:
    return f"""
Tu es l’agent News Analysis pour {symbol}.
Les items bruts sont fournis dans la variable `news_items` (string JSON).
Appelle UNIQUEMENT le tool `analyze_news(news_items)`.

Règles de sortie :
- Réponds UNIQUEMENT avec un JSON valide (aucun texte, aucun backtick).
- Si le tool échoue (ok=false), renvoie {{"global_bias":"neutral","reason":"tool failed"}}.

Schéma de sortie strict :
{{
  "global_bias":"<bullish|bearish|neutral>",
  "reason":"<explication textuelle du choix>",
  "summary": {{
    "positive": <float>, 
    "negative": <float>, 
    "neutral": <float>
  }},
  "global_score": <float>, 
  "top_influential_titles": ["titre1", "titre2", ...]
}}
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
