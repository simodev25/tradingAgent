# news_mcp_server.py
import json
import requests
from bs4 import BeautifulSoup
from typing import Annotated, Any, Dict, List
from loguru import logger
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

mcp = FastMCP("News Analysis MCP Server", log_level="DEBUG")

# -------------------------------
# Sentiment pipeline avec FinBERT
# -------------------------------
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def fetch_article_text(url: str) -> str:
    """Télécharge et extrait le texte principal d’un article depuis une URL."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")

        # Exemples de balises courantes où se trouve le texte
        paragraphs = soup.find_all(["p", "article", "div"])
        text = " ".join([p.get_text(" ", strip=True) for p in paragraphs])
        return text.strip()
    except Exception as e:
        logger.error(f"Erreur fetch {url}: {e}")
        return ""

@mcp.tool()
def analyze_news(
    news_items: Annotated[List[Dict[str, Any]], Field(
        description="Liste de news au format {title, publisher, link, timestamp}"
    )]
) -> str:
    """
    Analyse sentiment des news financières avec FinBERT,
    en récupérant le texte complet via 'link'.
    """
    results = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    for item in news_items:
        try:
            title = item.get("title", "").strip()
            link = item.get("link", "")
            publisher = item.get("publisher", "")
            ts = item.get("timestamp")

            # récupère le texte complet
            content = fetch_article_text(link)
            if not content:
                content = title  # fallback si pas de texte trouvé

            # Tronquer pour rester dans les limites du modèle
            snippet = content[:1000]

            sentiment = sentiment_analyzer(snippet,truncation=True, max_length=512)[0]
            label = sentiment["label"].lower()
            score = float(sentiment["score"])

            # impact simple
            impact = "low"
            if score > 0.7 and label in ("positive", "negative"):
                impact = "high"
            elif score > 0.5:
                impact = "medium"

            counts[label] = counts.get(label, 0) + 1

            results.append({
                "title": title,
                "publisher": publisher,
                "timestamp": ts,
                "link": link,
                "text_excerpt": snippet[:300] + "...",
                "sentiment": label,
                "score": round(score, 3),
                "impact": impact
            })
        except Exception as e:
            logger.error(f"Erreur sur analyse news: {e}")
            continue

    total = sum(counts.values()) or 1
    summary = {
        "positive": round(counts["positive"] / total, 2),
        "negative": round(counts["negative"] / total, 2),
        "neutral": round(counts["neutral"] / total, 2),
    }

    if summary["positive"] > 0.5:
        bias = "bullish"
    elif summary["negative"] > 0.5:
        bias = "bearish"
    else:
        bias = "neutral"

    return json.dumps({
        "ok": True,
        "summary": summary,
        "global_bias": bias,
        "data": results
    }, ensure_ascii=False, indent=2)
    
@mcp.prompt()
def news_agent(
    symbol: Annotated[str, Field(description="Ticker symbol associé aux news")],
    news_items: Annotated[str, Field(description="Liste de news en JSON (ex: '[{title, publisher, link, timestamp}]')")]
) -> str:
    # Ici, news_items est une string JSON
    return f"""
Tu es l’agent News Analysis pour {symbol}.
Voici les `news_items` (JSON):
news_items = {news_items}
1. Utilise `analyze_news(news_items)` pour analyser chaque article.
2. analyze les champs: sentiment, score et impact.
3. Basé sur les résultats, calcule un biais global : bullish / bearish / neutral.
4. Retourne JSON (global_bias).
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
