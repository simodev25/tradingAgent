import json
from loguru import logger
from mcp.server.fastmcp import FastMCP
from xtb import XTB   # ta classe que tu as déjà écrite

mcp = FastMCP("XTB MCP Server")

xtb_client: XTB = XTB()



@mcp.tool()
def get_symbol_info(symbol: str) -> str:
    """Retrieve technical information about a trading symbol (spread, lot size, precision, etc.)."""
    if not xtb_client:
        return json.dumps({"ok": False, "error": "Not logged in"})
    data = xtb_client.get_Symbol(symbol)
    return json.dumps({"ok": True, "symbol": symbol, "info": data}, ensure_ascii=False)


@mcp.tool()
def get_candles(symbol: str, period: str = "M1", qty_candles: int = 50) -> str:
    """Retrieve OHLC candles for a symbol."""
    if not xtb_client:
        return json.dumps({"ok": False, "error": "Not logged in"})
    candles = xtb_client.get_Candles(period, symbol, qty_candles=qty_candles)
    return json.dumps({"ok": True, "symbol": symbol, "candles": candles}, ensure_ascii=False)


@mcp.tool()
def get_today_history(symbol: str) -> str:
    """Retrieve today's trade history for the account (related to a symbol)."""
    if not xtb_client:
        return json.dumps({"ok": False, "error": "Not logged in"})
    history = xtb_client.get_today_history()
    # filtrer par symbole
    filtered = [h for h in history if h.get("symbol") == symbol]
    return json.dumps({"ok": True, "symbol": symbol, "history": filtered}, ensure_ascii=False)


@mcp.prompt()
def fetch_symbol_context(symbol: str, period: str = "M15", qty_candles: int = 200) -> str:
    """
    Assemble un paquet JSON avec toutes les infos TECHNIQUES d’un symbole depuis XTB.
    Cet agent est **uniquement Data**, il ne doit donner AUCUNE recommandation de trading.
    """
    return f"""
Tu es un **agent Data Technique (XTB)**. 
Ta mission est uniquement de collecter et assembler les informations techniques pour {symbol}.
**NE FOURNIS AUCUNE RECOMMANDATION**.

Appelle UNIQUEMENT ces tools :
- get_symbol_info("{symbol}")
- get_candles("{symbol}", period="{period}", qty_candles={qty_candles})
- get_today_history("{symbol}")

Assemble ensuite un unique objet JSON au format :

{{
  "symbol": "{symbol}",
  "info": <objet JSON retourné par get_symbol_info.data>,
  "candles": <liste OHLC retournée par get_candles.data>,
  "today_history": <liste JSON retournée par get_today_history.data>
}}

Si un tool échoue (ok=false), inclue quand même la clé correspondante avec **null** 
et ajoute une clé "errors" dans l’objet racine avec le message.
"""
    

if __name__ == "__main__":
    mcp.run(transport="stdio")
