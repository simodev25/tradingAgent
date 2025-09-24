# execution_mcp.py
import json
from typing import Any, Optional, Tuple, List
from datetime import datetime
import sys
import os
import math
from loguru import logger
from string import Template
from mcp.server.fastmcp import FastMCP

# --- Import meta_api (relative to this file) ---
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_fetch/"))
)
try:
    import meta_api as meta_api  # type: ignore
except Exception as e:  # pragma: no cover
    meta_api = None  # type: ignore
    logger.error(f"[INIT] meta_api import failed: {e}")

mcp = FastMCP("Trading Execution MCP Server", log_level="WARNING")

logger.remove()
logger.add(sys.stderr, level="DEBUG")


# ---------- Helpers ----------
def _ok(payload: Any) -> str:
    return json.dumps({"ok": True, "data": payload}, ensure_ascii=False)


def _err(msg: str, **extra: Any) -> str:
    return json.dumps({"ok": False, "error": msg, **extra}, ensure_ascii=False)


def _get(d: dict, path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_json_like(x: Any) -> Optional[dict]:
    """Best effort: return dict if x is JSON string or dict; else None."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def _safe_isfinite(*nums: float) -> bool:
    try:
        return all(isinstance(n, (int, float)) and math.isfinite(float(n)) for n in nums)
    except Exception:
        return False


def _fetch_symbol_spec(symbol: str) -> Optional[dict]:
    if meta_api is None:
        return None
    try:
        raw = meta_api.get_symbol_spec(symbol)  # type: ignore
        return _parse_json_like(raw)
    except Exception:
        return None


# --------- PRICE helpers ---------
def _unwrap_data(obj: Any) -> Any:
    """Déballe des enveloppes type {'ok': True, 'data': X} ou {'result': X}."""
    cur = obj
    while isinstance(cur, dict):
        for k in ("data", "result", "Data"):
            if k in cur:
                cur = cur[k]
                break
        else:
            break
    return cur


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _symbol_aliases(symbol: str) -> List[str]:
    """Retourne une petite liste d’alias (avec/sans suffixe .pro, case-insensitive)."""
    s = symbol
    cand = {s, s.upper()}
    for suf in (".pro", ".PRO"):
        if s.endswith(suf):
            base = s[: -len(suf)]
            cand |= {base, base.upper()}
    return list(cand)


def _fetch_current_quote(symbol: str) -> Optional[dict]:
    """
    Renvoie un dict 'quote' normalisé:
    {
      'symbol': str, 'bid': float|None, 'ask': float|None, 'price': float|None,
      'time': str|None, 'brokerTime': str|None, 'raw': <payload brut>
    }
    Compatible avec formats: {'symbol':'XXX','bid':...,'ask':...} ou imbriqués {'quote':{...}}.
    """
    if meta_api is None:
        return None
    for sym in _symbol_aliases(symbol):
        try:
            raw = meta_api.get_current_price(sym)  # type: ignore
            data_or_raw = _parse_json_like(raw)
            data = _unwrap_data(data_or_raw if data_or_raw is not None else raw)

            if isinstance(data, dict):
                bid = _first_not_none(data.get("bid"), data.get("Bid"))
                ask = _first_not_none(data.get("ask"), data.get("Ask"))
                px = _first_not_none(data.get("price"), data.get("Price"), data.get("last"), data.get("Last"))

                # formats imbriqués: {"quote":{...}} ou {"tick":{...}}
                if (bid is None and ask is None and px is None) and any(k in data for k in ("quote", "tick")):
                    inner = _unwrap_data(data.get("quote") or data.get("tick"))
                    if isinstance(inner, dict):
                        bid = _first_not_none(inner.get("bid"), inner.get("Bid"))
                        ask = _first_not_none(inner.get("ask"), inner.get("Ask"))
                        px = _first_not_none(inner.get("price"), inner.get("Price"), inner.get("last"), inner.get("Last"))

                return {
                    "symbol": str(_first_not_none(data.get("symbol"), sym)),
                    "bid": float(bid) if bid is not None else None,
                    "ask": float(ask) if ask is not None else None,
                    "price": float(px) if px is not None else None,
                    "time": data.get("time"),
                    "brokerTime": data.get("brokerTime"),
                    "raw": data,
                }

            # scalaire: juste un prix
            if isinstance(data, (int, float)):
                return {
                    "symbol": sym,
                    "bid": None,
                    "ask": None,
                    "price": float(data),
                    "time": None,
                    "brokerTime": None,
                    "raw": data,
                }
        except Exception:
            continue
    return None


def _fetch_current_price(symbol: str, action: Optional[str] = None) -> Optional[float]:
    """
    Préfère ASK pour BUY, BID pour SELL ; sinon MID=(bid+ask)/2 ; sinon 'price/last', à défaut bid/ask isolés.
    """
    q = _fetch_current_quote(symbol)
    if not q:
        return None

    bid, ask, px = q["bid"], q["ask"], q["price"]

    if action:
        a = action.upper()
        if a.startswith("BUY") and ask is not None:
            return float(ask)
        if a.startswith("SELL") and bid is not None:
            return float(bid)

    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if px is not None:
        return float(px)
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return None


def _extract_tick_and_digits(spec: Optional[dict]) -> Tuple[Optional[float], Optional[int]]:
    if not spec:
        return None, None
    tick = (
        spec.get("tickSize")
        or spec.get("tick_size")
        or spec.get("step")
        or spec.get("point")
        or spec.get("minTick")
        or spec.get("points")
    )
    digits = spec.get("digits") or spec.get("precision") or spec.get("pricePrecision")
    try:
        return (float(tick) if tick is not None else None, int(digits) if digits is not None else None)
    except Exception:
        return None, None


def _round_price(v: float, tick: Optional[float], digits: Optional[int]) -> float:
    if tick and tick > 0:
        v = round(v / tick) * tick
    if digits is not None and digits >= 0:
        v = round(v, digits)
    return float(v)


# ---------- MCP INFO Tools ----------
@mcp.tool()
def get_account_info() -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_account_info()  # type: ignore
    except Exception as e:
        return _err("account_info_failed", exc=str(e))


@mcp.tool()
def get_positions() -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_positions()  # type: ignore
    except Exception as e:
        return _err("positions_failed", exc=str(e))


@mcp.tool()
def get_orders() -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_orders()  # type: ignore
    except Exception as e:
        return _err("orders_failed", exc=str(e))


@mcp.tool()
def get_symbol_spec(symbol: str) -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_symbol_spec(symbol)  # type: ignore
    except Exception as e:
        return _err("symbol_spec_failed", symbol=symbol, exc=str(e))


@mcp.tool()
def get_current_price(symbol: str) -> str:
    if meta_api is None:
        return _err("meta_api_unavailable")
    try:
        return meta_api.get_current_price(symbol)  # type: ignore
    except Exception as e:
        return _err("current_price_failed", symbol=symbol, exc=str(e))


_ALLOWED_ACTIONS = {"BUY", "SELL", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"}
_MARKET_ACTIONS = {"BUY", "SELL"}
_PENDING_ACTIONS = _ALLOWED_ACTIONS - _MARKET_ACTIONS


def _validate_directional_levels(
    action: str,
    entry: Optional[float],
    sl: float,
    tp: float,
    symbol: str,
    tick: Optional[float] = None,
    stops_level_points: int = 0,
    epsilon_ticks: int = 2,
) -> Tuple[bool, Optional[str]]:
    """
    Valide la cohérence SL/TP par rapport au sens de l'ordre, avec :
    - prix de référence = entry si fourni, sinon quote courante cohérente (ASK pour BUY, BID pour SELL)
    - tolérance de epsilon_ticks * tick (pour arrondis/float)
    - respect de la distance mini broker (stopsLevel, exprimée en 'points' MT5)
    Si pas de prix disponible -> validation laissée au serveur (OK).
    """
    if not _safe_isfinite(sl, tp) or (entry is not None and not _safe_isfinite(entry)):
        return False, "invalid_number"

    # Prix de référence
    price_ref = entry if entry is not None else _fetch_current_price(symbol, action=action)
    if price_ref is None:
        # pas de quote fiable -> on ne bloque pas côté client
        return True, None

    # Distances minimales
    try:
        t = float(tick) if (tick is not None) else None
    except Exception:
        t = None
    eps = (epsilon_ticks * t) if (t and t > 0) else 0.0
    min_dist = max(float(stops_level_points or 0) * (t or 0.0), eps)

    a = action.upper()
    if a in ("BUY", "BUY_LIMIT", "BUY_STOP"):
        if tp <= price_ref + min_dist:
            return False, "tp_must_be_above_price_for_buy"
        if sl >= price_ref - min_dist:
            return False, "sl_must_be_below_price_for_buy"
    elif a in ("SELL", "SELL_LIMIT", "SELL_STOP"):
        if tp >= price_ref - min_dist:
            return False, "tp_must_be_below_price_for_sell"
        if sl <= price_ref + min_dist:
            return False, "sl_must_be_above_price_for_sell"
    else:
        return False, "invalid_action"

    # Cohérence LIMIT/STOP vs marché (si entry fourni et quote dispo)
    if a in _PENDING_ACTIONS and entry is not None:
        cur = _fetch_current_price(symbol, action=action)
        if cur is not None:
            if a == "BUY_LIMIT" and not (entry < cur - eps):
                return False, "buy_limit_entry_must_be_below_market"
            if a == "SELL_LIMIT" and not (entry > cur + eps):
                return False, "sell_limit_entry_must_be_above_market"
            if a == "BUY_STOP" and not (entry > cur + eps):
                return False, "buy_stop_entry_must_be_above_market"
            if a == "SELL_STOP" and not (entry < cur - eps):
                return False, "sell_stop_entry_must_be_below_market"

    return True, None


@mcp.tool()
def execute_trade(
    symbol: str,
    action: str,
    entry: float,
    sl: float,
    tp: float,
    volume: float = 0.01,
    comment: Optional[str] = None,
    client_id: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    """
    Exécute une transaction via l’API MetaApi.
    - action: "BUY"/"SELL" (market) ou "BUY_LIMIT"/"SELL_LIMIT"/"BUY_STOP"/"SELL_STOP"
    - entry: utilisé pour LIMIT/STOP (ignoré pour BUY/SELL)
    - dry_run: si True, ne PAS envoyer l’ordre (retourne payload validé ou résultat de validation SDK si dispo)
    """
    try:
        if meta_api is None:
            return _err("meta_api_unavailable")

        normalized_action = action.upper()
        if normalized_action not in _ALLOWED_ACTIONS:
            return _err("invalid_action", got=action, allowed=sorted(_ALLOWED_ACTIONS))

        if not _safe_isfinite(sl, tp, volume):
            return _err("invalid_number", reason="non_finite_sl_tp_or_volume")
        if volume <= 0:
            return _err("invalid_volume", volume=volume)

        spec = _fetch_symbol_spec(symbol)
        tick, digits = _extract_tick_and_digits(spec)

        # stopsLevel (points MT5) pour distance minimale broker
        stops_level_points = 0
        try:
            stops_level_points = int(
                (spec or {}).get("stopsLevel")
                or (spec or {}).get("stops_level")
                or 0
            )
        except Exception:
            stops_level_points = 0

        # ===== Quotes & métriques RR / spread =====
        quote = _fetch_current_quote(symbol)
        bid = (quote or {}).get("bid")
        ask = (quote or {}).get("ask")
        price_mid = (None if (bid is None or ask is None) else (bid + ask) / 2.0)
        spread = (None if (bid is None or ask is None) else (ask - bid))

        # Entry de référence pour le calcul (market)
        # - pour market BUY → entry_ref = ask
        # - pour market SELL → entry_ref = bid
        # - sinon (pending) → entry est la référence
        action_upper = normalized_action
        if action_upper in _MARKET_ACTIONS:
            entry_ref = ask if action_upper == "BUY" else bid
        else:
            entry_ref = float(entry) if _safe_isfinite(entry) else None

        # ===== Auto-bump du TP pour atteindre RR ≥ 1.2 (si bump <= 5×spread) =====
        RR_MIN = 1.2
        MAX_TP_BUMP_SPREADS = 5  # ne dépasse pas 5× le spread
        sl0, tp0 = float(sl), float(tp)  # pour logs

        if entry_ref is not None and spread is not None and spread > 0 and _safe_isfinite(sl0, tp0):
            if action_upper.startswith("BUY"):
                risk = max(entry_ref - sl0, 0.0)
                reward = max(tp0 - entry_ref, 0.0)
            else:
                risk = max(sl0 - entry_ref, 0.0)
                reward = max(entry_ref - tp0, 0.0)

            rr_now = (reward / risk) if (risk and risk > 0) else None
            if rr_now is not None and 1.0 <= rr_now < RR_MIN:
                needed_reward = RR_MIN * risk
                bump = needed_reward - reward  # combien il manque
                max_bump = MAX_TP_BUMP_SPREADS * spread
                if bump > 0 and bump <= max_bump:
                    # on déplace le TP dans le sens de l’action
                    if action_upper.startswith("BUY"):
                        tp0 = entry_ref + needed_reward
                    else:
                        tp0 = entry_ref - needed_reward
                    # arrondir après ajustement
                    tp0 = _round_price(tp0, tick, digits)

        # ===== Arrondi final des niveaux =====
        sl_r = _round_price(sl0, tick, digits)
        tp_r = _round_price(tp0, tick, digits)

        entry_for_payload: Optional[float] = None
        entry_for_call: Optional[float] = None
        if action_upper in _PENDING_ACTIONS:
            if not _safe_isfinite(entry):
                return _err("invalid_number", reason="non_finite_entry_for_pending")
            entry_for_payload = _round_price(float(entry), tick, digits)
            entry_for_call = entry_for_payload

        # ===== Validation directionnelle + stopsLevel + tolérance =====
        ok, why = _validate_directional_levels(
            action_upper, entry_for_call, sl_r, tp_r, symbol,
            tick=tick, stops_level_points=stops_level_points, epsilon_ticks=2
        )
        if not ok:
            return _err("values_incoherent", reason=why)

        # ===== Calcul RR et ratio TP vs spread (pour logs/retour) =====
        tp_vs_spread_ratio = None
        rr_out = None
        if entry_ref is not None and spread is not None and spread > 0 and _safe_isfinite(sl_r, tp_r):
            if action_upper.startswith("BUY"):
                risk = max(entry_ref - sl_r, 0.0)
                reward = max(tp_r - entry_ref, 0.0)
            else:
                risk = max(sl_r - entry_ref, 0.0)
                reward = max(entry_ref - tp_r, 0.0)
            rr_out = (reward / risk) if (risk and risk > 0) else None
            tp_vs_spread_ratio = (reward / spread) if reward is not None else None

        trade = {
            "actionType": action_upper,
            "symbol": symbol,
            "volume": float(volume),
            "openPrice": entry_for_payload,
            "stopLoss": sl_r,
            "takeProfit": tp_r,
            "comment": comment,
            "client_id": client_id,
        }

        # Log quote complète pour debug
        try:
            if quote:
                logger.debug(f"[PRICE] {quote['symbol']} bid={quote['bid']} ask={quote['ask']} time={quote.get('time')} brokerTime={quote.get('brokerTime')}")
            else:
                logger.debug(f"[PRICE] no quote for {symbol}")
        except Exception:
            pass

        logger.debug(f"[MCP:EXECUTE] dry_run={dry_run} trade={trade} rr={rr_out} tp_vs_spread={tp_vs_spread_ratio}")

        if dry_run:
            # Si le SDK propose une validation locale, l'utiliser
            if hasattr(meta_api, "trade_execute"):
                try:
                    return meta_api.trade_execute(trade, client_id=client_id, dry_run=True)  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning(f"[MCP:EXECUTE] meta_api.trade_execute failed in dry_run: {e}")
            return _ok({"trade": trade, "rr": rr_out, "tp_vs_spread_ratio": tp_vs_spread_ratio})

        # Exécution live
        try:
            result = meta_api.execute_trade(  # type: ignore[call-arg]
                symbol,
                action_upper,
                entry_for_call,
                sl_r,
                tp_r,
                volume=float(volume),
                comment=comment,
                client_id=client_id,
            )
        except TypeError:
            # Compat: certains SDK exigent 'entry' même pour market (on passe un None/valeur sûre)
            result = meta_api.execute_trade(  # type: ignore[call-arg]
                symbol,
                action_upper,
                entry if entry_for_call is None else entry_for_call,
                sl_r,
                tp_r,
                volume=float(volume),
                comment=comment,
                client_id=client_id,
            )
        except Exception as e:
            logger.exception("[MCP:EXECUTE] trade_failed")
            return _err("trade_failed", exc=str(e), symbol=symbol, action=action)

        return result

    except Exception as e:  # pragma: no cover
        logger.exception("[MCP:EXECUTE] trade_failed_unexpected")
        return _err("trade_failed", exc=str(e), symbol=symbol, action=action)


# ---------- Contexte minifié (optionnel) ----------
def thin_exec_context(full_ctx: dict) -> dict:
    td = (full_ctx or {}).get("technical_decision", {}) or {}
    ex = (full_ctx or {}).get("execution", {}) or {}
    d = td.get("decision", {}) or {}
    plan = ex.get("plan", {}) or {}
    snaps = plan.get("snapshots", {}) or {}
    return {
        "symbol": full_ctx.get("symbol"),
        "decision": {
            "action": d.get("action"),
            "entry": d.get("entry"),
            "sl": d.get("sl"),
            "tp": d.get("tp"),
            "confidence": d.get("confidence"),
            "risk_level": d.get("risk_level"),
            "regime": td.get("regime"),
        },
        "volatility": td.get("volatility"),
        "portfolio": plan.get("portfolio"),
        "account": snaps.get("account"),
        "symbol_spec": snaps.get("symbol_spec"),
        "price": snaps.get("price"),
        "positions": snaps.get("positions"),
        "orders": snaps.get("orders"),
        "news_bias": (full_ctx.get("news_sentiment") or {}).get("global_bias"),
    }


# =========================
#  PROMPT HEDGE-AWARE LLM
# =========================
@mcp.prompt()
def execution_agent(
    context: Any,
    default_volume: str = "0.01",
    min_confidence: str = "55",     # <- assoupli à 55
    honor_hold: str = "True",
    dry_run: str = "True",
    # --- flags hedge ---
    HEDGE_MODE: str = "true",
    HEDGE_MAX_PAIRS: str = "2",
    HEDGE_MAX_NET_USD_MULT: str = "2.0",
    HEDGE_RISK_SPLIT: str = "0.6",
) -> str:
    """
    Version hedge-aware : garde-fous (HOLD/confiance, TP vs spread, RR min, specs, marge)
    + autorise un hedge contrôlé (paires admissibles, caps USD, split volume).
    Retourne UNIQUEMENT un objet JSON (voir gabarit plus bas), sans texte.
    """
    # stringify context (minifié ou complet)
    ctx_str = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)

    tmpl = Template(r"""
Tu es un trader pro. Tu reçois un CONTEXTE JSON MINIMAL pour UN SEUL symbole.
Décide d’ENVOYER, AJUSTER ou ANNULER l’ordre.

PARAMS:
- default_volume = $DEFAULT_VOLUME
- min_confidence = $MIN_CONFIDENCE
- honor_hold = $HONOR_HOLD
- dry_run = $DRY_RUN

HEDGE FLAGS:
- HEDGE_MODE = $HEDGE_MODE
- HEDGE_MAX_PAIRS = $HEDGE_MAX_PAIRS
- HEDGE_MAX_NET_USD_MULT = $HEDGE_MAX_NET_USD_MULT
- HEDGE_RISK_SPLIT = $HEDGE_RISK_SPLIT
- REQUIRE_HTF_ON_EDGES = true

HEDGE_PAIRS (admissibles, sym sans suffixe .pro):
{
  "EURUSD": ["USDCHF","GBPUSD"],
  "GBPUSD": ["USDCHF","EURUSD"],
  "AUDUSD": ["USDCHF","NZDUSD"],
  "NZDUSD": ["USDCHF","AUDUSD"],
  "USDJPY": ["CHFJPY","EURJPY"],
  "USDCHF": ["EURUSD","GBPUSD","AUDUSD","NZDUSD"],
  "EURJPY": ["USDJPY"],
  "CHFJPY": ["USDJPY"]
}

CONTEXTE FOURNI (minifié):
$CTX

OUTILS DISPONIBLES (appelle-les si nécessaire):
- get_account_info(), get_positions(), get_orders(), get_symbol_spec(symbol), get_current_price(symbol)
- execute_trade(symbol, action, entry, sl, tp, volume, comment, client_id, dry_run=$DRY_RUN)

RÈGLES OBLIGATOIRES

1) Respect HOLD/Confiance/No-trade
- Si action == "HOLD" OU confidence < $MIN_CONFIDENCE OU regime == "no-trade" → ANNULER.

2) Edge vs SPREAD
- Obtiens bid/ask via get_current_price(symbol). Si indisponible → ANNULER (prudence).
- spread = ask - bid. entry_ref = ask pour BUY*, bid pour SELL* (mid si besoin).
- tp_dist = (tp - entry_ref) si BUY*, sinon (entry_ref - tp). Si <= 0 → ANNULER.
- Si volatility.band == "HIGH" → exiger `tp_dist ≥ 6 × spread`, sinon `tp_dist ≥ 4 × spread`.

3) RR minimal
- RR (si calculable) ≥ 1.2 sinon ANNULER.
- Tu peux **pousser le TP** si déplacer ≤ 5× spread permet d’atteindre RR=1.2 (ré-autorise ensuite).

4) Spécifications symbole
- Ajuste niveaux au tick/digits. Si incohérents (SL/TP côté opposé, stopsLevel), ANNULER.

5) Marges
- Vérifie account.freeMargin; si indisponible/insuffisante → ANNULER.

6) Gestion exposition (NON-HEDGE)
- Si HEDGE_MODE == "false" :
  - ANNULER si positions ouvertes ≥ 4.
  - ANNULER si positions sur le même symbole ≥ 1.
  - ANNULER si nombre de positions contenant "USD" ≥ 3.

7) Mode HEDGE (HEDGE_MODE == "true")
- Autoriser une position “opposée” seulement si:
  a) Le symbole courant a au moins UNE position ouverte dont le symbole est dans HEDGE_PAIRS[symbol] (ignore .pro).
  b) Caps respectés :
     - Total USD legs < 6.
     - |USD net notional| ≤ (HEDGE_MAX_NET_USD_MULT × equity) [approxime le notionnel USD par volume*contractSize et signe USD].
     - Nombre de paires hedgées simultanées ≤ HEDGE_MAX_PAIRS.
  c) Si volatility.band == "HIGH" → exiger confluence HTF (en cas de doute ANNULER).
- Si hedge OK:
  - Réduire le volume de la jambe hedge à max(0.01, (1 - HEDGE_RISK_SPLIT) × default_volume).
  - Ajouter un champ "hedge": {"enabled": true, "paired_with": "<symbole existant>"}.
- Sinon, appliquer les règles NON-HEDGE ci-dessus.

8) Sécurité
- Si un outil échoue → ANNULER (raison explicite).
- Éviter d’ouvrir dans les 2 minutes suivant un changement de bougie H1/M30 si l’heure est disponible.

SORTIE JSON STRICTE (AUCUN TEXTE HORS JSON)
{
  "symbol": "<string>",
  "source": {"news_bias": "<neutral|positive|negative|null>", "confidence": <number>, "risk_level": "<low|medium|high|null>"},
  "portfolio": {"floating_pnl_total": <number|null>, "symbol_floating_pnl": <number|null>, "symbol_net_exposure": <number|null>, "symbol_direction": "<long|short|flat|null>"},
  "account_checks": {"equity": <number|null>, "free_margin": <number|null>, "margin_required_est": <number|null>, "margin_ok": <true|false>, "margin_reason": "<string>"},
  "tools_used": {"get_account_info": <true|false>, "get_positions": <true|false>, "get_orders": <true|false>, "get_symbol_spec": <true|false>, "get_current_price": <true|false>},
  "checks": {"values_ok": <true|false>, "direction_ok": <true|false>, "spec_ok": <true|false>, "rr": <number|null>, "tp_vs_spread_ratio": <number|null>, "hedge_ok": <true|false|null>},
  "order_policy": {"conflict_with_existing": "<none|duplicate|opposite|overexposed>", "action": "<proceed|adjust|cancel>", "adjust_reason": "<string|null>", "cancel_reason": "<string|null>"},
  "normalized_order": null,
  "decision": {"send_order": <true|false>, "reason": "<detail la en francais règle> ", "hedge": {"enabled": <true|false>, "paired_with": "<string|null>", "volume": <number|null>}},
  "execution_result": "<execute_trade result>"
}

IMPORTANT: Réponds UNIQUEMENT avec l’objet JSON. Pas de prose. Pas de code fences.
Taille maximale de sortie : 700 tokens.
""")

    prompt = tmpl.substitute(
        CTX=ctx_str,
        DEFAULT_VOLUME=default_volume,
        MIN_CONFIDENCE=min_confidence,
        HONOR_HOLD=honor_hold,
        DRY_RUN=dry_run,
        HEDGE_MODE=HEDGE_MODE,
        HEDGE_MAX_PAIRS=HEDGE_MAX_PAIRS,
        HEDGE_MAX_NET_USD_MULT=HEDGE_MAX_NET_USD_MULT,
        HEDGE_RISK_SPLIT=HEDGE_RISK_SPLIT,
    )
    return prompt


if __name__ == "__main__":
    mcp.run(transport="stdio")
