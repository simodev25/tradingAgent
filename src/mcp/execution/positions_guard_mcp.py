#!/usr/bin/env python3
"""
Positions Guard MCP
--------------------
Un serveur MCP (Model Context Protocol) dédié à la *surveillance* et à l'*ajustement* des positions ouvertes.

Fonctions exposées (tools):
- list_positions() -> JSON: liste normalisée des positions ouvertes.
- review_positions(rules_json: str | None) -> JSON: propose des actions (tighten SL / BE / partial / close).
- execute_position_actions(plan_json: str, dry_run: bool = True) -> JSON: applique les actions via MetaApi, si dispo.

⚠️ Intégration prévue avec votre `meta_api.py` existant. Ce fichier essaye de l'importer depuis
   (a) data_fetch/meta_api.py, puis (b) ./meta_api.py. Si vos fonctions ont des noms différents,
   adaptez la classe `MetaAdapter` ci-dessous.

JSON de sortie (_ok / _err):
{
  "ok": true/false,
  "message": "...",
  "data": {...}
}

Action plan (produit par review_positions):
{
  "generated_at": "2025-10-12T10:00:00Z",
  "rules": {...},
  "positions": [
    {
      "position_id": "123456",
      "symbol": "EURUSD",
      "side": "long|short",
      "volume": 0.10,
      "entry_price": 1.2345,
      "sl": 1.2300,
      "tp": 1.2380,
      "open_time": "2025-10-12T07:51:00Z",
      "unrealized": 25.73,
      "current_price": 1.2358,
      "age_minutes": 128,
      "action": "hold|move_sl|move_sl_to_be|partial_close|close",
      "action_params": {"new_sl": 1.2346, "fraction": 0.5},
      "reason": "PnL > 1R, lock BE",
      "warnings": []
    }
  ]
}

Règles (overrides via env):
- GUARD_MAX_AGE_MIN=720           # âge max (min) si PnL <= 0 → close
- GUARD_BE_TRIGGER_R=1.0          # seuil R multiple pour SL=BE (si R calculable)
- GUARD_TRAIL_TRIGGER_R=1.5       # seuil R multiple pour trailing SL serré
- GUARD_FORCE_CLOSE_R=-1.2        # fermer si perte < -1.2R
- GUARD_PARTIAL_TP_R=1.0          # prendre partielle à 1R (si R calculable)
- GUARD_PARTIAL_FRACTION=0.5      # fraction à fermer partiellement
- GUARD_FALLBACK_BE_PCT=0.003     # 0.3% move → BE si R non calculable

La notion de R = |entry - SL|; si SL manquant ou prix courant indispo, on emploie des *fallbacks* (pourcentage).

Dépendances: mcp (FastMCP), loguru, pydantic (facultatif), python-dotenv (optionnel), votre meta_api.py
"""


import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Annotated

from loguru import logger

# ---- MCP server (FastMCP) ----------------------------------------------------
try:
    # FastMCP (https://github.com/modelcontextprotocol/fastmcp)
    from mcp.server.fastmcp import FastMCP
except Exception as e:  # pragma: no cover
    raise RuntimeError("FastMCP introuvable. Installez 'mcp' / 'fastmcp'.") from e

mcp = FastMCP("PositionsGuardMCP")

# ---- Meta API adapter ---------------------------------------------------------
try:
    from data_fetch import meta_api as meta  # type: ignore
except Exception:
    try:
        import meta_api as meta  # type: ignore
    except Exception:
        meta = None  # sera géré par l'adapter


@dataclass
class Position:
    id: str
    symbol: str
    side: str  # "long" or "short"
    volume: float
    entry_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    open_time: Optional[datetime] = None
    unrealized: Optional[float] = None
    current_price: Optional[float] = None
    raw: Dict[str, Any] = None

    def age_minutes(self) -> Optional[int]:
        if not self.open_time:
            return None
        return int((datetime.now(timezone.utc) - self.open_time).total_seconds() // 60)


class MetaAdapter:
    """Minuscule couche d'abstraction pour votre meta_api.

    Adaptez *ici* si vos fonctions ont des signatures différentes.
    """

    def __init__(self) -> None:
        self.available = meta is not None
        if not self.available:
            logger.warning("meta_api.py introuvable. Mode analyse uniquement.")

    # --- helpers
    @staticmethod
    def _parse_time(ts: Any) -> Optional[datetime]:
        if not ts:
            return None
        if isinstance(ts, (int, float)):
            # epoch seconds or ms
            if ts > 10_000_000_000:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            # try ISO
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_side(side: str) -> str:
        s = (side or "").lower()
        if s in {"buy", "long", "b", "1"}:
            return "long"
        if s in {"sell", "short", "s", "-1"}:
            return "short"
        return s or "unknown"

    def list_positions(self) -> List[Position]:
        if not self.available:
            return []
        # --- Essayez dans l'ordre des fonctions possibles ---
        positions: List[Dict[str, Any]] = []
        try:
            if hasattr(meta, "get_open_positions"):
                positions = meta.get_open_positions()  # type: ignore
            elif hasattr(meta, "list_open_positions"):
                positions = meta.list_open_positions()  # type: ignore
            elif hasattr(meta, "fetch_open_positions"):
                positions = meta.fetch_open_positions()  # type: ignore
            else:
                logger.error("Aucune fonction de récupération des positions dans meta_api.")
                return []
        except Exception as e:
            logger.exception("Erreur meta_api lors de la récupération des positions: {}", e)
            return []

        norm: List[Position] = []
        for p in positions:
            pos_id = str(p.get("id") or p.get("positionId") or p.get("ticket") or p.get("orderId") or "")
            sym = p.get("symbol") or p.get("instrument") or p.get("ticker")
            side = self._normalize_side(p.get("side") or p.get("type") or p.get("direction") or "")
            vol = float(p.get("volume") or p.get("lots") or p.get("qty") or 0.0)
            entry = p.get("entryPrice") or p.get("price") or p.get("openPrice")
            sl = p.get("sl") or p.get("stopLoss")
            tp = p.get("tp") or p.get("takeProfit")
            ot = p.get("openTime") or p.get("time") or p.get("openTimestamp")
            pnl = p.get("unrealized") or p.get("unrealizedProfit") or p.get("profit")
            last = p.get("currentPrice") or p.get("priceCurrent") or p.get("lastPrice")
            try:
                entry = float(entry) if entry is not None else None
                sl = float(sl) if sl is not None else None
                tp = float(tp) if tp is not None else None
                pnl = float(pnl) if pnl is not None else None
                last = float(last) if last is not None else None
            except Exception:
                pass
            norm.append(
                Position(
                    id=pos_id,
                    symbol=str(sym) if sym else "",
                    side=side,
                    volume=vol,
                    entry_price=entry,
                    sl=sl,
                    tp=tp,
                    open_time=self._parse_time(ot),
                    unrealized=pnl,
                    current_price=last,
                    raw=p,
                )
            )
        return norm

    # -- Exec ops (best-effort; adaptez aux signatures de meta_api) ------------
    def move_sl(self, position_id: str, new_sl: float) -> Tuple[bool, str]:
        if not self.available:
            return False, "meta_api indisponible"
        try:
            if hasattr(meta, "modify_position"):
                meta.modify_position(position_id=position_id, sl=new_sl)  # type: ignore
                return True, "OK"
            if hasattr(meta, "update_position"):
                meta.update_position(position_id, sl=new_sl)  # type: ignore
                return True, "OK"
        except Exception as e:
            logger.exception("Erreur move_sl: {}", e)
            return False, str(e)
        return False, "Fonction de modification non trouvée dans meta_api"

    def partial_close(self, position_id: str, fraction: float) -> Tuple[bool, str]:
        if not self.available:
            return False, "meta_api indisponible"
        try:
            if hasattr(meta, "close_position"):
                meta.close_position(position_id=position_id, fraction=fraction)  # type: ignore
                return True, "OK"
            if hasattr(meta, "close_position_partial"):
                meta.close_position_partial(position_id, fraction)  # type: ignore
                return True, "OK"
        except Exception as e:
            logger.exception("Erreur partial_close: {}", e)
            return False, str(e)
        return False, "Fonction de fermeture partielle non trouvée dans meta_api"

    def close(self, position_id: str) -> Tuple[bool, str]:
        if not self.available:
            return False, "meta_api indisponible"
        try:
            if hasattr(meta, "close_position"):
                meta.close_position(position_id=position_id)  # type: ignore
                return True, "OK"
            if hasattr(meta, "close_position_full"):
                meta.close_position_full(position_id)  # type: ignore
                return True, "OK"
        except Exception as e:
            logger.exception("Erreur close: {}", e)
            return False, str(e)
        return False, "Fonction de fermeture non trouvée dans meta_api"


adapter = MetaAdapter()

# ---- Utils JSON ---------------------------------------------------------------
def _ok(message: str, data: Any | None = None) -> str:
    return json.dumps({"ok": True, "message": message, "data": data}, ensure_ascii=False)


def _err(message: str, details: Any | None = None) -> str:
    return json.dumps({"ok": False, "message": message, "data": details}, ensure_ascii=False)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


# ---- Rules & heuristics -------------------------------------------------------
@dataclass
class GuardRules:
    max_age_min: int = 720
    be_trigger_r: float = 1.0
    trail_trigger_r: float = 1.5
    force_close_r: float = -1.2
    partial_tp_r: float = 1.0
    partial_fraction: float = 0.5
    fallback_be_pct: float = 0.003  # 0.3%

    @classmethod
    def from_env(cls) -> "GuardRules":
        return cls(
            max_age_min=_env_int("GUARD_MAX_AGE_MIN", 720),
            be_trigger_r=_env_float("GUARD_BE_TRIGGER_R", 1.0),
            trail_trigger_r=_env_float("GUARD_TRAIL_TRIGGER_R", 1.5),
            force_close_r=_env_float("GUARD_FORCE_CLOSE_R", -1.2),
            partial_tp_r=_env_float("GUARD_PARTIAL_TP_R", 1.0),
            partial_fraction=_env_float("GUARD_PARTIAL_FRACTION", 0.5),
            fallback_be_pct=_env_float("GUARD_FALLBACK_BE_PCT", 0.003),
        )


def _compute_r_multiple(pos: Position) -> Tuple[Optional[float], List[str]]:
    """Retourne (R_multiple, warnings). R = |entry - SL|. Nécessite entry, SL et current_price.
    Signe positif si trade en gain, négatif en perte.
    """
    warnings: List[str] = []
    if pos.entry_price is None or pos.sl is None or pos.current_price is None:
        warnings.append("R multiple non calculable (entry/sl/last manquant)")
        return None, warnings
    risk = abs(pos.entry_price - pos.sl)
    if risk <= 0:
        warnings.append("R=0 (SL=entry?)")
        return None, warnings
    if pos.side == "long":
        r_mult = (pos.current_price - pos.entry_price) / risk
    elif pos.side == "short":
        r_mult = (pos.entry_price - pos.current_price) / risk
    else:
        warnings.append("side inconnu")
        return None, warnings
    return r_mult, warnings


def _propose_action(pos: Position, rules: GuardRules) -> Tuple[str, Dict[str, Any], str, List[str]]:
    """Retourne (action, params, reason, warnings)."""
    warnings: List[str] = []
    # 1) Hard stop: trop vieux et pas en gain → close
    age = pos.age_minutes()
    if age is not None and age >= rules.max_age_min:
        if (pos.unrealized or 0.0) <= 0.0:
            return "close", {}, f"Age {age}m ≥ {rules.max_age_min}m et PnL≤0", warnings

    # 2) Si pas de SL → proposer un SL au break-even si prix courant connu
    if pos.sl is None and pos.entry_price is not None and pos.current_price is not None:
        # Si déjà en gain d'au moins fallback_be_pct → BE
        move = (pos.current_price - pos.entry_price) / pos.entry_price
        move = move if pos.side == "long" else -move
        if move >= rules.fallback_be_pct:
            new_sl = pos.entry_price
            return "move_sl_to_be", {"new_sl": new_sl}, f">= {rules.fallback_be_pct*100:.2f}% en gain → SL=BE", warnings

    # 3) Logique basée sur R si disponible
    r_mult, warn = _compute_r_multiple(pos)
    warnings.extend(warn)
    if r_mult is not None:
        if r_mult <= rules.force_close_r:
            return "close", {}, f"Perte {r_mult:.2f}R ≤ {rules.force_close_r}R → close", warnings
        if r_mult >= rules.trail_trigger_r:
            # trail: lock au moins 0.5R (simple)
            assert pos.entry_price is not None and pos.sl is not None
            risk = abs(pos.entry_price - pos.sl)
            if pos.side == "long":
                new_sl = pos.entry_price + 0.5 * risk
            else:
                new_sl = pos.entry_price - 0.5 * risk
            return "move_sl", {"new_sl": new_sl}, f"Trail: {r_mult:.2f}R ≥ {rules.trail_trigger_r}R → lock 0.5R", warnings
        if r_mult >= rules.be_trigger_r:
            return "move_sl_to_be", {"new_sl": pos.entry_price}, f"BE: {r_mult:.2f}R ≥ {rules.be_trigger_r}R", warnings
        if r_mult >= rules.partial_tp_r:
            return "partial_close", {"fraction": rules.partial_fraction}, f"Partielle: {r_mult:.2f}R ≥ {rules.partial_tp_r}R", warnings

    # 4) Fallback: si PnL très négatif (si dispo) → hold (on note seulement)
    if pos.unrealized is not None and pos.unrealized < 0:
        warnings.append("Perte mais seuil inconnu sans R; pas d'action dure")

    return "hold", {}, "Pas d'action requise", warnings


# ---- Tools -------------------------------------------------------------------
@mcp.tool()
async def list_positions() -> str:
    """Retourne les positions ouvertes normalisées."""
    positions = adapter.list_positions()
    data = []
    for p in positions:
        data.append(
            {
                "position_id": p.id,
                "symbol": p.symbol,
                "side": p.side,
                "volume": p.volume,
                "entry_price": p.entry_price,
                "sl": p.sl,
                "tp": p.tp,
                "open_time": p.open_time.isoformat() if p.open_time else None,
                "unrealized": p.unrealized,
                "current_price": p.current_price,
            }
        )
    return _ok(f"{len(data)} position(s)", {"positions": data})


@mcp.tool()
async def review_positions(rules_json: str | None = None) -> str:
    """Analyse les positions et propose un plan d'actions.

    :param rules_json: JSON optionnel pour override des règles (ex: {"partial_fraction": 0.33})
    """
    rules = GuardRules.from_env()
    if rules_json:
        try:
            overrides = json.loads(rules_json)
            for k, v in overrides.items():
                if hasattr(rules, k):
                    setattr(rules, k, v)
        except Exception as e:
            return _err("rules_json invalide", str(e))

    positions = adapter.list_positions()
    plan_positions = []
    for p in positions:
        action, params, reason, warnings = _propose_action(p, rules)
        plan_positions.append(
            {
                "position_id": p.id,
                "symbol": p.symbol,
                "side": p.side,
                "volume": p.volume,
                "entry_price": p.entry_price,
                "sl": p.sl,
                "tp": p.tp,
                "open_time": p.open_time.isoformat() if p.open_time else None,
                "unrealized": p.unrealized,
                "current_price": p.current_price,
                "age_minutes": p.age_minutes(),
                "action": action,
                "action_params": params,
                "reason": reason,
                "warnings": warnings,
            }
        )

    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "max_age_min": rules.max_age_min,
            "be_trigger_r": rules.be_trigger_r,
            "trail_trigger_r": rules.trail_trigger_r,
            "force_close_r": rules.force_close_r,
            "partial_tp_r": rules.partial_tp_r,
            "partial_fraction": rules.partial_fraction,
            "fallback_be_pct": rules.fallback_be_pct,
        },
        "positions": plan_positions,
    }
    return _ok("Plan généré", plan)


@mcp.tool()
async def execute_position_actions(plan_json: str, dry_run: bool = True) -> str:
    """Applique un *plan* produit par review_positions.

    Pour chaque position, supporte les actions: move_sl, move_sl_to_be, partial_close, close.
    Si `dry_run=True`, ne fait que valider et simuler.
    """
    try:
        plan = json.loads(plan_json)
    except Exception as e:
        return _err("plan_json invalide", str(e))

    positions = plan.get("positions") or []
    if not isinstance(positions, list):
        return _err("plan.positions doit être une liste")

    results: List[Dict[str, Any]] = []
    for item in positions:
        pid = str(item.get("position_id"))
        action = (item.get("action") or "").lower()
        params = item.get("action_params") or {}
        status: str = "skipped"
        detail: str = ""

        if action in {"move_sl", "move_sl_to_be"}:
            new_sl = params.get("new_sl")
            if new_sl is None:
                status, detail = "error", "new_sl manquant"
            else:
                if dry_run:
                    status, detail = "ok", "dry_run"
                else:
                    ok, msg = adapter.move_sl(pid, float(new_sl))
                    status, detail = ("ok", msg) if ok else ("error", msg)

        elif action == "partial_close":
            frac = float(params.get("fraction") or 0.0)
            if not (0.0 < frac < 1.0):
                status, detail = "error", "fraction (0,1) requise"
            else:
                if dry_run:
                    status, detail = "ok", "dry_run"
                else:
                    ok, msg = adapter.partial_close(pid, frac)
                    status, detail = ("ok", msg) if ok else ("error", msg)

        elif action == "close":
            if dry_run:
                status, detail = "ok", "dry_run"
            else:
                ok, msg = adapter.close(pid)
                status, detail = ("ok", msg) if ok else ("error", msg)

        else:
            status, detail = "ignored", f"action inconnue: {action}"

        results.append({"position_id": pid, "action": action, "status": status, "detail": detail})

    return _ok("Execution terminée" if not dry_run else "Simulation terminée", {"results": results})


# ---- Prompt(s) MCP ------------------------------------------------------------
try:
    from pydantic import Field  # type: ignore
except Exception:  # pragma: no cover
    # Fallback léger si pydantic non dispo; on définit Field de façon minimale
    class Field:  # type: ignore
        def __init__(self, description: str = ""):
            self.description = description


@mcp.prompt()
def positions_guard_prompt(
    strategy: Annotated[
        str,
        Field(description="Ligne directrice de gestion de position (ex: conservatrice, trend-following, mean-reversion)."),
    ] = "Gestion du risque conservatrice: protéger le capital, lock BE à 1R, trailing progressif.",
    rules_json: Annotated[Optional[str], Field(description="JSON d'override des règles (cf. review_positions).")] = None,
    dry_run: Annotated[bool, Field(description="Simuler l'exécution sans envoyer d'ordres.")] = True,
) -> dict:
    """Prompt orchestrateur pour auditer et (éventuellement) agir sur les positions ouvertes.

    Conseille l'agent à:
      1) Lister les positions (tool: list_positions)
      2) Générer un plan d'actions (tool: review_positions(rules_json))
      3) Si validé, exécuter le plan (tool: execute_position_actions(plan_json, dry_run))
    """
    now = datetime.now(timezone.utc).isoformat()
    rules_text = rules_json if rules_json else "{}"

    lines = [
        f"NOW(UTC): {now}",
        "STRATÉGIE:",
        strategy,
        "",
        "TÂCHES:",
        "1) Appelle `list_positions` pour obtenir les positions ouvertes.",
        "2) Appelle `review_positions(rules_json)` avec ces overrides (JSON):",
        rules_text,
        "3) Vérifie la cohérence du plan (SL/TP non nuls, fractions (0,1), ids valides).",
        f"4) Si validé, appelle `execute_position_actions(plan_json, dry_run={dry_run}).",
        "",
        "CONTRAINTE DE SORTIE:",
        "- Fournis un objet JSON final avec: { 'plan': <plan JSON>, 'ready_to_execute': bool, 'dry_run': bool, 'notes': str }.",
    ]
    content = "\n".join(lines)

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es *Positions Guard*, un assistant de gestion de trade. "
                "Objectif: réduire le risque, sécuriser les gains, et éviter les pertes extrêmes. "
                "Tu peux appeler les tools MCP: list_positions, review_positions, execute_position_actions. "
                "Toujours produire un résumé opérationnel clair, et un plan en JSON prêt pour execute_position_actions."
            ),
        },
        {"role": "user", "content": content},
    ]
    return {"messages": messages}


# ---- Entrée serveur -----------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")

