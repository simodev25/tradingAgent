from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..mcp.analyse.analysis_core import (
    CONFIG,
    _atr_pct_bands_from_df,
    _confidence,
    _compute_indicators_df,
    _df_from_ohlcv,
    _directional_score,
    _last_row,
    _regime_from_df,
    _volatility_gate,
)
from ..mcp.analyse.analyze_tec_mcp import plan_raw_from_json


@dataclass
class BacktestConfig:
    symbol: str
    data_path: str
    interval: str = "15m"
    horizon: str = "scalping"
    risk_level: str = "high"
    cost_bps: float = 5.0
    slippage_bps: float = 1.0
    warmup_bars: int = 200
    initial_equity: float = 100_000.0
    risk_allocation: float = 1.0
    max_holding_bars: int = 32

    def validate(self) -> None:
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Historical data file not found: {path}")
        if self.risk_allocation <= 0 or self.risk_allocation > 1.0:
            raise ValueError("risk_allocation must be in (0, 1]")
        if self.warmup_bars < 100:
            raise ValueError("warmup_bars should be >= 100 to stabilise indicators")
        if self.max_holding_bars <= 0:
            raise ValueError("max_holding_bars must be > 0")


@dataclass
class TradeResult:
    symbol: str
    entry_index: int
    exit_index: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    outcome: str
    bars_held: int
    gross_return: float
    net_return: float
    confidence: int
    regime: str
    volatility_band: Optional[str]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class BacktestReport:
    config: BacktestConfig
    trades: List[TradeResult]
    equity_curve: List[Tuple[pd.Timestamp, float]]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "config": vars(self.config),
            "trades": [
                {
                    "symbol": t.symbol,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "outcome": t.outcome,
                    "bars_held": t.bars_held,
                    "gross_return": t.gross_return,
                    "net_return": t.net_return,
                    "confidence": t.confidence,
                    "regime": t.regime,
                    "volatility_band": t.volatility_band,
                    "metadata": t.metadata,
                }
                for t in self.trades
            ],
            "equity_curve": [
                {"timestamp": ts.isoformat(), "equity": eq} for ts, eq in self.equity_curve
            ],
            "metrics": self.metrics,
        }


def _interval_to_minutes(interval: str) -> int:
    s = interval.strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    if s.endswith("d"):
        return int(s[:-1]) * 60 * 24
    raise ValueError(f"Unsupported interval: {interval}")


def _resample_htf(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    resampled = (
        df.set_index("Date")
        .resample(freq)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum" if "Volume" in df.columns else "first",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
        .reset_index()
    )
    return resampled


def _load_history(config: BacktestConfig) -> pd.DataFrame:
    raw = pd.read_csv(config.data_path)
    if raw.empty:
        raise ValueError(f"Historical file {config.data_path} is empty")
    parsed = _df_from_ohlcv(raw.to_dict(orient="records"))
    if parsed is None or parsed.empty:
        raise ValueError("Unable to parse OHLCV columns from historical file")
    parsed = parsed.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if len(parsed) <= config.warmup_bars + 1:
        raise ValueError("Not enough rows to satisfy warmup_bars")
    return parsed


def _call_plan(ltf: pd.DataFrame, risk_level: str, horizon: str, action: str) -> Optional[Dict[str, object]]:
    payload = ltf.tail(400).to_json(orient="records", date_format="iso")
    raw = plan_raw_from_json.__wrapped__(
        ohlcv_json=payload,
        risk_level=risk_level,
        direction="long" if action == "BUY" else "short",
        horizon=horizon,
    )
    data = json.loads(raw)
    if not data.get("ok"):
        logger.debug(f"[backtest] plan generation failed: {data}")
        return None
    return data["data"]


def _compute_drawdown(points: Sequence[Tuple[pd.Timestamp, float]]) -> float:
    max_dd = 0.0
    peak = None
    for _, eq in points:
        if peak is None or eq > peak:
            peak = eq
        drawdown = 0.0 if peak is None else (eq / peak) - 1.0
        if drawdown < max_dd:
            max_dd = drawdown
    return abs(max_dd)


class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.config.validate()
        self.history = _load_history(config)
        self.interval_minutes = _interval_to_minutes(config.interval)
        self.htf_interval = CONFIG["HTF_INTERVAL"]

    def run(self) -> BacktestReport:
        trades: List[TradeResult] = []
        equity_curve: List[Tuple[pd.Timestamp, float]] = []
        equity = self.config.initial_equity

        idx = self.config.warmup_bars
        while idx < len(self.history) - 1:
            window = self.history.iloc[: idx + 1]
            signal = self._generate_signal(window)
            if signal is None or signal["action"] == "HOLD":
                idx += 1
                continue

            trade = self._simulate_trade(idx, signal)
            if trade is None:
                idx += 1
                continue

            trades.append(trade)
            realised_return = self.config.risk_allocation * trade.net_return
            equity *= max(0.0, 1.0 + realised_return)
            equity_curve.append((trade.exit_time, equity))

            idx = trade.exit_index + 1

        metrics = self._compute_metrics(trades, equity_curve)
        return BacktestReport(config=self.config, trades=trades, equity_curve=equity_curve, metrics=metrics)

    def _generate_signal(self, window: pd.DataFrame) -> Optional[Dict[str, object]]:
        if len(window) < self.config.warmup_bars:
            return None

        ltf = _compute_indicators_df(window.copy())
        if ltf is None or ltf.empty:
            return None

        htf_raw = _resample_htf(window.copy(), self.htf_interval)
        if htf_raw.empty:
            return None
        htf = _compute_indicators_df(htf_raw)
        if htf is None or htf.empty:
            return None

        last_ltf = _last_row(ltf)
        vol_meta = None
        if CONFIG.get("VOL_ENABLED", True):
            bands = _atr_pct_bands_from_df(
                ltf,
                lookback_days=CONFIG["VOL_LOOKBACK_DAYS"],
                low_pct=CONFIG["VOL_LOW_PCT"],
                high_pct=CONFIG["VOL_HIGH_PCT"],
                extreme_pct=CONFIG["VOL_EXTREME_PCT"],
            )
            if bands:
                allowed, band, size_factor, reason = _volatility_gate(
                    bands["atr_now"],
                    bands["p10"],
                    bands["p90"],
                    bands.get("p95"),
                    size_high=CONFIG["VOL_SIZE_HIGH"],
                )
                vol_meta = {
                    "atr_pct_now": bands["atr_now"],
                    "p10": bands["p10"],
                    "p90": bands["p90"],
                    "p95": bands.get("p95"),
                    "band": band,
                    "size_factor": size_factor,
                    "reason": reason,
                }
                if not allowed:
                    return {
                        "action": "HOLD",
                        "reason": f"Volatility gate {band} ({reason})",
                        "regime": _regime_from_df(ltf, htf),
                        "confidence": 0,
                        "volatility": vol_meta,
                    }

        regime = _regime_from_df(ltf, htf)
        if regime == "no-trade":
            return {
                "action": "HOLD",
                "reason": "Regime flagged as no-trade",
                "regime": regime,
                "confidence": 0,
                "volatility": vol_meta,
            }

        score = _directional_score(last_ltf, df_context=ltf)
        confidence = _confidence(last_ltf, score)

        ltf_up = (last_ltf.get("EMA_Fast") or 0) >= (last_ltf.get("EMA_Slow") or 0)
        htf_last = _last_row(htf)
        htf_up = (htf_last.get("EMA_Fast") or 0) >= (htf_last.get("EMA_Slow") or 0)

        action = "HOLD"
        if regime == "trend":
            if score >= CONFIG["DEC_TREND_BUY_SCORE"] and ltf_up and htf_up:
                action = "BUY"
            elif score <= CONFIG["DEC_TREND_SELL_SCORE"] and (not ltf_up) and (not htf_up):
                action = "SELL"
        else:
            if score <= CONFIG["DEC_RANGE_SELL_SCORE"] and (not htf_up):
                action = "SELL"
            elif score >= CONFIG["DEC_RANGE_BUY_SCORE"] and htf_up:
                action = "BUY"

        if (
            vol_meta
            and vol_meta["band"] == "HIGH"
            and CONFIG.get("REQUIRE_HTF_ON_EDGES", False)
            and (
                (action == "BUY" and not (ltf_up and htf_up))
                or (action == "SELL" and not ((not ltf_up) and (not htf_up)))
            )
        ):
            action = "HOLD"

        min_conf = CONFIG["PROFILES"].get(self.config.horizon, {}).get("min_confidence")
        if action != "HOLD" and min_conf is not None and confidence < float(min_conf):
            action = "HOLD"

        if action == "HOLD":
            return {
                "action": "HOLD",
                "reason": f"Score={score:.2f} / confidence={confidence}",
                "regime": regime,
                "confidence": confidence,
                "volatility": vol_meta,
            }

        plan = _call_plan(ltf, self.config.risk_level, self.config.horizon, action)
        if not plan:
            return {
                "action": "HOLD",
                "reason": "Unable to derive plan (entry/sl/tp)",
                "regime": regime,
                "confidence": confidence,
                "volatility": vol_meta,
            }

        return {
            "action": action,
            "entry": plan.get("entry"),
            "sl": plan.get("sl"),
            "tp": plan.get("tp"),
            "confidence": confidence,
            "regime": regime,
            "volatility": vol_meta,
            "reason": f"score={score:.2f}",
        }

    def _simulate_trade(self, decision_index: int, signal: Dict[str, object]) -> Optional[TradeResult]:
        entry_index = decision_index + 1
        if entry_index >= len(self.history):
            return None

        sl = signal.get("sl")
        tp = signal.get("tp")
        if sl is None or tp is None:
            return None

        direction = signal["action"]
        entry_row = self.history.iloc[entry_index]
        entry_price = float(entry_row["Open"])

        slip = self.config.slippage_bps / 10_000.0
        if direction == "BUY":
            entry_price *= (1 + slip)
        else:
            entry_price *= (1 - slip)

        exit_price = entry_price
        exit_reason = "timeout"
        exit_index = entry_index

        take_price = float(tp)
        stop_price = float(sl)
        max_exit = min(entry_index + self.config.max_holding_bars, len(self.history) - 1)

        for idx in range(entry_index, max_exit + 1):
            row = self.history.iloc[idx]
            high = float(row["High"])
            low = float(row["Low"])

            if direction == "BUY":
                if low <= stop_price:
                    exit_price = stop_price * (1 - slip)
                    exit_reason = "stop"
                    exit_index = idx
                    break
                if high >= take_price:
                    exit_price = take_price * (1 - slip)
                    exit_reason = "target"
                    exit_index = idx
                    break
            else:
                if high >= stop_price:
                    exit_price = stop_price * (1 + slip)
                    exit_reason = "stop"
                    exit_index = idx
                    break
                if low <= take_price:
                    exit_price = take_price * (1 + slip)
                    exit_reason = "target"
                    exit_index = idx
                    break

        if exit_index == entry_index:
            # neither stop nor target hit on first bar -> exit at close if timeout
            exit_row = self.history.iloc[max_exit]
            exit_price = float(exit_row["Close"])
            if direction == "BUY":
                exit_price *= (1 - slip)
            else:
                exit_price *= (1 + slip)
            exit_reason = "timeout"
            exit_index = max_exit

        gross_return = (
            (exit_price - entry_price) / entry_price if direction == "BUY" else (entry_price - exit_price) / entry_price
        )
        fee = (self.config.cost_bps / 10_000.0)
        net_return = gross_return - fee

        bars_held = exit_index - decision_index

        return TradeResult(
            symbol=self.config.symbol,
            entry_index=entry_index,
            exit_index=exit_index,
            entry_time=pd.to_datetime(entry_row["Date"]),
            exit_time=pd.to_datetime(self.history.iloc[exit_index]["Date"]),
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_price,
            take_profit=take_price,
            outcome=exit_reason,
            bars_held=bars_held,
            gross_return=gross_return,
            net_return=net_return,
            confidence=int(signal.get("confidence", 0)),
            regime=str(signal.get("regime", "")),
            volatility_band=(signal.get("volatility") or {}).get("band") if signal.get("volatility") else None,
            metadata={
                "volatility": signal.get("volatility"),
                "reason": signal.get("reason"),
            },
        )

    def _compute_metrics(
        self,
        trades: Sequence[TradeResult],
        equity_curve: Sequence[Tuple[pd.Timestamp, float]],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["total_trades"] = len(trades)

        if not trades:
            metrics["final_equity"] = self.config.initial_equity
            metrics["cagr"] = 0.0
            metrics["sharpe"] = 0.0
            metrics["win_rate"] = 0.0
            metrics["avg_return"] = 0.0
            metrics["p_value"] = 1.0
            metrics["t_stat"] = 0.0
            metrics["max_drawdown"] = 0.0
            metrics["edge_significant"] = False
            return metrics

        returns = np.array([t.net_return * self.config.risk_allocation for t in trades], dtype=float)
        wins = np.sum(returns > 0)
        metrics["win_rate"] = float(wins) / len(trades)
        metrics["avg_return"] = float(np.mean(returns))
        metrics["median_return"] = float(np.median(returns))
        metrics["best_trade"] = float(np.max(returns))
        metrics["worst_trade"] = float(np.min(returns))

        equity_final = equity_curve[-1][1] if equity_curve else self.config.initial_equity
        metrics["final_equity"] = equity_final

        start_date = pd.to_datetime(self.history.iloc[self.config.warmup_bars]["Date"])
        end_date = pd.to_datetime(self.history.iloc[-1]["Date"])
        span_days = max((end_date - start_date).days, 1)

        metrics["cagr"] = (equity_final / self.config.initial_equity) ** (365 / span_days) - 1

        std = float(np.std(returns, ddof=1)) if len(trades) > 1 else 0.0
        trades_per_year = len(trades) / (span_days / 365) if span_days > 0 else len(trades)
        if std > 1e-9 and trades_per_year > 0:
            metrics["sharpe"] = (float(np.mean(returns)) / std) * math.sqrt(trades_per_year)
        else:
            metrics["sharpe"] = 0.0

        if std > 1e-9 and len(trades) > 1:
            t_stat = float(np.mean(returns)) / (std / math.sqrt(len(trades)))
            metrics["t_stat"] = t_stat
            metrics["p_value"] = 1 - NormalDist().cdf(t_stat)
        else:
            metrics["t_stat"] = 0.0
            metrics["p_value"] = 1.0

        metrics["max_drawdown"] = _compute_drawdown(equity_curve) if equity_curve else 0.0
        metrics["edge_significant"] = metrics["avg_return"] > 0 and metrics["p_value"] < 0.05

        return metrics


def _format_report(report: BacktestReport) -> str:
    cfg = report.config
    lines = [
        f"Backtest summary for {cfg.symbol} ({cfg.interval}, horizon={cfg.horizon})",
        f"Data file          : {cfg.data_path}",
        f"Trades             : {report.metrics['total_trades']}",
        f"Win rate           : {report.metrics['win_rate']:.2%}",
        f"Average return     : {report.metrics['avg_return']:.4f}",
        f"Final equity       : {report.metrics['final_equity']:.2f}",
        f"CAGR               : {report.metrics['cagr']:.2%}",
        f"Sharpe (per trade) : {report.metrics['sharpe']:.2f}",
        f"t-stat             : {report.metrics['t_stat']:.2f}",
        f"p-value (H>0)      : {report.metrics['p_value']:.4f}",
        f"Max drawdown       : {report.metrics['max_drawdown']:.2%}",
        f"Edge significant   : {'yes' if report.metrics['edge_significant'] else 'no'}",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run backtest on historical OHLCV data")
    parser.add_argument("--symbol", required=True, help="Symbol name (informational)")
    parser.add_argument("--data", required=True, help="CSV file with columns Date/Open/High/Low/Close[/Volume]")
    parser.add_argument("--interval", default="15m", help="Base timeframe (e.g. 15m, 1h)")
    parser.add_argument("--horizon", default="scalping", help="Strategy horizon (matches CONFIG profiles)")
    parser.add_argument("--risk-level", default="high", help="Risk profile")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Round-trip transaction cost in bps")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Per-leg slippage in bps")
    parser.add_argument("--warmup", type=int, default=200, help="Bars used for indicator warmup")
    parser.add_argument("--max-hold", type=int, default=32, help="Maximum bars to keep a trade open")
    parser.add_argument("--initial-equity", type=float, default=100_000.0, help="Starting capital")
    parser.add_argument("--risk-allocation", type=float, default=1.0, help="Fraction of equity put at risk per trade")
    args = parser.parse_args(argv)

    cfg = BacktestConfig(
        symbol=args.symbol,
        data_path=args.data,
        interval=args.interval,
        horizon=args.horizon,
        risk_level=args.risk_level,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        warmup_bars=args.warmup,
        max_holding_bars=args.max_hold,
        initial_equity=args.initial_equity,
        risk_allocation=args.risk_allocation,
    )
    tester = Backtester(cfg)
    report = tester.run()
    print(_format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
