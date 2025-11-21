"""
Backtesting utilities for the trading agent.

Exposes the :class:`Backtester` class for programmatic usage and can be
invoked through ``python -m src.backtest.engine`` for a simple CLI.
"""

from .engine import Backtester, BacktestConfig, BacktestReport, TradeResult  # noqa: F401

