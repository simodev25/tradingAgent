# Trading Agent — News · Technicals · Execution

A scheduled trading pipeline that blends **news sentiment** (FinBERT), **multi-timeframe technicals** (EMA/RSI/MACD/ATR/Bollinger + robust levels), and a **safeguarded execution layer** (MetaApi), each isolated in MCP subprocesses.

---

## TL;DR (Quick Start)

```bash
cd tradingAgent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.exemple .env   # edit your credentials + URLs
# edit config.json     # symbols, scheduler, logging, etc.
python start_trading.py           # scheduler mode
# or
python start_trading.py --once    # single run
```

Daily results are persisted as `results_YYYYMMDD.json`.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Environment (.env)](#environment-env)
  - [Application (configjson)](#application-configjson)
- [Running](#running)
  - [Scheduler (recommended)](#scheduler-recommended)
  - [One-off run](#one-off-run)
  - [Manual debug](#manual-debug)
- [Risk & Gatekeeping](#risk--gatekeeping)
- [MCP Modules](#mcp-modules)
  - [News](#news-srcmcpanalyseanalyze_news_mcppy)
  - [Technical Analysis](#technical-analysis-srcmcpanalyseanalyze_tec_mcppy)
  - [Execution](#execution-srcmcpexecutionexecution_mcppy)
- [Outputs & Monitoring](#outputs--monitoring)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)
- [License](#license)

---

## Overview

- **Entry point:** `start_trading.py`
- **Orchestration:** `src/scheduler.py` (runs every _X_ minutes within trading hours)
- **Pipeline:** `src/agents.py` launches three MCPs in sequence:
  1. **News** → global bias with FinBERT
  2. **Technicals** → BUY/SELL/HOLD + robust SL/TP levels
  3. **Execution** → send/adjust/cancel with strict safeguards

---

## Architecture

```
┌────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  Scheduler     │──▶──│   News MCP        │──▶──│  Technicals MCP     │
│ (start_trading)│     │  (FinBERT bias)   │     │ (MTF + levels/RR)   │
└───────┬────────┘     └─────────┬─────────┘     └──────────┬─────────┘
        │                          │                         │
        ▼                          ▼                         ▼
    batching                strict JSON I/O             HOLD gating, RR,
   trading hours             timeouts/guardrails        symbol constraints
        │                                                   │
        └───────────────────────────────────────────────────▼
                                            ┌───────────────┴──────────────┐
                                            │   Execution MCP (MetaApi)    │
                                            │ info tools + safeguards      │
                                            └───────────────────────────────┘
```

---

## Requirements

- **Python** 3.10+
- **Hugging Face** access to download `ProsusAI/finbert` (or a local HF cache)
- **MetaApi** credentials (or run with `dry_run=true`)
- **LLM via LangChain** (default: `ChatOllama`, local Ollama supported)

---

## Installation

```bash
cd tradingAgent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note (Ollama):** install/start Ollama and ensure the API is reachable (e.g. `http://localhost:11434`).

---

## Configuration

### Environment (.env)

Copy the template and edit:
```bash
cp env.exemple .env
```

Useful keys:

| Key               | Example                      | Notes                                   |
|-------------------|------------------------------|-----------------------------------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434`     | Local Ollama endpoint                   |
| `OLLAMA_MODEL`    | `llama3.1`                   | Any installed model                     |
| `OLLAMA_API_KEY`  | *(optional)*                 | Usually not required for local Ollama   |
| MetaApi creds     | *(per your impl)*            | See `src/mcp/data_fetch/meta_api.py`    |

### Application (`config.json`)

Controls symbols, scheduler, and global settings.

**Minimal example:**
```json
{
  "symbols": [
    {
      "symbol": "BTCUSD",
      "period": "5d",
      "interval": "15m",
      "horizon": "scalping",
      "risk_level": "high",
      "default_volume": 0.01,
      "min_confidence": 40
    }
  ],
  "global_settings": {
    "honor_hold": true,
    "dry_run": true,
    "news_top_n": 10,
    "include_columns": "Open,High,Low,Close,Volume",
    "max_concurrent_symbols": 3
  },
  "scheduler": {
    "enabled": true,
    "interval_minutes": 15,
    "start_time": "09:00",
    "end_time": "17:00",
    "timezone": "UTC"
  },
  "logging": {
    "level": "INFO",
    "file": "trading_agent.log",
    "max_size": "10MB",
    "backup_count": 5
  }
}
```

---

## Running

### Scheduler (recommended)

```bash
python start_trading.py
```

### One-off run

```bash
python start_trading.py --once
```

### Manual debug

```bash
python src/agents.py
```

---

## Risk & Gatekeeping

- **HTF Confluence:** if `|score| ≥ 3`, fetch HTF (e.g., M15/H1). Require EMA_fast > EMA_slow and RSI aligned with the trade side. If missing, downgrade confidence (e.g., `≤ 20`) and tag `reason_no_trade=HTF_MISS`.
- **Volatility Bands (dynamic):** compute `ATR_PCT` percentiles (e.g., p10/p90 over 30 days) **per symbol**. Block trades if outside bands and tag `ATR_GATE(pXX)`.
- **Symbol Executability:** if `tradeMode=CLOSEONLY`, blacklist execution for 24h (still analyze). Tag `CLOSEONLY`.
- **Execution Minima:** `confidence ≥ 60`, `RR ≥ 1.5` (scalp), `spread ≤ k * median(5m)`, `slippage ≤ max_pips`. If a rule fails, cancel and log `RULE_FAIL_*`.
- **Post-Entry Mgmt:** at `+0.5R` → move SL to BE. Use ATR-based trailing (e.g., `ATR(14) * 0.8`) or last swing (M1/M5). Optionally take 50% at TP1.
- **Exposure Caps:** limit net exposure per currency (e.g., ≤ 0.6 lot/ccy) and alert on breaches.
- **Telemetry:** log normalized `reason_no_trade`, `htf_snapshot`, `atr_band`, `tradeMode`, `spread`, `slippage_est`. Track counts per reason in dashboards.

---

## MCP Modules

### News (`src/mcp/analyse/analyze_news_mcp.py`)
- **Tool:** `analyze_news(symbol)` — fetches Yahoo Finance headlines, scrapes article text when possible, runs batched FinBERT inference, and computes a weighted global bias.
- **Prompt:** `news_agent` — strict JSON output (bias, summary, score).

### Technical Analysis (`src/mcp/analyse/analyze_tec_mcp.py`)
- **Key tools:**
  - `get_historical_candles` (with caching)
  - `compute_indicators` (EMA/RSI/MACD/ATR/Bollinger + market metrics)
  - `levels_autonomous` (robust SL/TP using HTF anchoring & Fibonacci)
  - `intraday_decision` (regime: trend/range/no-trade + actionable decision)
  - `plan_raw` (ATR-based plan)
- **Hardening:** robust timestamp parsing (s/ms/us/ns/ISO), tick/digits inference, min stop/buffer/spread checks, RR targets per horizon.
- **Prompt:** `analysis_agent` — ReAct-guided rules, directional scoring, strict JSON.

### Execution (`src/mcp/execution/execution_mcp.py`)
- **Info tools:** `get_account_info`, `get_positions`, `get_orders`, `get_symbol_spec`, `get_current_price`
- **`execute_trade`:** rounds to symbol constraints, validates SL/TP directionality, LIMIT/STOP vs market, margin, dups/conflicts; supports `dry_run`.
- **Prompt:** `execution_agent` — enforces use of all info tools, applies safeguards (HOLD gating, confidence, margin, conflict checks), produces a structured plan (`proceed/adjust/cancel`).

---

## Outputs & Monitoring

- **Results:** `results_YYYYMMDD.json` (latest run of the day)
- **Logs:** as configured in `config.json` (console + rotating `trading_agent.log`)
- **Monitoring:** optional `monitor.py` (if present) to visualize recent runs and stats

---

## Troubleshooting

- **LLM / Ollama**
  - Set `OLLAMA_BASE_URL=http://localhost:11434` and pick an installed model (`ollama list`).
  - Temperature is 0; prompts demand strict JSON and the pipeline guards non-JSON output.

- **FinBERT / Transformers**
  - First run downloads the model; ensure time and disk space.
  - If unavailable, news bias falls back to **neutral**.

- **MetaApi**
  - Without valid credentials, keep `"dry_run": true` in `config.json`.
  - Check `src/mcp/data_fetch/meta_api.py` and corresponding environment variables.

- **Trading hours**
  - Adjust `scheduler.timezone`, `start_time`, `end_time`. On errors, the scheduler behaves permissively.

---

## Notes

- LangChain and related deps evolve quickly. Pin compatible versions in `requirements.txt`.
- MCP outputs are strict JSON. The parser handles code-fence blocks and bracket balancing with conservative fallbacks.

---

## License

Provided as-is for research and experimentation. Ensure regulatory compliance and always test with `dry_run` before production.
