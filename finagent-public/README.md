<p align="center">
  <h1 align="center">📊 FinAgent — Quantitative Financial Analysis Platform</h1>
  <p align="center">
    <strong>An AI-powered terminal interface for real-time financial data retrieval, quantitative analysis, and market intelligence.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/AI_Engine-Claude_Sonnet-7C3AED?style=for-the-badge&logo=anthropic&logoColor=white" alt="Claude">
    <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT License">
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/data-Yahoo_Finance-7B1FA2?style=flat-square" alt="Yahoo Finance">
    <img src="https://img.shields.io/badge/data-CoinGecko-8DC63F?style=flat-square" alt="CoinGecko">
    <img src="https://img.shields.io/badge/data-Polymarket-FF6B35?style=flat-square" alt="Polymarket">
  </p>
  <p align="center">
    <a href="https://github.com/Mnfisher93/FinAgent/tree/multi-provider">🔀 Multi-Provider Branch (GPT-4 · Gemini · Grok)</a>
  </p>
</p>

---

## 📋 Overview

FinAgent is a **terminal-based quantitative analysis assistant** that leverages Claude's function-calling architecture to provide institutional-grade financial data retrieval and analysis through natural language. The system integrates multiple data sources and executes structured API calls in a tool-use loop — delivering real-time market data, fundamental analysis, and cross-asset comparisons.

> **🔀 Multi-Provider Support:** A [multi-provider branch](https://github.com/Mnfisher93/FinAgent/tree/multi-provider) extends compatibility to **OpenAI GPT-4**, **Google Gemini**, and **xAI Grok** — all sharing the same tool-calling architecture with provider-specific adapters.

---

## ⚡ Capabilities

| Domain | Tools | Data Source |
|--------|-------|-------------|
| **Equities** | `get_stock_quote` · `get_stock_history` · `get_stock_info` | Yahoo Finance |
| **Cryptocurrencies** | `get_crypto_price` · `get_crypto_top_n` · `search_crypto` | CoinGecko |
| **Prediction Markets** | `get_polymarket_markets` | Polymarket |
| **Quantitative Analysis** | `analyze_asset` · `compare_assets` | Computed |

### 📈 Analysis Metrics
- **Returns** — Total return, annualized return over arbitrary periods
- **Risk** — Annualized volatility, maximum drawdown, drawdown duration
- **Risk-Adjusted** — Sharpe ratio (rf = 5%)
- **Cross-Asset** — Pearson correlation matrices for portfolio construction

---

## 🖥️ Demo

```
You → What's Apple trading at?
  🔧 get_stock_quote({"ticker": "AAPL"})

Agent → Apple Inc. (AAPL) is currently trading at $264.58.
        Market Cap: $4.05T | P/E: 33.2 | Day Range: $261.30 – $266.12
        52-Week: $164.08 – $267.26 | Volume: 48.2M

You → Compare Tesla, Nvidia, and Microsoft over the last year
  🔧 compare_assets({"tickers": ["TSLA", "NVDA", "MSFT"]})

Agent → Correlation matrix of daily returns (1Y):
        TSLA-NVDA: 0.52 | TSLA-MSFT: 0.38 | NVDA-MSFT: 0.68
        Highest correlation: NVDA-MSFT (0.68) — both driven by AI spending

You → Analyze Nvidia's risk profile
  🔧 analyze_asset({"ticker": "NVDA", "period": "2y"})

Agent → NVDA 2-Year Risk Profile:
        Annualized Return: +142.3% | Volatility: 58.7%
        Sharpe Ratio: 2.34 | Max Drawdown: -33.1%
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.13+ |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |
| **API Key** | [Anthropic Console](https://console.anthropic.com/) |

### Installation

**1. Install `uv`** (Python package manager — [docs](https://docs.astral.sh/uv/))

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# or via Homebrew
brew install uv
```

**2. Clone and run**

```bash
git clone https://github.com/Mnfisher93/FinAgent.git
cd FinAgent/finagent-public
git checkout multi-provider

cp .env.example .env
# → Add your API key (any provider)

uv run main.py
```

> `uv` handles all dependency resolution and virtual environment creation automatically — no manual `pip install` required.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Interface                    │
│                   (main.py — REPL)                  │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                   Agent Core                         │
│              (agent.py — Tool Loop)                  │
│                                                      │
│   User Message → Claude API → Tool Selection         │
│   Tool Result  → Claude API → Natural Language       │
└──────┬──────────────┬──────────────┬────────────────┘
       │              │              │
┌──────▼─────┐ ┌──────▼─────┐ ┌─────▼──────┐
│  Equities  │ │   Crypto   │ │ Prediction │
│  Scraper   │ │  Scraper   │ │  Markets   │
│  (Yahoo)   │ │ (CoinGecko)│ │(Polymarket)│
└────────────┘ └────────────┘ └────────────┘
```

The agent operates in a **tool-use loop**: Claude receives each user message, determines which data tools to invoke, executes them against live APIs, and synthesizes the results into a coherent analytical response. This cycle repeats per turn until no further tool calls are needed.

---

## 📁 Project Structure

```
finagent-public/
├── .env.example              # API key configuration template
├── pyproject.toml             # Dependency manifest (uv)
├── main.py                    # Interactive REPL interface
├── agent.py                   # Claude tool-calling engine (9 tools)
├── analysis.py                # Quantitative metrics (Sharpe, drawdown)
└── scrapers/
    ├── stock_scraper.py       # Yahoo Finance — equities
    ├── crypto_scraper.py      # CoinGecko — cryptocurrencies
    └── polymarket_scraper.py  # Polymarket — prediction markets
```

---

## 🔀 Multi-Provider Architecture

> **This branch (`multi-provider`) extends the agent to support 4 LLM providers** through a unified tool-calling interface. Gemini and Grok use OpenAI-compatible endpoints, requiring **zero additional dependencies**.

| Provider | Model | Endpoint | SDK |
|----------|-------|----------|-----|
| 🟣 Anthropic | Claude Sonnet 4 | `api.anthropic.com` | `anthropic` |
| 🟢 OpenAI | GPT-4o | `api.openai.com` | `openai` |
| 🔵 Google | Gemini 2.0 Flash | `generativelanguage.googleapis.com` (OpenAI-compatible) | `openai` |
| ⚫ xAI | Grok 3 | `api.x.ai` (OpenAI-compatible) | `openai` |

### Provider Auto-Detection

The agent scans environment variables in priority order and initializes the appropriate client:

```python
ANTHROPIC_API_KEY → Anthropic SDK → Claude
OPENAI_API_KEY   → OpenAI SDK    → GPT-4o
GEMINI_API_KEY   → OpenAI SDK    → Gemini (custom base_url)
XAI_API_KEY      → OpenAI SDK    → Grok   (custom base_url)
```

Tool definitions are stored in a **provider-agnostic format** and converted at runtime to each provider's schema (Anthropic `input_schema` vs. OpenAI `function.parameters`).

### Configuration

```bash
# .env — set ONE key
ANTHROPIC_API_KEY=sk-ant-...    # Claude (recommended)
OPENAI_API_KEY=sk-...           # GPT-4o
GEMINI_API_KEY=AI...            # Gemini
XAI_API_KEY=xai-...             # Grok

# Optional: override default model
MODEL=claude-sonnet-4-20250514
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API client |
| `openai` | OpenAI / Gemini / Grok API client |
| `yfinance` | Yahoo Finance data |
| `pandas` / `numpy` | Quantitative computation |
| `requests` | HTTP client (CoinGecko, Polymarket) |
| `python-dotenv` | Environment variable management |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
