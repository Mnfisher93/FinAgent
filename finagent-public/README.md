# 📊 Quantitative Fintech Agent

> Talk to an AI financial analyst in your terminal.
> Ask about any stock, cryptocurrency, or prediction market.

**Designed for Claude** — also works with GPT-4, Gemini, and Grok. Just set your API key and go.

| Provider | Model | Key Variable |
|----------|-------|-------------|
| 🟣 Anthropic | Claude Sonnet | `ANTHROPIC_API_KEY` |
| 🟢 OpenAI | GPT-4o | `OPENAI_API_KEY` |
| 🔵 Google | Gemini 2.0 Flash | `GEMINI_API_KEY` |
| ⚫ xAI | Grok 3 | `XAI_API_KEY` |

## ✨ Features

| Category | What You Can Do |
|----------|----------------|
| **US Stocks** | Real-time quotes, historical data, company fundamentals for any NYSE/NASDAQ ticker |
| **Crypto** | Price, market cap, volume for any cryptocurrency via CoinGecko |
| **Prediction Markets** | Browse active Polymarket markets with probabilities |
| **Analysis** | Returns, volatility, Sharpe ratio, max drawdown |
| **Comparison** | Cross-asset correlation matrices |

### Demo

```
You → What's AAPL trading at?

  🔧 get_stock_quote({"ticker": "AAPL"})

Agent → **Apple Inc. (AAPL)** is currently trading at **$264.58**.

  • Previous Close: $262.83 (+0.67%)
  • Day Range: $261.25 – $265.10
  • 52-Week Range: $164.08 – $267.26
  • Market Cap: $4.02T
  • P/E Ratio: 41.3
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- An API key from **any** supported provider

### Setup

```bash
git clone https://github.com/Mnfisher93/ClaudeFintechAgent.git
cd ClaudeFintechAgent/finagent-public

uv sync

cp .env.example .env
# Edit .env → paste your API key (any provider)

uv run main.py
```

The agent auto-detects which key you've set and uses that provider.

## 🏗️ How It Works

```
You type a question
    ↓
LLM reads your message
    ↓
LLM decides which tool(s) to call
    ↓
Tool fetches real data (Yahoo Finance / CoinGecko / Polymarket)
    ↓
Data is sent back to the LLM
    ↓
LLM writes a natural language response
```

The agent has **9 tools**: `get_stock_quote`, `get_stock_history`, `get_stock_info`, `get_crypto_price`, `get_crypto_top_n`, `search_crypto`, `get_polymarket_markets`, `analyze_asset`, `compare_assets`

## 📁 Project Structure

```
├── .env.example          # API key template (4 providers)
├── pyproject.toml         # Dependencies (managed by uv)
├── main.py                # Chat interface
├── agent.py               # Multi-provider LLM engine with tool-calling
├── analysis.py            # Financial metrics (Sharpe, drawdown, etc.)
└── scrapers/
    ├── stock_scraper.py   # Yahoo Finance (any US stock)
    ├── crypto_scraper.py  # CoinGecko (any cryptocurrency)
    └── polymarket_scraper.py  # Prediction markets
```

## 🔑 Configuration

Create a `.env` file (or copy `.env.example`) and set **one** key:

```bash
# Pick your provider:
ANTHROPIC_API_KEY=sk-ant-...    # Claude (recommended)
OPENAI_API_KEY=sk-...           # GPT-4o
GEMINI_API_KEY=AI...            # Gemini
XAI_API_KEY=xai-...             # Grok
```

Optionally override the model:
```bash
MODEL=claude-sonnet-4-20250514
```

## Built With

- [Anthropic Claude](https://anthropic.com) / [OpenAI](https://openai.com) / [Google Gemini](https://ai.google.dev/) / [xAI Grok](https://x.ai/) — AI reasoning & tool-calling
- [Yahoo Finance](https://finance.yahoo.com/) — US stock data
- [CoinGecko](https://www.coingecko.com/) — Cryptocurrency data
- [Polymarket](https://polymarket.com/) — Prediction market data
- [uv](https://docs.astral.sh/uv/) — Python package management

## License

MIT
