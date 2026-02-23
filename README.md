<p align="center">
  <h1 align="center">💹 FinAgent</h1>
  <p align="center">
    <strong>Talk to a quantative financial analyst in your terminal, leveraging the AI of your choice.</strong>
  </p>
  <p align="center">
    Ask about any stock, cryptocurrency, or prediction market — powered by Claude.
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.13+-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/AI-Claude-blueviolet" alt="Claude">
    <img src="https://img.shields.io/badge/data-Yahoo%20Finance-purple" alt="Yahoo Finance">
    <img src="https://img.shields.io/badge/data-CoinGecko-green" alt="CoinGecko">
    <img src="https://img.shields.io/badge/data-Polymarket-orange" alt="Polymarket">
  </p>
</p>

---

## What is this?

FinAgent is a terminal-based AI assistant that can look up **any stock or crypto in real-time** and answer your financial questions using Claude's tool-calling capabilities. Instead of manually checking Yahoo Finance or CoinGecko, just ask in plain English.

### Demo

```
You → What's Apple trading at?
  🔧 get_stock_quote({"ticker": "AAPL"})
Agent → Apple (AAPL) is currently trading at $264.58.
        Market cap: $4.05T | PE: 33.2 | Day range: $261.30 - $266.12

You → Compare Tesla, Nvidia, and Microsoft over the last year
  🔧 compare_assets({"tickers": ["TSLA", "NVDA", "MSFT"]})
Agent → Correlation matrix of daily returns:
        TSLA-NVDA: 0.52 | TSLA-MSFT: 0.38 | NVDA-MSFT: 0.68

You → Price of Solana?
  🔧 get_crypto_price({"coin_id": "solana"})
Agent → Solana (SOL) is at $172.40 | 24h: -1.1% | 7d: +8.3%
        Market cap: $84.2B (#5) | ATH: $293.31
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- An [Anthropic API key](https://console.anthropic.com/)

### Setup

```bash
git clone https://github.com/Mnfisher93/ClaudeFintechAgent.git
cd ClaudeFintechAgent/finagent-public

# Create your .env file
cp .env.example .env
# Edit .env → paste your ANTHROPIC_API_KEY

# Run (uv installs everything automatically)
uv run main.py
```

That's it. No `pip install`, no virtual environment setup — `uv` handles everything.

## 🏗️ How It Works

```
You type a question
        ↓
Claude reads your message
        ↓
Claude decides which tool(s) to call
        ↓
Tool fetches real data (Yahoo Finance / CoinGecko / Polymarket)
        ↓
Data is sent back to Claude
        ↓
Claude writes a natural language response
```

The agent has **9 tools** it can use:

| Tool | What it does |
|------|-------------|
| `get_stock_quote` | Real-time stock price, volume, market cap |
| `get_stock_history` | Historical prices over any period |
| `get_stock_info` | Company fundamentals and financials |
| `get_crypto_price` | Live crypto price from CoinGecko |
| `get_crypto_top_n` | Top cryptos by market cap |
| `search_crypto` | Find any coin by name or symbol |
| `get_polymarket_markets` | Prediction market probabilities |
| `analyze_asset` | Returns, volatility, Sharpe, max drawdown |
| `compare_assets` | Cross-asset correlation analysis |

## 📁 Project Structure

```
finagent/
├── .env.example          # API key template
├── pyproject.toml        # Dependencies (managed by uv)
├── main.py               # Chat interface
├── agent.py              # Claude tool-calling engine
├── analysis.py           # Basic financial metrics
└── scrapers/
    ├── stock_scraper.py   # Yahoo Finance (any US stock)
    ├── crypto_scraper.py  # CoinGecko (any cryptocurrency)
    └── polymarket_scraper.py  # Prediction markets
```

## 🔑 Configuration

Create a `.env` file (or copy `.env.example`):

```
ANTHROPIC_API_KEY=your-api-key-here
```

## Built With

- [Claude](https://anthropic.com) (Anthropic) — AI reasoning and tool-calling
- [Yahoo Finance](https://finance.yahoo.com/) — US stock data
- [CoinGecko](https://www.coingecko.com/) — Cryptocurrency data
- [Polymarket](https://polymarket.com/) — Prediction market data
- [uv](https://docs.astral.sh/uv/) — Python package management

## License

MIT
