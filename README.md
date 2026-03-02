<p align="center">
  <h1 align="center">💹 FinAgent</h1>
  <p align="center">
    <strong>Quantitative AI finance agent — multi-provider LLM architecture (Claude, GPT-4, Gemini, Grok) with 16 tool-calling capabilities for equities, crypto, prediction markets, options & algorithmic backtesting</strong>
  </p>
  <p align="center">
    Stocks · Crypto · Prediction Markets · Quantitative Analysis · ML Signals · Backtesting
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/AI-Claude_·_GPT--4_·_Gemini_·_Grok-7C3AED?style=for-the-badge" alt="Multi-Provider">
    <img src="https://img.shields.io/badge/tools-15-22C55E?style=for-the-badge" alt="15 Tools">
    <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT License">
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/data-Yahoo_Finance-7B1FA2?style=flat-square" alt="Yahoo Finance">
    <img src="https://img.shields.io/badge/data-CoinGecko-8DC63F?style=flat-square" alt="CoinGecko">
    <img src="https://img.shields.io/badge/data-Polymarket-FF6B35?style=flat-square" alt="Polymarket">
  </p>
</p>

---

## 📋 Overview

FinAgent is a **terminal-based quantitative analysis assistant** that leverages AI tool-calling to provide institutional-grade financial analysis through natural language. Set one API key, ask a question — the AI autonomously selects from **15 tools**, fetches live data, runs quantitative models, and delivers a synthesized response.

> 📋 **[Full Capabilities Reference →](CAPABILITIES.md)** — detailed breakdown of all 15 tools, analysis modules, and data sources.

**All data sources are free. No API keys needed for data.** You only need one LLM provider key.

---

## ⚡ Capabilities

| Domain | Tools | Source |
|--------|-------|--------|
| **Equities** | `get_stock_quote` · `get_stock_history` · `get_stock_info` | Yahoo Finance |
| **Crypto** | `get_crypto_price` · `get_crypto_top_n` · `search_crypto` | CoinGecko |
| **Prediction Markets** | `get_polymarket_markets` | Polymarket |
| **Quant Analysis** | `analyze_asset` · `analyze_time_series` · `compare_assets` | Computed |
| **Machine Learning** | `run_regression` · `run_clustering_pca` · `run_feature_importance` | Computed |
| **Signals & Backtesting** | `generate_signals` · `backtest_strategy` | Computed |
| **Options** | `get_options_chain` | Yahoo Finance |

### 📈 Analysis Depth
- **Risk Metrics** — Sharpe, Sortino, Calmar ratios, VaR (historical + parametric + Monte Carlo), CVaR
- **Time Series** — ADF/KPSS stationarity tests, ARIMA forecasting, GARCH volatility
- **Machine Learning** — Linear/logistic regression, K-Means clustering, PCA, Random Forest
- **Signals** — Composite buy/sell score from 5 technical indicators (SMA, EMA, RSI, MACD, Bollinger)
- **Backtesting** — SMA/EMA crossover strategies vs. buy-and-hold benchmark

---

## 🖥️ Demo

```
You → What's Apple trading at?
  🔧 get_stock_quote({"ticker": "AAPL"})

Agent → Apple Inc. (AAPL) is currently trading at $264.58.
        Market Cap: $4.05T | P/E: 33.2 | Day Range: $261.30 – $266.12
        52-Week: $164.08 – $267.26 | Volume: 48.2M

You → Run a regression of NVDA against SPY
  🔧 run_regression({"mode": "linear", "target_ticker": "NVDA", "factor_tickers": ["SPY"]})

Agent → NVDA has a beta of 1.82 relative to SPY. R² = 0.41 — SPY explains
        41% of NVDA's return variance. It's a high-beta growth stock.

You → Generate trading signals for Tesla
  🔧 generate_signals({"ticker": "TSLA"})

Agent → TSLA Composite Signal: +0.35 (BUY 🟢)
        ├── SMA Crossover: BUY  (50 > 200)
        ├── EMA Crossover: BUY  (12 > 26)
        ├── RSI: HOLD  (52.3 — neutral zone)
        ├── MACD: BUY  (above signal line)
        └── Bollinger: HOLD  (within bands)

You → Backtest SMA crossover on Nvidia over 5 years
  🔧 backtest_strategy({"ticker": "NVDA", "strategy": "sma"})

Agent → SMA 50/200 Crossover on NVDA (5Y):
        Strategy Return: +380% | Buy-and-Hold: +520%
        Strategy UNDERPERFORMS buy-and-hold ❌
        Sharpe: 1.42 | Max Drawdown: -28% | 12 trades
```

---

## 🚀 Getting Started

### What You'll Need

| Requirement | What It Is | Where to Get It |
|-------------|-----------|-----------------|
| **Python 3.13+** | Programming language | [python.org/downloads](https://www.python.org/downloads/) |
| **uv** | Package manager (handles all dependencies for you) | Step 1 below |
| **An LLM API Key** | Lets the AI brain work — pick **any one** provider | Step 4 below |

### Step 1 · Install `uv` (package manager)

> Skip this if you already have `uv` installed. You can check by running `uv --version`.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Or via Homebrew (macOS):**
```bash
brew install uv
```

### Step 2 · Clone the repository

```bash
git clone https://github.com/Mnfisher93/FinAgent.git
```

### Step 3 · Navigate into the project folder

```bash
cd FinAgent
```

### Step 4 · Set up your API key

You need **one** AI provider key. Pick whichever you prefer:

| Provider | Get Your Key | Free Tier? |
|----------|-------------|------------|
| 🟣 Anthropic (Claude) | [console.anthropic.com](https://console.anthropic.com/) | $5 credit |
| 🟢 OpenAI (GPT-4) | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Pay-as-you-go |
| 🔵 Google (Gemini) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | ✅ Free |
| ⚫ xAI (Grok) | [console.x.ai](https://console.x.ai/) | Free credits |

First, create your config file from the template:

```bash
cp .env.example .env
```

Now open the `.env` file and paste your API key. Choose **one** of these methods:

<details>
<summary><strong>Option A — Open in a text editor (easiest)</strong></summary>

**macOS:**
```bash
open .env
```

**Windows:**
```bash
notepad .env
```

**Linux:**
```bash
nano .env
```

This opens the file. Find the line for your provider and replace `your-api-key-here` with your actual key. Save and close.

</details>

<details>
<summary><strong>Option B — Set it directly from the terminal (one command)</strong></summary>

Pick the command that matches your provider:

**Claude (Anthropic):**
```bash
echo 'ANTHROPIC_API_KEY=sk-ant-your-key-here' > .env
```

**GPT-4 (OpenAI):**
```bash
echo 'OPENAI_API_KEY=sk-your-key-here' > .env
```

**Gemini (Google):**
```bash
echo 'GEMINI_API_KEY=your-key-here' > .env
```

**Grok (xAI):**
```bash
echo 'XAI_API_KEY=xai-your-key-here' > .env
```

</details>

### Step 5 · Run the agent

```bash
uv run main.py
```

> `uv` automatically installs all dependencies and creates a virtual environment for you — no manual `pip install` needed. The first run may take 30–60 seconds to set up.

---

## 🤖 Multi-Provider Support

The agent auto-detects which API key you've set and uses that provider:

| Provider | Model | Setup |
|----------|-------|-------|
| 🟣 Anthropic | Claude Sonnet 4 | `ANTHROPIC_API_KEY=...` |
| 🟢 OpenAI | GPT-4o | `OPENAI_API_KEY=...` |
| 🔵 Google | Gemini 2.0 Flash | `GEMINI_API_KEY=...` |
| ⚫ xAI | Grok 3 Mini | `XAI_API_KEY=...` |

All providers share the **same 15-tool architecture** — zero extra dependencies. Gemini and Grok use OpenAI-compatible endpoints, so the same two SDKs (`anthropic` + `openai`) handle everything.

---

## 🏗️ Architecture

```
finagent/
├── .env.example                  # API key template
├── pyproject.toml                # Dependencies (uv)
├── CAPABILITIES.md               # Full capabilities reference
├── main.py                       # Interactive chat REPL
├── agent.py                      # Multi-provider tool-calling engine (15 tools)
├── scrapers/
│   ├── stock_scraper.py          # Any NYSE/NASDAQ ticker (Yahoo Finance)
│   ├── crypto_scraper.py         # CoinGecko + optional CoinMarketCap
│   └── polymarket_scraper.py     # Prediction market data
└── analysis/
    ├── quant_analysis.py         # Returns, risk metrics, VaR, correlations
    ├── time_series.py            # ARIMA, GARCH, stationarity tests
    ├── advanced_stats.py         # Regression, clustering, PCA, Random Forest
    ├── signals.py                # Trading signal generators (5 indicators)
    ├── backtest.py               # Strategy backtesting engine
    └── visualizations.py         # Publication-quality charts
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API |
| `openai` | GPT-4 / Gemini / Grok |
| `yfinance` | Yahoo Finance data |
| `pandas` + `numpy` | Data manipulation |
| `scipy` + `statsmodels` | Statistical analysis + ARIMA |
| `arch` | GARCH volatility modeling |
| `scikit-learn` | ML (regression, clustering, PCA, Random Forest) |
| `ta` | Technical indicators (SMA, EMA, RSI, MACD, Bollinger) |
| `matplotlib` + `seaborn` | Charts |
| `requests` | HTTP (CoinGecko, Polymarket) |
| `python-dotenv` | Environment configuration |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with Claude · GPT-4 · Gemini · Grok · Yahoo Finance · CoinGecko · Polymarket
</p>
