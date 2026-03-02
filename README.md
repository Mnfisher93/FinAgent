<p align="center">
  <h1 align="center">💹 FinAgent</h1>
  <p align="center">
    <strong>Quantitative AI finance agent — multi-provider LLM architecture (Claude, GPT-4, Gemini, Grok) with 16 tool-calling capabilities for equities, crypto, options, prediction markets & algorithmic backtesting</strong>
  </p>
  <p align="center">
    Stocks · Crypto · Options · Prediction Markets · Quantitative Analysis · ML Signals · Backtesting
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/AI-Claude_·_GPT--4_·_Gemini_·_Grok-7C3AED?style=for-the-badge" alt="Multi-Provider">
    <img src="https://img.shields.io/badge/tools-16-22C55E?style=for-the-badge" alt="16 Tools">
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

FinAgent is a **terminal-based quantitative analysis assistant** that leverages AI tool-calling to provide institutional-grade financial analysis through natural language. Ask a question — the AI autonomously selects from **16 tools**, fetches live data, runs quantitative models, and delivers a synthesized response.

> 📋 **[Full Capabilities Reference →](CAPABILITIES.md)** — detailed breakdown of all 16 tools, analysis modules, and data sources.

**All data sources are free. No API keys needed for data.** You only need one LLM provider key.

---

## ⚡ Capabilities

| Domain | Tools | Source |
|--------|-------|--------|
| **Equities** | `get_stock_quote` · `get_stock_history` · `get_stock_info` | Yahoo Finance |
| **Crypto** | `get_crypto_price` · `get_crypto_top_n` · `search_crypto` | CoinGecko |
| **Options** | `get_options_chain` | Yahoo Finance |
| **Prediction Markets** | `get_polymarket_markets` | Polymarket |
| **Quant Analysis** | `analyze_asset` · `analyze_time_series` · `compare_assets` | Computed |
| **Machine Learning** | `run_regression` · `run_clustering_pca` · `run_feature_importance` | Computed |
| **Signals & Backtesting** | `generate_signals` · `backtest_strategy` | Computed |

### 📈 Analysis Depth
- **Risk Metrics** — Sharpe, Sortino, Calmar ratios, VaR (historical + parametric + Monte Carlo), CVaR
- **Time Series** — ADF/KPSS stationarity tests, ARIMA, GARCH(1,1) volatility modeling
- **Machine Learning** — Linear/logistic regression, K-Means clustering, PCA, Random Forest feature importance
- **Trading Signals** — 5-indicator composite score (SMA, EMA, RSI, MACD, Bollinger)
- **Options** — Put/call ratios, IV summary, ATM options, most active contracts, sentiment analysis

---

## 🎯 Demo

```
You → What's Apple trading at?
  🔧 get_stock_quote({"ticker": "AAPL"})

Agent → AAPL is at $264.72 (+1.2%). Market cap: $4.0T.
        52W range: $179.27 – $266.19. Slightly below the 50-day MA of $266.23.

You → Show me Tesla's options chain
  🔧 get_options_chain({"ticker": "TSLA"})

Agent → TSLA Options (expires 2026-03-06):
        Calls: 45 | Puts: 42 | P/C Ratio: 0.37 (BULLISH 🟢)
        Mean Call IV: 68.2% | Mean Put IV: 72.1%
        Most active: $350C (vol: 12,341) | $300P (vol: 8,229)

You → Backtest SMA crossover on Nvidia over 5 years
  🔧 backtest_strategy({"ticker": "NVDA", "strategy": "sma"})

Agent → SMA 50/200 Crossover on NVDA (5Y):
        Strategy Return: +380% | Buy-and-Hold: +520%
        Sharpe: 1.42 | Max Drawdown: -28% | 12 trades
```

---

## 🚀 Getting Started

### Step 1 · Install `uv`

> Skip this if you already have it. Check with `uv --version`.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2 · Run FinAgent

```bash
git clone https://github.com/Mnfisher93/FinAgent.git && cd FinAgent && uv run main.py
```

**That's it.** The program handles everything:

- 📦 Installs all dependencies automatically
- 🔑 Walks you through API key setup on first launch
- 🤖 Lets you pick your provider and model
- 💾 Saves your settings — never asks again

> **Tip:** Google Gemini has a generous **free tier** — great for trying FinAgent at zero cost. Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

---

## 🤖 Multi-Provider & Multi-Model Support

Choose your provider **and** your model. Every provider supports multiple models:

| Provider | Available Models | Get a Key |
|----------|-----------------|-----------|
| 🟣 **Anthropic** | `claude-sonnet-4` · `claude-haiku-3` | [console.anthropic.com](https://console.anthropic.com/) |
| 🟢 **OpenAI** | `gpt-4o` · `gpt-4o-mini` · `gpt-4-turbo` | [platform.openai.com](https://platform.openai.com/api-keys) |
| 🔵 **Google** | `gemini-2.5-flash` · `gemini-2.0-flash` · `gemini-2.5-pro` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| ⚫ **xAI** | `grok-4` · `grok-3` · `grok-3-fast` | [console.x.ai](https://console.x.ai/) |

All providers share the **same 16-tool architecture**. Gemini and Grok use OpenAI-compatible endpoints, so just two SDKs (`anthropic` + `openai`) handle everything.

---

## 🏗️ Architecture

```
FinAgent/
├── main.py                       # Interactive chat + first-run setup
├── agent.py                      # Multi-provider tool-calling engine (16 tools)
├── CAPABILITIES.md               # Full capabilities reference
├── scrapers/
│   ├── stock_scraper.py          # Any NYSE/NASDAQ ticker (Yahoo Finance)
│   ├── crypto_scraper.py         # 10,000+ cryptocurrencies (CoinGecko)
│   ├── polymarket_scraper.py     # Prediction market data
│   └── options_scraper.py        # Options chains, IV, Greeks (Yahoo Finance)
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
| `openai` | GPT-4 / Gemini / Grok (multi-provider) |
| `yfinance` | Yahoo Finance data (stocks + options) |
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
