# 📋 FinAgent — Full Capabilities

> **16 autonomous AI tools** across real-time market data, options chains, quantitative risk analysis, machine learning, trading signals, and strategy backtesting — powered by your choice of LLM provider.

---

## 🤖 Multi-Provider LLM Architecture

FinAgent uses a **tool-calling loop** — the LLM receives your natural language question, autonomously selects which tools to invoke, executes them against live APIs, and synthesizes the results into a coherent analytical response.

| Provider | Model | SDK | Setup |
|----------|-------|-----|-------|
| Anthropic Claude | claude-sonnet-4-20250514 | `anthropic` | Set `ANTHROPIC_API_KEY` |
| OpenAI GPT-4 | gpt-4o | `openai` | Set `OPENAI_API_KEY` |
| Google Gemini | gemini-2.0-flash | `openai` (compatible) | Set `GEMINI_API_KEY` |
| xAI Grok | grok-3-mini-fast | `openai` (compatible) | Set `XAI_API_KEY` |

Set **one** key in your `.env` — the agent auto-detects your provider.

---

## 📊 All 16 Tools

### Market Data (Tools 1–7)

| # | Tool | Description | Source | API Key? |
|---|------|-------------|--------|----------|
| 1 | `get_stock_quote` | Live price, volume, market cap, PE ratio, 52-week range, 50/200-day MAs | Yahoo Finance | ❌ Free |
| 2 | `get_stock_history` | Historical OHLCV with auto-computed SMA, EMA, RSI, MACD, Bollinger Bands | Yahoo Finance | ❌ Free |
| 3 | `get_stock_info` | Company fundamentals — sector, industry, margins, ROE, beta, dividends, FCF | Yahoo Finance | ❌ Free |
| 4 | `get_crypto_price` | Live price, market cap, volume, 24h/7d/30d changes, ATH/ATL, supply | CoinGecko | ❌ Free |
| 5 | `get_crypto_top_n` | Top N cryptocurrencies ranked by market cap (up to 250) | CoinGecko | ❌ Free |
| 6 | `search_crypto` | Fuzzy search any cryptocurrency by name or symbol | CoinGecko | ❌ Free |
| 7 | `get_polymarket_markets` | Active prediction markets — probabilities, trading volume, liquidity | Polymarket | ❌ Free |

### Quantitative Analysis (Tools 8–10)

| # | Tool | Description | Metrics |
|---|------|-------------|---------|
| 8 | `analyze_asset` | Comprehensive risk/return analysis | Annualized return, volatility, Sharpe, Sortino, Calmar, max drawdown, VaR (historical + parametric + Monte Carlo), CVaR, skewness, kurtosis |
| 9 | `analyze_time_series` | Time series modeling | ADF + KPSS stationarity tests, ARIMA (auto-order), GARCH(1,1) volatility |
| 10 | `compare_assets` | Multi-asset portfolio analysis | Cross-correlation matrix, annualized covariance matrix |

### Machine Learning (Tools 11–13)

| # | Tool | Description | Methods |
|---|------|-------------|---------|
| 11 | `run_regression` | Factor analysis & direction prediction | Linear (CAPM beta, R², multi-factor) · Logistic (next-day up/down) |
| 12 | `run_clustering_pca` | Asset grouping & risk decomposition | K-Means clustering · PCA principal risk factors |
| 13 | `run_feature_importance` | Factor driver identification | Random Forest feature importance ranking |

### Trading Signals & Backtesting (Tools 14–15)

| # | Tool | Description | Details |
|---|------|-------------|---------|
| 14 | `generate_signals` | Composite buy/sell signals | SMA crossover, EMA crossover, RSI, MACD, Bollinger → weighted score (−1.0 to +1.0) |
| 15 | `backtest_strategy` | Strategy backtesting engine | SMA/EMA crossover or compare all — vs. buy-and-hold, custom windows |

### Options (Tool 16)

| # | Tool | Description | Source | API Key? |
|---|------|-------------|--------|----------|
| 16 | `get_options_chain` | Full options chain: calls, puts, put/call ratio, IV summary, ATM options, most active contracts, sentiment | Yahoo Finance | ❌ Free |

---

## 🔌 Data Sources

| Source | Coverage | API Key | Rate Limits |
|--------|----------|---------|-------------|
| **Yahoo Finance** | Any NYSE/NASDAQ stock | None needed | ~2,000 req/hr (unofficial) |
| **CoinGecko** | 10,000+ cryptocurrencies | None needed | 10–30 req/min (free tier) |
| **Polymarket** | Active prediction markets | None needed | Public API |

> **All data sources are free. No API keys required for data.** You only need one LLM provider key.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API |
| `openai` | GPT-4 / Gemini / Grok (multi-provider) |
| `pandas` + `numpy` | Data manipulation |
| `yfinance` | Yahoo Finance API |
| `requests` | HTTP requests (CoinGecko, Polymarket) |
| `scipy` | Statistical tests |
| `statsmodels` | ARIMA modeling |
| `arch` | GARCH volatility |
| `scikit-learn` | ML (regression, clustering, PCA, Random Forest) |
| `ta` | Technical indicators |
| `matplotlib` + `seaborn` | Charts |
