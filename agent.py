"""
Quantitative Fintech Agent — AI-Powered Financial Analyst

Supports multiple LLM providers (designed for Claude, works with all):
  • Anthropic (Claude)  — set ANTHROPIC_API_KEY
  • OpenAI (GPT-4o)     — set OPENAI_API_KEY
  • Google (Gemini)     — set GEMINI_API_KEY
  • xAI (Grok)          — set XAI_API_KEY

The agent auto-detects which key you have and uses that provider.

15 tools: market data (stocks, crypto, prediction markets), quantitative
analysis, time series modeling, machine learning, trading signals,
strategy backtesting, and options chain analysis.
"""

import json
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from scrapers.stock_scraper import get_stock_quote, get_stock_history, get_stock_info
from scrapers.crypto_scraper import get_crypto_price, get_crypto_top_n, search_crypto
from scrapers.polymarket_scraper import fetch_polymarket_data
from scrapers.options_scraper import get_options_chain
from analysis.quant_analysis import QuantAnalyzer
from analysis.time_series import TimeSeriesAnalyzer
from analysis.advanced_stats import (
    run_linear_regression, run_multi_regression,
    run_logistic_regression, run_clustering, run_pca, run_feature_importance,
)
from analysis.signals import composite_signal
from analysis.backtest import backtest_sma_crossover, backtest_ema_crossover, compare_strategies

load_dotenv(Path(__file__).resolve().parent / ".env")

MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are a highly knowledgeable financial analyst AI assistant. You have access to real-time and historical data through specialized tools for:

• **US Stocks** (NYSE, NASDAQ) — quotes, historical prices, company fundamentals
• **Cryptocurrencies** — prices via CoinGecko for any coin (Bitcoin, Ethereum, Solana, etc.)
• **Prediction Markets** — Polymarket data on active markets
• **Quantitative Analysis** — Sharpe ratio, VaR, volatility, ARIMA, GARCH, correlations
• **Advanced Statistics** — linear/logistic regression, K-Means clustering, PCA, Random Forest
• **Trading Signals** — composite buy/sell scoring from 5 technical indicators
• **Backtesting** — strategy performance vs. buy-and-hold benchmarks

When answering:
1. Always use your tools to fetch real data — never make up prices or statistics.
2. Present numbers clearly with proper formatting (percentages, currency, etc.).
3. Provide context and interpretation, not just raw data.
4. For analysis questions, explain what the metrics mean in practical terms.
5. If asked to compare assets, use your compare_assets tool for correlation analysis.
6. For regression/clustering tasks, explain the statistical results in practical terms.
7. Be concise but thorough — a graduate-level financial analyst tone.

You are talking to a user in a terminal chat interface. Format your responses with clear structure using markdown-like formatting (bold with **, headers with #, etc.) that still reads well in a terminal."""

# ============================================================================
# Tool Definitions (provider-agnostic, converted per-provider at runtime)
# ============================================================================

TOOLS_SPEC = [
    {
        "name": "get_stock_quote",
        "description": "Current stock quote for any US stock: price, volume, market cap, PE ratio, 52-week range, moving averages.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker (e.g., 'AAPL', 'MSFT', 'TSLA')"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_stock_history",
        "description": "Historical OHLCV data for any US stock with auto-computed technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands).",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "period": {"type": "string", "description": "Period: '1mo', '3mo', '6mo', '1y', '2y', '5y'", "default": "1y"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_stock_info",
        "description": "Company fundamentals: sector, industry, description, PE ratio, margins, beta, dividends, debt-to-equity, FCF.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_crypto_price",
        "description": "Current crypto price, market cap, volume, and price changes (24h/7d/30d). Supports any coin via CoinGecko.",
        "parameters": {
            "type": "object",
            "properties": {
                "coin_id": {"type": "string", "description": "Crypto name or symbol (e.g., 'bitcoin', 'eth', 'solana')"}
            },
            "required": ["coin_id"]
        }
    },
    {
        "name": "get_crypto_top_n",
        "description": "Top N cryptocurrencies by market cap with prices and 24h changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of top coins (default: 20)", "default": 20}
            },
            "required": []
        }
    },
    {
        "name": "search_crypto",
        "description": "Search for a cryptocurrency by name or symbol. Returns matching coins with IDs and market cap ranks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_polymarket_markets",
        "description": "Active Polymarket prediction markets with outcome probabilities, volume, and liquidity.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of markets (default: 20)", "default": 20},
                "min_volume": {"type": "number", "description": "Minimum volume in USD (default: 10000)", "default": 10000}
            },
            "required": []
        }
    },
    {
        "name": "analyze_asset",
        "description": "Comprehensive quantitative analysis: annualized return, volatility, Sharpe, Sortino, Calmar ratios, max drawdown, VaR (historical + parametric + Monte Carlo), CVaR, skewness, kurtosis.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker (e.g., 'AAPL', 'BTC-USD')"},
                "period": {"type": "string", "description": "Analysis period: '6mo', '1y', '2y', '5y'", "default": "1y"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "analyze_time_series",
        "description": "Time series analysis: stationarity tests (ADF, KPSS), ARIMA model fitting, and GARCH volatility modeling. Returns model parameters, AIC/BIC, and interpretations.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker or crypto-USD pair"},
                "period": {"type": "string", "description": "Data period for analysis", "default": "2y"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "compare_assets",
        "description": "Compare multiple assets: compute cross-correlation matrix and annualized covariance matrix of daily returns.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "List of tickers to compare"},
                "period": {"type": "string", "description": "Data period", "default": "1y"}
            },
            "required": ["tickers"]
        }
    },
    {
        "name": "run_regression",
        "description": "Regression analysis. Linear regression finds beta/R² between two assets (like CAPM). Logistic regression predicts next-day up/down direction using lagged returns.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["linear", "logistic"], "description": "'linear' for beta/factor analysis, 'logistic' for direction prediction"},
                "target_ticker": {"type": "string", "description": "Target asset ticker (Y variable)"},
                "factor_tickers": {"type": "array", "items": {"type": "string"}, "description": "Factor tickers (X variables) — for linear regression. For logistic, leave empty."},
                "period": {"type": "string", "description": "Data period", "default": "2y"}
            },
            "required": ["mode", "target_ticker"]
        }
    },
    {
        "name": "run_clustering_pca",
        "description": "Cluster assets by behavior patterns (K-Means) and identify principal risk factors (PCA). Useful for portfolio construction.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "List of tickers to analyze (need at least 4)"},
                "n_clusters": {"type": "integer", "description": "Number of clusters (default: 3)", "default": 3},
                "period": {"type": "string", "description": "Data period", "default": "1y"}
            },
            "required": ["tickers"]
        }
    },
    {
        "name": "run_feature_importance",
        "description": "Use Random Forest to identify which assets/factors most influence a target asset's price direction. Shows feature importance scores.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_ticker": {"type": "string", "description": "Target asset to analyze"},
                "feature_tickers": {"type": "array", "items": {"type": "string"}, "description": "Potential driver tickers (e.g., ['SPY', 'QQQ', 'DXY', 'GLD'])"},
                "period": {"type": "string", "description": "Data period", "default": "2y"}
            },
            "required": ["target_ticker", "feature_tickers"]
        }
    },
    {
        "name": "generate_signals",
        "description": "Generate buy/sell trading signals. Runs 5 indicators (SMA crossover, EMA crossover, RSI, MACD, Bollinger Bands) and produces a composite signal score from -1 (strong sell) to +1 (strong buy).",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol (e.g., 'AAPL', 'NVDA', 'BTC-USD')"},
                "period": {"type": "string", "description": "Data period (need at least '1y' for SMA 200)", "default": "2y"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "backtest_strategy",
        "description": "Backtest a moving average crossover strategy against historical data. Compares strategy returns vs. buy-and-hold. Can test SMA or EMA crossovers, or compare multiple strategies.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker to backtest on"},
                "strategy": {"type": "string", "enum": ["sma", "ema", "compare_all"], "description": "'sma' for SMA crossover, 'ema' for EMA crossover, 'compare_all' to test all"},
                "short_window": {"type": "integer", "description": "Short MA period (default: 50 for SMA, 12 for EMA)"},
                "long_window": {"type": "integer", "description": "Long MA period (default: 200 for SMA, 26 for EMA)"},
                "period": {"type": "string", "description": "Historical data period", "default": "5y"},
                "initial_capital": {"type": "number", "description": "Starting capital in USD (default: 10000)", "default": 10000}
            },
            "required": ["ticker", "strategy"]
        }
    },
    {
        "name": "get_options_chain",
        "description": "Get options chain data for any US stock. Returns calls, puts, put/call ratio, implied volatility summary, ATM options, most active contracts, and sentiment. Uses Yahoo Finance — no API key needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker (e.g., 'AAPL', 'TSLA', 'NVDA')"},
                "expiration": {"type": "string", "description": "Expiration date (YYYY-MM-DD). If omitted, uses the nearest expiration."}
            },
            "required": ["ticker"]
        }
    },
]


# ============================================================================
# Tool Execution (same for all providers)
# ============================================================================

def execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return JSON result."""
    try:
        if name == "get_stock_quote":
            return json.dumps(get_stock_quote(args["ticker"]), indent=2, default=str)

        elif name == "get_stock_history":
            df = get_stock_history(args["ticker"], period=args.get("period", "1y"))
            if df.empty:
                return json.dumps({"error": f"No data for {args['ticker']}"})
            summary = {
                "ticker": args["ticker"].upper(),
                "period": args.get("period", "1y"),
                "rows": len(df),
                "date_range": f"{df.index.min()} to {df.index.max()}",
                "latest_close": float(df["Close"].iloc[-1]),
                "period_high": float(df["High"].max()),
                "period_low": float(df["Low"].min()),
                "avg_volume": int(df["Volume"].mean()),
                "total_return": f"{((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1):.2%}",
                "last_5_days": df[["Close", "Volume"]].tail(5).to_dict(),
            }
            if "RSI_14" in df.columns:
                summary["current_rsi"] = float(df["RSI_14"].iloc[-1]) if pd.notna(df["RSI_14"].iloc[-1]) else None
            if "MACD" in df.columns:
                summary["current_macd"] = float(df["MACD"].iloc[-1]) if pd.notna(df["MACD"].iloc[-1]) else None
            return json.dumps(summary, indent=2, default=str)

        elif name == "get_stock_info":
            return json.dumps(get_stock_info(args["ticker"]), indent=2, default=str)

        elif name == "get_crypto_price":
            return json.dumps(get_crypto_price(args["coin_id"]), indent=2, default=str)

        elif name == "get_crypto_top_n":
            return json.dumps(get_crypto_top_n(args.get("n", 20)), indent=2, default=str)

        elif name == "search_crypto":
            return json.dumps(search_crypto(args["query"]), indent=2, default=str)

        elif name == "get_polymarket_markets":
            limit = args.get("limit", 20)
            min_volume = args.get("min_volume", 10000)
            df = fetch_polymarket_data(limit=limit, min_volume=min_volume)
            if df.empty:
                return json.dumps({"message": "No markets found matching criteria."})
            records = df[["question", "outcome_prices", "volume", "volume_24hr", "liquidity"]].head(limit).to_dict(orient="records")
            return json.dumps(records, indent=2, default=str)

        elif name == "analyze_asset":
            import yfinance as yf
            ticker = args["ticker"]
            period = args.get("period", "1y")
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                return json.dumps({"error": f"No data for {ticker}"})
            quant = QuantAnalyzer()
            result = quant.summary(df["Close"], name=ticker)
            return json.dumps(result, indent=2, default=str)

        elif name == "analyze_time_series":
            import yfinance as yf
            ticker = args["ticker"]
            period = args.get("period", "2y")
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                return json.dumps({"error": f"No data for {ticker}"})
            ts = TimeSeriesAnalyzer()
            result = ts.full_analysis(df["Close"], name=ticker)
            serializable = {
                "name": result["name"],
                "stationarity": {
                    "adf_p_value": result["stationarity"]["adf"]["p_value"],
                    "adf_stationary": result["stationarity"]["adf"]["is_stationary"],
                    "adf_conclusion": result["stationarity"]["adf"]["conclusion"],
                    "kpss_p_value": result["stationarity"]["kpss"]["p_value"],
                    "kpss_stationary": result["stationarity"]["kpss"]["is_stationary"],
                    "kpss_conclusion": result["stationarity"]["kpss"]["conclusion"],
                    "interpretation": result["stationarity"]["interpretation"],
                },
                "arima": {
                    "order": str(result["arima"]["order"]),
                    "aic": result["arima"]["aic"],
                    "bic": result["arima"]["bic"],
                    "coefficients": {k: float(v) for k, v in result["arima"]["coefficients"].items()},
                },
                "garch": {
                    "model_type": result["garch"]["model_type"],
                    "distribution": result["garch"]["distribution"],
                    "aic": result["garch"]["aic"],
                    "bic": result["garch"]["bic"],
                    "params": {k: float(v) for k, v in result["garch"]["params"].items()},
                },
            }
            return json.dumps(serializable, indent=2, default=str)

        elif name == "compare_assets":
            import yfinance as yf
            tickers = args["tickers"]
            period = args.get("period", "1y")
            price_dict = {}
            for t in tickers:
                df = yf.Ticker(t).history(period=period)
                if not df.empty:
                    price_dict[t.upper()] = df["Close"]
            if len(price_dict) < 2:
                return json.dumps({"error": "Need at least 2 assets with data to compare."})
            quant = QuantAnalyzer()
            corr = quant.correlation_matrix(price_dict)
            cov = quant.covariance_matrix(price_dict)
            result = {
                "correlation_matrix": corr.to_dict(),
                "covariance_matrix_annualized": cov.to_dict(),
                "assets": list(price_dict.keys()),
                "period": period,
            }
            return json.dumps(result, indent=2, default=str)

        elif name == "run_regression":
            import yfinance as yf
            mode = args["mode"]
            target = args["target_ticker"]
            period = args.get("period", "2y")
            target_df = yf.Ticker(target).history(period=period)
            if target_df.empty:
                return json.dumps({"error": f"No data for {target}"})

            if mode == "logistic":
                result = run_logistic_regression(target_df["Close"], name=target)
                return json.dumps(result, indent=2, default=str)
            else:
                factor_tickers = args.get("factor_tickers", [])
                if not factor_tickers:
                    return json.dumps({"error": "Linear regression needs factor_tickers"})
                if len(factor_tickers) == 1:
                    factor_df = yf.Ticker(factor_tickers[0]).history(period=period)
                    if factor_df.empty:
                        return json.dumps({"error": f"No data for {factor_tickers[0]}"})
                    result = run_linear_regression(
                        target_df["Close"], factor_df["Close"],
                        y_name=target, x_name=factor_tickers[0]
                    )
                else:
                    x_dict = {}
                    for t in factor_tickers:
                        fdf = yf.Ticker(t).history(period=period)
                        if not fdf.empty:
                            x_dict[t] = fdf["Close"]
                    result = run_multi_regression(target_df["Close"], x_dict, y_name=target)
                return json.dumps(result, indent=2, default=str)

        elif name == "run_clustering_pca":
            import yfinance as yf
            tickers = args["tickers"]
            n_clusters = args.get("n_clusters", 3)
            period = args.get("period", "1y")
            price_dict = {}
            for t in tickers:
                df = yf.Ticker(t).history(period=period)
                if not df.empty:
                    price_dict[t.upper()] = df["Close"]
            if len(price_dict) < 3:
                return json.dumps({"error": "Need at least 3 assets with data."})
            cluster_result = run_clustering(price_dict, n_clusters=n_clusters)
            pca_result = run_pca(price_dict)
            return json.dumps({
                "clustering": cluster_result,
                "pca": pca_result,
            }, indent=2, default=str)

        elif name == "run_feature_importance":
            import yfinance as yf
            target = args["target_ticker"]
            features = args["feature_tickers"]
            period = args.get("period", "2y")
            target_df = yf.Ticker(target).history(period=period)
            if target_df.empty:
                return json.dumps({"error": f"No data for {target}"})
            feature_dict = {}
            for t in features:
                fdf = yf.Ticker(t).history(period=period)
                if not fdf.empty:
                    feature_dict[t.upper()] = fdf["Close"]
            if len(feature_dict) < 2:
                return json.dumps({"error": "Need at least 2 feature assets with data."})
            result = run_feature_importance(target_df["Close"], feature_dict, target_name=target)
            return json.dumps(result, indent=2, default=str)

        elif name == "generate_signals":
            import yfinance as yf
            ticker = args["ticker"]
            period = args.get("period", "2y")
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                return json.dumps({"error": f"No data for {ticker}"})
            result = composite_signal(df["Close"], name=ticker)
            return json.dumps(result, indent=2, default=str)

        elif name == "backtest_strategy":
            import yfinance as yf
            ticker = args["ticker"]
            strategy = args["strategy"]
            period = args.get("period", "5y")
            capital = args.get("initial_capital", 10000)
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                return json.dumps({"error": f"No data for {ticker}"})
            if strategy == "compare_all":
                result = compare_strategies(df["Close"], name=ticker, initial_capital=capital)
            elif strategy == "ema":
                short = args.get("short_window", 12)
                long = args.get("long_window", 26)
                result = backtest_ema_crossover(df["Close"], short, long, capital, ticker)
            else:
                short = args.get("short_window", 50)
                long = args.get("long_window", 200)
                result = backtest_sma_crossover(df["Close"], short, long, capital, ticker)
            return json.dumps(result, indent=2, default=str)


        elif name == "get_options_chain":
            ticker = args["ticker"]
            expiration = args.get("expiration")
            result = get_options_chain(ticker, expiration=expiration)
            return json.dumps(result, indent=2, default=str)

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


# ============================================================================
# Provider Adapters
# ============================================================================

def _tools_for_anthropic() -> list[dict]:
    """Convert tool specs to Anthropic format."""
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in TOOLS_SPEC
    ]


def _tools_for_openai() -> list[dict]:
    """Convert tool specs to OpenAI-compatible format (works for OpenAI, Gemini, Grok)."""
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in TOOLS_SPEC
    ]


# ============================================================================
# Provider configurations
# ============================================================================

PROVIDERS = {
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "label": "Anthropic (Claude)",
        "emoji": "🟣",
        "default_model": "claude-sonnet-4-20250514",
        "url": "https://console.anthropic.com/",
        "api_type": "anthropic",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "label": "OpenAI (GPT-4o)",
        "emoji": "🟢",
        "default_model": "gpt-4o",
        "url": "https://platform.openai.com/api-keys",
        "api_type": "openai",
        "base_url": None,
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "label": "Google (Gemini)",
        "emoji": "🔵",
        "default_model": "gemini-2.0-flash",
        "url": "https://aistudio.google.com/apikey",
        "api_type": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "xai": {
        "env_key": "XAI_API_KEY",
        "label": "xAI (Grok)",
        "emoji": "⚫",
        "default_model": "grok-3-mini-fast",
        "url": "https://console.x.ai/",
        "api_type": "openai",
        "base_url": "https://api.x.ai/v1",
    },
}


# ============================================================================
# Agent Class — auto-detects provider
# ============================================================================

class FinancialAgent:
    """
    Interactive financial agent — designed for Claude, works with any major LLM.

    Auto-detects which API key is set and uses that provider.
    Supports: Anthropic (Claude), OpenAI (GPT-4o), Google (Gemini), xAI (Grok).
    15 tools for market data, quantitative analysis, ML, signals, and backtesting.
    """

    def __init__(self):
        # Try providers in priority order (Claude first)
        self.provider_config = None
        self.api_key = None

        for name, config in PROVIDERS.items():
            key = os.environ.get(config["env_key"], "").strip()
            if key and key != "your-api-key-here":
                self.provider_config = config
                self.provider_name = name
                self.api_key = key
                break

        if not self.provider_config:
            key_list = "\n".join(
                f"  {c['env_key']}=...  ({c['label']})" for c in PROVIDERS.values()
            )
            url_list = "\n".join(
                f"  • {c['url']}" for c in PROVIDERS.values()
            )
            raise ValueError(
                f"No API key found! Set one of these in your .env file:\n"
                f"{key_list}\n\n"
                f"Get keys at:\n{url_list}"
            )

        self.model = os.environ.get("MODEL", self.provider_config["default_model"])
        self.api_type = self.provider_config["api_type"]
        print(f"  {self.provider_config['emoji']} Provider: {self.provider_config['label']} ({self.model})")

        # Initialize the client
        if self.api_type == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            from openai import OpenAI
            base_url = self.provider_config.get("base_url")
            if base_url:
                self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=self.api_key)

        self.history: list[dict] = []

    def chat(self, message: str) -> str:
        """Send a message and get a response (with tool-calling loop)."""
        if self.api_type == "anthropic":
            return self._chat_anthropic(message)
        else:
            return self._chat_openai(message)

    # ── Anthropic (Claude) ──────────────────────────────────────────────

    def _chat_anthropic(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=_tools_for_anthropic(),
                messages=self.history,
            )

            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  🔧 {block.name}({json.dumps(block.input, default=str)})")
                        result = execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                self.history.append({"role": "user", "content": tool_results})
            else:
                return "\n".join(b.text for b in response.content if hasattr(b, "text"))

    # ── OpenAI-compatible (GPT-4, Gemini, Grok) ────────────────────────

    def _chat_openai(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=_tools_for_openai(),
                tool_choice="auto",
            )

            choice = response.choices[0]
            assistant_msg = choice.message

            self.history.append(assistant_msg.model_dump())
            messages.append(assistant_msg.model_dump())

            if choice.finish_reason == "tool_calls" and assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    print(f"  🔧 {fn_name}({json.dumps(fn_args, default=str)})")
                    result = execute_tool(fn_name, fn_args)

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                    self.history.append(tool_msg)
                    messages.append(tool_msg)
            else:
                return assistant_msg.content or ""

    def clear(self):
        self.history = []
