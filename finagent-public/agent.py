"""
AI Agent powered by Claude — simplified version for public demo.

7 tools: stock quote/history/info, crypto price/top/search, polymarket.
Plus basic analysis (returns, Sharpe, drawdown) and comparison.
"""

import json
import os
import traceback

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from scrapers.stock_scraper import get_stock_quote, get_stock_history, get_stock_info
from scrapers.crypto_scraper import get_crypto_price, get_crypto_top_n, search_crypto
from scrapers.polymarket_scraper import fetch_polymarket_data
from analysis import analyze_returns, compare_assets

load_dotenv()

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are a financial analyst AI assistant with access to real-time data. You can:

• Look up any US stock (NYSE/NASDAQ) — quotes, history, company info
• Check cryptocurrency prices via CoinGecko
• Browse Polymarket prediction markets
• Analyze asset returns, volatility, Sharpe ratio, and drawdowns
• Compare assets with correlation analysis

Rules:
1. Always use your tools to fetch real data — never make up numbers.
2. Present data clearly with proper formatting.
3. Provide context and interpretation, not just raw data.
4. Be concise but insightful.

You're chatting in a terminal. Use clear text formatting."""

TOOLS = [
    {
        "name": "get_stock_quote",
        "description": "Current stock quote for any US stock: price, volume, market cap, PE ratio, 52-week range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker (e.g., 'AAPL', 'MSFT', 'TSLA')"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_stock_history",
        "description": "Historical OHLCV data for any US stock. Returns summary with price range, total return, and recent prices.",
        "input_schema": {
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
        "description": "Company fundamentals: sector, industry, description, PE ratio, margins, beta, dividends.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_crypto_price",
        "description": "Current crypto price, market cap, volume, and price changes (24h/7d/30d). Supports any coin.",
        "input_schema": {
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
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of top coins (default: 20)", "default": 20}
            },
            "required": []
        }
    },
    {
        "name": "search_crypto",
        "description": "Search for a cryptocurrency by name or symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_polymarket_markets",
        "description": "Active Polymarket prediction markets with probabilities, volume, and liquidity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of markets (default: 20)", "default": 20}
            },
            "required": []
        }
    },
    {
        "name": "analyze_asset",
        "description": "Analyze a stock or crypto: total return, annualized return, volatility, Sharpe ratio, and max drawdown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker (e.g., 'AAPL', 'BTC-USD')"},
                "period": {"type": "string", "description": "Period for analysis", "default": "1y"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "compare_assets",
        "description": "Compare multiple assets: compute correlation matrix of daily returns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "List of tickers to compare"},
                "period": {"type": "string", "description": "Data period", "default": "1y"}
            },
            "required": ["tickers"]
        }
    },
]


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return JSON result."""
    try:
        if name == "get_stock_quote":
            return json.dumps(get_stock_quote(args["ticker"]), indent=2, default=str)

        elif name == "get_stock_history":
            df = get_stock_history(args["ticker"], period=args.get("period", "1y"))
            if df.empty:
                return json.dumps({"error": f"No data for {args['ticker']}"})
            return json.dumps({
                "ticker": args["ticker"].upper(),
                "period": args.get("period", "1y"),
                "rows": len(df),
                "date_range": f"{df.index.min()} to {df.index.max()}",
                "latest_close": float(df["Close"].iloc[-1]),
                "period_high": float(df["High"].max()),
                "period_low": float(df["Low"].min()),
                "total_return": f"{((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1):.2%}",
                "last_5_days": df[["Close", "Volume"]].tail(5).to_dict(),
            }, indent=2, default=str)

        elif name == "get_stock_info":
            return json.dumps(get_stock_info(args["ticker"]), indent=2, default=str)

        elif name == "get_crypto_price":
            return json.dumps(get_crypto_price(args["coin_id"]), indent=2, default=str)

        elif name == "get_crypto_top_n":
            return json.dumps(get_crypto_top_n(args.get("n", 20)), indent=2, default=str)

        elif name == "search_crypto":
            return json.dumps(search_crypto(args["query"]), indent=2, default=str)

        elif name == "get_polymarket_markets":
            df = fetch_polymarket_data(limit=args.get("limit", 20))
            if df.empty:
                return json.dumps({"message": "No markets found."})
            return json.dumps(
                df[["question", "outcome_prices", "volume"]].head(20).to_dict(orient="records"),
                indent=2, default=str
            )

        elif name == "analyze_asset":
            import yfinance as yf
            df = yf.Ticker(args["ticker"]).history(period=args.get("period", "1y"))
            if df.empty:
                return json.dumps({"error": f"No data for {args['ticker']}"})
            return json.dumps(analyze_returns(df["Close"], name=args["ticker"]), indent=2, default=str)

        elif name == "compare_assets":
            import yfinance as yf
            price_dict = {}
            for t in args["tickers"]:
                df = yf.Ticker(t).history(period=args.get("period", "1y"))
                if not df.empty:
                    price_dict[t.upper()] = df["Close"]
            if len(price_dict) < 2:
                return json.dumps({"error": "Need at least 2 assets with data."})
            return json.dumps(compare_assets(price_dict), indent=2, default=str)

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


class FinancialAgent:
    """Interactive financial agent powered by Claude."""

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            raise ValueError(
                "ANTHROPIC_API_KEY not set!\n"
                "  1. Copy .env.example to .env\n"
                "  2. Paste your Anthropic API key\n"
                "  Get one at: https://console.anthropic.com/"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.history: list[dict] = []

    def chat(self, message: str) -> str:
        """Send a message and get a response (with tool-calling loop)."""
        self.history.append({"role": "user", "content": message})

        while True:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
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

    def clear(self):
        self.history = []
