"""
Quantitative Fintech Agent — AI-Powered Financial Analyst

Supports multiple LLM providers (designed for Claude, works with all):
  • Anthropic (Claude)  — set ANTHROPIC_API_KEY
  • OpenAI (GPT-4o)     — set OPENAI_API_KEY
  • Google (Gemini)     — set GEMINI_API_KEY
  • xAI (Grok)          — set XAI_API_KEY

The agent auto-detects which key you have and uses that provider.
9 tools: stock quote/history/info, crypto price/top/search, polymarket,
plus basic analysis and cross-asset comparison.
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
from analysis import analyze_returns, compare_assets

load_dotenv(Path(__file__).resolve().parent / ".env")

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

# ============================================================================
# Tool Definitions (provider-agnostic, converted per-provider at runtime)
# ============================================================================

TOOLS_SPEC = [
    {
        "name": "get_stock_quote",
        "description": "Current stock quote for any US stock: price, volume, market cap, PE ratio, 52-week range.",
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
        "description": "Historical OHLCV data for any US stock. Returns summary with price range, total return, and recent prices.",
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
        "description": "Company fundamentals: sector, industry, description, PE ratio, margins, beta, dividends.",
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
        "description": "Current crypto price, market cap, volume, and price changes (24h/7d/30d). Supports any coin.",
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
        "description": "Search for a cryptocurrency by name or symbol.",
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
        "description": "Active Polymarket prediction markets with probabilities, volume, and liquidity.",
        "parameters": {
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
        "parameters": {
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
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "List of tickers to compare"},
                "period": {"type": "string", "description": "Data period", "default": "1y"}
            },
            "required": ["tickers"]
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
        "api_type": "anthropic",       # uses anthropic SDK
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "label": "OpenAI (GPT-4o)",
        "emoji": "🟢",
        "default_model": "gpt-4o",
        "url": "https://platform.openai.com/api-keys",
        "api_type": "openai",          # uses openai SDK
        "base_url": None,              # default OpenAI endpoint
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "label": "Google (Gemini)",
        "emoji": "🔵",
        "default_model": "gemini-2.0-flash",
        "url": "https://aistudio.google.com/apikey",
        "api_type": "openai",          # Gemini has an OpenAI-compatible endpoint
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "xai": {
        "env_key": "XAI_API_KEY",
        "label": "xAI (Grok)",
        "emoji": "⚫",
        "default_model": "grok-3-mini-fast",
        "url": "https://console.x.ai/",
        "api_type": "openai",          # Grok uses OpenAI-compatible API
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
