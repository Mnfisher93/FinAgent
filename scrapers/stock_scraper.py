"""
US stock data scraper using Yahoo Finance.
Supports any NYSE/NASDAQ ticker.
"""

import pandas as pd
import yfinance as yf


def get_stock_quote(ticker: str) -> dict:
    """Get current stock quote: price, change, volume, market cap."""
    stock = yf.Ticker(ticker.upper())
    info = stock.info

    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName", ticker.upper()),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
        "open": info.get("open") or info.get("regularMarketOpen"),
        "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
        "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
        "volume": info.get("volume") or info.get("regularMarketVolume"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "dividend_yield": info.get("dividendYield"),
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow"),
        "exchange": info.get("exchange"),
    }


def get_stock_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Get historical OHLCV data for any US stock."""
    stock = yf.Ticker(ticker.upper())
    df = stock.history(period=period)

    if df.empty:
        return df

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.dropna(inplace=True)
    df["Daily_Return"] = df["Close"].pct_change()

    return df


def get_stock_info(ticker: str) -> dict:
    """Get company information: sector, industry, description, financials."""
    stock = yf.Ticker(ticker.upper())
    info = stock.info

    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName", ticker.upper()),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "description": (info.get("longBusinessSummary") or "")[:500],
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "profit_margins": info.get("profitMargins"),
        "return_on_equity": info.get("returnOnEquity"),
        "beta": info.get("beta"),
        "dividend_yield": info.get("dividendYield"),
        "revenue": info.get("totalRevenue"),
    }
