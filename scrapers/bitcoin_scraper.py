"""
Bitcoin (BTC-USD) data scraper.

Primary source: Yahoo Finance (BTC-USD).
Fallback source: CoinGecko free API for additional granularity.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


TICKER = "BTC-USD"
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


def fetch_bitcoin_data(
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    add_indicators: bool = True,
    save_csv: bool = True,
    output_dir: str = "data",
    use_coingecko_fallback: bool = True,
) -> pd.DataFrame:
    """
    Fetch Bitcoin historical OHLCV data.

    Tries Yahoo Finance first (BTC-USD), then falls back to CoinGecko
    if requested and the primary source fails.

    Parameters
    ----------
    period : str
        Data period for Yahoo Finance (e.g., '1y', '2y', '5y', 'max').
    interval : str
        Data interval (e.g., '1d', '1h', '1wk').
    add_indicators : bool
        If True, compute and append technical indicators.
    save_csv : bool
        If True, save the resulting DataFrame to CSV.
    output_dir : str
        Directory to save CSV output.
    use_coingecko_fallback : bool
        If True, attempt CoinGecko API when Yahoo Finance returns empty data.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with datetime index and OHLCV + optional indicators.
    """
    print(f"[BTC] Fetching {period} of {interval} data from Yahoo Finance...")

    try:
        df = _fetch_from_yfinance(period, interval)
        print(f"[BTC] Yahoo Finance returned {len(df)} rows.")
    except Exception as e:
        print(f"[BTC] Yahoo Finance failed: {e}")
        df = pd.DataFrame()

    if df.empty and use_coingecko_fallback:
        print("[BTC] Falling back to CoinGecko API...")
        df = _fetch_from_coingecko(period)

    if df.empty:
        raise ValueError("No Bitcoin data returned from any source.")

    # Compute returns
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Daily_Return"] = df["Close"].pct_change()

    if add_indicators:
        df = _add_technical_indicators(df)

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"BTC_USD_{period}_{interval}.csv")
        df.to_csv(filepath)
        print(f"[BTC] Saved → {filepath}")

    print(f"[BTC] {len(df)} rows ({df.index.min()} to {df.index.max()})")
    return df


def _fetch_from_yfinance(period: str, interval: str) -> pd.DataFrame:
    """Fetch BTC-USD data from Yahoo Finance."""
    ticker = yf.Ticker(TICKER)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        return df

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = df.index.tz_localize(None)
    df.dropna(inplace=True)
    return df


def _fetch_from_coingecko(period: str) -> pd.DataFrame:
    """
    Fetch Bitcoin market chart data from CoinGecko free API.

    CoinGecko returns prices, market_caps, total_volumes as arrays of
    [timestamp_ms, value] pairs.
    """
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "max": "max"}
    days = days_map.get(period, 365)

    url = f"{COINGECKO_BASE_URL}/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])

    if not prices:
        return pd.DataFrame()

    df_prices = pd.DataFrame(prices, columns=["Timestamp", "Close"])
    df_prices["Date"] = pd.to_datetime(df_prices["Timestamp"], unit="ms")
    df_prices.set_index("Date", inplace=True)
    df_prices.drop(columns=["Timestamp"], inplace=True)

    if volumes:
        df_vol = pd.DataFrame(volumes, columns=["Timestamp", "Volume"])
        df_vol["Date"] = pd.to_datetime(df_vol["Timestamp"], unit="ms")
        df_vol.set_index("Date", inplace=True)
        df_vol.drop(columns=["Timestamp"], inplace=True)
        df_prices = df_prices.join(df_vol, how="left")

    # CoinGecko doesn't provide OHLC in market_chart, fill Open/High/Low from Close
    df_prices["Open"] = df_prices["Close"]
    df_prices["High"] = df_prices["Close"]
    df_prices["Low"] = df_prices["Close"]

    df_prices.index = df_prices.index.tz_localize(None) if df_prices.index.tz else df_prices.index
    df_prices.index.name = "Date"
    df_prices.dropna(inplace=True)

    return df_prices[["Open", "High", "Low", "Close", "Volume"]]


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append SMA, EMA, RSI, MACD, and Bollinger Bands."""
    close = df["Close"]

    df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()
    df["EMA_12"] = EMAIndicator(close, window=12).ema_indicator()
    df["EMA_26"] = EMAIndicator(close, window=26).ema_indicator()

    df["RSI_14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()

    return df


if __name__ == "__main__":
    data = fetch_bitcoin_data()
    print(data.tail(10))
    print(f"\nColumns: {list(data.columns)}")
