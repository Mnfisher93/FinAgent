"""
SPY (S&P 500 ETF) data scraper using Yahoo Finance.

Fetches historical OHLCV data for SPY and computes basic technical indicators.
"""

import os
import pandas as pd
import yfinance as yf
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


TICKER = "SPY"
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"


def fetch_spy_data(
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    add_indicators: bool = True,
    save_csv: bool = True,
    output_dir: str = "data",
) -> pd.DataFrame:
    """
    Fetch SPY historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    period : str
        Data period (e.g., '1y', '2y', '5y', 'max').
    interval : str
        Data interval (e.g., '1d', '1h', '1wk').
    add_indicators : bool
        If True, compute and append technical indicators (SMA, EMA, RSI, MACD, Bollinger).
    save_csv : bool
        If True, save the resulting DataFrame to CSV.
    output_dir : str
        Directory to save CSV output.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with datetime index and OHLCV + optional indicators.
    """
    print(f"[SPY] Fetching {period} of {interval} data...")

    ticker = yf.Ticker(TICKER)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {TICKER}")

    # Clean up columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = df.index.tz_localize(None)  # Remove timezone for consistency
    df.dropna(inplace=True)

    # Compute log returns
    df["Log_Return"] = pd.Series.apply(df["Close"].pct_change().dropna(), lambda x: __import__("numpy").log(1 + x))
    df["Daily_Return"] = df["Close"].pct_change()

    if add_indicators:
        df = _add_technical_indicators(df)

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{TICKER}_{period}_{interval}.csv")
        df.to_csv(filepath)
        print(f"[SPY] Saved → {filepath}")

    print(f"[SPY] {len(df)} rows fetched ({df.index.min()} to {df.index.max()})")
    return df


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append SMA, EMA, RSI, MACD, and Bollinger Bands to the DataFrame."""
    close = df["Close"]

    # Simple & Exponential Moving Averages
    df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()
    df["EMA_12"] = EMAIndicator(close, window=12).ema_indicator()
    df["EMA_26"] = EMAIndicator(close, window=26).ema_indicator()

    # Relative Strength Index
    df["RSI_14"] = RSIIndicator(close, window=14).rsi()

    # MACD
    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()

    return df


if __name__ == "__main__":
    data = fetch_spy_data()
    print(data.tail(10))
    print(f"\nColumns: {list(data.columns)}")
