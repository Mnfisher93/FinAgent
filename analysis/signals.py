"""
Trading signal generators for algorithmic trading strategies.

Generates buy/sell signals from technical indicators:
- SMA Crossover (Golden Cross / Death Cross)
- EMA Crossover (faster-reacting)
- RSI Overbought/Oversold
- MACD Signal Line Crossover
- Bollinger Band Breakouts
- Composite Signal Score (multi-indicator)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ============================================================================
# Individual Signal Generators
# ============================================================================

def sma_crossover(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
) -> dict:
    """
    SMA Crossover signal (Golden Cross / Death Cross).

    - Golden Cross: short SMA crosses ABOVE long SMA → BUY
    - Death Cross: short SMA crosses BELOW long SMA → SELL

    Parameters
    ----------
    prices : pd.Series
        Close prices.
    short_window : int
        Short SMA period (default: 50).
    long_window : int
        Long SMA period (default: 200).

    Returns
    -------
    dict
        Signal data with crossover events and current position.
    """
    sma_short = prices.rolling(short_window).mean()
    sma_long = prices.rolling(long_window).mean()

    # Position: +1 when short > long, -1 when short < long
    position = pd.Series(0.0, index=prices.index)
    position[sma_short > sma_long] = 1.0
    position[sma_short < sma_long] = -1.0

    # Crossover events (transitions)
    transitions = position.diff()
    golden_crosses = prices.index[transitions == 2.0].tolist()
    death_crosses = prices.index[transitions == -2.0].tolist()

    current_signal = "BUY" if position.iloc[-1] == 1.0 else "SELL"

    return {
        "indicator": "SMA Crossover",
        "short_window": short_window,
        "long_window": long_window,
        "current_signal": current_signal,
        "current_short_sma": float(sma_short.iloc[-1]) if pd.notna(sma_short.iloc[-1]) else None,
        "current_long_sma": float(sma_long.iloc[-1]) if pd.notna(sma_long.iloc[-1]) else None,
        "golden_crosses": [str(d)[:10] for d in golden_crosses[-5:]],
        "death_crosses": [str(d)[:10] for d in death_crosses[-5:]],
        "total_golden_crosses": len(golden_crosses),
        "total_death_crosses": len(death_crosses),
        "signal_value": float(position.iloc[-1]),
    }


def ema_crossover(
    prices: pd.Series,
    short_window: int = 12,
    long_window: int = 26,
) -> dict:
    """
    EMA Crossover signal — same logic as SMA but with exponential weighting
    for faster reaction to recent price changes.
    """
    ema_short = prices.ewm(span=short_window, adjust=False).mean()
    ema_long = prices.ewm(span=long_window, adjust=False).mean()

    position = pd.Series(0.0, index=prices.index)
    position[ema_short > ema_long] = 1.0
    position[ema_short < ema_long] = -1.0

    transitions = position.diff()
    buy_signals = prices.index[transitions == 2.0].tolist()
    sell_signals = prices.index[transitions == -2.0].tolist()

    current_signal = "BUY" if position.iloc[-1] == 1.0 else "SELL"

    return {
        "indicator": "EMA Crossover",
        "short_window": short_window,
        "long_window": long_window,
        "current_signal": current_signal,
        "current_short_ema": float(ema_short.iloc[-1]),
        "current_long_ema": float(ema_long.iloc[-1]),
        "recent_buy_signals": [str(d)[:10] for d in buy_signals[-5:]],
        "recent_sell_signals": [str(d)[:10] for d in sell_signals[-5:]],
        "signal_value": float(position.iloc[-1]),
    }


def rsi_signal(
    prices: pd.Series,
    period: int = 14,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> dict:
    """
    RSI (Relative Strength Index) signal.

    - RSI < 30 → Oversold → BUY signal
    - RSI > 70 → Overbought → SELL signal
    - 30 < RSI < 70 → HOLD
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    current_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None

    if current_rsi is None:
        signal = "INSUFFICIENT DATA"
        signal_value = 0.0
    elif current_rsi < oversold:
        signal = "BUY (Oversold)"
        signal_value = 1.0
    elif current_rsi > overbought:
        signal = "SELL (Overbought)"
        signal_value = -1.0
    else:
        signal = "HOLD (Neutral)"
        signal_value = 0.0

    # Count recent extreme events
    oversold_days = rsi[rsi < oversold].index.tolist()
    overbought_days = rsi[rsi > overbought].index.tolist()

    return {
        "indicator": "RSI",
        "period": period,
        "current_rsi": round(current_rsi, 2) if current_rsi else None,
        "current_signal": signal,
        "overbought_threshold": overbought,
        "oversold_threshold": oversold,
        "recent_oversold_dates": [str(d)[:10] for d in oversold_days[-5:]],
        "recent_overbought_dates": [str(d)[:10] for d in overbought_days[-5:]],
        "signal_value": signal_value,
    }


def macd_signal(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> dict:
    """
    MACD (Moving Average Convergence Divergence) signal.

    - MACD crosses ABOVE signal line → BUY
    - MACD crosses BELOW signal line → SELL
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    # Position
    position = pd.Series(0.0, index=prices.index)
    position[macd_line > signal_line] = 1.0
    position[macd_line < signal_line] = -1.0

    transitions = position.diff()
    buy_signals = prices.index[transitions == 2.0].tolist()
    sell_signals = prices.index[transitions == -2.0].tolist()

    current_signal = "BUY" if position.iloc[-1] == 1.0 else "SELL"

    return {
        "indicator": "MACD",
        "fast": fast,
        "slow": slow,
        "signal_period": signal_period,
        "current_macd": round(float(macd_line.iloc[-1]), 4),
        "current_signal_line": round(float(signal_line.iloc[-1]), 4),
        "current_histogram": round(float(histogram.iloc[-1]), 4),
        "current_signal": current_signal,
        "recent_buy_signals": [str(d)[:10] for d in buy_signals[-5:]],
        "recent_sell_signals": [str(d)[:10] for d in sell_signals[-5:]],
        "signal_value": float(position.iloc[-1]),
    }


def bollinger_signal(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> dict:
    """
    Bollinger Band signal.

    - Price <= lower band → BUY (mean reversion expected)
    - Price >= upper band → SELL (overbought)
    - Price between bands → HOLD
    """
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)

    current_price = float(prices.iloc[-1])
    current_upper = float(upper.iloc[-1]) if pd.notna(upper.iloc[-1]) else None
    current_lower = float(lower.iloc[-1]) if pd.notna(lower.iloc[-1]) else None
    current_sma = float(sma.iloc[-1]) if pd.notna(sma.iloc[-1]) else None

    if current_lower and current_price <= current_lower:
        signal = "BUY (Below Lower Band)"
        signal_value = 1.0
    elif current_upper and current_price >= current_upper:
        signal = "SELL (Above Upper Band)"
        signal_value = -1.0
    else:
        signal = "HOLD (Within Bands)"
        signal_value = 0.0

    # Bandwidth (volatility indicator)
    bandwidth = ((upper - lower) / sma * 100).iloc[-1] if pd.notna(sma.iloc[-1]) else None

    return {
        "indicator": "Bollinger Bands",
        "window": window,
        "num_std": num_std,
        "current_price": current_price,
        "upper_band": round(current_upper, 2) if current_upper else None,
        "lower_band": round(current_lower, 2) if current_lower else None,
        "middle_band": round(current_sma, 2) if current_sma else None,
        "bandwidth_pct": round(float(bandwidth), 2) if bandwidth and pd.notna(bandwidth) else None,
        "current_signal": signal,
        "signal_value": signal_value,
    }


# ============================================================================
# Composite Signal Score
# ============================================================================

def composite_signal(
    prices: pd.Series,
    name: str = "Asset",
    weights: Optional[dict] = None,
) -> dict:
    """
    Combine multiple indicators into a single composite signal score.

    Score ranges from -1.0 (strong sell) to +1.0 (strong buy).

    Default weights:
    - SMA Crossover (50/200): 0.25
    - EMA Crossover (12/26): 0.15
    - RSI: 0.20
    - MACD: 0.20
    - Bollinger Bands: 0.20

    Parameters
    ----------
    prices : pd.Series
        Close prices (need at least 200 data points for full analysis).
    name : str
        Asset name for display.
    weights : dict, optional
        Custom weights for each indicator.

    Returns
    -------
    dict
        Composite score and individual indicator results.
    """
    if weights is None:
        weights = {
            "sma": 0.25,
            "ema": 0.15,
            "rsi": 0.20,
            "macd": 0.20,
            "bollinger": 0.20,
        }

    # Run all indicators
    sma_result = sma_crossover(prices, 50, 200)
    ema_result = ema_crossover(prices, 12, 26)
    rsi_result = rsi_signal(prices)
    macd_result = macd_signal(prices)
    boll_result = bollinger_signal(prices)

    # Compute weighted composite score
    composite = (
        sma_result["signal_value"] * weights["sma"]
        + ema_result["signal_value"] * weights["ema"]
        + rsi_result["signal_value"] * weights["rsi"]
        + macd_result["signal_value"] * weights["macd"]
        + boll_result["signal_value"] * weights["bollinger"]
    )

    # Interpret
    if composite > 0.5:
        verdict = "STRONG BUY 🟢🟢"
    elif composite > 0.1:
        verdict = "BUY 🟢"
    elif composite < -0.5:
        verdict = "STRONG SELL 🔴🔴"
    elif composite < -0.1:
        verdict = "SELL 🔴"
    else:
        verdict = "HOLD ⚪"

    return {
        "asset": name,
        "composite_score": round(composite, 3),
        "verdict": verdict,
        "indicators": {
            "sma_crossover": {
                "signal": sma_result["current_signal"],
                "value": sma_result["signal_value"],
                "weight": weights["sma"],
                "sma_50": sma_result["current_short_sma"],
                "sma_200": sma_result["current_long_sma"],
            },
            "ema_crossover": {
                "signal": ema_result["current_signal"],
                "value": ema_result["signal_value"],
                "weight": weights["ema"],
            },
            "rsi": {
                "signal": rsi_result["current_signal"],
                "value": rsi_result["signal_value"],
                "weight": weights["rsi"],
                "rsi_value": rsi_result["current_rsi"],
            },
            "macd": {
                "signal": macd_result["current_signal"],
                "value": macd_result["signal_value"],
                "weight": weights["macd"],
                "macd_value": macd_result["current_macd"],
                "histogram": macd_result["current_histogram"],
            },
            "bollinger": {
                "signal": boll_result["current_signal"],
                "value": boll_result["signal_value"],
                "weight": weights["bollinger"],
                "upper": boll_result["upper_band"],
                "lower": boll_result["lower_band"],
            },
        },
        "recent_crossovers": {
            "golden_crosses": sma_result["golden_crosses"],
            "death_crosses": sma_result["death_crosses"],
        },
    }
