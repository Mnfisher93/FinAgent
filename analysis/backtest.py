"""
Simple strategy backtester.

Simulates trading a signal strategy against historical data and computes
performance metrics vs. buy-and-hold benchmark.

Supports:
- SMA Crossover strategies
- EMA Crossover strategies
- Custom signal-based strategies
"""

import numpy as np
import pandas as pd
from typing import Optional
from analysis.signals import sma_crossover, ema_crossover, composite_signal


# ============================================================================
# Backtesting Engine
# ============================================================================

def backtest_sma_crossover(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
    initial_capital: float = 10000.0,
    name: str = "Asset",
) -> dict:
    """
    Backtest a simple SMA crossover strategy.

    Rules:
    - Go long (buy) when short SMA > long SMA
    - Go flat (sell) when short SMA < long SMA
    - No shorting, no leverage

    Parameters
    ----------
    prices : pd.Series
        Historical close prices.
    short_window : int
        Short SMA period.
    long_window : int
        Long SMA period.
    initial_capital : float
        Starting capital in USD.
    name : str
        Asset name.

    Returns
    -------
    dict
        Strategy performance vs. buy-and-hold.
    """
    sma_short = prices.rolling(short_window).mean()
    sma_long = prices.rolling(long_window).mean()

    # Position: 1 = long, 0 = flat (no shorting)
    position = pd.Series(0.0, index=prices.index)
    position[sma_short > sma_long] = 1.0
    position = position.shift(1).fillna(0)  # Can't act on today's signal today

    # Daily returns
    daily_returns = prices.pct_change().fillna(0)

    # Strategy returns = market returns × position
    strategy_returns = daily_returns * position

    # Cumulative returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    benchmark_cumulative = (1 + daily_returns).cumprod()

    strategy_final = float(strategy_cumulative.iloc[-1])
    benchmark_final = float(benchmark_cumulative.iloc[-1])

    # Count trades
    trades = position.diff().abs()
    num_trades = int(trades.sum() / 2)  # Each round-trip = buy + sell

    # Strategy metrics
    trading_days = 252
    n_days = len(strategy_returns)
    years = n_days / trading_days

    strat_total_return = strategy_final - 1
    bench_total_return = benchmark_final - 1

    strat_ann_return = (strategy_final ** (1 / years)) - 1
    bench_ann_return = (benchmark_final ** (1 / years)) - 1

    strat_volatility = strategy_returns.std() * np.sqrt(trading_days)
    strat_sharpe = (strat_ann_return - 0.05) / strat_volatility if strat_volatility > 0 else 0

    # Max drawdown
    strat_peak = strategy_cumulative.cummax()
    strat_drawdown = (strategy_cumulative - strat_peak) / strat_peak
    max_drawdown = float(strat_drawdown.min())

    # Win rate
    winning_days = (strategy_returns[position == 1] > 0).sum()
    total_trading_days = (position == 1).sum()
    win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0

    # Time in market
    time_in_market = position.mean()

    return {
        "strategy": f"SMA Crossover ({short_window}/{long_window})",
        "asset": name,
        "period": f"{str(prices.index.min())[:10]} to {str(prices.index.max())[:10]}",
        "initial_capital": f"${initial_capital:,.0f}",
        "strategy_performance": {
            "final_value": f"${initial_capital * strategy_final:,.2f}",
            "total_return": f"{strat_total_return:.2%}",
            "annualized_return": f"{strat_ann_return:.2%}",
            "volatility": f"{strat_volatility:.2%}",
            "sharpe_ratio": f"{strat_sharpe:.3f}",
            "max_drawdown": f"{max_drawdown:.2%}",
            "win_rate": f"{win_rate:.2%}",
            "num_trades": num_trades,
            "time_in_market": f"{time_in_market:.1%}",
        },
        "buy_and_hold": {
            "final_value": f"${initial_capital * benchmark_final:,.2f}",
            "total_return": f"{bench_total_return:.2%}",
            "annualized_return": f"{bench_ann_return:.2%}",
        },
        "outperformance": f"{(strat_total_return - bench_total_return):.2%}",
        "verdict": (
            "Strategy OUTPERFORMS buy-and-hold ✅"
            if strat_total_return > bench_total_return
            else "Strategy UNDERPERFORMS buy-and-hold ❌"
        ),
    }


def backtest_ema_crossover(
    prices: pd.Series,
    short_window: int = 12,
    long_window: int = 26,
    initial_capital: float = 10000.0,
    name: str = "Asset",
) -> dict:
    """Backtest an EMA crossover strategy (same logic, EMA instead of SMA)."""
    ema_short = prices.ewm(span=short_window, adjust=False).mean()
    ema_long = prices.ewm(span=long_window, adjust=False).mean()

    position = pd.Series(0.0, index=prices.index)
    position[ema_short > ema_long] = 1.0
    position = position.shift(1).fillna(0)

    daily_returns = prices.pct_change().fillna(0)
    strategy_returns = daily_returns * position

    strategy_cumulative = (1 + strategy_returns).cumprod()
    benchmark_cumulative = (1 + daily_returns).cumprod()

    strategy_final = float(strategy_cumulative.iloc[-1])
    benchmark_final = float(benchmark_cumulative.iloc[-1])

    trades = position.diff().abs()
    num_trades = int(trades.sum() / 2)

    trading_days = 252
    n_days = len(strategy_returns)
    years = n_days / trading_days

    strat_total = strategy_final - 1
    bench_total = benchmark_final - 1
    strat_ann = (strategy_final ** (1 / years)) - 1
    strat_vol = strategy_returns.std() * np.sqrt(trading_days)
    strat_sharpe = (strat_ann - 0.05) / strat_vol if strat_vol > 0 else 0

    strat_peak = strategy_cumulative.cummax()
    max_dd = float(((strategy_cumulative - strat_peak) / strat_peak).min())

    winning = (strategy_returns[position == 1] > 0).sum()
    total = (position == 1).sum()
    win_rate = winning / total if total > 0 else 0

    return {
        "strategy": f"EMA Crossover ({short_window}/{long_window})",
        "asset": name,
        "period": f"{str(prices.index.min())[:10]} to {str(prices.index.max())[:10]}",
        "initial_capital": f"${initial_capital:,.0f}",
        "strategy_performance": {
            "final_value": f"${initial_capital * strategy_final:,.2f}",
            "total_return": f"{strat_total:.2%}",
            "annualized_return": f"{strat_ann:.2%}",
            "volatility": f"{strat_vol:.2%}",
            "sharpe_ratio": f"{strat_sharpe:.3f}",
            "max_drawdown": f"{max_dd:.2%}",
            "win_rate": f"{win_rate:.2%}",
            "num_trades": num_trades,
        },
        "buy_and_hold": {
            "final_value": f"${initial_capital * benchmark_final:,.2f}",
            "total_return": f"{bench_total:.2%}",
        },
        "outperformance": f"{(strat_total - bench_total):.2%}",
        "verdict": (
            "Strategy OUTPERFORMS buy-and-hold ✅"
            if strat_total > bench_total
            else "Strategy UNDERPERFORMS buy-and-hold ❌"
        ),
    }


def compare_strategies(
    prices: pd.Series,
    name: str = "Asset",
    initial_capital: float = 10000.0,
) -> dict:
    """
    Compare multiple crossover strategies side-by-side.

    Runs: SMA 50/200, SMA 20/50, EMA 12/26, EMA 9/21
    """
    strategies = [
        ("SMA 50/200", backtest_sma_crossover(prices, 50, 200, initial_capital, name)),
        ("SMA 20/50", backtest_sma_crossover(prices, 20, 50, initial_capital, name)),
        ("EMA 12/26", backtest_ema_crossover(prices, 12, 26, initial_capital, name)),
        ("EMA 9/21", backtest_ema_crossover(prices, 9, 21, initial_capital, name)),
    ]

    comparison = []
    for strat_name, result in strategies:
        comparison.append({
            "strategy": strat_name,
            "total_return": result["strategy_performance"]["total_return"],
            "sharpe": result["strategy_performance"]["sharpe_ratio"],
            "max_drawdown": result["strategy_performance"]["max_drawdown"],
            "num_trades": result["strategy_performance"]["num_trades"],
            "vs_buy_hold": result["outperformance"],
        })

    best = max(strategies, key=lambda x: float(x[1]["strategy_performance"]["total_return"].strip('%')) / 100)

    return {
        "asset": name,
        "period": strategies[0][1]["period"],
        "buy_and_hold_return": strategies[0][1]["buy_and_hold"]["total_return"],
        "strategies": comparison,
        "best_strategy": best[0],
        "best_return": best[1]["strategy_performance"]["total_return"],
    }
