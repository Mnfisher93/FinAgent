"""
Analysis package — quantitative analysis, time series modeling, and visualization.
"""

import numpy as np
import pandas as pd

from .quant_analysis import QuantAnalyzer
from .time_series import TimeSeriesAnalyzer
from .visualizations import Visualizer


# ── Simple analysis functions (used by the agent's tool-calling) ────────

def analyze_returns(prices: pd.Series, name: str = "Asset") -> dict:
    """
    Compute basic return and risk metrics for a price series.

    Returns total return, annualized return, volatility, Sharpe ratio,
    and max drawdown.
    """
    daily_ret = prices.pct_change().dropna()
    trading_days = 252

    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    ann_return = (1 + daily_ret).prod() ** (trading_days / len(daily_ret)) - 1
    ann_vol = daily_ret.std() * np.sqrt(trading_days)
    sharpe = (ann_return - 0.05) / ann_vol if ann_vol > 0 else 0

    cumulative = (1 + daily_ret).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "asset": name,
        "total_return": f"{total_return:.2%}",
        "annualized_return": f"{ann_return:.2%}",
        "annualized_volatility": f"{ann_vol:.2%}",
        "sharpe_ratio": f"{sharpe:.3f}",
        "max_drawdown": f"{max_dd:.2%}",
        "avg_daily_return": f"{daily_ret.mean():.4%}",
        "best_day": f"{daily_ret.max():.2%}",
        "worst_day": f"{daily_ret.min():.2%}",
    }


def compare_assets(price_dict: dict[str, pd.Series]) -> dict:
    """Compute correlation matrix for multiple assets."""
    returns_df = pd.DataFrame({
        name: prices.pct_change().dropna()
        for name, prices in price_dict.items()
    }).dropna()

    corr = returns_df.corr()
    return {
        "correlation_matrix": corr.to_dict(),
        "assets": list(price_dict.keys()),
    }


__all__ = [
    "QuantAnalyzer",
    "TimeSeriesAnalyzer",
    "Visualizer",
    "analyze_returns",
    "compare_assets",
]

