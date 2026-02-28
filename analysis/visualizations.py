"""
Visualization module for financial data analysis.

Generates publication-quality charts:
- Price history with moving averages
- Returns distributions (histogram + KDE)
- Correlation heatmaps
- Drawdown plots
- Volatility cones
- GARCH conditional volatility overlays
- Rolling Sharpe ratio
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# Set global style
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "figure.figsize": (14, 7),
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


class Visualizer:
    """
    Financial data visualizer — generates and saves charts.

    Parameters
    ----------
    output_dir : str
        Directory to save chart images.
    """

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig, filename: str):
        """Save figure to output directory."""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Chart saved → {filepath}")

    # -------------------------------------------------------------------------
    # Price Charts
    # -------------------------------------------------------------------------

    def plot_price_history(
        self, df: pd.DataFrame, title: str = "Price History",
        show_sma: bool = True, filename: str = "price_history.png",
    ):
        """
        Plot OHLCV price history with optional moving averages.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'Close' column and datetime index.
            Optionally contains 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26'.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={"height_ratios": [3, 1]})

        # Price
        ax1.plot(df.index, df["Close"], label="Close", linewidth=1.5, color="#2196F3")

        if show_sma and "SMA_20" in df.columns:
            ax1.plot(df.index, df["SMA_20"], label="SMA 20", linewidth=1, alpha=0.7)
            ax1.plot(df.index, df["SMA_50"], label="SMA 50", linewidth=1, alpha=0.7)

        if show_sma and "BB_Upper" in df.columns:
            ax1.fill_between(df.index, df["BB_Lower"], df["BB_Upper"],
                            alpha=0.1, color="gray", label="Bollinger Bands")

        ax1.set_title(title, fontweight="bold")
        ax1.set_ylabel("Price ($)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # Volume
        if "Volume" in df.columns:
            colors = ["#4CAF50" if df["Close"].iloc[i] >= df["Open"].iloc[i]
                      else "#F44336" for i in range(len(df))]
            ax2.bar(df.index, df["Volume"], color=colors, alpha=0.6, width=1)
            ax2.set_ylabel("Volume")
            ax2.set_xlabel("Date")

        fig.tight_layout()
        self._save(fig, filename)

    def plot_multi_asset_prices(
        self, price_dict: dict[str, pd.Series],
        normalize: bool = True, filename: str = "multi_asset_prices.png",
    ):
        """
        Plot multiple assets on the same chart, optionally normalized to 100.

        Parameters
        ----------
        price_dict : dict[str, pd.Series]
            Mapping of asset name → closing price series.
        normalize : bool
            If True, normalize all series to start at 100 for comparison.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        for name, prices in price_dict.items():
            if normalize:
                normalized = (prices / prices.iloc[0]) * 100
                ax.plot(normalized.index, normalized, label=name, linewidth=1.5)
            else:
                ax.plot(prices.index, prices, label=name, linewidth=1.5)

        ax.set_title("Multi-Asset Price Comparison" + (" (Normalized)" if normalize else ""),
                     fontweight="bold")
        ax.set_ylabel("Normalized Price (Base=100)" if normalize else "Price ($)")
        ax.set_xlabel("Date")
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        fig.tight_layout()
        self._save(fig, filename)

    # -------------------------------------------------------------------------
    # Returns Analysis
    # -------------------------------------------------------------------------

    def plot_returns_distribution(
        self, returns: pd.Series, name: str = "Asset",
        filename: str = "returns_distribution.png",
    ):
        """
        Plot histogram + KDE of returns with normal distribution overlay.
        Annotates skewness, kurtosis, and Jarque-Bera test result.
        """
        from scipy import stats as sp_stats

        fig, ax = plt.subplots(figsize=(12, 6))

        # Histogram + KDE
        sns.histplot(returns.dropna(), bins=80, stat="density", kde=True, ax=ax,
                     color="#2196F3", alpha=0.5, edgecolor="none")

        # Normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 200)
        ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), "r--", linewidth=2,
                label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})")

        # Stats annotation
        skew = returns.skew()
        kurt = returns.kurtosis()
        jb_stat, jb_pval = sp_stats.jarque_bera(returns.dropna())

        stats_text = (
            f"Skewness: {skew:.4f}\n"
            f"Excess Kurtosis: {kurt:.4f}\n"
            f"Jarque-Bera: {jb_stat:.2f} (p={jb_pval:.4f})"
        )
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_title(f"{name} — Daily Returns Distribution", fontweight="bold")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Density")
        ax.legend()

        fig.tight_layout()
        self._save(fig, filename)

    # -------------------------------------------------------------------------
    # Correlation
    # -------------------------------------------------------------------------

    def plot_correlation_heatmap(
        self, corr_matrix: pd.DataFrame,
        filename: str = "correlation_heatmap.png",
    ):
        """Plot a styled correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".3f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
        )

        ax.set_title("Cross-Asset Correlation Matrix (Daily Returns)", fontweight="bold")
        fig.tight_layout()
        self._save(fig, filename)

    # -------------------------------------------------------------------------
    # Drawdown
    # -------------------------------------------------------------------------

    def plot_drawdown(
        self, prices: pd.Series, name: str = "Asset",
        filename: str = "drawdown.png",
    ):
        """Plot underwater (drawdown) chart."""
        returns = prices.pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                        gridspec_kw={"height_ratios": [2, 1]})

        # Price
        ax1.plot(prices.index, prices, linewidth=1.5, color="#2196F3")
        ax1.set_title(f"{name} — Price & Drawdown", fontweight="bold")
        ax1.set_ylabel("Price ($)")

        # Drawdown
        ax2.fill_between(drawdown.index, drawdown, 0, color="#F44336", alpha=0.4)
        ax2.plot(drawdown.index, drawdown, color="#F44336", linewidth=0.5)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")

        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax2.annotate(f"Max DD: {max_dd:.2%}", xy=(max_dd_date, max_dd),
                     fontsize=10, fontweight="bold", color="darkred",
                     xytext=(30, 20), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color="darkred"))

        fig.tight_layout()
        self._save(fig, filename)

    # -------------------------------------------------------------------------
    # Volatility
    # -------------------------------------------------------------------------

    def plot_rolling_volatility(
        self, prices: pd.Series, windows: list[int] = None,
        name: str = "Asset", filename: str = "rolling_volatility.png",
    ):
        """Plot annualized rolling volatility for multiple windows."""
        if windows is None:
            windows = [20, 60, 120]

        returns = prices.pct_change().dropna()
        fig, ax = plt.subplots(figsize=(14, 6))

        colors = ["#2196F3", "#FF9800", "#4CAF50"]
        for i, w in enumerate(windows):
            rolling_vol = returns.rolling(w).std() * np.sqrt(252)
            ax.plot(rolling_vol.index, rolling_vol, label=f"{w}-day",
                    linewidth=1.5, color=colors[i % len(colors)])

        ax.set_title(f"{name} — Rolling Annualized Volatility", fontweight="bold")
        ax.set_ylabel("Annualized Volatility")
        ax.set_xlabel("Date")
        ax.legend()

        fig.tight_layout()
        self._save(fig, filename)

    def plot_garch_volatility(
        self, prices: pd.Series, conditional_vol: pd.Series,
        name: str = "Asset", filename: str = "garch_volatility.png",
    ):
        """
        Overlay GARCH conditional volatility on the price chart.

        Parameters
        ----------
        prices : pd.Series
            Closing price series.
        conditional_vol : pd.Series
            Conditional volatility from a fitted GARCH model.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                        gridspec_kw={"height_ratios": [2, 1]})

        ax1.plot(prices.index, prices, linewidth=1.5, color="#2196F3")
        ax1.set_title(f"{name} — Price & GARCH Conditional Volatility", fontweight="bold")
        ax1.set_ylabel("Price ($)")

        ax2.plot(conditional_vol.index, conditional_vol, color="#FF5722", linewidth=1)
        ax2.fill_between(conditional_vol.index, conditional_vol, alpha=0.3, color="#FF5722")
        ax2.set_ylabel("Conditional Volatility")
        ax2.set_xlabel("Date")

        fig.tight_layout()
        self._save(fig, filename)

    # -------------------------------------------------------------------------
    # Technical Indicators
    # -------------------------------------------------------------------------

    def plot_rsi(
        self, df: pd.DataFrame, name: str = "Asset",
        filename: str = "rsi.png",
    ):
        """Plot RSI with overbought/oversold zones."""
        if "RSI_14" not in df.columns:
            print("  RSI_14 column not found, skipping RSI plot.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={"height_ratios": [2, 1]})

        ax1.plot(df.index, df["Close"], linewidth=1.5, color="#2196F3")
        ax1.set_title(f"{name} — Price & RSI", fontweight="bold")
        ax1.set_ylabel("Price ($)")

        ax2.plot(df.index, df["RSI_14"], linewidth=1, color="#9C27B0")
        ax2.axhline(70, linestyle="--", color="#F44336", alpha=0.7, label="Overbought (70)")
        ax2.axhline(30, linestyle="--", color="#4CAF50", alpha=0.7, label="Oversold (30)")
        ax2.fill_between(df.index, 70, 100, alpha=0.1, color="#F44336")
        ax2.fill_between(df.index, 0, 30, alpha=0.1, color="#4CAF50")
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=9)

        fig.tight_layout()
        self._save(fig, filename)

    # -------------------------------------------------------------------------
    # Summary Dashboard
    # -------------------------------------------------------------------------

    def generate_dashboard(
        self, df: pd.DataFrame, name: str = "Asset",
        corr_matrix: Optional[pd.DataFrame] = None,
    ):
        """
        Generate a full set of charts for a single asset.

        Creates: price history, returns distribution, drawdown,
        rolling volatility, and RSI plots.
        """
        prefix = name.lower().replace(" ", "_").replace("-", "_")

        print(f"\n  Generating charts for {name}...")

        self.plot_price_history(df, title=f"{name} — Price History",
                               filename=f"{prefix}_price_history.png")

        if "Daily_Return" in df.columns:
            self.plot_returns_distribution(
                df["Daily_Return"].dropna(), name=name,
                filename=f"{prefix}_returns_dist.png")

        self.plot_drawdown(df["Close"], name=name,
                          filename=f"{prefix}_drawdown.png")

        self.plot_rolling_volatility(df["Close"], name=name,
                                    filename=f"{prefix}_rolling_vol.png")

        self.plot_rsi(df, name=name, filename=f"{prefix}_rsi.png")

        if corr_matrix is not None:
            self.plot_correlation_heatmap(corr_matrix)
