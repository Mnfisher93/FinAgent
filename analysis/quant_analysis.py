"""
Quantitative analysis module.

Provides graduate-level risk metrics, return analysis, and cross-asset statistics.

Includes:
- Daily, log, and cumulative returns
- Annualized volatility and Sharpe/Sortino ratios
- Maximum drawdown analysis
- Value-at-Risk (historical, parametric, Monte Carlo)
- Correlation and covariance matrices
- Rolling statistics
"""

import numpy as np
import pandas as pd
from scipy import stats


class QuantAnalyzer:
    """
    Quantitative analysis engine for financial time series.

    Parameters
    ----------
    risk_free_rate : float
        Annualized risk-free rate for Sharpe/Sortino calculations (default: 0.05 = 5%).
    trading_days : int
        Number of trading days per year (default: 252).
    """

    def __init__(self, risk_free_rate: float = 0.05, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    # -------------------------------------------------------------------------
    # Returns
    # -------------------------------------------------------------------------

    def daily_returns(self, prices: pd.Series) -> pd.Series:
        """Simple daily returns: (P_t - P_{t-1}) / P_{t-1}."""
        return prices.pct_change().dropna()

    def log_returns(self, prices: pd.Series) -> pd.Series:
        """Continuously compounded (log) returns: ln(P_t / P_{t-1})."""
        return np.log(prices / prices.shift(1)).dropna()

    def cumulative_returns(self, prices: pd.Series) -> pd.Series:
        """Cumulative return series from a price series."""
        return (1 + self.daily_returns(prices)).cumprod() - 1

    def total_return(self, prices: pd.Series) -> float:
        """Total return over the full period."""
        return (prices.iloc[-1] / prices.iloc[0]) - 1

    # -------------------------------------------------------------------------
    # Risk Metrics
    # -------------------------------------------------------------------------

    def annualized_volatility(self, prices: pd.Series) -> float:
        """Annualized volatility (std of daily returns * sqrt(252))."""
        return self.daily_returns(prices).std() * np.sqrt(self.trading_days)

    def annualized_return(self, prices: pd.Series) -> float:
        """Annualized return (geometric mean)."""
        daily_ret = self.daily_returns(prices)
        geo_mean = (1 + daily_ret).prod() ** (self.trading_days / len(daily_ret)) - 1
        return geo_mean

    def sharpe_ratio(self, prices: pd.Series) -> float:
        """
        Annualized Sharpe Ratio.

        SR = (R_p - R_f) / σ_p
        """
        excess_return = self.annualized_return(prices) - self.risk_free_rate
        vol = self.annualized_volatility(prices)
        return excess_return / vol if vol != 0 else 0.0

    def sortino_ratio(self, prices: pd.Series) -> float:
        """
        Sortino Ratio — penalizes only downside volatility.

        Sortino = (R_p - R_f) / σ_downside
        """
        daily_ret = self.daily_returns(prices)
        excess_return = self.annualized_return(prices) - self.risk_free_rate
        downside = daily_ret[daily_ret < 0]
        downside_std = downside.std() * np.sqrt(self.trading_days)
        return excess_return / downside_std if downside_std != 0 else 0.0

    def max_drawdown(self, prices: pd.Series) -> dict:
        """
        Maximum drawdown — largest peak-to-trough decline.

        Returns
        -------
        dict
            Contains 'max_drawdown' (float), 'peak_date', 'trough_date'.
        """
        cumulative = (1 + self.daily_returns(prices)).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        trough_idx = drawdown.idxmin()
        peak_idx = cumulative.loc[:trough_idx].idxmax()

        return {
            "max_drawdown": max_dd,
            "peak_date": peak_idx,
            "trough_date": trough_idx,
        }

    def calmar_ratio(self, prices: pd.Series) -> float:
        """Calmar Ratio = Annualized Return / |Max Drawdown|."""
        ann_ret = self.annualized_return(prices)
        mdd = abs(self.max_drawdown(prices)["max_drawdown"])
        return ann_ret / mdd if mdd != 0 else 0.0

    # -------------------------------------------------------------------------
    # Value-at-Risk (VaR)
    # -------------------------------------------------------------------------

    def var_historical(self, prices: pd.Series, confidence: float = 0.95) -> float:
        """
        Historical VaR at given confidence level.

        Returns the loss threshold such that losses exceed this value
        only (1 - confidence)% of the time.
        """
        daily_ret = self.daily_returns(prices)
        return np.percentile(daily_ret, (1 - confidence) * 100)

    def var_parametric(self, prices: pd.Series, confidence: float = 0.95) -> float:
        """
        Parametric (Gaussian) VaR assuming normally distributed returns.
        """
        daily_ret = self.daily_returns(prices)
        mu = daily_ret.mean()
        sigma = daily_ret.std()
        z = stats.norm.ppf(1 - confidence)
        return mu + z * sigma

    def var_monte_carlo(
        self, prices: pd.Series, confidence: float = 0.95,
        n_simulations: int = 10000, horizon: int = 1,
    ) -> float:
        """
        Monte Carlo VaR via geometric Brownian motion simulation.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.95 or 0.99).
        n_simulations : int
            Number of Monte Carlo paths to simulate.
        horizon : int
            Forecast horizon in days.
        """
        daily_ret = self.daily_returns(prices)
        mu = daily_ret.mean()
        sigma = daily_ret.std()
        last_price = prices.iloc[-1]

        simulated_returns = np.random.normal(mu, sigma, (n_simulations, horizon))
        simulated_prices = last_price * np.exp(np.cumsum(simulated_returns, axis=1))
        final_returns = (simulated_prices[:, -1] - last_price) / last_price

        return np.percentile(final_returns, (1 - confidence) * 100)

    def cvar(self, prices: pd.Series, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall) — average loss beyond VaR."""
        daily_ret = self.daily_returns(prices)
        var = self.var_historical(prices, confidence)
        return daily_ret[daily_ret <= var].mean()

    # -------------------------------------------------------------------------
    # Correlation & Covariance
    # -------------------------------------------------------------------------

    def correlation_matrix(self, price_dict: dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute pairwise correlation matrix from a dict of price series.

        Parameters
        ----------
        price_dict : dict[str, pd.Series]
            Mapping of asset name → closing price series.

        Returns
        -------
        pd.DataFrame
            Correlation matrix of daily returns.
        """
        returns_df = pd.DataFrame({
            name: self.daily_returns(prices) for name, prices in price_dict.items()
        }).dropna()
        return returns_df.corr()

    def covariance_matrix(self, price_dict: dict[str, pd.Series]) -> pd.DataFrame:
        """Annualized covariance matrix of daily returns."""
        returns_df = pd.DataFrame({
            name: self.daily_returns(prices) for name, prices in price_dict.items()
        }).dropna()
        return returns_df.cov() * self.trading_days

    # -------------------------------------------------------------------------
    # Rolling Statistics
    # -------------------------------------------------------------------------

    def rolling_volatility(self, prices: pd.Series, window: int = 30) -> pd.Series:
        """Rolling annualized volatility."""
        daily_ret = self.daily_returns(prices)
        return daily_ret.rolling(window=window).std() * np.sqrt(self.trading_days)

    def rolling_sharpe(self, prices: pd.Series, window: int = 60) -> pd.Series:
        """Rolling annualized Sharpe ratio."""
        daily_ret = self.daily_returns(prices)
        rolling_mean = daily_ret.rolling(window=window).mean() * self.trading_days
        rolling_std = daily_ret.rolling(window=window).std() * np.sqrt(self.trading_days)
        return (rolling_mean - self.risk_free_rate) / rolling_std

    def rolling_beta(
        self, asset_prices: pd.Series, market_prices: pd.Series, window: int = 60,
    ) -> pd.Series:
        """Rolling beta of asset returns against market returns."""
        asset_ret = self.daily_returns(asset_prices)
        market_ret = self.daily_returns(market_prices)

        aligned = pd.DataFrame({"asset": asset_ret, "market": market_ret}).dropna()

        cov = aligned["asset"].rolling(window).cov(aligned["market"])
        var = aligned["market"].rolling(window).var()
        return cov / var

    # -------------------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------------------

    def summary(self, prices: pd.Series, name: str = "Asset") -> dict:
        """
        Generate a comprehensive summary of risk/return metrics.

        Returns
        -------
        dict
            All key metrics in a single dictionary.
        """
        dd = self.max_drawdown(prices)
        return {
            "asset": name,
            "total_return": f"{self.total_return(prices):.4%}",
            "annualized_return": f"{self.annualized_return(prices):.4%}",
            "annualized_volatility": f"{self.annualized_volatility(prices):.4%}",
            "sharpe_ratio": f"{self.sharpe_ratio(prices):.4f}",
            "sortino_ratio": f"{self.sortino_ratio(prices):.4f}",
            "calmar_ratio": f"{self.calmar_ratio(prices):.4f}",
            "max_drawdown": f"{dd['max_drawdown']:.4%}",
            "max_dd_peak": str(dd["peak_date"]),
            "max_dd_trough": str(dd["trough_date"]),
            "var_95_historical": f"{self.var_historical(prices, 0.95):.4%}",
            "var_99_historical": f"{self.var_historical(prices, 0.99):.4%}",
            "var_95_parametric": f"{self.var_parametric(prices, 0.95):.4%}",
            "cvar_95": f"{self.cvar(prices, 0.95):.4%}",
            "var_95_monte_carlo": f"{self.var_monte_carlo(prices, 0.95):.4%}",
            "skewness": f"{self.daily_returns(prices).skew():.4f}",
            "kurtosis": f"{self.daily_returns(prices).kurtosis():.4f}",
        }
