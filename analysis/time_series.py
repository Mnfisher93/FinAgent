"""
Time series analysis module.

Provides graduate-level time series modeling tools:
- Stationarity testing (ADF, KPSS)
- ARIMA model fitting and forecasting
- GARCH volatility modeling
- Rolling window predictions
- Autocorrelation analysis
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


class TimeSeriesAnalyzer:
    """
    Time series analysis engine for financial data.

    Parameters
    ----------
    trading_days : int
        Number of trading days per year (default: 252).
    """

    def __init__(self, trading_days: int = 252):
        self.trading_days = trading_days

    # -------------------------------------------------------------------------
    # Stationarity Tests
    # -------------------------------------------------------------------------

    def adf_test(self, series: pd.Series, significance: float = 0.05) -> dict:
        """
        Augmented Dickey-Fuller test for unit root (non-stationarity).

        H0: Series has a unit root (non-stationary).
        If p-value < significance → reject H0 → series is stationary.

        Parameters
        ----------
        series : pd.Series
            Time series to test.
        significance : float
            Significance level (default: 0.05).

        Returns
        -------
        dict
            Test statistic, p-value, critical values, and conclusion.
        """
        clean = series.dropna()
        result = adfuller(clean, autolag="AIC")

        return {
            "test": "Augmented Dickey-Fuller",
            "statistic": result[0],
            "p_value": result[1],
            "lags_used": result[2],
            "n_observations": result[3],
            "critical_values": result[4],
            "is_stationary": result[1] < significance,
            "conclusion": (
                f"Stationary (p={result[1]:.6f} < {significance})"
                if result[1] < significance
                else f"Non-stationary (p={result[1]:.6f} >= {significance})"
            ),
        }

    def kpss_test(self, series: pd.Series, significance: float = 0.05) -> dict:
        """
        KPSS test for stationarity.

        H0: Series is stationary.
        If p-value < significance → reject H0 → series is non-stationary.

        Note: KPSS has opposite null hypothesis from ADF.
        """
        clean = series.dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value, lags, critical_values = kpss(clean, regression="c", nlags="auto")

        return {
            "test": "KPSS",
            "statistic": stat,
            "p_value": p_value,
            "lags_used": lags,
            "critical_values": critical_values,
            "is_stationary": p_value >= significance,
            "conclusion": (
                f"Stationary (p={p_value:.6f} >= {significance})"
                if p_value >= significance
                else f"Non-stationary (p={p_value:.6f} < {significance})"
            ),
        }

    def stationarity_report(self, series: pd.Series, name: str = "Series") -> dict:
        """
        Run both ADF and KPSS tests and provide a combined interpretation.

        Interpretations:
        - Both stationary → Series is stationary
        - ADF stationary, KPSS non-stationary → Difference-stationary (trend)
        - ADF non-stationary, KPSS stationary → Stationary around a deterministic trend
        - Both non-stationary → Non-stationary, needs differencing
        """
        adf = self.adf_test(series)
        kpss_result = self.kpss_test(series)

        if adf["is_stationary"] and kpss_result["is_stationary"]:
            interpretation = "Stationary — no differencing needed"
        elif adf["is_stationary"] and not kpss_result["is_stationary"]:
            interpretation = "Difference-stationary — has a stochastic trend"
        elif not adf["is_stationary"] and kpss_result["is_stationary"]:
            interpretation = "Trend-stationary — stationary around deterministic trend"
        else:
            interpretation = "Non-stationary — differencing required"

        return {
            "name": name,
            "adf": adf,
            "kpss": kpss_result,
            "interpretation": interpretation,
        }

    # -------------------------------------------------------------------------
    # Autocorrelation
    # -------------------------------------------------------------------------

    def autocorrelation(self, series: pd.Series, nlags: int = 40) -> dict:
        """
        Compute ACF and PACF for lag selection.

        Returns
        -------
        dict
            Contains 'acf' and 'pacf' arrays and suggested AR/MA orders.
        """
        clean = series.dropna()
        acf_vals = acf(clean, nlags=nlags)
        pacf_vals = pacf(clean, nlags=nlags)

        # Confidence interval: ±1.96/√n
        ci = 1.96 / np.sqrt(len(clean))

        # Suggest AR order from PACF (first lag outside CI after which it drops)
        significant_pacf = np.where(np.abs(pacf_vals[1:]) > ci)[0]
        suggested_ar = int(significant_pacf[-1] + 1) if len(significant_pacf) > 0 else 0

        # Suggest MA order from ACF
        significant_acf = np.where(np.abs(acf_vals[1:]) > ci)[0]
        suggested_ma = int(significant_acf[-1] + 1) if len(significant_acf) > 0 else 0

        return {
            "acf": acf_vals,
            "pacf": pacf_vals,
            "confidence_interval": ci,
            "suggested_ar_order": min(suggested_ar, 5),
            "suggested_ma_order": min(suggested_ma, 5),
        }

    # -------------------------------------------------------------------------
    # ARIMA Modeling
    # -------------------------------------------------------------------------

    def fit_arima(
        self, series: pd.Series,
        order: tuple = None,
        auto_order: bool = True,
    ) -> dict:
        """
        Fit an ARIMA(p,d,q) model to the series.

        Parameters
        ----------
        series : pd.Series
            Time series (typically log returns or differenced prices).
        order : tuple
            (p, d, q) order. If None and auto_order=True, will be estimated.
        auto_order : bool
            If True and order is None, estimate order from ACF/PACF.

        Returns
        -------
        dict
            Model results including coefficients, AIC, BIC, residuals, and fitted model.
        """
        clean = series.dropna()

        if order is None and auto_order:
            # Determine differencing order
            adf = self.adf_test(clean)
            d = 0 if adf["is_stationary"] else 1

            if d > 0:
                differenced = clean.diff().dropna()
            else:
                differenced = clean

            ac = self.autocorrelation(differenced)
            p = min(ac["suggested_ar_order"], 3)
            q = min(ac["suggested_ma_order"], 3)
            order = (p, d, q)

        print(f"  Fitting ARIMA{order}...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(clean, order=order)
            fitted = model.fit()

        return {
            "order": order,
            "aic": fitted.aic,
            "bic": fitted.bic,
            "coefficients": dict(fitted.params),
            "residuals": fitted.resid,
            "fitted_model": fitted,
            "summary": str(fitted.summary()),
        }

    def arima_forecast(
        self, fitted_model, steps: int = 30,
    ) -> pd.DataFrame:
        """
        Generate out-of-sample forecasts from a fitted ARIMA model.

        Parameters
        ----------
        fitted_model : ARIMAResults
            The fitted ARIMA model object.
        steps : int
            Number of steps ahead to forecast.

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame with columns: forecast, lower_ci, upper_ci.
        """
        forecast_result = fitted_model.get_forecast(steps=steps)
        forecast_df = pd.DataFrame({
            "forecast": forecast_result.predicted_mean,
            "lower_ci": forecast_result.conf_int().iloc[:, 0],
            "upper_ci": forecast_result.conf_int().iloc[:, 1],
        })
        return forecast_df

    # -------------------------------------------------------------------------
    # GARCH Volatility Modeling
    # -------------------------------------------------------------------------

    def fit_garch(
        self, returns: pd.Series,
        p: int = 1, q: int = 1,
        vol: str = "Garch", dist: str = "normal",
    ) -> dict:
        """
        Fit a GARCH(p,q) model to return series for volatility modeling.

        Parameters
        ----------
        returns : pd.Series
            Return series (daily returns, typically scaled by 100).
        p : int
            GARCH lag order (default: 1).
        q : int
            ARCH lag order (default: 1).
        vol : str
            Volatility model: 'Garch', 'EGARCH', 'TARCH'.
        dist : str
            Error distribution: 'normal', 't', 'skewt'.

        Returns
        -------
        dict
            Model results including params, conditional volatility, and forecasts.
        """
        clean = returns.dropna() * 100  # Scale for numerical stability

        print(f"  Fitting {vol}({p},{q}) with {dist} distribution...")

        model = arch_model(clean, vol=vol, p=p, q=q, dist=dist, mean="Constant")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = model.fit(disp="off")

        cond_vol = fitted.conditional_volatility / 100  # Rescale back

        return {
            "model_type": f"{vol}({p},{q})",
            "distribution": dist,
            "params": dict(fitted.params),
            "aic": fitted.aic,
            "bic": fitted.bic,
            "conditional_volatility": cond_vol,
            "standardized_residuals": fitted.resid / fitted.conditional_volatility,
            "fitted_model": fitted,
            "summary": str(fitted.summary()),
        }

    def garch_forecast(self, fitted_model, horizon: int = 10) -> pd.DataFrame:
        """
        Forecast conditional variance from a fitted GARCH model.

        Parameters
        ----------
        fitted_model : ARCHModelResult
            The fitted GARCH model object.
        horizon : int
            Forecast horizon in days.

        Returns
        -------
        pd.DataFrame
            Forecast variance and volatility.
        """
        forecast = fitted_model.forecast(horizon=horizon)
        variance = forecast.variance.iloc[-1]
        vol = np.sqrt(variance) / 100  # Rescale

        return pd.DataFrame({
            "forecast_variance": variance.values / 10000,
            "forecast_volatility": vol.values,
        }, index=range(1, horizon + 1))

    # -------------------------------------------------------------------------
    # Rolling Window Analysis
    # -------------------------------------------------------------------------

    def rolling_forecast(
        self, series: pd.Series,
        window: int = 252, horizon: int = 1,
        arima_order: tuple = (1, 0, 1),
    ) -> pd.DataFrame:
        """
        Rolling window ARIMA forecast — walk-forward validation.

        At each step, fit on the trailing `window` observations and
        forecast `horizon` steps ahead, then compare to actuals.

        Parameters
        ----------
        series : pd.Series
            Full time series.
        window : int
            Training window size.
        horizon : int
            Steps ahead to forecast at each roll.
        arima_order : tuple
            (p, d, q) for ARIMA.

        Returns
        -------
        pd.DataFrame
            Columns: actual, forecast, error.
        """
        clean = series.dropna()
        n = len(clean)
        results = []

        print(f"  Rolling forecast: window={window}, horizon={horizon}, n={n}...")

        for i in range(window, n - horizon + 1):
            train = clean.iloc[i - window:i]
            actual = clean.iloc[i + horizon - 1]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(train, order=arima_order)
                    fitted = model.fit()
                    pred = fitted.forecast(steps=horizon).iloc[-1]
            except Exception:
                pred = np.nan

            results.append({
                "date": clean.index[i + horizon - 1],
                "actual": actual,
                "forecast": pred,
            })

        df = pd.DataFrame(results).set_index("date")
        df["error"] = df["actual"] - df["forecast"]
        df["abs_error"] = df["error"].abs()
        df["pct_error"] = (df["error"] / df["actual"]).abs()

        mae = df["abs_error"].mean()
        rmse = np.sqrt((df["error"] ** 2).mean())
        print(f"  Rolling forecast — MAE: {mae:.6f}, RMSE: {rmse:.6f}")

        return df

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def full_analysis(self, prices: pd.Series, name: str = "Asset") -> dict:
        """
        Run a complete time series analysis pipeline on a price series.

        Returns
        -------
        dict
            Combined results from stationarity tests, ARIMA, and GARCH.
        """
        print(f"\n{'='*60}")
        print(f"  Time Series Analysis: {name}")
        print(f"{'='*60}")

        log_returns = np.log(prices / prices.shift(1)).dropna()

        # Stationarity
        print("\n[1] Stationarity Tests...")
        stationarity = self.stationarity_report(log_returns, name=f"{name} Log Returns")
        print(f"    ADF: {stationarity['adf']['conclusion']}")
        print(f"    KPSS: {stationarity['kpss']['conclusion']}")
        print(f"    → {stationarity['interpretation']}")

        # ARIMA
        print("\n[2] ARIMA Modeling...")
        arima_result = self.fit_arima(log_returns)
        print(f"    Order: {arima_result['order']}, AIC: {arima_result['aic']:.2f}")

        # GARCH
        print("\n[3] GARCH Volatility Modeling...")
        garch_result = self.fit_garch(log_returns)
        print(f"    Type: {garch_result['model_type']}, AIC: {garch_result['aic']:.2f}")

        return {
            "name": name,
            "stationarity": stationarity,
            "arima": arima_result,
            "garch": garch_result,
        }
