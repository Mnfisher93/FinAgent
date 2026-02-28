"""
Advanced statistical analysis module.

Implements methods from juliajerke/advanced-statistics-with-python,
adapted for financial data analysis:

- Linear Regression (factor analysis, trend modeling)
- Logistic Regression (up/down day prediction)
- K-Means Clustering (stock behavior grouping)
- PCA (dimensionality reduction on multi-asset returns)
- Random Forest (feature importance for price drivers)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report


# ============================================================================
# Linear Regression
# ============================================================================

def run_linear_regression(
    y_prices: pd.Series,
    x_prices: pd.Series,
    y_name: str = "Y",
    x_name: str = "X",
) -> dict:
    """
    Fit a linear regression: Y returns = α + β * X returns.

    Use cases:
    - CAPM beta estimation (Y = stock, X = market index)
    - Factor analysis (how much does X explain Y?)

    Parameters
    ----------
    y_prices : pd.Series
        Dependent variable (price series).
    x_prices : pd.Series
        Independent variable (price series).

    Returns
    -------
    dict
        Alpha, beta, R², and statistical summary.
    """
    y_ret = y_prices.pct_change().dropna()
    x_ret = x_prices.pct_change().dropna()

    # Align series
    aligned = pd.DataFrame({"y": y_ret, "x": x_ret}).dropna()

    X = aligned["x"].values.reshape(-1, 1)
    y = aligned["y"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    return {
        "type": "linear_regression",
        "dependent": y_name,
        "independent": x_name,
        "alpha": float(model.intercept_),
        "beta": float(model.coef_[0]),
        "r_squared": float(r2),
        "n_observations": len(aligned),
        "interpretation": (
            f"{y_name} has a beta of {model.coef_[0]:.4f} relative to {x_name}. "
            f"R² = {r2:.4f} — {x_name} explains {r2*100:.1f}% of {y_name}'s return variance."
        ),
    }


def run_multi_regression(
    y_prices: pd.Series,
    x_dict: dict[str, pd.Series],
    y_name: str = "Y",
) -> dict:
    """
    Multi-factor linear regression: Y = α + β₁X₁ + β₂X₂ + ...

    Parameters
    ----------
    y_prices : pd.Series
        Dependent variable.
    x_dict : dict[str, pd.Series]
        Multiple independent variables.

    Returns
    -------
    dict
        Coefficients, R², and factor contributions.
    """
    y_ret = y_prices.pct_change().dropna()
    x_rets = pd.DataFrame({
        name: prices.pct_change().dropna()
        for name, prices in x_dict.items()
    })

    aligned = pd.concat([y_ret.rename("y"), x_rets], axis=1).dropna()
    X = aligned.drop(columns=["y"]).values
    y = aligned["y"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    factors = dict(zip(x_dict.keys(), [float(c) for c in model.coef_]))

    return {
        "type": "multi_factor_regression",
        "dependent": y_name,
        "alpha": float(model.intercept_),
        "factor_betas": factors,
        "r_squared": float(r2),
        "n_observations": len(aligned),
    }


# ============================================================================
# Logistic Regression (Up/Down Prediction)
# ============================================================================

def run_logistic_regression(
    prices: pd.Series,
    lookback_days: list[int] = None,
    name: str = "Asset",
) -> dict:
    """
    Predict whether next day returns will be positive (up) or negative (down)
    using lagged returns as features.

    Parameters
    ----------
    prices : pd.Series
        Price series.
    lookback_days : list[int]
        Lag periods to use as features (default: [1, 2, 3, 5, 10]).

    Returns
    -------
    dict
        Accuracy, feature importances, and predictions.
    """
    if lookback_days is None:
        lookback_days = [1, 2, 3, 5, 10]

    returns = prices.pct_change().dropna()

    # Build feature matrix: lagged returns
    features = pd.DataFrame()
    for lag in lookback_days:
        features[f"ret_lag_{lag}"] = returns.shift(lag)

    # Target: 1 if next day is up, 0 if down
    target = (returns > 0).astype(int)

    data = pd.concat([features, target.rename("target")], axis=1).dropna()

    X = data.drop(columns=["target"]).values
    y = data["target"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)

    feature_importance = dict(zip(
        [f"ret_lag_{d}" for d in lookback_days],
        [float(c) for c in model.coef_[0]]
    ))

    return {
        "type": "logistic_regression",
        "asset": name,
        "target": "next_day_direction (1=up, 0=down)",
        "features": [f"ret_lag_{d}" for d in lookback_days],
        "accuracy": f"{accuracy:.2%}",
        "baseline_accuracy": f"{(y_test.mean() if y_test.mean() > 0.5 else 1 - y_test.mean()):.2%}",
        "feature_importance": feature_importance,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "interpretation": (
            f"Logistic regression predicts {name}'s next-day direction with "
            f"{accuracy:.1%} accuracy (baseline: {max(y_test.mean(), 1-y_test.mean()):.1%})."
        ),
    }


# ============================================================================
# K-Means Clustering
# ============================================================================

def run_clustering(
    price_dict: dict[str, pd.Series],
    n_clusters: int = 3,
) -> dict:
    """
    Cluster assets by their return characteristics using K-Means.

    Features used: mean return, volatility, skewness, kurtosis.

    Parameters
    ----------
    price_dict : dict[str, pd.Series]
        Mapping of asset name → price series.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    dict
        Cluster assignments and cluster centers.
    """
    features = []
    names = []

    for name, prices in price_dict.items():
        ret = prices.pct_change().dropna()
        features.append({
            "mean_return": ret.mean(),
            "volatility": ret.std(),
            "skewness": ret.skew(),
            "kurtosis": ret.kurtosis(),
        })
        names.append(name)

    df = pd.DataFrame(features, index=names)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    n_clusters = min(n_clusters, len(names))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Group assets by cluster
    clusters = {}
    for i, name in enumerate(names):
        cluster_id = int(labels[i])
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(name)

    return {
        "type": "kmeans_clustering",
        "n_clusters": n_clusters,
        "n_assets": len(names),
        "clusters": clusters,
        "asset_features": df.to_dict(orient="index"),
        "feature_columns": ["mean_return", "volatility", "skewness", "kurtosis"],
    }


# ============================================================================
# Principal Component Analysis (PCA)
# ============================================================================

def run_pca(
    price_dict: dict[str, pd.Series],
    n_components: int = None,
) -> dict:
    """
    PCA on multi-asset daily returns to find principal risk factors.

    Parameters
    ----------
    price_dict : dict[str, pd.Series]
        Mapping of asset name → price series.
    n_components : int, optional
        Number of components (default: min of n_assets, 5).

    Returns
    -------
    dict
        Explained variance, component loadings, and interpretation.
    """
    returns_df = pd.DataFrame({
        name: prices.pct_change().dropna()
        for name, prices in price_dict.items()
    }).dropna()

    if n_components is None:
        n_components = min(len(price_dict), 5)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(returns_df.values)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=returns_df.columns,
    )

    return {
        "type": "pca",
        "n_components": n_components,
        "n_assets": len(price_dict),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cumulative_variance": [float(v) for v in np.cumsum(pca.explained_variance_ratio_)],
        "loadings": loadings.to_dict(),
        "interpretation": (
            f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance. "
            f"Top {n_components} components explain "
            f"{sum(pca.explained_variance_ratio_)*100:.1f}% total."
        ),
    }


# ============================================================================
# Random Forest Feature Importance
# ============================================================================

def run_feature_importance(
    target_prices: pd.Series,
    feature_dict: dict[str, pd.Series],
    target_name: str = "Target",
) -> dict:
    """
    Use Random Forest to identify which factors most influence an asset's direction.

    Parameters
    ----------
    target_prices : pd.Series
        Target asset prices.
    feature_dict : dict[str, pd.Series]
        Potential driver assets.

    Returns
    -------
    dict
        Feature importances ranked by influence.
    """
    target_ret = target_prices.pct_change().dropna()
    feature_rets = pd.DataFrame({
        name: prices.pct_change().dropna()
        for name, prices in feature_dict.items()
    })

    aligned = pd.concat([target_ret.rename("target"), feature_rets], axis=1).dropna()

    X = aligned.drop(columns=["target"]).values
    y = (aligned["target"] > 0).astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    importances = dict(zip(
        feature_dict.keys(),
        [float(v) for v in model.feature_importances_]
    ))
    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    return {
        "type": "random_forest_feature_importance",
        "target": target_name,
        "accuracy": f"{accuracy:.2%}",
        "feature_importances": importances,
        "n_estimators": 100,
        "interpretation": (
            f"Most important factor for {target_name}: "
            f"{list(importances.keys())[0]} ({list(importances.values())[0]:.3f})"
        ),
    }
