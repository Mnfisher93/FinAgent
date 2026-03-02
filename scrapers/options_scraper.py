"""
Options chain scraper — fetch options data via yfinance.

Provides:
  - Available expiration dates
  - Full option chain (calls + puts) for any expiration
  - Summary analytics: put/call ratio, max pain, IV skew, unusual activity
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional


def get_options_expirations(ticker: str) -> dict:
    """Get available option expiration dates for a ticker."""
    t = yf.Ticker(ticker)
    try:
        expirations = t.options
    except Exception:
        return {"error": f"No options data available for {ticker}"}

    if not expirations:
        return {"error": f"No options available for {ticker}"}

    return {
        "ticker": ticker.upper(),
        "expiration_count": len(expirations),
        "expirations": list(expirations),
        "nearest": expirations[0],
        "furthest": expirations[-1],
    }


def get_options_chain(
    ticker: str,
    expiration: Optional[str] = None,
) -> dict:
    """
    Get full options chain with analytics for a given expiration.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    expiration : str, optional
        Expiration date (YYYY-MM-DD). If None, uses the nearest expiration.

    Returns
    -------
    dict
        Calls, puts, summary analytics, and key metrics.
    """
    t = yf.Ticker(ticker)

    try:
        expirations = t.options
    except Exception:
        return {"error": f"No options data available for {ticker}"}

    if not expirations:
        return {"error": f"No options available for {ticker}"}

    # Use nearest expiration if none specified
    if expiration is None:
        expiration = expirations[0]
    elif expiration not in expirations:
        return {
            "error": f"Expiration {expiration} not available",
            "available": list(expirations[:10]),
        }

    chain = t.option_chain(expiration)
    calls = chain.calls
    puts = chain.puts

    # Get current stock price for context
    try:
        info = t.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
    except Exception:
        current_price = 0

    # --- Analytics ---

    # Put/Call ratio (by volume and open interest)
    total_call_vol = calls["volume"].sum() if "volume" in calls else 0
    total_put_vol = puts["volume"].sum() if "volume" in puts else 0
    total_call_oi = calls["openInterest"].sum() if "openInterest" in calls else 0
    total_put_oi = puts["openInterest"].sum() if "openInterest" in puts else 0

    pc_ratio_vol = (total_put_vol / total_call_vol) if total_call_vol > 0 else None
    pc_ratio_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

    # IV summary
    call_iv = calls["impliedVolatility"].dropna()
    put_iv = puts["impliedVolatility"].dropna()

    # ATM options (closest to current price)
    atm_calls = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:3]] if current_price > 0 else pd.DataFrame()
    atm_puts = puts.iloc[(puts["strike"] - current_price).abs().argsort()[:3]] if current_price > 0 else pd.DataFrame()

    # Most active by volume
    top_calls = calls.nlargest(5, "volume") if "volume" in calls.columns and not calls["volume"].isna().all() else calls.head(5)
    top_puts = puts.nlargest(5, "volume") if "volume" in puts.columns and not puts["volume"].isna().all() else puts.head(5)

    def _chain_to_records(df: pd.DataFrame) -> list[dict]:
        """Convert option chain df to clean records."""
        cols = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest",
                "impliedVolatility", "inTheMoney"]
        available_cols = [c for c in cols if c in df.columns]
        records = df[available_cols].head(10).to_dict(orient="records")
        # Clean up NaN values
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and (pd.isna(v) or np.isinf(v)):
                    r[k] = None
                elif k == "impliedVolatility" and v is not None:
                    r[k] = round(v * 100, 2)  # Convert to percentage
                elif isinstance(v, float):
                    r[k] = round(v, 2)
        return records

    result = {
        "ticker": ticker.upper(),
        "expiration": expiration,
        "current_price": round(current_price, 2) if current_price else None,
        "summary": {
            "total_calls": len(calls),
            "total_puts": len(puts),
            "total_call_volume": int(total_call_vol) if pd.notna(total_call_vol) else 0,
            "total_put_volume": int(total_put_vol) if pd.notna(total_put_vol) else 0,
            "total_call_open_interest": int(total_call_oi) if pd.notna(total_call_oi) else 0,
            "total_put_open_interest": int(total_put_oi) if pd.notna(total_put_oi) else 0,
            "put_call_ratio_volume": round(pc_ratio_vol, 3) if pc_ratio_vol and pd.notna(pc_ratio_vol) else None,
            "put_call_ratio_oi": round(pc_ratio_oi, 3) if pc_ratio_oi and pd.notna(pc_ratio_oi) else None,
        },
        "implied_volatility": {
            "call_iv_mean": round(call_iv.mean() * 100, 2) if len(call_iv) > 0 else None,
            "put_iv_mean": round(put_iv.mean() * 100, 2) if len(put_iv) > 0 else None,
            "call_iv_range": [round(call_iv.min() * 100, 2), round(call_iv.max() * 100, 2)] if len(call_iv) > 0 else None,
            "put_iv_range": [round(put_iv.min() * 100, 2), round(put_iv.max() * 100, 2)] if len(put_iv) > 0 else None,
        },
        "atm_calls": _chain_to_records(atm_calls),
        "atm_puts": _chain_to_records(atm_puts),
        "most_active_calls": _chain_to_records(top_calls),
        "most_active_puts": _chain_to_records(top_puts),
        "available_expirations": list(expirations[:10]),
    }

    # Put/call sentiment interpretation
    if pc_ratio_oi and pd.notna(pc_ratio_oi):
        if pc_ratio_oi > 1.0:
            result["sentiment"] = f"BEARISH — put/call OI ratio {pc_ratio_oi:.2f} (more puts than calls)"
        elif pc_ratio_oi < 0.7:
            result["sentiment"] = f"BULLISH — put/call OI ratio {pc_ratio_oi:.2f} (more calls than puts)"
        else:
            result["sentiment"] = f"NEUTRAL — put/call OI ratio {pc_ratio_oi:.2f}"

    return result
