"""
Polymarket prediction market scraper.
Fetches active markets from the Gamma Markets API.
"""

import json
from typing import Optional
import pandas as pd
import requests


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"


def fetch_polymarket_data(limit: int = 20, min_volume: float = 10000) -> pd.DataFrame:
    """Fetch active prediction markets sorted by volume."""
    url = f"{GAMMA_BASE_URL}/markets"
    params = {"limit": limit, "order": "volume24hr", "ascending": "false",
              "active": "true", "closed": "false"}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    markets = resp.json()

    records = []
    for m in markets:
        outcomes = m.get("outcomes", "")
        prices = m.get("outcomePrices", "")
        if isinstance(outcomes, str):
            try: outcomes = json.loads(outcomes)
            except: outcomes = []
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []

        vol = float(m.get("volume", 0) or 0)
        if vol >= min_volume:
            records.append({
                "question": m.get("question", ""),
                "outcomes": str(outcomes),
                "outcome_prices": str(prices),
                "volume": vol,
                "volume_24hr": float(m.get("volume24hr", 0) or 0),
                "liquidity": float(m.get("liquidity", 0) or 0),
                "active": m.get("active", False),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values("volume", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df
