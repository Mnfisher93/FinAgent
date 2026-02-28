"""
Nasdaq Data Link scraper — integrates all three Nasdaq APIs.

1. Tables API (free with API key) — historical fundamentals, bulk datasets
2. REST API (premium) — real-time/delayed: last sale, quote, snapshot, trends, bars
3. Streaming API (premium) — real-time tick data via Python SDK

Free API key: https://data.nasdaq.com/sign-up
Premium access: contact Nasdaq sales team
"""

import os
import json
from typing import Optional
from datetime import datetime, timedelta

import requests
import nasdaqdatalink
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

TABLES_BASE_URL = "https://data.nasdaq.com/api/v3/datatables"
REALTIME_BASE_URL = None  # Set from NASDAQ_BASE_URL env var when premium


def _get_api_key() -> Optional[str]:
    return os.environ.get("NASDAQ_API_KEY")


def _get_realtime_token() -> Optional[str]:
    """Get OAuth2 token for real-time REST API (premium)."""
    client_id = os.environ.get("NASDAQ_CLIENT_ID")
    client_secret = os.environ.get("NASDAQ_CLIENT_SECRET")
    token_url = os.environ.get("NASDAQ_TOKEN_URL")

    if not all([client_id, client_secret, token_url]):
        return None

    resp = requests.post(token_url, data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json().get("access_token")


# ============================================================================
# 1. TABLES API (Free)
# ============================================================================

def get_nasdaq_table(
    table_code: str,
    filters: Optional[dict] = None,
    columns: Optional[list[str]] = None,
    rows: int = 100,
) -> pd.DataFrame:
    """
    Query any Nasdaq Data Link table.

    Parameters
    ----------
    table_code : str
        Table code (e.g., 'WIKI/PRICES', 'ZACKS/FC', 'MER/F1').
    filters : dict, optional
        Row filters (e.g., {'ticker': 'AAPL', 'date.gte': '2024-01-01'}).
    columns : list[str], optional
        Columns to return.
    rows : int
        Max rows to return.

    Returns
    -------
    pd.DataFrame
        Query results.

    Example
    -------
    >>> get_nasdaq_table('WIKI/PRICES', filters={'ticker': 'AAPL'}, rows=10)
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("NASDAQ_API_KEY not set in .env")

    url = f"{TABLES_BASE_URL}/{table_code}.json"
    params = {"api_key": api_key, "qopts.per_page": rows}

    if filters:
        params.update(filters)
    if columns:
        params["qopts.columns"] = ",".join(columns)

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    dt = data.get("datatable", {})
    col_names = [c["name"] for c in dt.get("columns", [])]
    rows_data = dt.get("data", [])

    if not rows_data:
        return pd.DataFrame()

    return pd.DataFrame(rows_data, columns=col_names)


def get_nasdaq_fundamentals(ticker: str) -> dict:
    """
    Get company fundamentals from free Nasdaq datasets.

    Tries multiple free data sources in order of availability.
    """
    api_key = _get_api_key()
    if not api_key:
        return {"error": "NASDAQ_API_KEY not set"}

    # Try Zacks Fundamentals (ZACKS/FC) — common free dataset
    try:
        df = get_nasdaq_table(
            "ZACKS/FC",
            filters={"ticker": ticker.upper()},
            rows=5,
        )
        if not df.empty:
            return {
                "source": "ZACKS/FC",
                "ticker": ticker.upper(),
                "data": df.to_dict(orient="records"),
            }
    except Exception:
        pass

    # Try Sharadar fundamentals (SHARADAR/SF1)
    try:
        df = get_nasdaq_table(
            "SHARADAR/SF1",
            filters={"ticker": ticker.upper(), "dimension": "MRY"},
            rows=5,
        )
        if not df.empty:
            return {
                "source": "SHARADAR/SF1",
                "ticker": ticker.upper(),
                "data": df.to_dict(orient="records"),
            }
    except Exception:
        pass

    return {
        "ticker": ticker.upper(),
        "message": "No free fundamental data found. Some datasets require premium access.",
    }


def search_nasdaq_datasets(query: str) -> list[dict]:
    """
    Search for available Nasdaq Data Link datasets.

    Parameters
    ----------
    query : str
        Search term (e.g., 'Apple', 'oil prices', 'GDP').

    Returns
    -------
    list[dict]
        Matching datasets with codes and descriptions.
    """
    api_key = _get_api_key()
    if not api_key:
        return [{"error": "NASDAQ_API_KEY not set"}]

    url = "https://data.nasdaq.com/api/v3/datasets.json"
    params = {"api_key": api_key, "query": query, "per_page": 10}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    return [
        {
            "code": ds.get("dataset_code"),
            "database_code": ds.get("database_code"),
            "name": ds.get("name"),
            "description": (ds.get("description") or "")[:200],
            "frequency": ds.get("frequency"),
            "newest_date": ds.get("newest_available_date"),
        }
        for ds in data.get("datasets", [])
    ]


# ============================================================================
# 2. REST API — Real-Time/Delayed (Premium)
# ============================================================================

def _realtime_request(endpoint: str, params: dict = None) -> Optional[dict]:
    """Make an authenticated request to the Nasdaq real-time REST API."""
    base_url = os.environ.get("NASDAQ_BASE_URL")
    if not base_url:
        return None

    token = _get_realtime_token()
    if not token:
        return None

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    url = f"{base_url}/{endpoint}"

    resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_last_sale(symbol: str) -> dict:
    """
    Get latest last sale eligible transaction for a symbol.
    (Premium — requires Nasdaq real-time API credentials)

    Returns: timestamp, symbol, price, size, conditions, exchange, change
    """
    result = _realtime_request(f"last-sale/{symbol.upper()}")
    if result is None:
        return {
            "error": "Nasdaq real-time API not configured. "
                     "Set NASDAQ_CLIENT_ID, NASDAQ_CLIENT_SECRET, NASDAQ_BASE_URL in .env. "
                     "Contact Nasdaq sales for access.",
            "alternative": "Use get_stock_quote() from Yahoo Finance instead.",
        }
    return result


def get_last_quote(symbol: str) -> dict:
    """
    Get latest bid/ask quote for a symbol.
    (Premium — requires Nasdaq real-time API credentials)

    Returns: timestamp, symbol, bidPrice, bidSize, askPrice, askSize, condition
    """
    result = _realtime_request(f"last-quote/{symbol.upper()}")
    if result is None:
        return {"error": "Nasdaq real-time API not configured.", "symbol": symbol.upper()}
    return result


def get_snapshot(symbol: str) -> dict:
    """
    Get latest market snapshot: OHLCV + change stats.
    (Premium — requires Nasdaq real-time API credentials)

    Returns: open, high, low, close, lastTrade, volume, previousClose, netChange, percentChange
    """
    result = _realtime_request(f"snapshot/{symbol.upper()}")
    if result is None:
        return {"error": "Nasdaq real-time API not configured.", "symbol": symbol.upper()}
    return result


def get_market_trends() -> dict:
    """
    Get top 5 gainers and decliners.
    (Premium — requires Nasdaq real-time API credentials)

    Returns: gainers and decliners with symbol, lastTrade, lastSale, netChange, percentChange
    """
    result = _realtime_request("trends")
    if result is None:
        return {"error": "Nasdaq real-time API not configured."}
    return result


def get_bars(symbol: str, precision: str = "1min", from_date: str = None) -> dict:
    """
    Get aggregated bar data (OHLC + volume) for a symbol.
    (Premium — requires Nasdaq real-time API credentials)

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    precision : str
        '1min' or '5sec'.
    from_date : str
        Start date (YYYY-MM-dd), defaults to today.
    """
    params = {"precision": precision}
    if from_date:
        params["from"] = from_date

    result = _realtime_request(f"bars/{symbol.upper()}", params)
    if result is None:
        return {"error": "Nasdaq real-time API not configured.", "symbol": symbol.upper()}
    return result


# ============================================================================
# 3. STREAMING API (Premium) — Stub
# ============================================================================

class NasdaqStreamClient:
    """
    Real-time streaming client for Nasdaq data.

    Requires premium credentials:
    - NASDAQ_CLIENT_ID
    - NASDAQ_CLIENT_SECRET
    - NASDAQ_BOOTSTRAP_SERVERS

    Usage:
        client = NasdaqStreamClient()
        client.subscribe(['AAPL', 'MSFT'])
        client.start()  # Starts receiving ticks in background
        latest = client.get_latest('AAPL')
    """

    def __init__(self):
        self.client_id = os.environ.get("NASDAQ_CLIENT_ID")
        self.client_secret = os.environ.get("NASDAQ_CLIENT_SECRET")
        self.bootstrap_servers = os.environ.get("NASDAQ_BOOTSTRAP_SERVERS")
        self._latest_data: dict[str, dict] = {}
        self._running = False

        if not all([self.client_id, self.client_secret, self.bootstrap_servers]):
            self._configured = False
        else:
            self._configured = True

    @property
    def is_configured(self) -> bool:
        return self._configured

    def subscribe(self, symbols: list[str]):
        """Subscribe to real-time data for given symbols."""
        if not self._configured:
            print("  [Nasdaq Streaming] Not configured — set credentials in .env")
            return
        self._symbols = [s.upper() for s in symbols]
        print(f"  [Nasdaq Streaming] Subscribed to: {', '.join(self._symbols)}")

    def get_latest(self, symbol: str) -> Optional[dict]:
        """Get the latest cached tick for a symbol."""
        return self._latest_data.get(symbol.upper())

    def start(self):
        """Start the streaming connection (placeholder — requires SDK setup)."""
        if not self._configured:
            return
        # In production, this would use the Nasdaq Cloud Data Service Python SDK:
        # https://github.com/Nasdaq/NasdaqCloudDataService-SDK-Python
        print("  [Nasdaq Streaming] Stream started (placeholder — install Nasdaq SDK for live data)")
        self._running = True

    def stop(self):
        """Stop the streaming connection."""
        self._running = False
        print("  [Nasdaq Streaming] Stream stopped")
