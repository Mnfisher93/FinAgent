"""
Scrapers package — modules for fetching financial data from various sources.
"""

from .spy_scraper import fetch_spy_data
from .nq_scraper import fetch_nq_data
from .bitcoin_scraper import fetch_bitcoin_data
from .polymarket_scraper import fetch_polymarket_data
from .stock_scraper import get_stock_quote, get_stock_history, get_stock_info
from .crypto_scraper import get_crypto_price, get_crypto_top_n, search_crypto
from .nasdaq_scraper import (
    get_nasdaq_table, get_nasdaq_fundamentals, search_nasdaq_datasets,
    get_last_sale, get_last_quote, get_snapshot, get_market_trends, get_bars,
)

__all__ = [
    "fetch_spy_data",
    "fetch_nq_data",
    "fetch_bitcoin_data",
    "fetch_polymarket_data",
    "get_stock_quote",
    "get_stock_history",
    "get_stock_info",
    "get_crypto_price",
    "get_crypto_top_n",
    "search_crypto",
    "get_nasdaq_table",
    "get_nasdaq_fundamentals",
    "search_nasdaq_datasets",
    "get_last_sale",
    "get_last_quote",
    "get_snapshot",
    "get_market_trends",
    "get_bars",
]
