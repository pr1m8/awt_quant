"""
Data Fetching Module

This module provides functions for retrieving financial and macroeconomic data 
from external sources such as Yahoo Finance.

Available Functions:
- `fetch_macro_data`: Retrieves macroeconomic indicators.
- `fetch_yfinance_data`: Fetches historical stock price data from Yahoo Finance.
"""

from .macro import get_macro,plot_macro_series,get_bulk_macro,get_fred_series_info,get_fred_series
from .yf_fetch import download_data 

__all__ = ["get_macro", "plot_macro_series", "get_bulk_macro", "get_fred_series_info", "get_fred_series", "download_data"]
