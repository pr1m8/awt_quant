"""
FRED Macroeconomic Data Fetching and Visualization

This module provides functions to fetch macroeconomic time series data from the Federal Reserve Economic Data (FRED) API
and visualize the results using Plotly. It supports retrieving both historical data and metadata for various economic indicators.

Functions:
    - get_fred_series(series_id, shorten=False): Fetches time series data from FRED.
    - get_fred_series_info(series_id, as_dict=True): Retrieves metadata for a FRED series.
    - get_macro(series_id, data=True, info=True, shorten=False): Fetches both time series and metadata for a FRED series.
    - get_bulk_macro(series_dict): Bulk fetches macroeconomic data for predefined indicators.
    - plot_macro_series(data, meta_data, dropna=False, y_axis_format=None): Plots macroeconomic time series data.
    - plot_macro_series_forecast(forecast_data, actual_data, meta_data, dropna=False, y_axis_format=None):
      Plots actual and forecasted macroeconomic data with confidence intervals.
    - fetch_and_plot(series_id, with_data=False): Fetches and plots a macroeconomic time series.

Usage:
    df, meta = get_macro("GDP")
    fig = plot_macro_series(df, meta)
    fig.show()
"""

import requests
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

# Load API Key from environment variables
load_dotenv(".env.example")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# API Endpoints
SERIES_TS_API_STR = 'https://api.stlouisfed.org/fred/series/observations?series_id={}&api_key={}&file_type=json'
SERIES_INFO_API_STR = 'https://api.stlouisfed.org/fred/series?series_id={}&api_key={}&file_type=json'

def get_fred_series(series_id, shorten=False):
    """
    Fetches time series data from FRED.

    Args:
        series_id (str): The FRED series ID.
        shorten (bool, optional): If True, returns only the last 30 observations. Defaults to False.

    Returns:
        pd.DataFrame: A dataframe containing the date and value columns.
    """
    response = requests.get(SERIES_TS_API_STR.format(series_id, FRED_API_KEY))
    while response.status_code == 429:
        response = requests.get(SERIES_TS_API_STR.format(series_id, FRED_API_KEY))
    if response.status_code == 200:
        data = json.loads(response.content)
        df = pd.DataFrame(data['observations']).replace('.', np.NaN)
        df['date'] = pd.to_datetime(df['date'])
        return df.iloc[-30:] if shorten else df[['date', 'value']]
    return None

def get_fred_series_info(series_id, as_dict=True):
    """
    Retrieves metadata for a FRED series.

    Args:
        series_id (str): The FRED series ID.
        as_dict (bool, optional): If True, returns metadata as a dictionary; otherwise, returns a DataFrame.

    Returns:
        dict or pd.DataFrame: Metadata about the series.
    """
    response = requests.get(SERIES_INFO_API_STR.format(series_id, FRED_API_KEY))
    if response.status_code == 200:
        data = json.loads(response.content)
        series_info = data.get('seriess', [{}])[0]
        return series_info if as_dict else pd.DataFrame([series_info])
    print(f"Error fetching series {series_id} info: {response.status_code}")
    return None

def get_macro(series_id, data=True, info=True, shorten=False):
    """
    Fetches both time series data and metadata for a given FRED series.

    Args:
        series_id (str): The FRED series ID.
        data (bool, optional): Whether to fetch time series data. Defaults to True.
        info (bool, optional): Whether to fetch metadata. Defaults to True.
        shorten (bool, optional): If True, returns only the last 30 observations. Defaults to False.

    Returns:
        tuple: (pd.DataFrame, dict) or single return depending on arguments.
    """
    if data and info:
        return get_fred_series(series_id, shorten), get_fred_series_info(series_id)
    if data:
        return get_fred_series(series_id, shorten)
    if info:
        return get_fred_series_info(series_id)
    return None

MACRO_INDICATORS = {
    "Gross Domestic Product (GDP)": "GDP",
    "Unemployment Rate": "UNRATE",
    "Consumer Price Index (CPI)": "CPIAUCSL",
    "Federal Funds Rate": "FEDFUNDS",
    "Retail Sales": "RSXFS",
}

def get_bulk_macro(series_dict=MACRO_INDICATORS):
    """
    Bulk fetch of major macroeconomic series data.

    Args:
        series_dict (dict, optional): Dictionary of macroeconomic indicators and their FRED series IDs.

    Returns:
        dict: Dictionary containing time series data and metadata for each indicator.
    """
    bulk_data = {}
    for key, series_id in series_dict.items():
        try:
            time_series, meta_data = get_macro(series_id, shorten=True)
            if time_series is not None and meta_data is not None:
                bulk_data[key] = {"time_series": time_series.to_dict(orient='records'), "meta_data": meta_data}
        except Exception as e:
            print(f"Error fetching data for {key}: {str(e)}")
    return bulk_data

def plot_macro_series(data, meta_data, dropna=False, y_axis_format=None):
    """
    Plots macroeconomic time series data.

    Args:
        data (pd.DataFrame): The time series data.
        meta_data (dict): The metadata of the series.
        dropna (bool, optional): Whether to drop NaN values. Defaults to False.
        y_axis_format (str, optional): Y-axis tick format. Defaults to None.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object.
    """
    if dropna:
        data = data.dropna()
    fig = go.Figure(data=go.Scatter(x=data['date'], y=data['value']))
    fig.update_layout(title=meta_data['title'], xaxis_title=meta_data['units'])
    if y_axis_format:
        fig.update_yaxes(tickformat=y_axis_format)
    return fig
