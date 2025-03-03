"""
Yahoo Finance Data Fetching

This module provides a function to fetch historical stock price data from Yahoo Finance.
It preprocesses the data for use in stochastic differential equation models.

Functions:
    - download_data(ticker, start_date, end_date, train_test_split): Fetches and splits stock price data.

Usage:
    train_data, test_data, meta_data = download_data("AAPL", "2022-01-01", "2023-01-01", train_test_split=0.8)
"""

import yfinance as yf
import numpy as np
import torch
import pandas as pd

def download_data(ticker, start_date, end_date, train_test_split):
    """
    Downloads stock price data from Yahoo Finance and processes it for training/testing.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        train_test_split (float): Fraction of data to use for training (e.g., 0.8 for 80% training data).

    Returns:
        tuple: (train_data, test_data, meta_data)
            - train_data (pd.DataFrame): Training set containing stock close prices.
            - test_data (pd.DataFrame): Testing set containing stock close prices.
            - meta_data (dict): Dictionary with additional information (dates, S0, T, N).
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.index = data.index.date
    data = data[['Close']]  # Keep only the closing prices

    # Train-test split
    L = len(data)
    split_idx = int(np.floor(train_test_split * L))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    # Metadata
    S0 = torch.tensor(train_data['Close'].iloc[-1])
    T = len(test_data)
    N = int(T)  # Assuming dt=1

    meta_data = {
        "dates": data.index,
        "dates_train": train_data.index,
        "dates_pred": test_data.index,
        "S0": S0,
        "T": T,
        "N": N,
    }

    return train_data, test_data, meta_data
