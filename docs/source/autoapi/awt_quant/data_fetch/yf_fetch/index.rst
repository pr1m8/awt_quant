awt_quant.data_fetch.yf_fetch
=============================

.. py:module:: awt_quant.data_fetch.yf_fetch

.. autoapi-nested-parse::

   Yahoo Finance Data Fetching

   This module provides a function to fetch historical stock price data from Yahoo Finance.
   It preprocesses the data for use in stochastic differential equation models.

   Functions:
       - download_data(ticker, start_date, end_date, train_test_split): Fetches and splits stock price data.

   Usage:
       train_data, test_data, meta_data = download_data("AAPL", "2022-01-01", "2023-01-01", train_test_split=0.8)





Module Contents
---------------

.. py:function:: download_data(ticker, start_date, end_date, train_test_split)

   Downloads stock price data from Yahoo Finance and processes it for training/testing.

   :param ticker: Stock ticker symbol.
   :type ticker: str
   :param start_date: Start date in 'YYYY-MM-DD' format.
   :type start_date: str
   :param end_date: End date in 'YYYY-MM-DD' format.
   :type end_date: str
   :param train_test_split: Fraction of data to use for training (e.g., 0.8 for 80% training data).
   :type train_test_split: float

   :returns:

             (train_data, test_data, meta_data)
                 - train_data (pd.DataFrame): Training set containing stock close prices.
                 - test_data (pd.DataFrame): Testing set containing stock close prices.
                 - meta_data (dict): Dictionary with additional information (dates, S0, T, N).
   :rtype: tuple


