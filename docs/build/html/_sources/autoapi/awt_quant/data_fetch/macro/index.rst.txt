awt_quant.data_fetch.macro
==========================

.. py:module:: awt_quant.data_fetch.macro

.. autoapi-nested-parse::

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







Module Contents
---------------

.. py:data:: FRED_API_KEY

.. py:data:: SERIES_TS_API_STR
   :value: 'https://api.stlouisfed.org/fred/series/observations?series_id={}&api_key={}&file_type=json'


.. py:data:: SERIES_INFO_API_STR
   :value: 'https://api.stlouisfed.org/fred/series?series_id={}&api_key={}&file_type=json'


.. py:function:: get_fred_series(series_id, shorten=False)

   Fetches time series data from FRED.

   :param series_id: The FRED series ID.
   :type series_id: str
   :param shorten: If True, returns only the last 30 observations. Defaults to False.
   :type shorten: bool, optional

   :returns: A dataframe containing the date and value columns.
   :rtype: pd.DataFrame


.. py:function:: get_fred_series_info(series_id, as_dict=True)

   Retrieves metadata for a FRED series.

   :param series_id: The FRED series ID.
   :type series_id: str
   :param as_dict: If True, returns metadata as a dictionary; otherwise, returns a DataFrame.
   :type as_dict: bool, optional

   :returns: Metadata about the series.
   :rtype: dict or pd.DataFrame


.. py:function:: get_macro(series_id, data=True, info=True, shorten=False)

   Fetches both time series data and metadata for a given FRED series.

   :param series_id: The FRED series ID.
   :type series_id: str
   :param data: Whether to fetch time series data. Defaults to True.
   :type data: bool, optional
   :param info: Whether to fetch metadata. Defaults to True.
   :type info: bool, optional
   :param shorten: If True, returns only the last 30 observations. Defaults to False.
   :type shorten: bool, optional

   :returns: (pd.DataFrame, dict) or single return depending on arguments.
   :rtype: tuple


.. py:data:: MACRO_INDICATORS

.. py:function:: get_bulk_macro(series_dict=MACRO_INDICATORS)

   Bulk fetch of major macroeconomic series data.

   :param series_dict: Dictionary of macroeconomic indicators and their FRED series IDs.
   :type series_dict: dict, optional

   :returns: Dictionary containing time series data and metadata for each indicator.
   :rtype: dict


.. py:function:: plot_macro_series(data, meta_data, dropna=False, y_axis_format=None)

   Plots macroeconomic time series data.

   :param data: The time series data.
   :type data: pd.DataFrame
   :param meta_data: The metadata of the series.
   :type meta_data: dict
   :param dropna: Whether to drop NaN values. Defaults to False.
   :type dropna: bool, optional
   :param y_axis_format: Y-axis tick format. Defaults to None.
   :type y_axis_format: str, optional

   :returns: A Plotly figure object.
   :rtype: plotly.graph_objects.Figure


