awt_quant.forecast.macro_forecast
=================================

.. py:module:: awt_quant.forecast.macro_forecast

.. autoapi-nested-parse::

   Macro Data Forecasting and Visualization

   This module provides functionality for forecasting macroeconomic time series data using AutoTS and TimeGPT.
   It includes preprocessing utilities, automated forecasting methods, and visualization tools.

   Functions:
       - MacroDataForecasting: A class for managing time series data and forecasting.
       - convert_numpy_floats(obj): Converts NumPy float64 values to native Python floats.
       - api_forecast(series_id): Fetches macroeconomic data and forecasts future values using TimeGPT.

   Usage:
       time_series_data, meta_data = get_macro("GDP")
       forecasting = MacroDataForecasting(time_series_data)
       forecast_results = forecasting.execute_forecasts()









Module Contents
---------------

.. py:data:: TIMEGPT_API_KEY

.. py:data:: timegpt

.. py:class:: MacroDataForecasting(time_series, meta_data=None)

   A class for forecasting macroeconomic time series data using AutoTS.


   .. py:attribute:: time_series


   .. py:attribute:: meta_data
      :value: None



   .. py:attribute:: forecast_results


   .. py:method:: preprocess_data(method='average', normalize=False, return_type=None, na_method='drop')

      Preprocesses the time series data by handling missing values and formatting dates.

      :param method: Method to handle missing values ('average', 'interpolate'). Defaults to 'average'.
      :type method: str, optional
      :param normalize: Whether to normalize the data. Defaults to False.
      :type normalize: bool, optional
      :param return_type: Type of return calculation ('log', 'percent') or None. Defaults to None.
      :type return_type: str, optional
      :param na_method: Method to handle missing values ('drop', 'ffill', 'interpolate'). Defaults to 'drop'.
      :type na_method: str, optional



   .. py:method:: forecast_with_autots(forecast_length=30, frequency='infer', prediction_interval=0.9, model_list='superfast', transformer_list='superfast', ensemble='distance', max_generations=4, num_validations=1, validation_method='backward', metric_weighting={'smape_weighting': 0.5, 'mae_weighting': 0.5}, drop_most_recent=0, n_jobs='auto')

      Generates forecasts using the AutoTS library with enhanced parameterization.

      :param forecast_length: Number of periods to forecast. Defaults to 30.
      :type forecast_length: int, optional
      :param frequency: Frequency of the time series data. Defaults to 'infer'.
      :type frequency: str, optional
      :param prediction_interval: Prediction interval for the forecast. Defaults to 0.9.
      :type prediction_interval: float, optional
      :param model_list: Models to be used in the search. Defaults to 'superfast'.
      :type model_list: list or str, optional
      :param transformer_list: Data transformations to be applied. Defaults to 'superfast'.
      :type transformer_list: list or str, optional
      :param ensemble: Ensemble method to use. Defaults to 'distance'.
      :type ensemble: str, optional
      :param max_generations: Number of generations for the model search. Defaults to 4.
      :type max_generations: int, optional
      :param num_validations: Number of validation sets used in model selection. Defaults to 1.
      :type num_validations: int, optional
      :param validation_method: Method for time series cross-validation. Defaults to 'backward'.
      :type validation_method: str, optional
      :param metric_weighting: Weighting of different performance metrics. Defaults to {'smape_weighting': 0.5, 'mae_weighting': 0.5}.
      :type metric_weighting: dict, optional
      :param drop_most_recent: Number of most recent data points to drop. Defaults to 0.
      :type drop_most_recent: int, optional
      :param n_jobs: Number of jobs to run in parallel. Defaults to 'auto'.
      :type n_jobs: int or str, optional

      :returns: Dictionary containing forecast results, lower and upper bounds.
      :rtype: dict



   .. py:method:: execute_forecasts(na_method='drop')

      Executes the full forecasting pipeline including preprocessing and forecasting.

      :param na_method: Method to handle missing values. Defaults to 'drop'.
      :type na_method: str, optional

      :returns: Forecast results.
      :rtype: dict



.. py:function:: convert_numpy_floats(obj)

   Recursively converts NumPy float64 values to Python native float.

   :param obj: Object containing NumPy floats.
   :type obj: any

   :returns: Object with converted float values.
   :rtype: any


.. py:function:: api_forecast(series_id)
   :async:


   Fetches macroeconomic data and forecasts future values using TimeGPT.

   :param series_id: The macroeconomic series ID.
   :type series_id: str

   :returns: Dictionary containing forecasted values.
   :rtype: dict


