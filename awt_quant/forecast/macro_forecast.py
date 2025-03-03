"""
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
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from autots import AutoTS
from dotenv import load_dotenv
import os
from awt_quant.data_fetch.macro import get_macro
from nixtlats import TimeGPT

# Load API Key from environment variables
load_dotenv(".env.example")
TIMEGPT_API_KEY = os.getenv("TIMEGPT_API_KEY")
timegpt = TimeGPT(token=TIMEGPT_API_KEY)


class MacroDataForecasting:
    """
    A class for forecasting macroeconomic time series data using AutoTS.
    """

    def __init__(self, time_series, meta_data=None):
        """
        Initializes the MacroDataForecasting class.

        Args:
            time_series (pd.DataFrame): The macroeconomic time series data.
            meta_data (dict, optional): Metadata related to the time series.
        """
        self.time_series = time_series
        self.meta_data = meta_data
        self.forecast_results = {}

    def preprocess_data(self, method='average', normalize=False, return_type=None, na_method='drop'):
        """
        Preprocesses the time series data by handling missing values and formatting dates.

        Args:
            method (str, optional): Method to handle missing values ('average', 'interpolate'). Defaults to 'average'.
            normalize (bool, optional): Whether to normalize the data. Defaults to False.
            return_type (str, optional): Type of return calculation ('log', 'percent') or None. Defaults to None.
            na_method (str, optional): Method to handle missing values ('drop', 'ffill', 'interpolate'). Defaults to 'drop'.
        """
        df = self.time_series.copy()

        # Identify date and value columns
        date_col = 'date' if 'date' in df.columns else 'Date' if 'Date' in df.columns else None
        value_col = 'value' if 'value' in df.columns else None

        if not value_col:
            raise ValueError("The 'value' column is missing in the dataset.")

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])

            if not df.index.equals(df[date_col]):
                df.set_index(date_col, drop=True, inplace=True)
                df.index = pd.to_datetime(df.index)

        # Handle missing values
        if na_method == 'drop':
            df = df.dropna()
        elif na_method == 'ffill':
            df = df.ffill()
        elif na_method == 'interpolate':
            df = df.interpolate()

        self.time_series = df

    def forecast_with_autots(self, forecast_length=30, frequency='infer', prediction_interval=0.9,
                             model_list='superfast', transformer_list='superfast', ensemble='distance',
                             max_generations=4, num_validations=1, validation_method='backward',
                             metric_weighting={'smape_weighting': 0.5, 'mae_weighting': 0.5},
                             drop_most_recent=0, n_jobs='auto'):
        """
        Generates forecasts using the AutoTS library with enhanced parameterization.

        Args:
            forecast_length (int, optional): Number of periods to forecast. Defaults to 30.
            frequency (str, optional): Frequency of the time series data. Defaults to 'infer'.
            prediction_interval (float, optional): Prediction interval for the forecast. Defaults to 0.9.
            model_list (list or str, optional): Models to be used in the search. Defaults to 'superfast'.
            transformer_list (list or str, optional): Data transformations to be applied. Defaults to 'superfast'.
            ensemble (str, optional): Ensemble method to use. Defaults to 'distance'.
            max_generations (int, optional): Number of generations for the model search. Defaults to 4.
            num_validations (int, optional): Number of validation sets used in model selection. Defaults to 1.
            validation_method (str, optional): Method for time series cross-validation. Defaults to 'backward'.
            metric_weighting (dict, optional): Weighting of different performance metrics. Defaults to {'smape_weighting': 0.5, 'mae_weighting': 0.5}.
            drop_most_recent (int, optional): Number of most recent data points to drop. Defaults to 0.
            n_jobs (int or str, optional): Number of jobs to run in parallel. Defaults to 'auto'.

        Returns:
            dict: Dictionary containing forecast results, lower and upper bounds.
        """
        model = AutoTS(
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            model_list=model_list,
            transformer_list=transformer_list,
            max_generations=max_generations,
            num_validations=num_validations,
            validation_method=validation_method,
            metric_weighting=metric_weighting,
            drop_most_recent=drop_most_recent,
            n_jobs=n_jobs
        )

        # Fit the model
        model = model.fit(self.time_series)

        # Generate forecast
        prediction = model.predict()
        forecast = prediction.forecast
        lower_bound = prediction.lower_forecast
        upper_bound = prediction.upper_forecast

        # Store forecast results
        self.forecast_results = {
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
        }

        return self.forecast_results

    def execute_forecasts(self, na_method='drop'):
        """
        Executes the full forecasting pipeline including preprocessing and forecasting.

        Args:
            na_method (str, optional): Method to handle missing values. Defaults to 'drop'.

        Returns:
            dict: Forecast results.
        """
        self.preprocess_data(na_method=na_method)
        return self.forecast_with_autots()


def convert_numpy_floats(obj):
    """
    Recursively converts NumPy float64 values to Python native float.

    Args:
        obj (any): Object containing NumPy floats.

    Returns:
        any: Object with converted float values.
    """
    if isinstance(obj, list):
        return [convert_numpy_floats(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_floats(value) for key, value in obj.items()}
    elif isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj


async def api_forecast(series_id):
    """
    Fetches macroeconomic data and forecasts future values using TimeGPT.

    Args:
        series_id (str): The macroeconomic series ID.

    Returns:
        dict: Dictionary containing forecasted values.
    """
    time_series_data, _ = get_macro(series_id)
    forecast = timegpt.forecast(df=time_series_data.dropna(), h=10, time_col='date', target_col='value').to_dict()
    return {"date": list(forecast["date"].values()), "forecast": list(forecast["TimeGPT"].values())}
