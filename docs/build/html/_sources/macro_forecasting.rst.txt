====================================
Macro Data Forecasting and Visualization
====================================

This module provides functionality for forecasting macroeconomic time series data using **AutoTS** and **TimeGPT**.
It includes **preprocessing utilities**, **automated forecasting methods**, and **visualization tools**.

Features:
---------
- **Preprocessing**: Handling missing values, normalizing data, and formatting time series.
- **Forecasting**: AutoTS-based automated forecasting and TimeGPT API for macroeconomic time series.
- **Visualization**: Interactive plotting with Plotly.

Dependencies:
-------------
- `autots`
- `pandas`
- `plotly`
- `nixtlats`
- `dotenv`
- `numpy`

Functions:
----------
1. **MacroDataForecasting**
   - A class for managing macroeconomic time series data and forecasting.

2. **convert_numpy_floats(obj)**
   - Converts NumPy `float64` values to native Python floats.

3. **api_forecast(series_id)**
   - Fetches macroeconomic data and forecasts future values using **TimeGPT**.

Usage:
------
Example of how to use **MacroDataForecasting**:

.. code-block:: python

    from awt_quant.data_fetch.macro import get_macro
    from awt_quant.forecast.macro_forecast import MacroDataForecasting

    # Fetch macroeconomic data (example: GDP)
    time_series_data, meta_data = get_macro("GDP")

    # Initialize forecasting object
    forecasting = MacroDataForecasting(time_series_data)

    # Execute forecasts
    forecast_results = forecasting.execute_forecasts()

    # Access forecasted values
    print(forecast_results['forecast'])

Class: MacroDataForecasting
---------------------------
.. code-block:: python

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

Methods:
--------

**preprocess_data(method='average', normalize=False, return_type=None, na_method='drop')**
    - Handles missing values and formats time series data.
    - Methods include:
      - `drop`: Remove missing values.
      - `ffill`: Forward-fill missing values.
      - `interpolate`: Interpolate missing values.

**forecast_with_autots(forecast_length=30, frequency='infer', prediction_interval=0.9, model_list='superfast', ...)**
    - Runs **AutoTS** to generate time series forecasts.
    - Supports different model and transformation options.

**execute_forecasts(na_method='drop')**
    - Executes the full pipeline of preprocessing and forecasting.

Additional Functions:
---------------------
**convert_numpy_floats(obj)**
    - Recursively converts NumPy float64 values to native Python floats.

**api_forecast(series_id)**
    - Uses **TimeGPT** API to forecast macroeconomic time series data.

Example API Forecast:
---------------------
.. code-block:: python

    from awt_quant.forecast.macro_forecast import api_forecast

    forecast_data = api_forecast("GDP")
    print(forecast_data)

This module leverages over **80,000+** FRED macroeconomic time series datasets and provides advanced forecasting tools.
