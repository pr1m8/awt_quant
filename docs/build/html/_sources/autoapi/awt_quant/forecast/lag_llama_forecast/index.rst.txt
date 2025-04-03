awt_quant.forecast.lag_llama_forecast
=====================================

.. py:module:: awt_quant.forecast.lag_llama_forecast






Module Contents
---------------

.. py:data:: LAG_LLAMA_CKPT_PATH
   :value: 'resources/lag_llama/model/lag_llama.ckpt'


.. py:function:: get_device()

   Returns the appropriate device for computation.

   Uses CUDA if available, otherwise falls back to CPU.

   :returns: The device to use for model computations.
   :rtype: torch.device


.. py:function:: fetch_stock_data(ticker, start_date, end_date)

   Fetches stock price data from Yahoo Finance and formats it for Lag-Llama.

   :param ticker: Stock symbol.
   :type ticker: str
   :param start_date: Start date in 'YYYY-MM-DD' format.
   :type start_date: str
   :param end_date: End date in 'YYYY-MM-DD' format.
   :type end_date: str

   :returns: The dataset formatted for Lag-Llama.
   :rtype: PandasDataset


.. py:function:: get_lag_llama_predictions(dataset, prediction_length, num_samples=100, context_length=32, use_rope_scaling=False)

   Runs Lag-Llama predictions on a given dataset.

   :param dataset: The dataset for forecasting.
   :type dataset: PandasDataset
   :param prediction_length: Forecast horizon.
   :type prediction_length: int
   :param num_samples: Number of Monte Carlo samples per timestep. Defaults to 100.
   :type num_samples: int, optional
   :param context_length: Context length for model. Defaults to 32.
   :type context_length: int, optional
   :param use_rope_scaling: Whether to use RoPE scaling for extended context. Defaults to False.
   :type use_rope_scaling: bool, optional

   :returns: Forecasts and actual time series.
   :rtype: Tuple[list, list]


.. py:function:: plot_forecasts(forecasts, tss, ticker, prediction_length)

   Plots actual stock prices along with forecasted values.

   :param forecasts: List of forecasted series.
   :type forecasts: list
   :param tss: List of actual time series.
   :type tss: list
   :param ticker: Stock ticker symbol.
   :type ticker: str
   :param prediction_length: Forecast horizon.
   :type prediction_length: int


.. py:function:: evaluate_forecasts(forecasts, tss)

   Evaluates forecasts using GluonTS Evaluator.

   :param forecasts: Forecasted time series.
   :type forecasts: list
   :param tss: Actual time series.
   :type tss: list

   :returns: Aggregated evaluation metrics including CRPS.
   :rtype: dict


.. py:function:: backtest(forecasts, actual_series)

   Computes backtest evaluation metrics by comparing forecasts against actual values.

   :param forecasts: List of forecasted time series.
   :type forecasts: list
   :param actual_series: List of actual time series.
   :type actual_series: list

   :returns: Evaluation metrics including mean error and quantiles.
   :rtype: dict


.. py:function:: main()

   Runs the end-to-end pipeline:
   - Fetches stock data
   - Runs Lag-Llama forecasting with context length 32
   - Evaluates and plots the forecasts
   - Performs backtesting


