awt_quant.forecast.garch_forecast
=================================

.. py:module:: awt_quant.forecast.garch_forecast

.. autoapi-nested-parse::

   GARCH Model Optimization and Volatility Forecasting.

   This module finds the best GARCH-like model for a given time series, fits it, and visualizes
   its conditional volatility.

   Classes:
       - GARCHOptimizer: Handles GARCH model selection, fitting, and volatility forecasting.

   Usage:
       garch = GARCHOptimizer(series, dates_train, ticker)
       best_model = garch.fit()
       fig = garch.plot_volatility()
       fig.show()





Module Contents
---------------

.. py:class:: GARCHOptimizer(series, dates_train, ticker, plot_vol=True)

   A class to find and optimize a GARCH-like model for a given time series.

   .. attribute:: series

      Time series data of asset returns.

      :type: pd.Series

   .. attribute:: dates_train

      Corresponding date index for the series.

      :type: pd.Series

   .. attribute:: ticker

      Stock ticker symbol.

      :type: str

   .. attribute:: plot_vol

      Whether to plot the volatility.

      :type: bool

   .. attribute:: best_model

      The best identified GARCH model.

      :type: str

   .. attribute:: best_p

      Optimal p lag order.

      :type: int

   .. attribute:: best_q

      Optimal q lag order.

      :type: int

   .. attribute:: fitted_model

      The fitted GARCH model.

      :type: arch.univariate.base.ARCHModelResult


   .. py:attribute:: series


   .. py:attribute:: dates_train


   .. py:attribute:: ticker


   .. py:attribute:: plot_vol
      :value: True



   .. py:attribute:: best_model
      :value: None



   .. py:attribute:: best_p
      :value: None



   .. py:attribute:: best_q
      :value: None



   .. py:attribute:: fitted_model
      :value: None



   .. py:method:: fit()

      Finds the best GARCH model using Bayesian Information Criterion (BIC).

      :returns: The fitted optimal GARCH model.
      :rtype: arch.univariate.base.ARCHModelResult



   .. py:method:: plot_volatility()

      Plots the conditional volatility of the fitted GARCH model.

      :returns: A Plotly figure displaying the volatility plot.
      :rtype: plotly.graph_objects.Figure



   .. py:method:: forecast(horizon=10)

      Generates a volatility forecast for the next `horizon` periods.

      :param horizon: Number of future periods to forecast.
      :type horizon: int

      :returns: A DataFrame with the forecasted conditional variances.
      :rtype: pd.DataFrame



