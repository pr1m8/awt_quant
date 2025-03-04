awt_quant.forecast.stochastic.pde_forecast
==========================================

.. py:module:: awt_quant.forecast.stochastic.pde_forecast






Module Contents
---------------

.. py:class:: SPDEMCSimulator(ticker, equation, start_date='2022-01-01', end_date='2022-03-01', dt=1, num_paths=1000, plot_vol=True)

   Stochastic Process & GARCH-based Forecasting Simulator.

   .. attribute:: ticker

      Stock ticker symbol.

      :type: str

   .. attribute:: equation

      Stochastic model (`CIR`, `GBM`, `Heston`, `OU`).

      :type: str

   .. attribute:: start_date

      Start date for fetching historical data.

      :type: str

   .. attribute:: end_date

      End date for fetching historical data.

      :type: str

   .. attribute:: dt

      Time increment (default: 1 for daily, 1/252 for annual).

      :type: float

   .. attribute:: num_paths

      Number of Monte Carlo simulation paths.

      :type: int


   .. py:attribute:: ticker


   .. py:attribute:: start_date


   .. py:attribute:: end_date


   .. py:attribute:: dt
      :value: 1



   .. py:attribute:: num_paths
      :value: 1000



   .. py:attribute:: equation


   .. py:attribute:: plot_vol
      :value: True



   .. py:attribute:: forecasted_vol
      :value: None



   .. py:attribute:: GARCH_fit
      :value: None



   .. py:attribute:: device
      :value: 'cpu'



   .. py:method:: download_data(train_test_split)

      Downloads historical stock data and splits into train-test sets.



   .. py:method:: simulate()

      Runs stochastic simulation based on the selected model.



   .. py:method:: backwards(strike_price, option)

      Calculates backward pricing probability for options.



   .. py:method:: plot_simulation()

      Plots the quantile paths for simulated stock price.



   .. py:method:: error_estimation(num_sim=100)

      Estimates the error of stock price forecasts.



   .. py:method:: backtest()

      Performs backtesting on the simulated data.



.. py:data:: sim

