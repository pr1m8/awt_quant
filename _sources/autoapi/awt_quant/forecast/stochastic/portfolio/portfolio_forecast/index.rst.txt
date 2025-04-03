awt_quant.forecast.stochastic.portfolio.portfolio_forecast
==========================================================

.. py:module:: awt_quant.forecast.stochastic.portfolio.portfolio_forecast

.. autoapi-nested-parse::

   Portfolio Forecasting using Monte Carlo Simulations & Copula Models.

   This module extends SPDEMCSimulator to forecast a portfolio of assets.

   Classes:
       - PortfolioForecast: Forecasts portfolio performance using copula-based simulations.





Module Contents
---------------

.. py:class:: PortfolioForecast(portfolio, equation, train_test_split, start_date='2022-01-01', end_date='2022-03-01', dt=1, num_paths=1000, plot_vol=False, plot_sim=False)

   Bases: :py:obj:`awt_quant.forecast.stochastic.pde_forecast.SPDEMCSimulator`


   Forecasts a portfolio using Monte Carlo simulations and copula models.

   .. attribute:: - portfolio

      Dictionary with stock symbols, positions, and quantities.

   .. attribute:: - equation

      Stochastic Differential Equation model (CIR, GBM, Heston, OU).

   .. attribute:: - assets

      List of SPDEMCSimulator instances for each stock.

   .. attribute:: - train_data, test_data

      Aggregated portfolio data.


   .. py:attribute:: portfolio


   .. py:attribute:: symbols


   .. py:attribute:: assets
      :value: []



   .. py:attribute:: quantity


   .. py:attribute:: position
      :value: []



   .. py:attribute:: equation


   .. py:method:: copula_simulation()

      Simulates a copula model to generate dependent return quantiles.

      :returns: Copula-simulated quantiles (shape: n_stocks x n_simulations x T)
      :rtype: - torch.Tensor



   .. py:method:: forecast()

      Generates portfolio price forecasts based on copula-simulated quantiles.



   .. py:method:: plot_forecast()

      Plots portfolio simulation with quantile paths and percent errors.



   .. py:method:: backtest()

      Backtests the forecast and prints ARMA model summaries.



