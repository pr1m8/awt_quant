awt_quant.portfolio.optimization.optimize
=========================================

.. py:module:: awt_quant.portfolio.optimization.optimize

.. autoapi-nested-parse::

   Portfolio Optimization and Risk Analysis

   This module provides functions to optimize a portfolio of stocks based on Sharpe Ratio and Value at Risk (VaR).
   It allows for portfolio weight optimization and visualization of the efficient frontier.

   Functions:
       - portfolio_sharpe(stocks_list, n=1000): Computes the optimal portfolio weights to maximize the Sharpe Ratio.
       - portfolio_var(stocks_list, n=1000): Computes the optimal portfolio weights to minimize Value at Risk (VaR).
       - plot_efficient_frontier(mean_variance_pairs, return_shp_max, vol_shp_max):
         Visualizes the efficient frontier using randomly generated portfolios.

   Usage:
       weights, sharpe_ratio, return_shp, vol_shp = portfolio_sharpe(stocks_list)
       weights_var, min_var, return_var = portfolio_var(stocks_list)
       fig = plot_efficient_frontier(mean_variance_pairs, return_shp_max, vol_shp_max)
       fig.show()





Module Contents
---------------

.. py:function:: portfolio_sharpe(stocks_list, n=1000)

   Computes the optimal portfolio allocation to maximize the Sharpe Ratio.

   :param stocks_list: List of stock tickers.
   :type stocks_list: list
   :param n: Number of randomly generated portfolios. Defaults to 1000.
   :type n: int, optional

   :returns:

             (dict, float, float, float)
                 - dict: Optimal portfolio weights.
                 - float: Maximum Sharpe Ratio.
                 - float: Expected return of the optimal portfolio.
                 - float: Expected volatility of the optimal portfolio.
   :rtype: tuple


.. py:function:: portfolio_var(stocks_list, n=1000)

   Computes the optimal portfolio allocation to minimize Value at Risk (VaR).

   :param stocks_list: List of stock tickers.
   :type stocks_list: list
   :param n: Number of randomly generated portfolios. Defaults to 1000.
   :type n: int, optional

   :returns:

             (dict, float, float)
                 - dict: Optimal portfolio weights.
                 - float: Minimum Value at Risk (VaR).
                 - float: Expected return of the minimum VaR portfolio.
   :rtype: tuple


.. py:function:: plot_efficient_frontier(mean_variance_pairs, return_shp_max, vol_shp_max)

   Plots the efficient frontier of randomly generated portfolios.

   :param mean_variance_pairs: List of tuples (expected return, variance).
   :type mean_variance_pairs: list
   :param return_shp_max: Expected return of the optimal Sharpe Ratio portfolio.
   :type return_shp_max: float
   :param vol_shp_max: Expected volatility of the optimal Sharpe Ratio portfolio.
   :type vol_shp_max: float

   :returns: A Plotly figure object displaying the efficient frontier.
   :rtype: plotly.graph_objects.Figure


