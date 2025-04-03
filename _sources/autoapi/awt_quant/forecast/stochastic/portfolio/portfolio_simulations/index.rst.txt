awt_quant.forecast.stochastic.portfolio.portfolio_simulations
=============================================================

.. py:module:: awt_quant.forecast.stochastic.portfolio.portfolio_simulations




Module Contents
---------------

.. py:function:: run_portfolio_simulation(portfolio, equation, start_date, end_date, train_test_split, num_paths=1000, plot_vol=False, plot_sim=False, num_sim=100)

   Runs a single portfolio simulation using the chosen stochastic differential equation.

   :param portfolio: Dictionary containing symbols, positions, and quantities.
   :type portfolio: dict
   :param equation: Chosen stochastic model (CIR, GBM, Heston, OU).
   :type equation: str
   :param start_date: Start date for simulation.
   :type start_date: str
   :param end_date: End date for simulation.
   :type end_date: str
   :param train_test_split: Ratio of training data.
   :type train_test_split: float
   :param num_paths: Number of simulation paths (default: 1000).
   :type num_paths: int
   :param plot_vol: Whether to plot volatility models (default: False).
   :type plot_vol: bool
   :param plot_sim: Whether to plot individual stock simulations (default: False).
   :type plot_sim: bool
   :param num_sim: Number of simulations for error estimation (default: 100).
   :type num_sim: int


.. py:function:: compare_multiple_portfolio_simulations(portfolios, equation_classes, end_dates, forecast_periods, train_test_splits, num_paths=1000, num_sim=100, plot_vol=False, plot_sim=False)

   Compares multiple portfolio simulations across different stochastic models and settings.

   :param portfolios: List of portfolios with stock symbols and positions.
   :type portfolios: list[dict]
   :param equation_classes: List of stochastic models to test.
   :type equation_classes: list[str]
   :param end_dates: End dates for different simulations.
   :type end_dates: list[str]
   :param forecast_periods: Forecasting periods in days.
   :type forecast_periods: list[int]
   :param train_test_splits: Different train-test split ratios.
   :type train_test_splits: list[float]
   :param num_paths: Number of Monte Carlo paths (default: 1000).
   :type num_paths: int
   :param num_sim: Number of simulations for error estimation (default: 100).
   :type num_sim: int
   :param plot_vol: Whether to plot volatility models (default: False).
   :type plot_vol: bool
   :param plot_sim: Whether to plot individual stock simulations (default: False).
   :type plot_sim: bool

   :returns: Dataframe containing forecast errors and summaries.
   :rtype: pd.DataFrame


