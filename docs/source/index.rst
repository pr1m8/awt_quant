.. AWT-Quant documentation master file, created by
   sphinx-quickstart on 2025.

Welcome to **AWT-Quant's Documentation!**
==========================================

**AWT-Quant** is an advanced quantitative finance library specializing in:

- **Stochastic modeling & forecasting** (SPDE, GARCH, Jump Diffusion, Monte Carlo).
- **Multi-factor analysis (MFA)** for financial markets.
- **Portfolio optimization** (Mean-Variance, Black-Litterman, Monte Carlo).
- **Risk management & stress testing**.

.. image:: _static/images/logo.png
   :width: 200px
   :align: center
   :alt: AWT Quant Logo

🚀 **GitHub Repository:** `AWT-Quant <https://github.com/pr1m8/awt_quant>`_  
📦 **PyPI Package:** `awt-quant on PyPI <https://pypi.org/project/awt-quant/>`_  
📖 **Full Documentation:** `ReadTheDocs <https://awt-quant.readthedocs.io/en/latest/>`_  

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   installation
   quickstart
   forecasting
   portfolio_optimization
   risk_management
   multi_factor_analysis
   api_reference
   contributing
   changelog

----

Getting Started
===============

If you're new to AWT-Quant, start with:

1. **Installation Guide** → :doc:`installation`
2. **Quick Start Tutorials** → :doc:`quickstart`
3. **API Reference & Modules** → :doc:`api_reference`

----

Core Features
=============

📊 **Stochastic PDE Forecasting**
---------------------------------
- Supports **Geometric Brownian Motion (GBM), Heston, CIR, OU, Jump Diffusion** models.
- Monte Carlo simulations & likelihood estimation.

.. code-block:: python

   from awt_quant.forecast.stochastic.run_simulations import SPDEMCSimulator
   
   # Initialize stochastic simulator with Heston model
   sim = SPDEMCSimulator(
       symbol='AAPL',
       start_date='2022-01-01',
       end_date='2022-03-01',
       dt=1,
       num_paths=1000,
       eq='heston'
   )
   
   sim.download_data()
   sim.set_parameters()
   sim.simulate()
   sim.plot_simulation()

📈 **Multi-Factor Analysis (MFA)**
----------------------------------
- Constructs **factors from macro & historical data**.
- Uses **ML-based feature selection** (Random Forest, PCA).
- **K-Means clustering & stress testing**.

.. code-block:: python

   from awt_quant.portfolio.multi_factor_analysis.main import MultiFactorAnalysis
   
   # Run factor analysis with clustering and stress testing
   mfa = MultiFactorAnalysis(
       assets=["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"],
       factors=["Market", "Size", "Value", "Momentum", "Quality"]
   )
   
   mfa.collect_data()
   mfa.construct_factors()
   mfa.run_clustering()
   mfa.run_stress_sensitivity()
   mfa.plot_factor_attribution()

💰 **Portfolio Optimization**
-----------------------------
- **Mean-Variance Optimization (MPT)** & Efficient Frontier.
- **Black-Litterman Bayesian Portfolio Optimization**.
- **Monte Carlo Portfolio Simulations**.

.. code-block:: python

   from awt_quant.portfolio.optimization.optimize import PortfolioOptimizer
   
   # Set up portfolio optimizer with constraints
   optimizer = PortfolioOptimizer(
       assets=["AAPL", "MSFT", "AMZN", "TSLA", "BND"],
       objective="sharpe",
       constraints={
           "max_volatility": 0.15,
           "max_per_asset": 0.25,
           "min_per_asset": 0.05
       }
   )
   
   weights = optimizer.optimize()
   optimizer.plot_efficient_frontier()
   optimizer.plot_allocation()

📉 **Risk Management & Stress Testing**
---------------------------------------
- **VaR, CVaR, Sharpe Ratio, Maximum Drawdown**.
- **Factor exposure analysis & custom performance tear sheets**.

.. code-block:: python

   from awt_quant.risk.tearsheet import RiskTearsheet
   
   # Generate comprehensive risk report
   tearsheet = RiskTearsheet(
       assets=["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"],
       weights=[0.2, 0.2, 0.2, 0.2, 0.2],
       start_date="2020-01-01",
       benchmark="SPY"
   )
   
   tearsheet.generate(
       include_drawdowns=True,
       include_stress_tests=True,
       include_factor_attribution=True
   )

----

Installation
============

To install AWT-Quant:

.. code-block:: bash

   # Using pip
   pip install awt-quant
   
   # Using Poetry
   poetry add awt-quant
   
   # For CUDA support
   poetry install --with cuda

----

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`