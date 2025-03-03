.. _portfolio_forecast:

##################################
Portfolio Forecasting Documentation
##################################

Portfolio Forecasting is a module designed for **multi-asset Monte Carlo simulations**, integrating **stochastic processes** and **GARCH modeling** for advanced financial forecasting.

This module enables:

- **Multi-Asset Price Forecasting** using various stochastic models.
- **Portfolio Risk Analysis** with volatility estimation and simulations.
- **Integration of Factor-Based and Copula Models** for enhanced prediction accuracy.

------------

.. contents:: Table of Contents
   :local:
   :depth: 2

------------

.. _portfolio_forecast_overview:

Overview
========

The `PortfolioForecast` class extends `SPDEMCSimulator`, supporting **multi-asset simulations** by incorporating dependencies between assets and their volatilities.

**Key Features:**
- Supports **Heston, Geometric Brownian Motion (GBM), Ornstein-Uhlenbeck (OU), and CIR** models.
- Integrates **copula-based dependence structures** between assets.
- Uses **Monte Carlo simulations** for price forecasting.
- Implements **GARCH-based volatility forecasting** for improved accuracy.
- Supports **stress testing and sensitivity analysis** to assess risk.

------------

.. _portfolio_forecast_structure:

Module Structure
================

The `portfolio_forecast` module consists of the following key components:

- **PortfolioForecast**: Core class for multi-asset Monte Carlo simulations.
- **Copula Simulation**: Uses `GaussianMultivariate` copula to model asset dependencies.
- **Factor-Based Forecasting**: Integrates multi-factor models for enhanced predictions.
- **Volatility Modeling**: Utilizes GARCH models to improve stochastic forecasts.

------------

.. _portfolio_forecast_class:

PortfolioForecast Class
=======================

.. autoclass:: awt_quant.forecast.stochastic.portfolio.PortfolioForecast
   :members:
   :undoc-members:
   :show-inheritance:

------------

.. _portfolio_forecast_usage:

Usage
=====

Below is an example demonstrating how to use `PortfolioForecast` to simulate a multi-asset portfolio:

.. code-block:: python

    from awt_quant.forecast.stochastic.portfolio import PortfolioForecast
    
    portfolio_data = {
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'quantity': [10, 5, 8],
        'position': ['Long', 'Short', 'Long']
    }
    
    simulator = PortfolioForecast(
        portfolio=portfolio_data,
        equation='Heston',
        train_test_split=0.75,
        start_date='2020-01-01',
        end_date='2022-01-01',
        dt=1,
        num_paths=1000,
        plot_vol=True,
        plot_sim=True
    )
    
    simulator.forecast()
    simulator.plot_forecast()
    simulator.backtest()

------------

.. _portfolio_forecast_stress_tests:

Stress Testing & Sensitivity Analysis
=====================================

The `PortfolioForecast` class includes built-in methods for stress testing and sensitivity analysis:

- **Noise Injection Analysis**
- **Extreme Value Testing**
- **Feature Importance Analysis**
- **Global Economic Downturn Simulations**
- **Inflation Shock Modeling**

Example:

.. code-block:: python

    from awt_quant.forecast.stochastic.portfolio import PortfolioForecast
    
    stress_results = simulator.run_stress_tests()
    sensitivity_results = simulator.run_sensitivity_analysis()

------------

.. _portfolio_forecast_visualization:

Visualization
=============

`PortfolioForecast` provides advanced visualization tools using `plotly` and `matplotlib`.

- **Quantile Forecast Plots**
- **Conditional Volatility Plots**
- **Correlation Heatmaps**
- **Factor Analysis Charts**

Example:

.. code-block:: python

    simulator.plot_forecast()
    simulator.plot_factor_heatmap()

------------

.. _portfolio_forecast_future:

Future Enhancements
====================

Planned enhancements include:

- **Enhanced Copula Models** (e.g., Student’s t, Vine Copulas)
- **Deep Learning Integrations** for factor forecasting
- **Live Market Data Integration**
- **Scenario-Based Portfolio Stress Testing**

------------

.. _portfolio_forecast_conclusion:

Conclusion
==========

The `PortfolioForecast` module is a powerful tool for simulating **multi-asset portfolios** using **stochastic PDEs**, **copulas**, and **volatility models**. It enables risk analysis, stress testing, and enhanced forecasting, making it ideal for financial research and trading strategies.

For more details, visit the **[AWT_MF Documentation](../index.rst)**.

