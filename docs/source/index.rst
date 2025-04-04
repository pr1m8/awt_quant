.. AWT-Quant documentation master file, created by
   sphinx-quickstart on 2025.

Welcome to **AWT-Quant’s Documentation!**
==========================================

**AWT-Quant** is an advanced quantitative finance library specializing in:

- **Stochastic modeling & forecasting** (SPDE, GARCH, Jump Diffusion, Monte Carlo).
- **Multi-factor analysis (MFA)** for financial markets.
- **Portfolio optimization** (Mean-Variance, Black-Litterman, Monte Carlo).
- **Risk management & stress testing**.

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

1. **Installation Guide** → `installation.rst`
2. **Quick Start Tutorials** → `quickstart.rst`
3. **API Reference & Modules** → `api_reference.rst`

----

Core Features
=============

📊 **Stochastic PDE Forecasting**
---------------------------------
- Supports **Geometric Brownian Motion (GBM), Heston, CIR, OU, Jump Diffusion** models.
- Monte Carlo simulations & likelihood estimation.

📈 **Multi-Factor Analysis (MFA)**
----------------------------------
- Constructs **factors from macro & historical data**.
- Uses **ML-based feature selection** (Random Forest, PCA).
- **K-Means clustering & stress testing**.

💰 **Portfolio Optimization**
-----------------------------
- **Mean-Variance Optimization (MPT)** & Efficient Frontier.
- **Black-Litterman Bayesian Portfolio Optimization**.
- **Monte Carlo Portfolio Simulations**.

📉 **Risk Management & Stress Testing**
---------------------------------------
- **VaR, CVaR, Sharpe Ratio, Maximum Drawdown**.
- **Factor exposure analysis & custom performance tear sheets**.

----

Installation
============

To install AWT-Quant:

**Using pip**:
```bash
pip install awt-quant
```

**Using Poetry**:
```bash
poetry add awt-quant
```
For CUDA support:
```bash
poetry install --with cuda
```
# Modules