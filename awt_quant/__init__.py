"""
AWT Quant - Algorithmic Wealth Trading Quantitative Library

This package provides tools for financial modeling, stochastic simulations, risk assessment, portfolio optimization,
and macroeconomic forecasting.

Modules:
    - `data_fetch`: Market data retrieval and preprocessing.
    - `forecast`: Time-series forecasting and stochastic modeling.
    - `portfolio`: Portfolio analysis, optimization, and factor models.
    - `risk`: Risk modeling and performance analysis.
    - `utils`: Utility functions for quantitative finance.
"""

__version__ = "0.1.0"

# Expose high-level modules for easy imports
from .data_fetch import *
from .forecast import *
from .portfolio import *
from .risk import *
from .utils import *

__all__ = ["data_fetch", "forecast", "portfolio", "risk", "utils"]
