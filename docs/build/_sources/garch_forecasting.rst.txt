============================================
GARCH Model Optimization & Volatility Forecasting
============================================

This module provides **automated GARCH model selection, fitting, and volatility forecasting**  
for financial time series data, particularly **stock returns**.

**Key Features:**
-----------------
- **Automated Model Selection**
  - Finds the **best GARCH-type model** (GARCH, ARCH, EGARCH, APARCH, HARCH) based on **BIC**.
  - Optimizes `p` and `q` lag orders for the best fit.
- **Conditional Volatility Forecasting**
  - Uses the selected model to **forecast future volatility**.
- **Visualization**
  - Plots **actual returns vs. conditional volatility** using **Plotly**.

-------------------

Dependencies:
-------------
This module relies on:

- **`arch`**: For estimating GARCH-type models.
- **`plotly`**: To visualize **volatility trends**.
- **`pandas` & `numpy`**: For time-series data handling.

-------------------

Why Use GARCH for Volatility Forecasting?
-----------------------------------------
GARCH models capture **heteroskedasticity** (time-varying volatility),  
which is **crucial for risk modeling and options pricing**.

| **Feature**              | **GARCH Models**                           | **Simple Moving Averages** |
|--------------------------|-------------------------------------------|----------------------------|
| **Captures Volatility Clustering** | ✅ Yes | ❌ No |
| **Probability-Based Forecasts** | ✅ Yes (Conditional Variance) | ❌ No |
| **Time-Varying Volatility** | ✅ Yes | ❌ No |
| **Used in Financial Markets** | ✅ Yes (Options, Risk Models) | ✅ Yes (Basic Smoothing) |

-------------------

Class: `GARCHOptimizer`
------------------------
This module provides a **Python class** for automating GARCH model selection and forecasting.

.. code-block:: python

    from awt_quant.forecast.garch_forecast import GARCHOptimizer

    # Example: Fit GARCH model to stock returns
    garch = GARCHOptimizer(series=returns, dates_train=dates, ticker="AAPL")
    best_model = garch.fit()

    # Plot volatility trends
    fig = garch.plot_volatility()
    fig.show()

-------------------

Functions:
----------

**`fit()`** - Automatically selects the best GARCH model based on Bayesian Information Criterion (BIC).

- **Finds the best model** from:
  - **GARCH**
  - **ARCH**
  - **EGARCH**
  - **APARCH**
  - **HARCH**
- Optimizes **p** (lags for past variance) & **q** (lags for past squared residuals).
- Returns **fitted model**.

**`plot_volatility()`** - Plots **actual returns vs. conditional volatility**.

- Uses **Plotly** for interactive visualization.
- Helps **identify volatility spikes**.

**`forecast(horizon=10)`** - Forecasts **future volatility**.

- Generates conditional variance forecasts for the next `horizon` periods.

-------------------

End-to-End Example:
-------------------

1. **Fetch Stock Data**
    - Use Yahoo Finance or another source to obtain **log returns**.

2. **Run GARCH Model Selection**
    - Automatically **chooses the best volatility model**.

3. **Forecast Future Volatility**
    - Predicts **next 10-day volatility**.

4. **Visualize Volatility Trends**
    - Plots **conditional volatility vs. stock returns**.

.. code-block:: python

    from awt_quant.forecast.garch_forecast import GARCHOptimizer
    import pandas as pd

    # Load historical returns (assumed log-returns of stock prices)
    returns = pd.read_csv("aapl_returns.csv", index_col=0, parse_dates=True)["returns"]

    # Instantiate GARCH optimizer
    garch = GARCHOptimizer(series=returns, dates_train=returns.index, ticker="AAPL")

    # Fit optimal GARCH model
    garch.fit()

    # Plot volatility
    fig = garch.plot_volatility()
    fig.show()

    # Forecast future volatility
    forecasted_volatility = garch.forecast(horizon=10)
    print(forecasted_volatility)

-------------------

Visualization:
-------------
The module generates an interactive **Plotly** chart comparing **log-returns vs. conditional volatility**.

.. code-block:: python

    fig = garch.plot_volatility()
    fig.show()

-------------------

Conclusion:
-----------
- **GARCH models** capture **volatility clustering** in stock returns.
- **Automated model selection** ensures **optimal fitting**.
- **Forecasting helps estimate future market risk**.

This module enables **data-driven volatility modeling** for risk analysis and financial forecasting.

