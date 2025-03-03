============================================
Lag-Llama: Stock Price Forecasting
============================================

This module implements time series forecasting using **Lag-Llama**, a **large language model (LLM) for time series forecasting**.  
It leverages **GluonTS**, **Monte Carlo sampling**, and **RoPE scaling** for long-context forecasting.

**Key Features:**
-----------------
- **Data Fetching**: Downloads stock price data from Yahoo Finance.
- **Lag-Llama Model**:
  - Context-aware **time-series forecasting** with transformer-based LLMs.
  - **Monte Carlo sampling** for probabilistic predictions.
  - **RoPE Scaling** for extended forecast horizons.
- **Backtesting & Evaluation**:
  - Uses **Continuous Ranked Probability Score (CRPS)** for forecast evaluation.
  - Computes **mean error** and **quantile errors** over backtested forecasts.
- **Visualization**:
  - Plots **actual vs. predicted stock prices** using Matplotlib.
  - Displays **forecast uncertainty bands**.

-------------------

Dependencies:
-------------
This module relies on the following key dependencies:

- **Lag-Llama**: A specialized time-series forecasting model built on LLMs.
- **GluonTS**: Provides dataset handling and evaluation functions.
- **Yahoo Finance (yfinance)**: Fetches stock price data.
- **PyTorch**: Runs deep learning models efficiently.
- **Matplotlib**: Visualizes actual vs. forecasted stock prices.

-------------------

Why Use Lag-Llama for Forecasting?
-----------------------------------
Lag-Llama is designed for **time-series prediction** using **LLM architectures**.  
Unlike traditional forecasting models, it can capture **long-term dependencies** while handling **non-stationary trends**.

| **Feature**            | **Lag-Llama**                         | **Traditional Models (ARIMA, Prophet, etc.)** |
|------------------------|--------------------------------------|----------------------------------------------|
| **Time-Series Type**   | Non-stationary, long-sequence       | Mostly stationary, short-sequence           |
| **Monte Carlo Sampling** | Yes (uncertainty quantification) | No                                          |
| **Long Context Forecasting** | Yes (RoPE scaling)             | No                                          |
| **Deep Learning**      | Transformer-based LLM               | Limited (LSTMs, GRUs)                       |

-------------------

Class: Lag-Llama Forecasting Pipeline
--------------------------------------
This module provides functions to **fetch, forecast, evaluate, and visualize stock prices**.

.. code-block:: python

    from awt_quant.forecast.lag_llama import main

    # Run full forecasting pipeline
    main()

-------------------

Functions:
----------

**fetch_stock_data(ticker, start_date, end_date)**
    - Fetches stock price data from Yahoo Finance.
    - Formats the dataset for **GluonTS compatibility**.

**get_lag_llama_predictions(dataset, prediction_length, num_samples=100, context_length=32, use_rope_scaling=False)**
    - Runs **Lag-Llama forecasting** with Monte Carlo sampling.
    - **RoPE scaling** (Relative Positional Encoding) allows longer context forecasting.

**plot_forecasts(forecasts, tss, ticker, prediction_length)**
    - Plots **actual vs. predicted stock prices** with forecast uncertainty bands.

**evaluate_forecasts(forecasts, tss)**
    - Computes **CRPS (Continuous Ranked Probability Score)** as an evaluation metric.

**backtest(forecasts, actual_series)**
    - Compares forecasts against actual stock prices.
    - Computes **mean error** and **quantile errors**.

-------------------

End-to-End Forecasting Example:
-------------------------------

1. **Download Stock Data**
    - Uses `yfinance` to fetch historical prices.

2. **Run Lag-Llama Forecasting**
    - Generates forecasts using **Monte Carlo sampling**.

3. **Evaluate Model Performance**
    - Computes **forecast accuracy using CRPS**.

4. **Visualize Forecasts**
    - Plots **actual vs. predicted stock prices**.

.. code-block:: python

    from awt_quant.forecast.lag_llama import main

    main()

-------------------

Visualization:
-------------
Forecasts can be visualized using **Matplotlib**.

.. code-block:: python

    dataset = fetch_stock_data("AAPL", "2023-01-01", "2024-01-01")
    forecasts, tss = get_lag_llama_predictions(dataset, prediction_length=30)

    plot_forecasts(forecasts, tss, ticker="AAPL", prediction_length=30)

-------------------

Conclusion:
-----------
- **Lag-Llama** is a powerful **LLM-based forecasting model** for **stock price prediction**.
- **Monte Carlo sampling** provides **uncertainty-aware forecasts**.
- **RoPE Scaling** allows **long-context sequence modeling**.
- **CRPS & quantile errors** provide robust **evaluation metrics**.

This module enables **advanced, deep-learning-based stock forecasting**, integrating **transformer architectures with GluonTS**.

