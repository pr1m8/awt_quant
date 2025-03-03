import os
import yfinance as yf
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import islice

# Constant for Lag-Llama checkpoint storage
LAG_LLAMA_CKPT_PATH = "resources/lag_llama/model/lag_llama.ckpt"

def get_device():
    """
    Returns the appropriate device for computation.

    Uses CUDA if available, otherwise falls back to CPU.

    Returns:
        torch.device: The device to use for model computations.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches stock price data from Yahoo Finance and formats it for Lag-Llama.

    Args:
        ticker (str): Stock symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        PandasDataset: The dataset formatted for Lag-Llama.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df.reset_index()

    # Format for Lag-Llama
    df["target"] = df["Close"].astype("float32")  # Set target variable
    df["item_id"] = ticker  # Stock identifier
    df = df[["Date", "target", "item_id"]]

    return PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")

def get_lag_llama_predictions(dataset, prediction_length, num_samples=100, context_length=32, use_rope_scaling=False):
    """
    Runs Lag-Llama predictions on a given dataset.

    Args:
        dataset (PandasDataset): The dataset for forecasting.
        prediction_length (int): Forecast horizon.
        num_samples (int, optional): Number of Monte Carlo samples per timestep. Defaults to 100.
        context_length (int, optional): Context length for model. Defaults to 32.
        use_rope_scaling (bool, optional): Whether to use RoPE scaling for extended context. Defaults to False.

    Returns:
        Tuple[list, list]: Forecasts and actual time series.
    """
    device = get_device()

    if not os.path.exists(LAG_LLAMA_CKPT_PATH):
        raise FileNotFoundError(f"Lag-Llama checkpoint not found at {LAG_LLAMA_CKPT_PATH}. Download it before proceeding.")

    ckpt = torch.load(LAG_LLAMA_CKPT_PATH, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_args = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=LAG_LLAMA_CKPT_PATH,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_args if use_rope_scaling else None,
        batch_size=1,
        num_parallel_samples=num_samples,
        device=device,
    )

    predictor = estimator.create_predictor(
        estimator.create_transformation(),
        estimator.create_lightning_module()
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )

    return list(forecast_it), list(ts_it)

def plot_forecasts(forecasts, tss, ticker, prediction_length):
    """
    Plots actual stock prices along with forecasted values.

    Args:
        forecasts (list): List of forecasted series.
        tss (list): List of actual time series.
        ticker (str): Stock ticker symbol.
        prediction_length (int): Forecast horizon.
    """
    plt.figure(figsize=(12, 6))
    date_formatter = mdates.DateFormatter('%b %d')

    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 1):
        plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="Actual", color='black')
        forecast.plot(color='g')  # Forecasted path
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(date_formatter)
        plt.title(f"Lag-Llama Forecast for {ticker}")

    plt.legend()
    plt.show()

def evaluate_forecasts(forecasts, tss):
    """
    Evaluates forecasts using GluonTS Evaluator.

    Args:
        forecasts (list): Forecasted time series.
        tss (list): Actual time series.

    Returns:
        dict: Aggregated evaluation metrics including CRPS.
    """
    evaluator = Evaluator()
    agg_metrics, _ = evaluator(iter(tss), iter(forecasts))
    return agg_metrics

def backtest(forecasts, actual_series):
    """
    Computes backtest evaluation metrics by comparing forecasts against actual values.

    Args:
        forecasts (list): List of forecasted time series.
        actual_series (list): List of actual time series.

    Returns:
        dict: Evaluation metrics including mean error and quantiles.
    """
    forecast_vals = forecasts[0].samples.mean(axis=0)  # Mean forecast values
    actual_vals = actual_series[0].to_numpy()[-len(forecast_vals):]  # Align with forecast length

    error = 100 * (actual_vals - forecast_vals) / actual_vals  # Percent error

    quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
    quantile_errors = {q: 100 * (actual_vals - forecasts[0].quantile(q)) / actual_vals for q in quantiles}

    return {
        "Mean Error": error.mean(),
        "Quantile Errors": quantile_errors
    }

def main():
    """
    Runs the end-to-end pipeline:
    - Fetches stock data
    - Runs Lag-Llama forecasting with context length 32
    - Evaluates and plots the forecasts
    - Performs backtesting
    """
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    prediction_length = 30
    num_samples = 100

    dataset = fetch_stock_data(ticker, start_date, end_date)

    # Forecast with context length 32
    forecasts_ctx_len_32, tss_ctx_len_32 = get_lag_llama_predictions(
        dataset, prediction_length, num_samples, context_length=32, use_rope_scaling=False
    )

    plot_forecasts(forecasts_ctx_len_32, tss_ctx_len_32, ticker, prediction_length)

    # Run Evaluator
    eval_metrics = evaluate_forecasts(forecasts_ctx_len_32, tss_ctx_len_32)
    print("CRPS:", eval_metrics["mean_wQuantileLoss"])

    # Run Backtest
    backtest_results = backtest(forecasts_ctx_len_32, tss_ctx_len_32)
    print("Backtest Results:", backtest_results)

if __name__ == "__main__":
    main()
