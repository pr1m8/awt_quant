import timeit
import numpy as np
import pandas as pd
import torch
from awt_quant.utils import financial_calendar_days_before
from awt_quant.forecast.stochastic.portfolio.portfolio_forecast import PortfolioForecast



def run_portfolio_simulation(portfolio, equation, start_date, end_date, train_test_split, num_paths=1000,
                             plot_vol=False, plot_sim=False, num_sim=100):
    """
    Runs a single portfolio simulation using the chosen stochastic differential equation.

    Args:
        portfolio (dict): Dictionary containing symbols, positions, and quantities.
        equation (str): Chosen stochastic model (CIR, GBM, Heston, OU).
        start_date (str): Start date for simulation.
        end_date (str): End date for simulation.
        train_test_split (float): Ratio of training data.
        num_paths (int): Number of simulation paths (default: 1000).
        plot_vol (bool): Whether to plot volatility models (default: False).
        plot_sim (bool): Whether to plot individual stock simulations (default: False).
        num_sim (int): Number of simulations for error estimation (default: 100).
    """
    dt = 1
    print(f"Running Portfolio Simulation: {portfolio['symbol']} | Equation: {equation}")

    start_time = timeit.default_timer()
    sim = PortfolioForecast(portfolio, equation, train_test_split, start_date, end_date, dt, num_paths, plot_vol, plot_sim)

    # Forecast and visualize results
    sim.forecast()
    sim.plot_forecast()
    forecasts, error_df = sim.backtest()

    # Run error estimation
    error_fig, error_summaries, error_summary_df = sim.error_estimation(equation, num_sim)

    end_time = timeit.default_timer()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")

    return forecasts, error_df, error_summary_df


def compare_multiple_portfolio_simulations(portfolios, equation_classes, end_dates, forecast_periods, train_test_splits,
                                           num_paths=1000, num_sim=100, plot_vol=False, plot_sim=False):
    """
    Compares multiple portfolio simulations across different stochastic models and settings.

    Args:
        portfolios (list[dict]): List of portfolios with stock symbols and positions.
        equation_classes (list[str]): List of stochastic models to test.
        end_dates (list[str]): End dates for different simulations.
        forecast_periods (list[int]): Forecasting periods in days.
        train_test_splits (list[float]): Different train-test split ratios.
        num_paths (int): Number of Monte Carlo paths (default: 1000).
        num_sim (int): Number of simulations for error estimation (default: 100).
        plot_vol (bool): Whether to plot volatility models (default: False).
        plot_sim (bool): Whether to plot individual stock simulations (default: False).

    Returns:
        pd.DataFrame: Dataframe containing forecast errors and summaries.
    """
    calendar = "NYSE"
    results = []

    for portfolio in portfolios:
        for end_date in end_dates:
            for forecast_period in forecast_periods:
                for train_test_split in train_test_splits:
                    total_observations = round(forecast_period / (1 - train_test_split))
                    start_date = financial_calendar_days_before(end_date, total_observations, calendar)

                    for equation in equation_classes:
                        print(f"\nRunning simulation for {portfolio['symbol']} | Equation: {equation} | End Date: {end_date}")

                        # Run portfolio simulation
                        forecasts, error_df, error_summary_df = run_portfolio_simulation(
                            portfolio, equation, start_date, end_date, train_test_split, num_paths, plot_vol, plot_sim, num_sim
                        )

                        # Store results
                        error_df["portfolio"] = ", ".join(portfolio["symbol"])
                        error_df["equation"] = equation
                        error_df["end_date"] = end_date
                        error_df["forecast_period"] = forecast_period
                        error_df["train_test_split"] = train_test_split

                        results.append(error_df)

    final_results = pd.concat(results, ignore_index=True)
    return final_results
