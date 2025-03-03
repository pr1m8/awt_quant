"""
Run Stock Forecast Simulations using SPDEMCSimulator.

This script allows running single and multiple stock simulations with different configurations.

Usage:
    python run_simulations.py --symbol AAPL --mode single
    python run_simulations.py --mode multi
"""

import argparse
import timeit
import numpy as np
import pandas as pd
import torch

from awt_quant.utils import financial_calendar_days_before

# Default Parameters
calendar = 'NYSE'
end_dates = ['2023-10-13', '2022-08-10', '2019-06-02', '2021-02-02']
forecast_periods = [14, 30, 60, 90, 180, 252]  # Trading days
train_test_splits = [0.75]
dt = 1
num_paths = 1000
num_sim = 100
plot_vol = True
plot_sim = False
eq_classes = ['Heston']
eq_class = 'Heston'


def run_single_simulation(symbol):
    """
    Runs a single simulation for a given stock symbol.
    """
    print(f"\nRunning single simulation for {symbol}")

    # Dynamic training split
    train_test_split = np.random.uniform(0.685, 0.8)
    total_observations = round(forecast_periods[-1] / (1 - train_test_split))
    start_date = financial_calendar_days_before(end_dates[0], total_observations, calendar)

    start_time = timeit.default_timer()
    from awt_quant.forecast.stochastic.pde_forecast import SPDEMCSimulator
    sim = SPDEMCSimulator(symbol, eq_class, start_date, end_dates[0], dt, num_paths, plot_vol)
    sim.download_data(train_test_split)
    sim.set_parameters()
    sim.simulate(eq_class)
    sim.plot_simulation(eq_class)
    forecast, errors = sim.backtest()

    print(f"Execution Time: {timeit.default_timer() - start_time:.2f} seconds")
    return forecast, errors


def run_multiple_simulations(symbols):
    """
    Runs multiple simulations across different stock symbols.
    """
    print("\nRunning multiple simulations for symbols:", symbols)

    results = []
    for symbol in symbols:
        for end_date in end_dates:
            for train_test_split in train_test_splits:
                total_observations = round(forecast_periods[-1] / (1 - train_test_split))
                start_date = financial_calendar_days_before(end_date, total_observations, calendar)

                start_time = timeit.default_timer()
                print(f"\nSimulating {symbol} for end date {end_date}, split {train_test_split:.2f}")
                from awt_quant.forecast.stochastic.pde_forecast import SPDEMCSimulator
                sim = SPDEMCSimulator(symbol, eq_class, start_date, end_date, dt, num_paths, plot_vol)
                sim.download_data(train_test_split)
                sim.set_parameters()
                sim.simulate(eq_class)

                forecast, errors = sim.backtest()
                errors['symbol'] = symbol
                errors['end_date'] = end_date
                results.append(errors)

                print(f"Execution Time: {timeit.default_timer() - start_time:.2f} seconds")

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPDEMCSimulator for stock forecasting")
    parser.add_argument("--symbol", type=str, help="Stock symbol to simulate (default: AAPL)", default="AAPL")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="single",
                        help="Choose simulation mode: 'single' (one stock) or 'multi' (multiple)")

    args = parser.parse_args()

    if args.mode == "single":
        run_single_simulation(args.symbol)
    else:
        stock_list = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        df_results = run_multiple_simulations(stock_list)
        print(df_results)
