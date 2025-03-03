import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tqdm
from scipy.stats import norm
from datetime import datetime
from plotly.subplots import make_subplots
from arch import arch_model
import warnings

# Import from `awt_quant`
from awt_quant.utils import financial_calendar_days_before
from awt_quant.data_fetch.yf_fetch import download_data
from awt_quant.forecast.garch_forecast import GARCHOptimizer
from awt_quant.forecast.stochastic.stochastic_models import StochasticSimulator
from awt_quant.forecast.stochastic.stochastic_likelihoods import neg_log_likelihood_ou, neg_log_likelihood_cir

class SPDEMCSimulator:
    """
    Stochastic Process & GARCH-based Forecasting Simulator.

    Attributes:
        ticker (str): Stock ticker symbol.
        equation (str): Stochastic model (`CIR`, `GBM`, `Heston`, `OU`).
        start_date (str): Start date for fetching historical data.
        end_date (str): End date for fetching historical data.
        dt (float): Time increment (default: 1 for daily, 1/252 for annual).
        num_paths (int): Number of Monte Carlo simulation paths.
    """

    def __init__(self, ticker, equation, start_date='2022-01-01', end_date='2022-03-01',
                 dt=1, num_paths=1000, plot_vol=True):
        self.ticker = ticker
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.dt = dt
        self.num_paths = num_paths
        self.equation = equation
        self.plot_vol = plot_vol

        # Data placeholders
        self.train_data, self.test_data, self.dates = None, None, None
        self.S, self.S0, self.T, self.N = None, None, None, None
        self.prices, self.returns, self.mu, self.sigma = None, None, None, None
        self.forecasted_vol = None
        self.GARCH_fit = None
        self.device = "cpu"

    def download_data(self, train_test_split):
        """Downloads historical stock data and splits into train-test sets."""
        data = download_data(self.ticker, self.start_date, self.end_date)
        self.dates = data.index
        self.train_data, self.test_data = np.split(data, [int(train_test_split * len(data))])
        self.dates_train = self.train_data.index
        self.dates_pred = self.test_data.index

        # Compute log-returns
        self.prices = torch.tensor(self.train_data['Close'])
        log_returns = torch.log(self.prices)
        self.returns = log_returns[1:] - log_returns[:-1]
        self.mu = self.returns.mean()
        self.sigma = self.returns.std()

        self.S0 = self.train_data['Close'].iloc[-1]
        self.T = len(self.test_data)
        self.N = int(self.T / self.dt)

        # Run GARCH model for Heston model only
        if self.equation == 'Heston':
            self.GARCH_fit = GARCHOptimizer(self.returns, self.plot_vol)
            self.forecasted_vol = self.GARCH_fit.forecast()

    def simulate(self):
        """Runs stochastic simulation based on the selected model."""
        simulator = StochasticSimulator(
            self.equation, self.mu, self.sigma, self.dt, self.num_paths, self.N, self.device
        )
        self.S = simulator.run_simulation(self.S0, self.returns, self.GARCH_fit)

    def backwards(self, strike_price, option):
        """Calculates backward pricing probability for options."""
        weights = torch.full((self.num_paths, self.N), 1 / self.N)
        expiration_value = torch.zeros((self.num_paths, self.N))

        if option == 'Call':
            expiration_value[:, -1] = torch.maximum(self.S[:, -1] - strike_price, torch.tensor(0.0))

    def plot_simulation(self):
        """Plots the quantile paths for simulated stock price."""
        dates = [self.dates_train[-1]] + list(self.dates_pred)
        quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
        colors = ['red', 'orange', 'green', 'orange', 'red']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.dates, y=self.train_data['Close'],
                                 mode='lines', name='Historical Prices', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=dates, y=self.S.mean(0),
                                 mode='lines', name='Mean Forecast', line=dict(color='#4B75AB')))

        for i, q in enumerate(quantiles):
            fig.add_trace(go.Scatter(x=dates, y=self.S.quantile(q, dim=0),
                                     mode='lines', name=f'Quantile {q}', line=dict(color=colors[i])))

        fig.update_layout(title=f"{self.ticker} Stock Price Simulation ({self.equation})",
                          xaxis_title="Time", yaxis_title="Stock Price")
        fig.show()

    def error_estimation(self, num_sim=100):
        """Estimates the error of stock price forecasts."""
        actual_value = torch.tensor(self.test_data['Close'])[-1]
        mean_forecast, median_forecast = torch.zeros(num_sim), torch.zeros(num_sim)

        for _ in tqdm.tqdm(range(num_sim)):
            self.simulate()
            mean_forecast[_] = self.S[:, -1].mean()
            median_forecast[_] = self.S[:, -1].quantile(0.5)

        error_mean = 100 * (actual_value - mean_forecast) / actual_value
        error_median = 100 * (actual_value - median_forecast) / actual_value

        # Generate plots
        linspace = np.linspace(error_mean.min().item(), error_mean.max().item(), 100)
        pdf_mean = norm.pdf(linspace, error_mean.mean().item(), error_mean.std().item())

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Mean Error', 'Median Error'))
        fig.add_trace(go.Histogram(x=error_mean.cpu().numpy(), name='Error Distribution',
                                   marker=dict(color='#4B75AB'), histnorm='probability'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=linspace, y=pdf_mean, mode='lines', name='Fitted Normal',
                                 line=dict(color='black')), row=1, col=1)
        fig.show()

        return error_mean.mean().item(), error_median.mean().item()

    def backtest(self):
        """Performs backtesting on the simulated data."""
        actual_data = torch.tensor(self.test_data['Close'])
        predicted_data = self.S
        quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]

        forecasts = self.test_data['Close'].to_frame().rename(columns={'Close': 'actual'})

        # Calculate percent errors
        errors = []
        for q in quantiles:
            quantile_forecast = predicted_data.quantile(q, dim=0)
            quantile_error = 100 * (actual_data - quantile_forecast) / actual_data
            errors.append(quantile_error.mean().item())
            forecasts[f"{q}_forecast"] = quantile_forecast

        mean_forecast = predicted_data.mean(0)
        mean_error = 100 * (actual_data - mean_forecast) / actual_data
        errors.append(mean_error.mean().item())
        forecasts['mean_forecast'] = mean_forecast

        error_df = pd.DataFrame({"Quantile": quantiles + ["Mean"], "Error": errors})

        return forecasts, error_df

if __name__ == "__main__":
    sim = SPDEMCSimulator("AAPL", "Heston", start_date="2022-01-01", end_date="2023-01-01")
    sim.download_data(0.75)
    sim.simulate()
    sim.plot_simulation()
    sim.backtest()
    sim.error_estimation()
