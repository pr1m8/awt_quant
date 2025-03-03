"""
Portfolio Forecasting using Monte Carlo Simulations & Copula Models.

This module extends SPDEMCSimulator to forecast a portfolio of assets.

Classes:
    - PortfolioForecast: Forecasts portfolio performance using copula-based simulations.

"""

from awt_quant.forecast.stochastic.pde_forecast import SPDEMCSimulator
import numpy as np
import torch
import matplotlib.pyplot as plt
import pmdarima as pm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from copulas.multivariate import GaussianMultivariate
from sklearn.neighbors import KernelDensity
from awt_quant.utils import plot_correlogram


class PortfolioForecast(SPDEMCSimulator):
    """
    Forecasts a portfolio using Monte Carlo simulations and copula models.

    Attributes:
        - portfolio: Dictionary with stock symbols, positions, and quantities.
        - equation: Stochastic Differential Equation model (CIR, GBM, Heston, OU).
        - assets: List of SPDEMCSimulator instances for each stock.
        - train_data, test_data: Aggregated portfolio data.
    """

    def __init__(self, portfolio, equation, train_test_split, start_date='2022-01-01',
                 end_date='2022-03-01', dt=1, num_paths=1000, plot_vol=False, plot_sim=False):
        """
        Initializes PortfolioForecast with a list of stocks and positions.

        Args:
            portfolio (dict): Dictionary containing symbols, quantities, and positions.
            equation (str): Chosen stochastic model (CIR, GBM, Heston, OU).
            train_test_split (float): Ratio of training data.
            start_date (str): Start date for data collection.
            end_date (str): End date for data collection.
            dt (int): Time step size.
            num_paths (int): Number of simulation paths.
            plot_vol (bool): Plot volatility models.
            plot_sim (bool): Plot individual stock simulations.
        """
        self.portfolio = portfolio
        self.symbols = portfolio['symbol']
        self.assets = []
        self.quantity = portfolio['quantity']
        self.position = []
        self.S0, self.S = 0, []
        self.equation = equation
        self.train_data, self.test_data, self.data = 0, 0, 0
        self.dates_pred, self.dates_train, self.dates = None, None, None
        self.dt, self.num_paths = dt, num_paths

        # Load individual assets
        for idx, ticker in enumerate(self.symbols):
            asset = SPDEMCSimulator(ticker, equation, start_date, end_date, dt, num_paths, plot_vol)
            asset.download_data(train_test_split)
            asset.set_parameters()
            asset.simulate(equation)
            if plot_sim:
                asset.plot_simulation(equation)

            self.assets.append(asset)
            self.S.append(asset.S)
            self.position.append(1.0 if portfolio['position'][idx] == 'Long' else -1.0)

            self.train_data += self.quantity[idx] * asset.train_data
            self.test_data += self.quantity[idx] * asset.test_data['Close']
            self.data += self.quantity[idx] * asset.data['Close']
            self.S0 += self.quantity[idx] * asset.S0
            self.dates, self.dates_train, self.dates_pred = asset.dates, asset.dates_train, asset.dates_pred

    def copula_simulation(self):
        """
        Simulates a copula model to generate dependent return quantiles.

        Returns:
            - torch.Tensor: Copula-simulated quantiles (shape: n_stocks x n_simulations x T)
        """
        num_paths, n_stocks, T = self.num_paths, len(self.assets), self.assets[0].N
        log_returns = [asset.returns for asset in self.assets]
        kdes = [KernelDensity(kernel="gaussian").fit(asset.reshape(-1, 1)) for asset in log_returns]

        uniform_data = []
        for kde, asset in zip(kdes, log_returns):
            scores = kde.score_samples(asset.reshape(-1, 1))
            cdf_values = np.exp(scores - scores.max())  # Avoid underflow
            cdf_values = cdf_values.cumsum() / cdf_values.sum()
            uniform_data.append(cdf_values)

        uniform_data = np.array(uniform_data)
        copula = GaussianMultivariate()
        copula.fit(uniform_data.T)
        copula_samples = copula.sample(num_paths * T).values
        return copula_samples.reshape(num_paths, T, n_stocks).transpose(2, 0, 1)

    def forecast(self):
        """
        Generates portfolio price forecasts based on copula-simulated quantiles.
        """
        n_stocks, num_paths, T = len(self.quantity), self.num_paths, self.assets[0].N
        quantiles = torch.tensor(self.copula_simulation())
        simulation = torch.stack(self.S)

        forecasted_paths = torch.zeros((n_stocks, num_paths, T)).to(torch.float64)
        for stock_idx in range(n_stocks):
            for t in range(T):
                forecasted_paths[stock_idx, :, t] = torch.quantile(
                    simulation[stock_idx, :, t].to(torch.float64),
                    quantiles[stock_idx, :, t].to(torch.float64)
                )

        self.S = (torch.tensor(self.quantity).view(n_stocks, 1, 1) * forecasted_paths).sum(0)

    def plot_forecast(self):
        """
        Plots portfolio simulation with quantile paths and percent errors.
        """
        redundancy_column = self.S0 * torch.ones(self.num_paths)
        pred_data = torch.tensor(self.test_data)
        shifted_simulations = torch.column_stack((redundancy_column, self.S))
        dates = [self.dates_train[-1]] + list(self.dates_pred)
        quantiles, colors = [0.01, 0.25, 0.5, 0.75, 0.99], ['red', 'orange', 'green', 'orange', 'red']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Portfolio Forecast", "Error Mean", "Error Median"),
            specs=[[{"colspan": 2}, None], [{}, {}]]
        )

        # Portfolio Forecast
        fig.add_trace(go.Scatter(x=self.dates, y=self.data, mode='lines', name='Historical Prices',
                                 line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=shifted_simulations.mean(0), mode='lines', name='Mean Forecast',
                                 line=dict(color='blue')), row=1, col=1)
        for i, q in enumerate(quantiles):
            fig.add_trace(go.Scatter(x=dates, y=shifted_simulations.quantile(q, dim=0),
                                     mode='lines', name=f'Quantile {q}', line=dict(color=colors[i])),
                          row=1, col=1)

        # Mean Error
        mean_error = 100 * (pred_data - self.S.mean(0)) / pred_data
        fig.add_trace(go.Scatter(x=self.dates_pred, y=mean_error, mode='lines', name='Mean Error',
                                 line=dict(color='black')), row=2, col=1)

        # Median Error
        median_error = 100 * (pred_data - self.S.quantile(0.5, dim=0)) / pred_data
        fig.add_trace(go.Scatter(x=self.dates_pred, y=median_error, mode='lines', name='Median Error',
                                 line=dict(color='black')), row=2, col=2)

        fig.update_layout(title_text="Portfolio Forecast and Error Analysis")
        fig.show()

    def backtest(self):
        """
        Backtests the forecast and prints ARMA model summaries.
        """
        for idx, asset in enumerate(self.assets):
            returns = np.log(asset.train_data['Close']).diff().dropna()
            plot_correlogram(returns, lags=len(returns) // 2 - 1,
                             title=f'Correlogram for {self.symbols[idx]}')

            # Fit ARMA model
            arma_model = pm.auto_arima(asset.returns, d=0, start_p=1, start_q=1, max_p=4, max_q=4,
                                       seasonal=False, information_criterion='bic', trace=True)
            print(f"ARMA Model Summary for {self.symbols[idx]}:\n", arma_model.summary())
