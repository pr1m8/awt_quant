"""
GARCH Model Optimization and Volatility Forecasting.

This module finds the best GARCH-like model for a given time series, fits it, and visualizes 
its conditional volatility.

Classes:
    - GARCHOptimizer: Handles GARCH model selection, fitting, and volatility forecasting.

Usage:
    garch = GARCHOptimizer(series, dates_train, ticker)
    best_model = garch.fit()
    fig = garch.plot_volatility()
    fig.show()
"""

import numpy as np
import plotly.graph_objects as go
from arch import arch_model

class GARCHOptimizer:
    """
    A class to find and optimize a GARCH-like model for a given time series.

    Attributes:
        series (pd.Series): Time series data of asset returns.
        dates_train (pd.Series): Corresponding date index for the series.
        ticker (str): Stock ticker symbol.
        plot_vol (bool): Whether to plot the volatility.
        best_model (str): The best identified GARCH model.
        best_p (int): Optimal p lag order.
        best_q (int): Optimal q lag order.
        fitted_model (arch.univariate.base.ARCHModelResult): The fitted GARCH model.
    """

    def __init__(self, series, dates_train, ticker, plot_vol=True):
        self.series = series
        self.dates_train = dates_train
        self.ticker = ticker
        self.plot_vol = plot_vol
        self.best_model = None
        self.best_p = None
        self.best_q = None
        self.fitted_model = None

    def fit(self):
        """
        Finds the best GARCH model using Bayesian Information Criterion (BIC).

        Returns:
            arch.univariate.base.ARCHModelResult: The fitted optimal GARCH model.
        """
        volatility_models = ['GARCH', 'ARCH', 'EGARCH', 'APARCH', 'HARCH']
        models = {vol: [(), np.inf] for vol in volatility_models}
        
        p_max, q_max = 5, 5
        for vol in models:
            for p in range(1, p_max):
                for q in range(1, q_max):
                    model = arch_model(self.series, mean='Constant', vol=vol, p=p, q=q, dist='ged')
                    results = model.fit(disp='off')
                    if results.bic < models[vol][1]:
                        models[vol] = [(p, q), results.bic]
        
        self.best_model = min(models, key=lambda v: models[v][1])
        self.best_p, self.best_q = models[self.best_model][0]

        print(f'Selected GARCH Model: {self.best_model}({self.best_p}, {self.best_q})')

        self.fitted_model = arch_model(
            self.series, mean='Constant', vol=self.best_model, p=self.best_p, q=self.best_q, dist='ged'
        ).fit()

        return self.fitted_model

    def plot_volatility(self):
        """
        Plots the conditional volatility of the fitted GARCH model.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure displaying the volatility plot.
        """
        if not self.plot_vol or self.fitted_model is None:
            return None
        
        cond_vol = self.fitted_model.conditional_volatility

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.dates_train[1:], y=cond_vol, mode='lines', name='Conditional Volatility',
            line=dict(color='#4B75AB')
        ))
        fig.add_trace(go.Scatter(
            x=self.dates_train[1:], y=self.series, mode='lines', name='Returns',
            line=dict(color='black')
        ))
        fig.update_layout(
            title=f"Log-Returns and {self.best_model}({self.best_p}, {self.best_q}) Conditional Volatility for {self.ticker}",
            xaxis_title="Time",
            yaxis_title="Value"
        )
        
        return fig

    def forecast(self, horizon=10):
        """
        Generates a volatility forecast for the next `horizon` periods.

        Args:
            horizon (int): Number of future periods to forecast.

        Returns:
            pd.DataFrame: A DataFrame with the forecasted conditional variances.
        """
        if self.fitted_model is None:
            raise ValueError("GARCH model has not been fitted yet. Call `fit()` first.")

        forecast = self.fitted_model.forecast(horizon=horizon)
        return forecast.variance.iloc[-1]
