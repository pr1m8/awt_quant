import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd
# Need to get real interest rate .

def compute_beta(portfolio_value_series, ticker='^GSPC'):
    """
    Calculate the beta of the portfolio against a benchmark index.

    Args:
    portfolio_value_series (pandas.Series): Time series data of portfolio values.
    ticker (str): Ticker symbol of the benchmark index. Default is S&P 500 ('^GSPC').

    Returns:
    float: Beta value of the portfolio.
    """
    # Ensure the portfolio series is a Pandas Series
    if not isinstance(portfolio_value_series, pd.Series):
        raise ValueError("portfolio_value_series must be a pandas Series")

    # Calculate percent change for the portfolio
    portfolio_returns = portfolio_value_series.pct_change().dropna()

    # Check if the portfolio returns series is empty
    if portfolio_returns.empty:
        raise ValueError("The portfolio returns series is empty after percent change calculation.")

    # Download S&P500 data for the same date range as the portfolio
    start_date = portfolio_returns.index.min().strftime('%Y-%m-%d')
    end_date = portfolio_returns.index.max().strftime('%Y-%m-%d')
    sp500_data = yf.download(ticker, start=start_date, end=end_date)['Close'].pct_change().dropna()

    # Ensure that both series have a common index
    common_dates = portfolio_returns.index.intersection(sp500_data.index)
    portfolio_returns = portfolio_returns[common_dates]
    sp500_returns = sp500_data[common_dates]

    # Calculate the beta
    covariance = np.cov(portfolio_returns, sp500_returns)[0, 1]
    variance = np.var(sp500_returns)
    beta = covariance / variance

    return beta
def common_index(series1, series2):
    """
    Returns the common index values of two series.

    Args:
    series1 (pandas.Series): The first series.
    series2 (pandas.Series): The second series.

    Returns:
    pandas.Index: The common index values of the two series.
    """
    return series1.index.intersection(series2.index)

def risk_tearSheet(data, time_input='2y', risk_free_rate=0.02, confidence_level=0.95, benchmark_ticker='^GSPC'):
    # Determine the period for the analysis
    start_date, end_date = None, None
    if '-' in time_input:
        start_date, end_date = time_input.split('-')

    # Handle ticker input for data
    if isinstance(data, str):
        data = yf.download(data, start=start_date, end=end_date, period=(None if start_date else time_input))['Close']

    # Download benchmark data
    market_data = yf.download(benchmark_ticker, start=start_date, end=end_date, period=(None if start_date else time_input))
    
    market_returns = market_data['Close'].pct_change().dropna()

    # Calculate various metrics
    daily_returns = data.pct_change().dropna()
    annual_return = (data[-1] / data[0]) ** (252 / len(data)) - 1
    cumulative_returns = data[-1] / data[0] - 1
    annual_volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = (daily_returns.mean() - risk_free_rate / 252) / daily_returns.std() * np.sqrt(252)
    rolling_max = data.cummax()
    drawdown = (data - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    calmar_ratio = annual_return / abs(max_drawdown)
    # stability = np.exp(np.mean(np.log(1 + daily_returns))) - 1
    omega_ratio = np.sum(daily_returns[daily_returns > 0]) / np.abs(np.sum(daily_returns[daily_returns < 0]))
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (annual_return - risk_free_rate / 252) / np.sqrt(np.mean(downside_returns ** 2) * 252)
    skewness = daily_returns.skew()
    kurtosis = daily_returns.kurtosis()
    tail_ratio = np.abs(daily_returns[daily_returns < 0].mean()) / daily_returns[daily_returns > 0].mean()
    common_sense_ratio = annual_return / abs(max_drawdown)
    gross_leverage = (1 + daily_returns.abs()).mean()
    daily_turnover = np.mean(daily_returns.abs() / gross_leverage)
    semi_std_dev = np.sqrt(np.mean(np.square(daily_returns[daily_returns < daily_returns.mean()])))
    standard_error = np.std(daily_returns) / np.sqrt(len(daily_returns))
    var_99 = norm.ppf(1 - 0.01, daily_returns.mean(), daily_returns.std())
    var_95 = norm.ppf(1 - 0.05, daily_returns.mean(), daily_returns.std())
    ivar = np.mean(daily_returns[daily_returns <= var_95])
    cvar = daily_returns[daily_returns <= var_95].mean()
    var_gaussian = -norm.ppf(1 - confidence_level, daily_returns.mean(), daily_returns.std())

    # Calculate Beta and Alpha
    beta = compute_beta(data, ticker=benchmark_ticker)
    benchmark_annual_return = (market_returns[-1] / market_returns[0]) ** (252 / len(market_returns)) - 1
    alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))

    
     # Historical Volatility
    yearly_volatility = np.std(daily_returns) * np.sqrt(252)
    monthly_volatility = np.std(daily_returns) * np.sqrt(21)

    # Pain Gain Ratio
    gains = daily_returns[daily_returns > 0].sum()
    pains = -daily_returns[daily_returns < 0].sum()
    pain_gain_ratio = gains / pains

    # Downside Deviation
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))

    # Upside Potential Ratio
    upside_returns = daily_returns[daily_returns > 0]
    upside_potential_ratio = np.mean(upside_returns) / downside_deviation

    # Treynor Ratio
    treynor_ratio = (annual_return - risk_free_rate) / beta

    # Tracking Error
    tracking_error = np.sqrt(np.mean(np.square(daily_returns - market_returns)))

    # Information Ratio
    information_ratio = (annual_return - benchmark_annual_return) / tracking_error

    # R-Squared
    correlation_matrix = np.corrcoef(daily_returns, market_returns)
    r_squared = correlation_matrix[0, 1] ** 2
    
    # Maximum Drawdown Duration
    drawdown_end = np.argmin(drawdown)  # End of the max drawdown period
    drawdown_start = np.argmax(data[:drawdown_end])  # Start of the max drawdown period
    max_drawdown_duration = drawdown_end - drawdown_start
    
    # Compile metrics
    metrics = {
        'Annual Return': annual_return,
        'Cumulative Returns': cumulative_returns,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Calmar Ratio': calmar_ratio,
        # 'Stability': stability,
        'Omega Ratio': omega_ratio,
        'Sortino Ratio': sortino_ratio,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Tail Ratio': tail_ratio,
        'Common Sense Ratio': common_sense_ratio,
        'Gross Leverage': gross_leverage,
        'Daily Turnover': daily_turnover,
        'Semi Standard Deviation': semi_std_dev,
        'Standard Error': standard_error,
        'VaR 99%': var_99,
        'VaR 95%': var_95,
        'IVaR': ivar,
        'CVaR': cvar,
        f'Gaussian VaR({confidence_level})': var_gaussian,
        'Max Drawdown': max_drawdown,
        'Beta': beta,
        'Alpha': alpha
    }
    # Adding new metrics to the dictionary
    metrics['Yearly Volatility'] = yearly_volatility
    metrics['Monthly Volatility'] = monthly_volatility
    metrics['Pain Gain Ratio'] = pain_gain_ratio
    metrics['Downside Deviation'] = downside_deviation
    metrics['Upside Potential Ratio'] = upside_potential_ratio
    
    
    # Adding new metrics to the dictionary
    metrics.update({
        'Treynor Ratio': treynor_ratio,
        'Tracking Error': tracking_error,
        'Information Ratio': information_ratio,
        'R-Squared': r_squared,
        'Max Drawdown Duration': max_drawdown_duration,
        # 'Active Share': active_share,  # Uncomment when holdings data is available
    })

    
    return metrics
# Example usage
if __name__ == "__main__":
    data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')['Close']
    metrics = risk_tearSheet(data)
    print(metrics)