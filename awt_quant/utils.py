import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from scipy.stats import probplot, moment
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from numpy import sqrt, log, polyfit, std, subtract
#from torch import sqrt, log, polyfit, std, subtract
def hurst(ts, lag):
    """
    Calculates the Hurst Exponent for a given time series.

    The Hurst Exponent is a measure of long-term memory in a time series:
    - Near 0.5: Random series.
    - Near 0: Mean reverting.
    - Near 1: Trending.

    Args:
        ts (array-like): Time series data.
        lag (int): Maximum lag to compute.

    Returns:
        float: Hurst exponent value.
    """
    lags = range(2, lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return max(0.00, poly[0] * 2.0)

def hurst(ts, lag):
    """Returns the Hurst Exponent of the time series vector ts
    The Hurst Exponent is a statistical measure used to classify time series and infer the level of difficulty in predicting and
    choosing an appropriate model for the series at hand. The Hurst exponent is used as a measure of long-term memory of time series.
    It relates to the autocorrelations of the time series, and the rate at which these decrease as the lag between pairs of
    values increases.

    Value near 0.5 indicates a random series.
    Value near 0 indicates a mean reverting series.
    Value near 1 indicates a trending series."""
    lags = range(2, lag)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = polyfit(log(lags), log(tau), 1)
    return max(0.00, poly[0] * 2.0)

def financial_calendar_days_before(date_str, T, calendar_name='NYSE'):
    """
    Gets the T-th market day occurring before a given date.

    Args:
        date_str (str): End date in 'YYYY-MM-DD' format.
        T (int): Number of market days to go back.
        calendar_name (str): Market calendar name (default: 'NYSE').

    Returns:
        str: Computed start date in 'YYYY-MM-DD' format.
    """
    calendar = mcal.get_calendar(calendar_name)
    end_date = pd.Timestamp(date_str)
    delta = pd.Timedelta(days=T * 3)
    start_date = end_date - delta
    market_days = calendar.schedule(start_date=start_date, end_date=end_date)
    result_date = market_days.iloc[-T].market_open.date()
    return result_date.strftime('%Y-%m-%d')


def plot_correlogram(x, lags=None, title=None):
    """
    Plots the correlogram for a given time series.

    The output consists of:
    - Time series plot.
    - Q-Q plot.
    - Autocorrelation Function (ACF).
    - Partial Autocorrelation Function (PACF).

    Args:
        x (pd.Series): Time series data.
        lags (int, optional): Number of lags in ACF/PACF.
        title (str, optional): Plot title.
    """
    lags = min(10, int(len(x) / 5), len(x) - 1) if lags is None else min(lags, len(x) - 1)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    
    # Time series plot
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f} \nHurst: {round(hurst(x.values, lags), 2)}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)

    # Q-Q plot
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)

    # ACF/PACF
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')

    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    