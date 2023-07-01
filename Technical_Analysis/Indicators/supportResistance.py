from matplotlib import pyplot as plt



import numpy as np
from scipy.signal import argrelextrema
import pandas as pd

def find_support_resistance(df, c='close', n=5):
    # Find local minima (support)
    df['support'] = df.iloc[argrelextrema(df[c].values, np.less_equal, order=n)[0]][c]

    # Find local maxima (resistance)
    df['resistance'] = df.iloc[argrelextrema(df[c].values, np.greater_equal, order=n)[0]][c]

    return df


def fibonacci_retracement(stock_df, high_col='High', low_col='Low', level_cols=['38.2%', '50%', '61.8%']):
    """
    Computes the Fibonacci retracement levels of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data
    high_col (str): The name of the column in the stock dataframe containing the high prices (default is 'High')
    low_col (str): The name of the column in the stock dataframe containing the low prices (default is 'Low')
    level_cols (list): A list of the names of the columns in the output dataframe (default is ['38.2%', '50%', '61.8%'])

    Returns:
    pd.DataFrame: A pandas dataframe containing the Fibonacci retracement levels
    """
    # Compute the highest and lowest prices
    highest_price = stock_df[high_col].max()
    lowest_price = stock_df[low_col].min()

    # Compute the range of the price movement
    price_range = highest_price - lowest_price

    # Compute the Fibonacci retracement levels
    level_values = [highest_price - price_range * level for level in [0.382, 0.5, 0.618]]

    # Create a pandas dataframe to store the Fibonacci retracement levels
    retracement_df = pd.DataFrame({level_cols[i]: [level_values[i]] for i in range(len(level_cols))})

    return retracement_df

def fibonacci_arcs(df, low_col='Low', high_col='High', levels=[0, 0.236, 0.382, 0.5, 0.618, 1]):
    """
    Computes and returns the Fibonacci Arcs indicator for a given stock DataFrame.
    """
    # Compute the high and low values for the indicator
    low_val = df[low_col].min()
    high_val = df[high_col].max()

    # Compute the range and mid point
    range_val = high_val - low_val
    mid_point = low_val + range_val / 2

    # Compute the levels
    levels_val = [mid_point - level * range_val for level in levels[::-1]]

    # Plot the levels
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close')
    for level in levels_val:
        plt.plot(df.index, [level] * len(df.index), label=f'Fibonacci Arcs {round(mid_point - level, 2)}')
    plt.legend()
    plt.title('Fibonacci Arcs')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def fibonacci_fan(df, low_col='Low', high_col='High', levels=[0, 0.236, 0.382, 0.5, 0.618, 1]):
    """
    Computes and returns the Fibonacci Fan indicator for a given stock DataFrame.
    """
    # Compute the high and low values for the indicator
    low_val = df[low_col].min()
    high_val = df[high_col].max()

    # Compute the range and mid point
    range_val = high_val - low_val
    mid_point = low_val + range_val / 2

    # Compute the levels
    levels_val = [mid_point - level * range_val for level in levels[::-1]]

    # Plot the levels
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close')
    for i, level in enumerate(levels_val):
        plt.plot(df.index, [(i + 1) * level - i * mid_point] * len(df.index),
                 label=f'Fibonacci Fan {round(mid_point - level, 2)}')
    plt.legend()
    plt.title('Fibonacci Fan')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def fibonacci_time_zone(df, start_date, end_date, high='High', low='Low'):
    # Filter the dataframe based on the start and end dates
    df = df.loc[start_date:end_date]

    # Compute the highest high and lowest low over the time period
    max_high = df[high].max()
    min_low = df[low].min()

    # Compute the range of prices over the time period
    price_range = max_high - min_low

    # Compute the Fibonacci time zones
    time_zones = [min_low + price_range * level for level in
                  [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.236, 1.382, 1.618]]

    return time_zones



def max_drawdown(stock_df):
    """
    Calculates the maximum drawdown of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data

    Returns:
    float: The maximum drawdown as a decimal value
    """
    # Calculate the cumulative maximum value of the stock prices
    cummax = stock_df['Close'].cummax()

    # Calculate the relative drawdown
    drawdown = (stock_df['Close'] - cummax) / cummax

    # Find the maximum drawdown and return it
    return abs(drawdown.min())


import pandas as pd


def relative_extremes(stock_df):
    """
    Calculates the relative maximums and minimums of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data

    Returns:
    pd.DataFrame: A pandas dataframe containing the relative maximums and minimums
    """
    # Calculate the rolling maximum and minimum values of the stock prices
    roll_max = stock_df['Close'].rolling(window=100, min_periods=1).max()
    roll_min = stock_df['Close'].rolling(window=100, min_periods=1).min()

    # Calculate the relative maximum and minimum values
    rel_max = stock_df['Close'] / roll_max - 1
    rel_min = stock_df['Close'] / roll_min - 1

    # Combine the relative maximum and minimum values into a single dataframe and return it
    return pd.concat([rel_max, rel_min], axis=1)


import pandas as pd
import numpy as np


def support_resistance_points(stock_df, lookback_window=100, threshold=0.02):
    """
    Computes all of the support and resistance points of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data
    lookback_window (int): The number of periods to look back when identifying support and resistance points
    threshold (float): The minimum percentage change in price required to identify a support or resistance point

    Returns:
    dict: A dictionary containing the support and resistance points
    """
    # Calculate the rolling maximum and minimum values of the stock prices
    roll_max = stock_df['Close'].rolling(window=lookback_window, min_periods=1).max()
    roll_min = stock_df['Close'].rolling(window=lookback_window, min_periods=1).min()

    # Calculate the percentage change in price relative to the rolling maximum and minimum
    rel_max = stock_df['Close'] / roll_max - 1
    rel_min = stock_df['Close'] / roll_min - 1

    # Identify support points as local minima where the percentage change is greater than the threshold
    support_points = stock_df.loc[rel_min >= threshold, 'Low']

    # Identify resistance points as local maxima where the percentage change is greater than the threshold
    resistance_points = stock_df.loc[rel_max >= threshold, 'High']

    # Return the support and resistance points as a dictionary
    return {'support': support_points, 'resistance': resistance_points}


import pandas as pd
import numpy as np
def calculate_pivot_points(df, period=1):
    """
    Calculates pivot points for a given time period based on daily data.
    :param df: A pandas DataFrame containing the stock data.
    :param period: The time period in days.
    :return: A pandas DataFrame containing the pivot points and support/resistance levels.
    """
    high_prices = df['High']
    low_prices = df['Low']
    close_prices = df['Close']

    # Calculate pivot points
    pivot_points = (high_prices + low_prices + close_prices) / 3

    # Calculate support and resistance levels
    resistance_1 = (2 * pivot_points) - low_prices.min()
    support_1 = (2 * pivot_points) - high_prices.max()
    resistance_2 = pivot_points + (high_prices.max() - low_prices.min())
    support_2 = pivot_points - (high_prices.max() - low_prices.min())
    resistance_3 = high_prices.max() + 2 * (pivot_points - low_prices.min())
    support_3 = low_prices.min() - 2 * (high_prices.max() - pivot_points)

    # Create a DataFrame with the pivot points and support/resistance levels
    pp = pd.DataFrame({
        'Pivot': pivot_points,
        'R1': resistance_1,
        'S1': support_1,
        'R2': resistance_2,
        'S2': support_2,
        'R3': resistance_3,
        'S3': support_3
    })

    # Return the pivot points DataFrame
    return pp


def pivot_points(stock_df):
    """
    Computes the pivot points of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data

    Returns:
    dict: A dictionary containing the pivot points
    """
    # Calculate the pivot point, support, and resistance levels
    pivot = (stock_df['High'] + stock_df['Low'] + stock_df['Close']) / 3
    support1 = 2 * pivot - stock_df['High']
    resistance1 = 2 * pivot - stock_df['Low']
    support2 = pivot - stock_df['High'] + stock_df['Low']
    resistance2 = pivot + stock_df['High'] - stock_df['Low']
    support3 = pivot - 2 * (stock_df['High'] - stock_df['Low'])
    resistance3 = pivot + 2 * (stock_df['High'] - stock_df['Low'])

    # Return the pivot points as a dictionary
    return {'pivot': pivot, 'support1': support1, 'resistance1': resistance1,
            'support2': support2, 'resistance2': resistance2, 'support3': support3,
            'resistance3': resistance3}
