import pandas as pd
from scipy.stats import stats




import pandas as pd


import pandas as pd
import numpy as np
from scipy.signal import hilbert, butter, filtfilt




'''
def advance_decline_line(stock_df):
    """
    Computes the Advance-Decline Line of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data

    Returns:
    pd.Series: A pandas series containing the Advance-Decline Line
    """
    # Compute the Advances and Declines
    advances = ((stock_df['Close'] - stock_df['Open']) > 0).cumsum()
    declines = ((stock_df['Close'] - stock_df['Open']) < 0).cumsum()

    # Compute the Advance-Decline Line
    advance_decline_line = advances - declines

    return advance_decline_line
'''

#-----------ELLIOT
import pandas as pd


def larger_trend(stock_df):
    """
    Identifies the larger trend of the stock

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data

    Returns:
    str: A string indicating the larger trend of the stock (Bullish or Bearish)
    """
    # Compute the long-term trend using the moving average
    long_term_ma = stock_df['Close'].rolling(window=200).mean()

    # Determine the direction of the long-term trend
    if long_term_ma.iloc[-1] > long_term_ma.iloc[0]:
        return "Bullish"
    else:
        return "Bearish"


import pandas as pd


def major_waves(stock_df):
    """
    Identifies the major waves of the stock

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data

    Returns:
    pd.DataFrame: A pandas dataframe containing the major waves of the stock
    """
    # Compute the difference between the high and low prices
    price_diff = stock_df['High'] - stock_df['Low']

    # Identify the highest price difference over the past 30 days
    max_diff = price_diff.rolling(window=30).max()

    # Identify the major waves
    waves = pd.cut(price_diff,
                   bins=[0, max_diff.quantile(0.25), max_diff.quantile(0.5), max_diff.quantile(0.75), max_diff.max()],
                   labels=['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4'])

    # Combine the major waves with the stock dataframe
    waves_df = pd.concat([stock_df, waves], axis=1)
    waves_df.rename(columns={0: 'Major Wave'}, inplace=True)

    return waves_df


import pandas as pd


def corrective_waves(stock_df):
    """
    Identifies the corrective waves of the stock

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data

    Returns:
    pd.DataFrame: A pandas dataframe containing the corrective waves of the stock
    """
    # Compute the difference between the high and low prices
    price_diff = stock_df['High'] - stock_df['Low']

    # Identify the highest price difference over the past 10 days
    max_diff = price_diff.rolling(window=10).max()

    # Identify the corrective waves
    waves = pd.cut(price_diff,
                   bins=[0, max_diff.quantile(0.25), max_diff.quantile(0.5), max_diff.quantile(0.75), max_diff.max()],
                   labels=['Wave A', 'Wave B', 'Wave C', 'Wave D'])

    # Combine the corrective waves with the stock dataframe
    waves_df = pd.concat([stock_df, waves], axis=1)
    waves_df.rename(columns={0: 'Corrective Wave'}, inplace=True)

    return waves_df


import pandas as pd


def minor_waves(stock_df):
    """
    Identifies the minor waves of the stock

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data

    Returns:
    pd.DataFrame: A pandas dataframe containing the minor waves of the stock
    """
    # Compute the difference between the high and low prices
    price_diff = stock_df['High'] - stock_df['Low']

    # Identify the highest price difference over the past 5 days
    max_diff = price_diff.rolling(window=5).max()

    # Identify the minor waves
    waves = pd.cut(price_diff,
                   bins=[0, max_diff.quantile(0.25), max_diff.quantile(0.5), max_diff.quantile(0.75), max_diff.max()],
                   labels=['Wave i', 'Wave ii', 'Wave iii', 'Wave iv'])

    # Combine the minor waves with the stock dataframe
    waves_df = pd.concat([stock_df, waves], axis=1)
    waves_df.rename(columns={0: 'Minor Wave'}, inplace=True)

    return waves_df


import pandas as pd


import pandas as pd

def elliot_wave_steps_3_4(major_wave_df, wave_dfs):
    # Step 3: Determine the corrective waves
    corrective_waves = []
    for i, wave_df in enumerate(wave_dfs):
        if len(wave_df) > 0:
            if major_wave_df.iloc[i]['type'] == 'Bullish':
                if wave_df.iloc[-1]['close'] < wave_df.iloc[-2]['close']:
                    corrective_waves.append(wave_df)
            elif major_wave_df.iloc[i]['type'] == 'Bearish':
                if wave_df.iloc[-1]['close'] > wave_df.iloc[-2]['close']:
                    corrective_waves.append(wave_df)

    # Step 4: Determine the price targets
    price_targets = []
    for i, wave_df in enumerate(corrective_waves):
        if len(wave_df) > 0:
            if major_wave_df.iloc[i]['type'] == 'Bullish':
                wave_high = wave_df['close'].max()
                wave_low = wave_df['close'].min()
                price_targets.append(wave_high + (wave_high - wave_low))
            elif major_wave_df.iloc[i]['type'] == 'Bearish':
                wave_high = wave_df['close'].max()
                wave_low = wave_df['close'].min()
                price_targets.append(wave_low - (wave_high - wave_low))

    return corrective_waves, price_targets
def elliot_wave_steps_5_6(price_targets, current_price):
    # Step 5: Buy or sell when price reaches the target
    buy_or_sell = []
    for target in price_targets:
        if current_price > target:
            buy_or_sell.append('Sell')
        elif current_price < target:
            buy_or_sell.append('Buy')
        else:
            buy_or_sell.append('Hold')

    # Step 6: Set stop loss and take profit levels
    stop_loss_levels = []
    take_profit_levels = []
    for target in price_targets:
        if current_price > target:
            stop_loss_levels.append(target + (target - current_price))
            take_profit_levels.append(current_price - (target - current_price))
        elif current_price < target:
            stop_loss_levels.append(target - (current_price - target))
            take_profit_levels.append(current_price + (current_price - target))
        else:
            stop_loss_levels.append(current_price - (target - current_price))
            take_profit_levels.append(current_price + (target - current_price))

    return buy_or_sell, stop_loss_levels, take_profit_levels
import pandas as pd
import numpy as np











import pandas as pd
import numpy as np
'''
def fischer_transform(df, n=10):
    """
    Calculates the Fischer Transformation for a given stock dataframe.
    Args:
        df (pd.DataFrame): The stock dataframe.
        n (int): The number of periods to use for calculation.
    Returns:
        pd.DataFrame: The Fischer Transform values.
    """
    df['max'] = df['High'].rolling(n).max()
    df['min'] = df['Low'].rolling(n).min()
    df['range'] = df['max'] - df['min']
    df['center'] = (df['max'] + df['min']) / 2
    df['x'] = 2 * ((df['Close'] - df['center']) / df['range'])
    df['fischer'] = np.log((np.exp(2 * df['x']) + 1) / (np.exp(2 * df['x']) - 1))
    return df['fischer']
'''

import numpy as np
import pandas as pd

def hurst_exponent(df, lags=None):
    """
    Calculates the Hurst Exponent for a given stock dataframe.
    Args:
        df (pd.DataFrame): The stock dataframe.
        lags (list): The list of lags to use for calculation.
    Returns:
        float: The Hurst Exponent value.
    """
    if not lags:
        lags = range(2, 100)
    ts = df['Close'].values
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
import pandas as pd




import pandas as pd
import numpy as np






def renko(df, brick_size):
    """
    Calculates and returns Renko chart data for a given dataframe and brick size

    Args:
    df: pandas DataFrame containing OHLC data
    brick_size: float representing the brick size in the Renko chart

    Returns:
    pandas DataFrame containing Renko chart data

    """
    renko_df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close'])
    current_price = df.iloc[0]['close']
    prev_price = current_price
    direction = 0  # 1 for up, -1 for down
    for i in range(len(df)):
        if abs(current_price - prev_price) >= brick_size:
            num_bricks = int(abs(current_price - prev_price) // brick_size)
            for j in range(num_bricks):
                if current_price > prev_price:
                    direction = 1
                elif current_price < prev_price:
                    direction = -1
                renko_df = renko_df.append(
                    {'date': df.iloc[i]['date'], 'open': prev_price, 'high': prev_price + direction * brick_size,
                     'low': prev_price - direction * brick_size, 'close': current_price}, ignore_index=True)
                prev_price += direction * brick_size
        current_price = df.iloc[i]['close']
    return renko_df


def std_dev_channel(df, n, num_std_dev=2):
    """
    Calculates and returns Standard Deviation Channel data for a given dataframe and window size

    Args:
    df: pandas DataFrame containing OHLC data
    n: int representing the window size
    num_std_dev: int representing the number of standard deviations to use for upper and lower bands. Default is 2.

    Returns:
    pandas DataFrame containing Standard Deviation Channel data

    """
    rolling_mean = df['close'].rolling(window=n).mean()
    rolling_std = df['close'].rolling(window=n).std()
    upper_band = rolling_mean + num_std_dev * rolling_std
    lower_band = rolling_mean - num_std_dev * rolling_std
    return pd.DataFrame(
        {'date': df['date'], 'rolling_mean': rolling_mean, 'upper_band': upper_band, 'lower_band': lower_band})


'''
def chandelier_exit(high, low, close, atr_period=22, multiplier=3.0):
    atr = ta.average_true_range(high, low, close, atr_period)
    long_exit = close - (atr * multiplier)
    short_exit = close + (atr * multiplier)
    return long_exit, short_exit
def bressert_dss(close, length=8, smoothing_length=8, double_smoothing_length=8):
    smoothed = ta.sma(close, length)
    double_smoothed = ta.sma(smoothed, length)
    stoch_k = (close - smoothed) / (double_smoothed - smoothed)
    stoch_d = ta.sma(stoch_k, smoothing_length)
    dss = ta.sma(stoch_d, double_smoothing_length)
    return dss
def chop_zone_oscillator(high, low, close, period=14, multiplier=1.5):
    atr = ta.average_true_range(high, low, close, period)
    tr = ta.true_range(high, low, close)
    cz = atr / (multiplier * np.sqrt(tr))
    return cz
'''

def standard_error(close, period=20):
    return close.rolling(period).std() / np.sqrt(period)


import pandas as pd
import numpy as np


def correlation_coefficient(df, x_col, y_col, period):
    """
    Computes the correlation coefficient between two columns in a dataframe over a given period.

    Parameters:
    df (pandas.DataFrame): Input dataframe.
    x_col (str): Column name for the first variable.
    y_col (str): Column name for the second variable.
    period (int): Period for which to calculate the correlation coefficient.

    Returns:
    pandas.DataFrame: Dataframe containing the correlation coefficient for each row over the given period.
    """
    rolling_corr = df[x_col].rolling(window=period).corr(df[y_col])
    return pd.DataFrame(rolling_corr, columns=['correlation_coefficient'])






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



'''
def pvt(df):
    close, volume = df['Close'], df['Volume']
    pvt = ((close - close.shift(1)) / close.shift(1)) * volume
    return pvt.cumsum()
'''




import numpy as np



import pandas as pd

'''
def pivot_points(stock_df):
    """
    Computes and returns pivot points, support and resistance levels
    based on the previous period's high, low and close prices of a stock.

    Parameters:
    stock_df (pandas.DataFrame): DataFrame containing the OHLC data of a stock

    Returns:
    pandas.DataFrame: DataFrame containing the pivot points, support and resistance levels
    """
    prev_close = stock_df['Close'].shift(1)
    prev_high = stock_df['High'].shift(1)
    prev_low = stock_df['Low'].shift(1)

    pivot_point = (prev_high + prev_low + prev_close) / 3
    resistance_1 = (2 * pivot_point) - prev_low
    resistance_2 = pivot_point + (prev_high - prev_low)
    resistance_3 = prev_high + 2 * (pivot_point - prev_low)
    support_1 = (2 * pivot_point) - prev_high
    support_2 = pivot_point - (prev_high - prev_low)
    support_3 = prev_low - 2 * (prev_high - pivot_point)

    pivot_df = pd.DataFrame({'Pivot': pivot_point, 'R1': resistance_1, 'R2': resistance_2,
                             'R3': resistance_3, 'S1': support_1, 'S2': support_2, 'S3': support_3})

    return pivot_df
'''


'''
def swing_index(high, low, close, prev_high, prev_low, prev_close):
    """
    Calculates the Swing Index for a given stock using the High, Low, and Close prices.

    Parameters:
    -----------
    high: pandas Series
        Series containing the High prices of the stock.
    low: pandas Series
        Series containing the Low prices of the stock.
    close: pandas Series
        Series containing the Close prices of the stock.
    prev_high: pandas Series
        Series containing the previous day's High prices of the stock.
    prev_low: pandas Series
        Series containing the previous day's Low prices of the stock.
    prev_close: pandas Series
        Series containing the previous day's Close prices of the stock.

    Returns:
    --------
    pandas Series
        Series containing the Swing Index values for each day in the input Series.
    """
    pivot = ((high + low + close) / 3).shift(1)
    prev_pivot = ((prev_high + prev_low + prev_close) / 3).shift(1)
    r1 = abs(high - prev_pivot)
    r2 = abs(low - prev_pivot)
    r3 = abs(high - low)
    r = pd.concat([r1, r2, r3], axis=1).max(axis=1)
    k = pd.Series(0.0, index=high.index)
    for i in range(1, len(high)):
        if high[i] >= prev_high[i]:
            bp = high[i]
        else:
            bp = prev_high[i]
        if low[i] <= prev_low[i]:
            sp = low[i]
        else:
            sp = prev_low[i]
        if close[i] > prev_close[i]:
            t = max(high[i] - prev_close[i], prev_close[i] - prev_low[i])
        elif close[i] < prev_close[i]:
            t = max(low[i] - prev_close[i], prev_close[i] - high[i])
        else:
            t = high[i] - low[i]
        if t == 0:
            rsi = 0
        else:
            rsi = 50 * ((close[i] - prev_close[i]) + 0.5 * (close[i] - prev_open[i]) + 0.25 * (
                        prev_close[i] - prev_open[i])) / t
        si = bp - sp
        si = si + (0.5 * si / r[i]) * abs(si / r[i]) + 0.25 * k[i - 1]
        if abs(si) < 1:
            si = 0
        k[i] = si * rsi + (1 - rsi) * k[i - 1]
    return k
'''
import pandas as pd


def alpha_beta_ratio(df, window):
    close = df['Close']
    returns = close.pct_change()
    alpha = returns.rolling(window).apply(lambda x: np.sum(x * (x.index - x.index.mean())) / np.sum((x.index - x.index.mean())**2))
    beta = returns.rolling(window).cov(returns.index.to_series().apply(lambda x: x.value)).apply(lambda x: x.iloc[0,1] / x.iloc[1,1])
    return alpha / beta


def atr_percent(df, n=14):
    tr = pd.DataFrame(index=df.index)
    tr['tr1'] = abs(df['High'] - df['Low'])
    tr['tr2'] = abs(df['High'] - df['Close'].shift())
    tr['tr3'] = abs(df['Low'] - df['Close'].shift())
    tr['tr'] = tr[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr = tr['tr'].rolling(n).sum()
    atr_percent = atr / df['Close'] * 100
    return atr_percent
def adaptive_ma(df, n=10, fast_w=2 / (2 + 1), slow_w=2 / (30 + 1)):
    l = len(df)
    ema1 = df['Close'].rolling(n).mean()
    ema2 = pd.Series(np.zeros(l))
    ema3 = pd.Series(np.zeros(l))
    for i in range(n, l):
        ema2[i] = ema1[i] * fast_w + ema2[i - 1] * (1 - fast_w)
        ema3[i] = ema1[i] * slow_w + ema3[i - 1] * (1 - slow_w)
    dm = abs(ema2 - ema3)
    c1 = fast_w / slow_w
    c2 = (fast_w / slow_w) ** 2
    am = pd.Series(np.zeros(l))
    for i in range(n, l):
        am[i] = (c1 - c2 + 1) * ema3[i] - c1 * ema2[i] + c2 * df['Close'][i]
    return am
def alpha_beta_ratio(df, n=14):
    x = np.log(df['Close'])
    x = pd.Series(x)
    y = x.diff(1).dropna()
    y = pd.Series(y)
    x = x.iloc[1:]
    x = pd.Series(x)
    reg = np.polyfit(x, y, deg=1)
    alpha = reg[0]
    beta = reg[1]
    ab_ratio = alpha / beta
    return ab_ratio
'''
def atrp(df, n=14):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=n)
    return atr / df['Close'] * 100
def atrp(df, n=14):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=n)
    return atr / df['Close'] * 100
def envelope_bands(df, n=20, m=0.02):
    sma = talib.SMA(df['Close'], timeperiod=n)
    upper_band = sma * (1 + m)
    lower_band = sma * (1 - m)
    return upper_band, lower_band
def atr_percent(df, n=14):
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=n)
    tr = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()))
    tr = np.maximum(tr, np.abs(df['Low'] - df['Close'].shift()))
    return atr / tr * 100


'''
def lri(df, n=14):
    x = np.arange(n)
    y = df['Close'][-n:].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return intercept
def arms_adi(df):
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    adi = clv * df['Volume']
    return adi.cumsum()





def polarized_fractal_efficiency(df, n=10):
    """Calculate the Polarized Fractal Efficiency for a given DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame containing stock data, including 'High' and 'Low' columns.
    n (int, optional): Number of periods to use for the oscillator calculation. Default is 10.

    Returns:
    pandas.Series: The Polarized Fractal Efficiency values.
    """
    hl_avg = (df['High'] + df['Low']) / 2
    change = hl_avg.diff()
    total_range = df['High'] - df['Low']
    up = (change > 0) & (total_range > 0)
    down = (change < 0) & (total_range > 0)
    up_fe = np.abs(change[up] / total_range[up])
    down_fe = np.abs(change[down] / total_range[down])
    up_fe[up_fe.isnull()] = 0
    down_fe[down_fe.isnull()] = 0
    polarized = (up_fe - down_fe).rolling(window=n, min_periods=1).mean()
    return polarized











def ergo(df, short_period=10, long_period=30, signal_period=20):
    df['change'] = df['close'] - df['close'].shift(1)
    df['volatility'] = df['change'].abs().rolling(window=short_period).sum()
    df['ema_volatility'] = df['volatility'].ewm(span=long_period, min_periods=long_period).mean()
    df['ergo'] = (df['change'] / df['ema_volatility']) * 100
    df['signal'] = df['ergo'].ewm(span=signal_period, min_periods=signal_period).mean()
    return df[['ergo', 'signal']]


def cfo(df, period=14):
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['roc'] = df['typical_price'].pct_change(period)
    df['forecast'] = df['typical_price'].shift(period) + (period * df['roc'])
    df['error'] = df['typical_price'] - df['forecast']
    df['deviation'] = df['error'].rolling(window=period).std()
    df['cfo'] = (df['error'] / df['deviation']) * 100
    return df['cfo']






def parabolic_sar_down(df, acceleration_factor=0.02, max_factor=0.2):
    # Initialize variables
    af = acceleration_factor
    max_af = max_factor
    ep = df['High'][0]
    sar = df['High'][0]
    trend = -1  # downtrend
    sar_list = [sar]

    for i in range(1, len(df)):
        # Update extreme price (EP) and acceleration factor (AF) if necessary
        if df['Low'][i] < ep:
            ep = df['Low'][i]
            af = min(af + acceleration_factor, max_factor)
        if trend == 1:
            if df['Low'][i] <= sar:
                # Switch to downtrend and reset variables
                trend = -1
                sar = ep
                ep = df['High'][i]
                af = acceleration_factor
        else:
            if df['High'][i] >= sar:
                # Start uptrend and reset variables
                trend = 1
                sar = ep
                ep = df['Low'][i]
                af = acceleration_factor
        if trend == 1:
            # Calculate SAR for uptrend
            sar = sar + af * (ep - sar)
            sar = min(sar, df['Low'][i - 1], df['Low'][i - 2])
        else:
            # Calculate SAR for downtrend
            sar = sar - af * (sar - ep)
            sar = max(sar, df['High'][i - 1], df['High'][i - 2])
        sar_list.append(sar)

    return sar_list















def elder_force_index(df, period_1=2, period_2=13):
    force_index = (df['Close'].diff(period_1) * df['Volume']) / 1000000000
    ema_force_index = force_index.rolling(period_2).mean()
    return ema_force_index

def chande_qstick(df, period=14):
    qstick = (df['Close'] - df['Open'].rolling(period).mean()).rolling(period).sum()
    return qstick



'''
def linear_regression_r2(df, window):
    x = np.arange(window)
    x_sum = window * (window - 1) / 2
    x_squared_sum = window * (window - 1) * (2 * window - 1) / 6
    y = df['Close'].rolling(window=window).mean()
    y_mean = y.mean()
    y_squared_sum = ((y - y_mean)**2).sum()
    xy_sum = ((df['Close'] * x).rolling(window=window).sum() - y_sum * x_sum) / window
    beta = xy_sum / ((x_squared_sum - x_sum**2 / window) / window)
    alpha = y_mean - beta * x_sum / window
    y_fit = alpha + beta * x
    ss_res = ((y - y_fit)**2).sum()
    ss_tot = y_squared_sum
    r2 = 1 - ss_res / ss_tot
    return r2
'''

