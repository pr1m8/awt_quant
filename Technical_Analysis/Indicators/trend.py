import pandas as pd

from movingAverages import *
def ta_trend_ParabolicSarUp(dataframe, acceleration=0.02, maximum=0.2):
    cols = list(dataframe.columns)
    dataframe = dataframe[list(dict.fromkeys(cols))].reset_index(drop=True)
    dataframe['High_Shift'] = dataframe['High'].shift(1)
    dataframe['Low_Shift'] = dataframe['Low'].shift(1)
    dataframe['SAR'] = dataframe['Low_Shift']
    dataframe['EP'] = dataframe['High']
    dataframe['ACC'] = acceleration
    dataframe['MAX_ACC'] = maximum
    dataframe['Trend'] = None
    dataframe.loc[0, 'Trend'] = 'Long'
    dataframe.loc[1, 'Trend'] = 'Long'
    for i in range(2, len(dataframe)):
        if dataframe.loc[i - 1, 'Trend'] == 'Long':
            if dataframe.loc[i, 'Low'] < dataframe.loc[i - 1, 'SAR']:
                dataframe.loc[i, 'Trend'] = 'Short'
                dataframe.loc[i, 'SAR'] = dataframe.loc[i - 1, 'EP']
                dataframe.loc[i, 'EP'] = dataframe.loc[i, 'Low']
                dataframe.loc[i, 'ACC'] = acceleration
            else:
                dataframe.loc[i, 'SAR'] = dataframe.loc[i - 1, 'SAR'] + \
                                         dataframe.loc[i - 1, 'ACC'] * \
                                         (dataframe.loc[i - 1, 'EP'] - dataframe.loc[i - 1, 'SAR'])
                dataframe.loc[i, 'EP'] = max(dataframe.loc[i, 'EP'], dataframe.loc[i, 'High'])
                dataframe.loc[i, 'ACC'] = min(dataframe.loc[i - 1, 'ACC'] + acceleration, maximum)
        else:
            if dataframe.loc[i, 'High'] > dataframe.loc[i - 1, 'SAR']:
                dataframe.loc[i, 'Trend'] = 'Long'
                dataframe.loc[i, 'SAR'] = dataframe.loc[i - 1, 'EP']
                dataframe.loc[i, 'EP'] = dataframe.loc[i, 'High']
                dataframe.loc[i, 'ACC'] = acceleration
            else:
                dataframe.loc[i, 'SAR'] = dataframe.loc[i - 1, 'SAR'] - \
                                         dataframe.loc[i - 1, 'ACC'] * \
                                         (dataframe.loc[i - 1, 'SAR'] - dataframe.loc[i - 1, 'EP'])
                dataframe.loc[i, 'EP'] = min(dataframe.loc[i, 'EP'], dataframe.loc[i, 'Low'])
                dataframe.loc[i, 'ACC'] = min(dataframe.loc[i - 1, 'ACC'] + acceleration, maximum)
    return dataframe['SAR']
def ta_trend_ParabolicSarDown(data, af=0.02, max_af=0.2):
    high = data['High']
    low = data['Low']
    sar = high.iloc[0]  # Initialize SAR with first high value
    ep = low.iloc[0]  # Initialize EP with first low value
    af_current = af  # Initialize acceleration factor

    # Initialize lists to store the values for each day
    sar_values = [sar]
    trend_direction = [-1]  # Downwards trend
    ep_values = [ep]

    for i in range(1, len(high)):
        prev_sar = sar
        prev_ep = ep

        # If current trend is upwards, switch to downwards trend
        if trend_direction[-1] == 1:
            sar = prev_ep
            trend_direction.append(-1)
            ep = low.iloc[i]
            af_current = af
        # If current trend is downwards, continue in same direction
        else:
            sar = prev_sar + af_current * (prev_ep - prev_sar)
            if sar > high.iloc[i-1]:
                sar = high.iloc[i-1]
            trend_direction.append(-1)
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af_current = min(af_current + af, max_af)

        sar_values.append(sar)
        ep_values.append(ep)

    return sar_values
def ta_trend_ParabolicSarDiff(data,af=0.02, max_af=0.2):
    up = ta_trend_ParabolicSarUp(data)
    down = ta_trend_ParabolicSarDown(data)
    return [u - d for u,d in zip(up,down)]

def ta_trend_MACD(data, fast_window=12, slow_window=26, signal_window=9):
    """
    The Moving Average Convergence Divergence (MACD) is a technical indicator that combines the use of moving averages and histograms to determine the trend and momentum of a stock's price.

The MACD is calculated as the difference between a fast exponential moving average (EMA) and a slow exponential moving average. The fast EMA is typically set to a 12-day moving average, while the slow EMA is set to a 26-day moving average. A 9-day EMA of the MACD, called the signal line, is then plotted along with the MACD line to help traders identify potential signals.
    :param data:
    :param fast_window:
    :param slow_window:
    :param signal_window:
    :return:
    """
    fast_ema = ta_ma_EMA(data, fast_window)
    slow_ema = ta_ma_EMA(data, slow_window)
    macd = [fast - slow for fast, slow in zip(fast_ema, slow_ema)]
    signal = ta_ma_EMA(macd, signal_window)
    histogram = [macd - sig for macd, sig in zip(macd, signal)]
    return macd, signal, histogram

def ta_trend_MACDDiff(data):
    macd,signal,hist =ta_trend_MACD(data)
    return [m-s for m,s in zip(macd,signal)]
#Change to trend

def ta_trend_VortexIndicator(df, n=14):
    """
    Computes the Vortex Indicator for a stock based on its dataframe.

    Parameters:
    df (pandas dataframe): Dataframe of the stock, containing 'High' and 'Low' columns.
    n (int): Length of the Vortex Indicator window.

    Returns:
    pandas series: Series of Vortex Indicator values for the stock.
    """
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff().abs()
    tr = high_diff.combine(low_diff, max)
    vi_pos = high_diff * low_diff
    vi_pos = vi_pos.rolling(window=n).sum()
    vi_neg = tr.rolling(window=n).sum()
    vi = vi_pos / vi_neg
    return vi,vi_pos,vi_neg

def ta_trend_IchimokuCloud(df, tenkan_period=9, kijun_period=26, senkou_b_period=52):
    """
    Computes the Ichimoku Cloud for a given DataFrame of OHLC data.

    Args:
        df (pandas.DataFrame): A DataFrame containing columns for "open", "high", "low", and "close".
        tenkan_period (int): The number of periods to use when computing the Tenkan-sen line. Default is 9.
        kijun_period (int): The number of periods to use when computing the Kijun-sen line. Default is 26.
        senkou_b_period (int): The number of periods to use when computing the Senkou B line. Default is 52.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - tenkan_sen
            - kijun_sen
            - senkou_a
            - senkou_b
            - chikou_span
    """
    # Compute the Tenkan-sen line
    high_prices = df["High"].rolling(tenkan_period).max()
    low_prices = df["Low"].rolling(tenkan_period).min()
    tenkan_sen = (high_prices + low_prices) / 2

    # Compute the Kijun-sen line
    high_prices = df["High"].rolling(kijun_period).max()
    low_prices = df["Low"].rolling(kijun_period).min()
    kijun_sen = (high_prices + low_prices) / 2

    # Compute the Senkou A line
    senkou_a = (tenkan_sen + kijun_sen) / 2

    # Compute the Senkou B line
    high_prices = df["High"].rolling(senkou_b_period).max()
    low_prices = df["Low"].rolling(senkou_b_period).min()
    senkou_b = (high_prices + low_prices) / 2

    # Compute the Chikou Span
    chikou_span = df["Close"].shift(-kijun_period)

    # Combine the results into a single DataFrame
    ichimoku_df = pd.DataFrame({
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b.shift(kijun_period),
        "chikou_span": chikou_span
    })

    #return ichimoku_df
    #try
    return tenkan_sen,kijun_sen,senkou_a,senkou_b.shift(kijun_period),chikou_span

#ADDD
def ta_trend_ElderRay(df, n_fast=13, n_slow=26, sma_window=10):
    """
    he Elder Ray indicator, also known as the Bull/Bear Power indicator, is a technical analysis tool used to measure the buying and selling pressure in a financial instrument. It was developed by Alexander Elder, a well-known technical analyst and trader.

The Elder Ray indicator consists of two lines, the "Bull Power" line and the "Bear Power" line. The Bull Power line represents the difference between the highest high of a given period and the current price, while the Bear Power line represents the difference between the current price and the lowest low of the same period. By subtracting the Bear Power line from the Bull Power line, traders can get an idea of the buying and selling pressure in a stock, commodity, or other financial instrument.

The Elder Ray indicator can be used in a variety of ways to help traders make informed trading decisions. For example, a high Bull Power reading and a low Bear Power reading could indicate that there is strong buying pressure in the market, while a low Bull Power reading and a high Bear Power reading could indicate that there is strong selling pressure. Traders may also use the indicator to confirm trend and momentum in a stock's price action, or to identify potential reversal points.
    :param df:
    :param n_fast:
    :param n_slow:
    :param sma_window:
    :return:
    """
    df = df.copy()
    bull_power = df['Close'] - df['Close'].rolling(n_fast).mean()
    bear_power = -df['Close'] + df['Close'].rolling(n_slow).mean()
    elder_ray = bull_power / bear_power
    elder_ray = elder_ray.rolling(sma_window).mean()
    return bull_power, bear_power, elder_ray


def ta_trend_ADX(df, n=14):
    '''
    Compute the Average Directional Index derived from Directional Movement Index (DMI) for a dataframe of stock prices.

    Parameters:
    df (pd.DataFrame) : dataframe of stock prices
    n (int) : number of periods for DMI calculation

    Returns:
    pd.DataFrame : dataframe of DMI values
    '''
    high = df['High']
    low = df['Low']
    close = df['Close']

    # Calculate the +DM and -DM values
    plus_DM = np.zeros(df.shape[0])
    minus_DM = np.zeros(df.shape[0])
    for i in range(1, df.shape[0]):
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_DM[i] = high[i] - high[i - 1]
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_DM[i] = low[i - 1] - low[i]

    # Calculate the +DI and -DI values
    plus_DI = pd.Series(plus_DM).rolling(window=n).mean()
    minus_DI = pd.Series(minus_DM).rolling(window=n).mean()

    # Calculate the DX value
    dx = 100 * np.abs((plus_DI - minus_DI) / (plus_DI + minus_DI))

    # Calculate the ADX value
    adx = dx.rolling(window=n).mean()

    # Combine results into a dataframe
    dmi = pd.concat([plus_DI, minus_DI, adx], axis=1)
    dmi.columns = ['+DI', '-DI', 'ADX']

    return dmi


def ta_trend_Zigzag(data,threshold=.05):
    """
    The ZigZag indicator is a trend-following indicator that uses swing highs and swing lows to identify significant price movements. It helps in filtering out market noise and finding trends by connecting significant price peaks and troughs.
    :param df:
    :param threshold:
    :return:
    """
    high = data['High']
    low = data['Low']
    zigzag = low.copy()
    changes = pd.Series(index=data.index)
    for i in range(1, len(data)):
        if high[i] >= high[i - 1] and low[i] >= low[i - 1]:
            zigzag[i] = zigzag[i - 1]
        elif high[i] <= high[i - 1] and low[i] <= low[i - 1]:
            zigzag[i] = zigzag[i - 1]
        else:
            if abs(high[i] - zigzag[i - 1]) >= abs(low[i] - zigzag[i - 1]):
                zigzag[i] = high[i]
            else:
                zigzag[i] = low[i]
        if abs(zigzag[i] - zigzag[i - 1]) / zigzag[i - 1] >= threshold:
            changes[i] = zigzag[i]
    return changes.fillna(method='ffill')

def ta_trend_KST(df, r1=10, r2=15, r3=20, r4=30, w1=10, w2=10, w3=10, w4=15):
    roc1 = df['Close'].pct_change(periods=r1)
    roc2 = df['Close'].pct_change(periods=r2)
    roc3 = df['Close'].pct_change(periods=r3)
    roc4 = df['Close'].pct_change(periods=r4)
    kst = w1 * roc1 + w2 * roc2 + w3 * roc3 + w4 * roc4
    kst_signal = kst.rolling(window=9).mean()
    kst_hist = kst - kst_signal
    return kst, kst_hist, kst_signal

def ta_trend_DPO(df, period=20, signal_period=9):
    dpo = df['Close'].shift(int(period/2)+1) - df['Close'].rolling(period).mean().shift(int(period/2)+1)
    signal = dpo.ewm(span=signal_period, adjust=False).mean()
    histogram = dpo - signal
    return dpo, histogram, signal

def ta_trend_Trix(df, n=24):
    ema1 = df['Close'].ewm(span=n, min_periods=n).mean()
    ema2 = ema1.ewm(span=n, min_periods=n).mean()
    ema3 = ema2.ewm(span=n, min_periods=n).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    return trix
def ta_trend_Aroon(df, n=20):
    """
    Computes the Aroon Indicator for a stock based on its dataframe.
    Parameters:
    df (pandas dataframe): Dataframe of the stock, containing 'High' and 'Low' columns.
    n (int): Length of the Aroon Indicator window.

    Returns:
    pandas dataframe: Dataframe containing Aroon Up and Aroon Down Indicator values for the stock.
    """
    df=df.copy()
    high_index = df['High'].rolling(window=n).apply(lambda x: np.argmax(x), raw=True)
    low_index = df['Low'].rolling(window=n).apply(lambda x: np.argmin(x), raw=True)
    aroon_up = 100 * (n - high_index) / n
    aroon_down = 100 * (n - low_index) / n
    return [u - d for u,d in zip(aroon_up,aroon_down)],aroon_up,aroon_down

def ta_trend_CCI(dataframe, n=20):
    dataframe=dataframe.copy()
    typical_price = (dataframe['High'] + dataframe['Low'] + dataframe['Close']) / 3
    moving_average = typical_price.rolling(window=n).mean()
    deviation = abs(typical_price - moving_average)
    mean_deviation = deviation.rolling(window=n).mean()
    cci = (typical_price - moving_average) / (0.015 * mean_deviation)
    dataframe['CCI'] = cci
    return dataframe['CCI']
def ta_trend_CTM(dataframe, window_size=14, sensitivity=1.5):
    """
    Calculates the Chande Trend Meter.

    Parameters:
        dataframe (Pandas DataFrame): DataFrame containing High, Low, and Close prices.
        window_size (int): Number of periods to use in calculation. Default is 14.
        sensitivity (float): Sensitivity factor. Default is 1.5.

    Returns:
        A Pandas DataFrame containing the Chande Trend Meter values.
    """
    high = dataframe['High']
    low = dataframe['Low']
    close = dataframe['Close']
    atr = pd.Series(abs(high - low), name='ATR')
    atr_smooth = atr.rolling(window=window_size).mean()
    tr = pd.DataFrame({'TR': abs(high - low)})
    up = pd.DataFrame({'UP': (high - high.shift(1))})
    dn = pd.DataFrame({'DN': (low.shift(1) - low)})
    pos_dm = pd.DataFrame({'PDM': (up.where((up > dn) & (up > 0), 0)).fillna(0)})
    neg_dm = pd.DataFrame({'NDM': (dn.where((dn > up) & (dn > 0), 0)).fillna(0)})
    pos_dm_smooth = pos_dm.rolling(window=window_size).mean()
    neg_dm_smooth = neg_dm.rolling(window=window_size).mean()
    chande_raw = pd.Series((pos_dm_smooth - neg_dm_smooth) / (pos_dm_smooth + neg_dm_smooth), name='CTM')
    chande_smooth = chande_raw.rolling(window=window_size).mean()
    sensitivity_factor = sensitivity * atr_smooth
    return pd.DataFrame({'CTM': chande_smooth, 'Upper': chande_smooth + sensitivity_factor,
                         'Lower': chande_smooth - sensitivity_factor})
from volatility import *

def ta_trend_RaffRegressionChannel(stock_df, n=10, width=2):
    """
    Computes and returns the upper and lower bounds of the Raff Regression Channel based on the
    least squares regression line of a stock's closing prices over a given period.

    Parameters:
    stock_df (pandas.DataFrame): DataFrame containing the OHLC data of a stock
    n (int): Number of periods to consider for the regression line
    width (int): Number of standard deviations to use for the channel width

    Returns:
    pandas.DataFrame: DataFrame containing the upper and lower bounds of the channel
    """
    x = np.arange(n)
    y = stock_df['Close'][-n:]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    std = np.std(y - (m * x + c))
    upper = (m * (n + 1)) + c + (width * std)
    lower = (m * (n + 1)) + c - (width * std)

    channel_df = pd.DataFrame({'Upper': upper, 'Lower': lower})

    return upper,lower



def ht_trendline(stock_df, period=10):
    """
    Computes the Hilbert Transform Instantaneous Trendline (HT Trendline) of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data
    period (int): The period of the Hilbert Transform

    Returns:
    pd.Series: A pandas series containing the HT Trendline
    """
    # Calculate the analytic signal using the Hilbert Transform
    analytic_signal = hilbert(stock_df['Close'])

    # Calculate the instantaneous phase of the analytic signal
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # Calculate the Hilbert Transform of the instantaneous phase
    hilbert_transform = hilbert(instantaneous_phase)

    # Smooth the Hilbert Transform with a low-pass filter
    b, a = butter(3, 0.025)
    smoothed_ht = filtfilt(b, a, np.abs(hilbert_transform))

    # Calculate the HT Trendline as the exponential moving average of the smoothed Hilbert Transform
    ht_trendline = pd.Series(smoothed_ht, index=stock_df.index).ewm(span=period, min_periods=period).mean()

    return ht_trendline


import pandas as pd
import numpy as np
from scipy.signal import hilbert, butter, filtfilt


def ht_dcperiod(stock_df, period=10):
    """
    Computes the Hilbert Transform Dominant Cycle Period (HT DCPERIOD) of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data
    period (int): The period of the Hilbert Transform

    Returns:
    pd.Series: A pandas series containing the HT DCPERIOD
    """
    # Calculate the analytic signal using the Hilbert Transform
    analytic_signal = hilbert(stock_df['Close'])

    # Calculate the instantaneous phase of the analytic signal
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # Calculate the Hilbert Transform of the instantaneous phase
    hilbert_transform = hilbert(instantaneous_phase)

    # Smooth the Hilbert Transform with a low-pass filter
    b, a = butter(3, 0.025)
    smoothed_ht = filtfilt(b, a, np.abs(hilbert_transform))

    # Calculate the instantaneous frequency of the Hilbert Transform
    instantaneous_frequency = np.diff(np.unwrap(np.angle(hilbert_transform)))
    instantaneous_frequency = np.insert(instantaneous_frequency, 0, instantaneous_frequency[0])
    instantaneous_frequency = pd.Series(instantaneous_frequency, index=stock_df.index)

    # Calculate the HT DCPERIOD as the exponential moving average of the period of the instantaneous frequency
    ht_dcperiod = pd.Series(period / instantaneous_frequency, index=stock_df.index).ewm(span=period,
                                                                                        min_periods=period).mean()

    return ht_dcperiod


import pandas as pd
import numpy as np
from scipy.signal import hilbert, butter, filtfilt


def ta_trend_HTDominantCyclePhase(stock_df, period=10):
    """
    Computes the Hilbert Transform Dominant Cycle Phase (HT DCPHASE) of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data
    period (int): The period of the Hilbert Transform

    Returns:
    pd.Series: A pandas series containing the HT DCPHASE
    """
    # Calculate the analytic signal using the Hilbert Transform
    analytic_signal = hilbert(stock_df['Close'])

    # Calculate the instantaneous phase of the analytic signal
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # Calculate the Hilbert Transform of the instantaneous phase
    hilbert_transform = hilbert(instantaneous_phase)

    # Smooth the Hilbert Transform with a low-pass filter
    b, a = butter(3, 0.025)
    smoothed_ht = filtfilt(b, a, np.abs(hilbert_transform))

    # Calculate the instantaneous frequency of the Hilbert Transform
    instantaneous_frequency = np.diff(np.unwrap(np.angle(hilbert_transform)))
    instantaneous_frequency = np.insert(instantaneous_frequency, 0, instantaneous_frequency[0])
    instantaneous_frequency = pd.Series(instantaneous_frequency, index=stock_df.index)

    # Calculate the HT DCPHASE as the instantaneous phase minus the phase of the exponential moving average of the smoothed Hilbert Transform
    ht_dcphase = instantaneous_phase - np.unwrap(np.angle(smoothed_ht.ewm(span=period, min_periods=period).mean()))

    return ht_dcphase


def ta_trend_HTSinewave(stock_df, period=10):
    """
    Computes the Hilbert Transform Sine Wave (HT SINEWAVE) of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data
    period (int): The period of the Hilbert Transform

    Returns:
    pd.Series: A pandas series containing the HT SINEWAVE
    """
    # Calculate the analytic signal using the Hilbert Transform
    analytic_signal = hilbert(stock_df['Close'])

    # Calculate the instantaneous phase of the analytic signal
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # Calculate the Hilbert Transform of the instantaneous phase
    hilbert_transform = hilbert(instantaneous_phase)

    # Smooth the Hilbert Transform with a low-pass filter
    b, a = butter(3, 0.025)
    smoothed_ht = filtfilt(b, a, np.abs(hilbert_transform))

    # Calculate the HT SINEWAVE as the real part of the analytic signal times the cosine of the instantaneous phase
    ht_sinewave = pd.Series(np.real(analytic_signal) * np.cos(instantaneous_phase), index=stock_df.index)

    return ht_sinewave


import pandas as pd
import numpy as np
from scipy.signal import hilbert, butter, filtfilt


def ta_trend_HTPhasor(stock_df, period=10):
    """
    Computes the Hilbert Transform Phasor Components (HT PHASOR) of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data
    period (int): The period of the Hilbert Transform

    Returns:
    pd.DataFrame: A pandas dataframe containing the HT PHASOR
    """
    # Calculate the analytic signal using the Hilbert Transform
    analytic_signal = hilbert(stock_df['Close'])

    # Calculate the instantaneous phase of the analytic signal
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # Calculate the Hilbert Transform of the instantaneous phase
    hilbert_transform = hilbert(instantaneous_phase)

    # Smooth the Hilbert Transform with a low-pass filter
    b, a = butter(3, 0.025)
    smoothed_ht = filtfilt(b, a, np.abs(hilbert_transform))

    # Calculate the instantaneous frequency of the Hilbert Transform
    instantaneous_frequency = np.diff(np.unwrap(np.angle(hilbert_transform)))
    instantaneous_frequency = np.insert(instantaneous_frequency, 0, instantaneous_frequency[0])
    instantaneous_frequency = pd.Series(instantaneous_frequency, index=stock_df.index)

    # Calculate the HT PHASOR components
    ht_phasor_real = np.real(smoothed_ht) * np.cos(instantaneous_phase)
    ht_phasor_imag = np.real(smoothed_ht) * np.sin(instantaneous_phase)

    # Create a pandas dataframe to store the HT PHASOR components
    ht_phasor_df = pd.DataFrame(
        {'Real': ht_phasor_real, 'Imaginary': ht_phasor_imag, 'Frequency': instantaneous_frequency},
        index=stock_df.index)

    return ht_phasor_real,ht_phasor_imag,instantaneous_frequency

def ta_trend_SwingIndex(df, high='High', low='Low', close='Close'):
    # Compute the various price differences
    hl_diff = abs(df[high] - df[low])
    hc_diff = abs(df[high] - df[close].shift())
    lc_diff = abs(df[low] - df[close].shift())

    # Compute the True Range
    tr = pd.concat([hl_diff, hc_diff, lc_diff], axis=1).max(axis=1)

    # Compute the Swing Index
    si = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        r = 0.0
        if df[close][i] > df[close][i - 1]:
            r = tr[i] - 0.5 * hc_diff[i - 1] - 0.25 * lc_diff[i - 1]
        elif df[close][i] < df[close][i - 1]:
            r = tr[i] - 0.5 * lc_diff[i - 1] - 0.25 * hc_diff[i - 1]
        else:
            r = tr[i]
        si[i] = si[i - 1] + (r * np.sign(si[i - 1] - si[i - 2]))

    return si
def ta_trend_EhlersFisherTransform(df, length=10, factor=0.33):
    close = df['Close']

    alpha = factor * 2.0 / (length + 1.0)
    # Calculate ema of prices
    ema = close.ewm(alpha=alpha, adjust=False).mean()

    # Calculate the delta
    delta = close.diff(1)

    # Calculate the sum of absolute deltas
    abs_delta = delta.abs().rolling(window=length).sum()

    # Calculate the max price over the last length periods
    max_price = close.rolling(window=length).max()

    # Calculate the min price over the last length periods
    min_price = close.rolling(window=length).min()

    # Calculate the range between the max and min price
    range_price = max_price - min_price

    # Calculate the normalized delta
    norm_delta = (delta / abs_delta) * range_price

    # Calculate the final value
    fisher = norm_delta.rolling(window=length).sum()

    return fisher


def ta_trend_AdaptiveLaguerreFilter(df, gamma=0.8):
    def ALF(price, gamma):
        alpha = 1 - gamma
        gamma_2 = gamma ** 2
        gamma_3 = gamma ** 3
        gamma_4 = gamma ** 4
        y1 = np.zeros(len(price))
        y2 = np.zeros(len(price))
        y3 = np.zeros(len(price))
        y4 = np.zeros(len(price))
        for i in range(1, len(price)):
            y1[i] = alpha * price[i] + (1 - alpha) * y1[i - 1]
            y2[i] = -gamma_2 * y1[i] + 2 * gamma * y2[i - 1] - (1 - gamma_2) * y2[i - 2]
            y3[i] = -gamma_3 * y2[i] + 3 * gamma_2 * y3[i - 1] - 3 * gamma * (1 - gamma_2) * y3[i - 2] + gamma_3 * y3[
                i - 3]
            y4[i] = -gamma_4 * y3[i] + 4 * gamma_3 * y4[i - 1] - 6 * gamma_2 * (1 - gamma_2) * y4[i - 2] + 4 * gamma * (
                        1 - gamma_3) * y4[i - 3] - (1 - gamma_4) * y4[i - 4]
        return (y1 + 2 * y2 + 2 * y3 + y4) / 6

    df['ALF'] = ALF(df['Close'].values, gamma)
    return df['ALF']

def ta_trend_AADI(df):
    c = df['Close']
    h = df['High']
    l = df['Low']
    vol = df['Volume']
    cm = ((c - l) - (h - c)) / (h - l)
    ad = cm * vol
    aadi = ad.cumsum()
    return aadi

# FIX
def ta_trend_ADXR(stock_df, period=14):
    high = stock_df['High']
    low = stock_df['Low']
    close = stock_df['Close']

    # Compute the true range (TR) for the given period
    tr_list = []
    for i in range(1, len(stock_df)):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        tr_list.append(tr)
    tr_s = pd.Series(tr_list)

    # Compute the directional movement (+DM, -DM) for the given period
    plus_dm_list = []
    minus_dm_list = []
    for i in range(1, len(stock_df)):
        plus_dm = max(high[i] - high[i - 1], 0)
        minus_dm = max(low[i - 1] - low[i], 0)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
    plus_dm_s = pd.Series(plus_dm_list)
    minus_dm_s = pd.Series(minus_dm_list)

    # Compute the smoothed true range (ATR) and the directional movement indicators (+DI, -DI)
    atr = tr_s.rolling(window=period).mean()
    plus_di = 100 * (plus_dm_s.rolling(window=period).sum() / atr)
    minus_di = 100 * (minus_dm_s.rolling(window=period).sum() / atr)

    # Compute the directional movement index (DX) and the ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()

    # Compute the ADXR
    adxr = (adx.shift(period) + adx) / 2

    return pd.DataFrame({'DI+': plus_di, 'DI-': minus_di, 'ADX': adx, 'ADXR': adxr})
def ta_trend_CE(df, period=22, atr_multiplier=3):
    atr =ta_vol_ATR(df, period)
    long_stop = df["High"] - atr_multiplier * atr
    short_stop = df["Low"] + atr_multiplier * atr
    chandelier_exit = pd.concat([long_stop, short_stop], axis=1).max(axis=1)
    chandelier_exit.fillna(method="ffill", inplace=True)
    return chandelier_exit
def ta_trend_LaguerreFilter(df, gamma=0.7):
    lag1 = (1 - gamma) * df['Close'] + gamma * df['Close'].shift(1)
    lag2 = (1 - gamma) * lag1 + gamma * lag1.shift(1)
    lag3 = (1 - gamma) * lag2 + gamma * lag2.shift(1)
    lag4 = (1 - gamma) * lag3 + gamma * lag3.shift(1)
    alf = (4 * lag1 - 6 * lag2 + 4 * lag3 - lag4) / 6
    return alf

def ta_trend_tsf(df, period=14):
    tsf = pd.Series(0.0, index=df.index)
    for i in range(period, len(df.index)):
        coef = np.polyfit(range(period), df['Close'][i-period:i], 1)
        tsf[i] = coef[-1] + coef[-2] * period
    return tsf
def ta_trend_LinearRegressionIndicator(df, period=14):
    lr = pd.Series(index=df.index)
    for i in range(period, len(df)):
        lr[i] = np.polyfit(np.arange(period), df['Close'][i-period+1:i+1], 1)[0]
    return lr

def ta_trend_STC(df, fast=23, slow=50, smooth=10):
    high = df['High']
    low = df['Low']
    close = df['Close']
    stoch1 = ((close - low.rolling(window=fast).min()) / (high.rolling(window=fast).max() - low.rolling(window=fast).min())) * 100
    stoch2 = stoch1.rolling(window=fast).mean()
    stoch3 = stoch2.rolling(window=fast).mean()
    stoch4 = ((close - low.rolling(window=slow).min()) / (high.rolling(window=slow).max() - low.rolling(window=slow).min())) * 100
    stoch5 = stoch4.rolling(window=slow).mean()
    stoch6 = stoch5.rolling(window=slow).mean()
    stch = stoch3 * (1 - (smooth / 2)) + stoch6 * (smooth / 2)
    return stch

def ta_trend_PriceChannel(df, n=20):
    """Computes the upper and lower price channels for a given period using the highest high and lowest low of the period."""
    high = df['High'].rolling(n).max()
    low = df['Low'].rolling(n).min()
    middle = (high + low) / 2
    upper_channel = middle + 2 * (middle - low)
    lower_channel = middle - 2 * (high - middle)
    return middle,upper_channel,lower_channel


def add_Trends(df):
    df['ta_trend_ParabolicSar_Upwards'] = ta_trend_ParabolicSarUp(df)
    df['ta_trend_ParabolicSar_Downwards'] = ta_trend_ParabolicSarDown(df)
    df['ta_trend_ParabolicSar_Diff']=ta_trend_ParabolicSarDiff(data=df)

    df['ta_trend_MACD'],df['ta_trend_MACD_signal'],df['ta_trend_MACD_histogram']=ta_trend_MACD(data=df)
    df['ta_trend_MACD_diff'] = ta_trend_MACDDiff(df)

    df['ta_trend_Vortex'],df['ta_trend_Vortex_Positive'],df['ta_trend_Vortex_Negative']=ta_trend_VortexIndicator(df)


    df['ta_trend_IchimokuCloud_tenkanSen'], df['ta_trend_IchimokuCloud_kijunSen'], \
    df['ta_trend_IchimokuCloud_senkouA'],df['ta_trend_IchimokuCloud_senkouB'],\
    df['ta_trend_IchimokuCloud_chikouSpan']= ta_trend_IchimokuCloud(df)


    df['ta_trend_ElderRay_Bull'], df['ta_trend_ElderRay_Bear'], \
    df['ta_trend_ElderRay']=ta_trend_ElderRay(df)

    df['ta_trend_ADX_Plus'],df['ta_trend_ADX_Minus'],\
    df['ta_trend_ADX'] = ta_trend_ADX(df)

    df['ta_trend_KST'], df['ta_trend_KST_hist'], df['ta_trend_KST_signal'] = ta_trend_KST(df)

    df['ta_trend_DPO'], df['ta_trend_DPO_hist'], df['ta_trend_DPO_signal'] = ta_trend_DPO(df)

    df['ta_trend_Trix']=ta_trend_Trix(df)
    df['ta_trend_CCI'] = ta_trend_CCI(df)

    #df['ta_trend_CTM'],df['ta_trend_CTM_Upper'],df['ta_trend_CTM_Lower'] = ta_trend_CTM(df)

    df['ta_trend_STC']=ta_trend_STC(df)
    df = pd.concat([df,ta_trend_CE(df)],axis=1)
    df['ta_trend_PriceChannel_mid'],df['ta_trend_PriceChannel_upper'],df['ta_trend_PriceChannel_lower']=ta_trend_PriceChannel(df)
    df['ta_trend_Zigzag']=ta_trend_Zigzag(df)
    df['ta_trend_Aroon'],df['ta_trend_Aroon_Up'],df['ta_trend_Aroon_Down']=ta_trend_Aroon(df)
    return df


import yfinance
'''
df = yf.Ticker('AAPL').history(period='MAX')
print(df.columns)
df= add_trends(df)
import plotly.express  as px
fig=px.line(df)
fig.show()
'''
