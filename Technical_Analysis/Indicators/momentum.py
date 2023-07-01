import numpy as np
import pandas as pd
from movingAverages import *
from volatility import *
from volume import *
def ta_mo_RVI(df, n=10):
    h = (df['High'] + df['Close'].shift(1)) / 2
    l = (df['Low'] + df['Close'].shift(1)) / 2
    c = (2 * df['Close'] + df['High'] + df['Low']) / 4
    rvi = ta_ma_SMA((h - l) / (c - l), n)
    rvi_signal = ta_ma_SMA(rvi, n)
    return rvi, rvi_signal


def ta_mo_WildersMovingAverage(df, column='Close', period=14):
    weighted_prices = df[column].rolling(period).apply(lambda x: np.sum(x * (1 - np.arange(period) / period)),
                                                       raw=True)
    divisor = np.cumsum(np.ones(period))[-1]
    return weighted_prices / divisor



def ta_mo_RSI(df, n=14):
    prices=df['Close']
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)
    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n
        rs = up / down
        rsi[i] = 100. - 100./(1. + rs)
    return rsi

def ta_mo_TSI(df, r=25, s=13):
    prices = df['Close']
    pc = np.diff(prices)
    ema1 = ta_ma_EMA(pc, r)
    ema2 = ta_ma_EMA(ema1, s)
    ema3 = ta_ma_EMA(np.abs(pc), r)
    ema4 = ta_ma_EMA(ema3, s)
    tsi = [100 * e2/e4 for e2,e4 in zip(ema2,ema4)]
    #tsi = 100. * ema2 / ema4
    return [np.NaN]+tsi



def ta_HurstSpectralOscillator(data, lag=14):
    price = data['Close']
    log_returns = np.log(price / price.shift(1)).dropna()
    tau = [np.sqrt(np.mean(np.power(log_returns[i:] - log_returns[:-i], 2))) for i in range(1, lag + 1)]
    poly = np.polyfit(np.log(range(1, lag + 1)), np.log(tau), 1)
    h = poly[0] * 2
    return h
def ta_mo_UltimateOscillator(data, period1=7, period2=14, period3=28):
    # stock_data = yf.Ticker(symbol).history(period="max")
    stock_data = data.copy()
    stock_data['bp'] = stock_data['Close'] - pd.concat([stock_data['Low'], stock_data['Close'].shift(1)], axis=1).min(
        axis=1)
    stock_data['tr'] = pd.concat([stock_data['High'], stock_data['Close'].shift(1)], axis=1).max(axis=1) - pd.concat(
        [stock_data['Low'], stock_data['Close'].shift(1)], axis=1).min(axis=1)
    stock_data['average_bp1'] = stock_data['bp'].rolling(window=period1).mean()
    stock_data['average_tr1'] = stock_data['tr'].rolling(window=period1).mean()
    stock_data['average_bp2'] = stock_data['bp'].rolling(window=period2).mean()
    stock_data['average_tr2'] = stock_data['tr'].rolling(window=period2).mean()
    stock_data['average_bp3'] = stock_data['bp'].rolling(window=period3).mean()
    stock_data['average_tr3'] = stock_data['tr'].rolling(window=period3).mean()
    stock_data['ultimate_oscillator'] = 100 * ((4 * stock_data['average_bp1'] / stock_data['average_tr1']) + (
                2 * stock_data['average_bp2'] / stock_data['average_tr2']) + (
                                                           stock_data['average_bp3'] / stock_data['average_tr3'])) / (
                                                    4 + 2 + 1)
    return stock_data['ultimate_oscillator']
def ta_mo_CMO(df, period=14):
    """
    The Chande Momentum Oscillator, on the other hand, is a momentum oscillator that measures the difference between the sum of the up closes and the sum of the down closes over a specified period. The CMO oscillates between +100 and -100, with readings above +50 indicating bullish momentum and readings below -50 indicating bearish momentum. Like the SMI, the CMO can be used to identify overbought and oversold conditions, as well as to signal possible trend reversals
    :param df:
    :param period:
    :return:
    """
    # Calculate the difference between the close price and previous close price
    delta = df['Close'].diff()

    # Calculate the up and down changes
    up_change = delta.where(delta > 0, 0)
    down_change = abs(delta.where(delta < 0, 0))

    # Calculate the sum of up and down changes over the specified period
    up_sum = up_change.rolling(window=period).sum()
    down_sum = down_change.rolling(window=period).sum()

    # Calculate the CMO value
    CMO = 100 * (up_sum - down_sum) / (up_sum + down_sum)

    return CMO

def ta_mo_CoppockCurve(df):
    """
    The Coppock curve is a technical analysis tool used in finance to determine long-term momentum in the stock market. It was developed by E.S. Coppock in the 1960s and measures the rate of change in a weighted moving average of the sum of the 11-month rate of change in the spot price of a stock, plus the 14-month rate of change in the spot price. The Coppock curve is often used by investors and traders to determine long-term buying or selling opportunities in the stock market, and is considered a long-term momentum indicator.
    :param df:
    :return:
    """
    if 'Adj Close' in df.columns:
        c = 'Adj Close'
    else:
        c = 'Close'
    # Calculate the rate of change (ROC) over a 11-period and 14-period
    roc11 = df[c].pct_change(periods=11)
    roc14 = df[c].pct_change(periods=14)

    # Calculate the weighted sum of ROC11 and ROC14
    coppock = (roc11 + roc14) * (11 * 14) / 2

    # Add the weighted sum to a 10-period weighted moving average of the sum
    coppock = coppock.rolling(window=10).mean()

    # Return the result
    return coppock

def ta_mo_PPO(df, n1=12, n2=26, n3=9):
    ema_short = df['Close'].ewm(span=n1, min_periods=n1).mean()
    ema_long = df['Close'].ewm(span=n2, min_periods=n2).mean()
    ppo = 100 * (ema_short - ema_long) / ema_long
    ppo_signal = ppo.ewm(span=n3, min_periods=n3).mean()
    ppo_hist = ppo - ppo_signal
    return ppo, ppo_hist, ppo_signal




def ta_mo_StochOscillator(df, n=14, k=3, d=3):
    low_min = df['Low'].rolling(window=n).min()
    high_max = df['High'].rolling(window=n).max()
    stoch_k = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    stoch_d = stoch_k.rolling(window=k).mean()
    stoch_ds = stoch_d.rolling(window=d).mean()
    stoch_sig = stoch_ds.rolling(window=3).mean()
    return stoch_k, stoch_d, stoch_ds,stoch_sig

def ta_mo_APO(df, fast_period=10, slow_period=30, signal_period=9):
    fast_ema = df['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['Close'].ewm(span=slow_period, adjust=False).mean()
    apo = fast_ema - slow_ema
    signal = apo.ewm(span=signal_period, adjust=False).mean()
    histogram = apo - signal
    return apo, histogram, signal


def ta_mo_KRI(dataframe, n=10):
    """
    Kairi index
    :param dataframe:
    :param n:
    :return:
    """
    dataframe=dataframe.copy()
    close = dataframe['Close']
    ma = close.rolling(n).mean()
    kairi = (close - ma) / ma
    dataframe['Kairi'] = kairi
    return dataframe['Kairi']

def ta_mo_ConnorsRSI(dataframe, n1=3, n2=2, n3=100):
    dataframe=dataframe.copy()
    close = dataframe['Close']
    delta = close.diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    sma1 = close.rolling(n1).mean()
    sma2 = close.rolling(n2).mean()
    rsi1 = sma1.rolling(n3).apply(lambda x: np.mean(up[-n3:]) / np.mean(down[-n3:]))
    rsi2 = sma2.rolling(n3).apply(lambda x: np.mean(up[-n3:]) / np.mean(down[-n3:]))
    rsi3 = (rsi1 + rsi2) / 2
    c_rsi = (rsi3 - rsi3.rolling(n1).min()) / (rsi3.rolling(n1).max() - rsi3.rolling(n1).min())
    dataframe['Connors_RSI'] = c_rsi
    return dataframe['Connors_RSI']



def ta_mo_PMO(df, short=10, long=35, signal=20):
    ema_short = df['Close'].ewm(span=short, adjust=False).mean()
    ema_long = df['Close'].ewm(span=long, adjust=False).mean()
    roc = ((ema_short - ema_long) / ema_long) * 100
    pmo = roc.ewm(span=signal, adjust=False).mean()
    pmo_hist = roc - pmo
    pmo_signal = pmo.ewm(span=signal, adjust=False).mean()
    return pmo, pmo_hist, pmo_signal

def ta_mo_CMO(df, period=14):
    """
    The Chande Momentum Oscillator, on the other hand, is a momentum oscillator that measures the difference between the sum of the up closes and the sum of the down closes over a specified period. The CMO oscillates between +100 and -100, with readings above +50 indicating bullish momentum and readings below -50 indicating bearish momentum. Like the SMI, the CMO can be used to identify overbought and oversold conditions, as well as to signal possible trend reversals
    :param df:
    :param period:
    :return:
    """
    # Calculate the difference between the close price and previous close price
    delta = df['Close'].diff()

    # Calculate the up and down changes
    up_change = delta.where(delta > 0, 0)
    down_change = abs(delta.where(delta < 0, 0))

    # Calculate the sum of up and down changes over the specified period
    up_sum = up_change.rolling(window=period).sum()
    down_sum = down_change.rolling(window=period).sum()

    # Calculate the CMO value
    CMO = 100 * (up_sum - down_sum) / (up_sum + down_sum)

    return CMO

def ta_mo_SpecialK(dataframe, short_window=10, long_window=30, smoothing_period=10):
    """
    Calculates Price's Special K.

    Parameters:
        dataframe (Pandas DataFrame): DataFrame containing High, Low, and Close prices.
        short_window (int): Number of periods to use in short-term moving average. Default is 10.
        long_window (int): Number of periods to use in long-term moving average. Default is 30.
        smoothing_period (int): Number of periods to use in smoothing. Default is 10.

    Returns:
        A Pandas DataFrame containing the Special K values.
    """
    high = dataframe['High']
    low = dataframe['Low']
    close = dataframe['Close']

    # Calculate the fast, slow, and smoothed versions of the stochastic oscillator
    fast_k = (close - low.rolling(window=short_window).min()) / (
                high.rolling(window=short_window).max() - low.rolling(window=short_window).min())
    slow_k = fast_k.rolling(window=long_window).mean()
    slow_d = slow_k.rolling(window=smoothing_period).mean()

    # Calculate the fast and slow versions of the stochastic oscillator signal line
    fast_d = fast_k.rolling(window=smoothing_period).mean()
    slow_sd = slow_d.rolling(window=smoothing_period).mean()

    # Calculate the special K value
    special_k = slow_sd + (slow_sd - fast_d)

    # Return a DataFrame containing the special K values
    return special_k

def ta_mo_WilliamsR(data, period=14):
    high = data['High']
    low=data['Low']
    close=data['Close']
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    wr = -100 * (hh - close) / (hh - ll)
    return wr

def ta_mo_CMB(df,roclen=14,momlen=14,smooth=10):
    """
    Calculate the CMB (Coppock's Modified Breadth) Composite Index for a given set of data.

    Parameters:
    - df: a pandas DataFrame containing the data for the stock or market index, with columns for the date, close price, volume, and other indicators as needed.
    - roclen: an integer representing the rate of change (ROC) period for the Advance-Decline Ratio calculation. Default is 14.
    - momlen: an integer representing the momentum period for the Upside-Downside Volume Ratio calculation. Default is 11.
    - smooth: an integer representing the smoothing period for the final CMB Composite Index calculation. Default is 10.

    Returns:
    - A pandas DataFrame with three columns: "Date", "CMB", and "Signal".
    """

    # Calculate the Advance-Decline Ratio
    adv = (df['Close'].rolling(roclen).apply(lambda x: sum(x > 0)) / roclen).fillna(0)
    dec = (df['Close'].rolling(roclen).apply(lambda x: sum(x < 0)) / roclen).fillna(0)
    adr = adv / (adv + dec)

    # Calculate the Upside-Downside Volume Ratio
    uv = ((df['Close'] - df['Close'].shift(1)).fillna(0) * df['Volume']).rolling(momlen).apply(
        lambda x: sum(x[x > 0])) / ((df['Close'] - df['Close'].shift(1)).fillna(0) * df['Volume']).rolling(
        momlen).apply(lambda x: sum(abs(x)))

    # Calculate the CMB Composite Index
    cmb = adr.ewm(span=smooth).mean() + uv.ewm(span=smooth).mean()
    signal = cmb.rolling(9).mean()
    empty = [np.NaN]*9
    print(len(cmb))
    print(len(signal))
    #signal=empty+list(signal)
    print(signal)
    print(len(signal))
    # Return the result as a pandas DataFrame
    #result = pd.DataFrame({'CMB': cmb, 'Signal': signal})
    return cmb,signal


# Change name
def ta_mo_TTMSqueeze(df, period=20, ma_period=20, bb_multiplier=2, kc_multiplier=1.5, kc_period=20):
    """
    Computes the TTM Squeeze indicator given a pandas dataframe

    Args:
    df (pandas.DataFrame): The dataframe to compute the TTM Squeeze for
    period (int): The period used for the Bollinger Bands and Keltner Channels
    ma_period (int): The period used for the moving average of the Bollinger Bands
    bb_multiplier (float): The multiplier used for the Bollinger Bands
    kc_multiplier (float): The multiplier used for the Keltner Channels
    kc_period (int): The period used for the Keltner Channels

    Returns:
    pandas.DataFrame: A dataframe containing the TTM Squeeze values
    """
    # Compute the Bollinger Bands
    df = df.copy()
    df['bb_middle'] = df['Close'].rolling(window=ma_period).mean()
    df['bb_std'] = df['Close'].rolling(window=ma_period).std()
    df['bb_upper'] = df['bb_middle'] + bb_multiplier * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - bb_multiplier * df['bb_std']

    # Compute the Keltner Channels
    df['kc_middle'] = df['Close'].rolling(window=ma_period).mean()
    df['kc_atr'] = pd.Series(abs(df['High'] - df['Low']), name='atr').rolling(window=ma_period).mean()
    df['kc_upper'] = df['kc_middle'] + kc_multiplier * df['kc_atr']
    df['kc_lower'] = df['kc_middle'] - kc_multiplier * df['kc_atr']

    # Compute the TTM Squeeze
    df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    df['squeeze_off'] = (df['bb_upper'] > df['kc_upper']) & (df['bb_lower'] < df['kc_lower'])
    df['ta_mo_TTMSqueeze'] = np.where(df['squeeze_on'], 1, np.where(df['squeeze_off'], -1, 0))
    df.drop(['bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'kc_middle', 'squeeze_on','squeeze_off','kc_atr', 'kc_upper', 'kc_lower' ], axis=1, inplace=True)

    return df

from trend import ta_trend_CCI

def ta_mo_IMI(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    vol = df['Volume']

    imi = pd.Series(0.0, index=df.index)

    for i in range(period, len(df)):
        pm = 0
        nm = 0
        for j in range(i - period, i):
            if close[j] > close[j - 1]:
                pm += vol[j] * (close[j] - ((high[j] + low[j]) / 2))
            elif close[j] < close[j - 1]:
                nm += vol[j] * (((high[j] + low[j]) / 2) - close[j])
        if pm + nm == 0:
            imi[i] = imi[i - 1]
        else:
            imi[i] = 100 * (pm / (pm + nm))

    return imi

def ta_mo_3DOscillator(df, n=5, m=8):
    """
    Computes the 3D Oscillator for a stock.

    Args:
    df (pd.DataFrame): Dataframe containing the stock data.
    n (int): Shorter lookback period for the indicator.
    m (int): Longer lookback period for the indicator.

    Returns:
    float: 3D Oscillator.
    """

    ema_n = df['Close'].ewm(span=n).mean()
    ema_m = df['Close'].ewm(span=m).mean()
    oscillator = (ema_n - ema_m) / ema_m

    return oscillator.iloc[-1]

def ta_mo_RMI(df, period=14):
    close = df['Close']
    rmi = pd.Series(0.0, index=df.index)
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = abs(delta.where(delta < 0, 0.0))

    ema_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    ema_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rsi = 100 - (100 / (1 + (ema_gain / ema_loss)))

    rmi[period:] = 100 - (100 / (1 + rsi))

    return rmi
def ta_mo_RSC(df, period=14):
    """
    Computes the Regression Slope Cross (RSC) indicator.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the stock data.
    period : int, optional (default=14)
        Number of periods for the moving average.

    Returns:
    --------
    pandas.DataFrame
        Dataframe containing the RSC values.
    """
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    # Calculate linear regression
    x = pd.DataFrame(range(1, len(df) + 1))
    y = df['Close']
    linreg = LinearRegression()
    linreg.fit(x, y)
    linreg_line = pd.Series(linreg.predict(x), index=df.index)

    # Calculate moving average of regression line
    rsc_ma = linreg_line.rolling(window=period).mean()

    # Compute RSC values
    rsc = df['Close'] - rsc_ma
    rsc = rsc.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    return rsc



def ta_mo_AdaptiveCyberCycle(df, period=10):
    alpha = 2 / (period + 1)
    diff = df['Close'].diff().abs()
    max_diff = diff.rolling(period).max()
    raw_cci = ((df['Close'] - df['Close'].rolling(period).mean()) / max_diff)
    cci = (1 - alpha) * ta_trend_CCI(df).shift(1) + alpha * raw_cci
    return cci

'''

def high_low_index(df, period=10):
    highs = df['High'].rolling(period).max().values
    lows = df['Low'].rolling(period).min().values
    hl_index = (df['Close'] - lows) / (highs - lows)
    return hl_index
def intraday_intensity_index(df):
    ii = (2 * df['Close'] - df['High'] - df['Low']) * df['Volume'] / (df['High'] - df['Low'])
    return ii
def market_facilitation_index(df):
    mfi = df['Volume'] / (df['High'] - df['Low'])
    return mfi

'''
def ta_mo_RainbowOscillator(df, n=10):
    close = df['Close']
    roc = ((close - close.shift(n)) / close.shift(n)) * 100
    ma1 = roc.rolling(window=n, min_periods=n).mean()
    ma2 = roc.rolling(window=n*2, min_periods=n*2).mean()
    ma3 = roc.rolling(window=n*4, min_periods=n*4).mean()
    return (ma1 + ma2 + ma3) / 3

def ta_mo_GannFan(df):
    highs = df['High'].values
    lows = df['Low'].values
    highs = np.sort(highs)
    lows = np.sort(lows)
    a = np.abs(highs[-1] - lows[-1])
    b = np.abs(highs[-1] - lows[0])
    c = np.abs(highs[0] - lows[-1])
    angles = [np.arctan(b/a), np.arctan(c/a)]
    return angles
def ta_mo_HighLowIndex(df, period=10):
    highs = df['High'].rolling(period).max().values
    lows = df['Low'].rolling(period).min().values
    hl_index = (df['Close'] - lows) / (highs - lows)
    return hl_index
def ta_mo_IntradayIntensityIndex(df):
    ii = (2 * df['Close'] - df['High'] - df['Low']) * df['Volume'] / (df['High'] - df['Low'])
    return ii
def ta_mo_MarketFacilitationIndex(df):
    mfi = df['Volume'] / (df['High'] - df['Low'])
    return mfi
def ta_mo_PolarizedFractalEfficiency(df, n=10):
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
'''
def bid_ratio(df):
    df['Bid Ratio'] = df['Bid Size'] / df['Ask Size']
    return df[['Close', 'Bid Ratio']]
'''
def ta_mo_AwesomeOscillator(df):
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_34'] = df['Close'].rolling(window=34).mean()
    df['AO'] = df['SMA_5'] - df['SMA_34']
    df['SMA_AO'] = df['AO'].rolling(window=5).mean()
    return df[['Close', 'AO', 'SMA_AO']]


def ta_mo_BressertDss(df, period=5, smoothing_period=14):
    """
    Computes the Bressert Double Smoothed Stochastic of a column in a dataframe.

    Parameters:
    df (pandas.DataFrame): Input dataframe.
    period (int): Period for the Bressert DSS.
    smoothing_period (int): Smoothing period for the Bressert DSS.

    Returns:
    pandas.DataFrame: Dataframe containing the Bressert DSS.
    """
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    k = (df['close'] - lowest_low) / (highest_high - lowest_low)
    k_smooth = k.rolling(window=smoothing_period).mean()
    dss = k_smooth.rolling(window=smoothing_period).mean()
    return dss

def ta_mo_BreadthThrustIndicator(stock_df):
    # Calculate advancing and declining issues
    advancers = stock_df['Close'] > stock_df['Open']
    decliners = stock_df['Close'] < stock_df['Open']

    # Calculate advancing and declining volume
    adv_volume = stock_df['Volume'] * advancers
    dec_volume = stock_df['Volume'] * decliners

    # Compute daily advancing issues and volume ratios
    adv_ratio = advancers.sum() / (advancers.sum() + decliners.sum())
    adv_volume_ratio = adv_volume.sum() / (adv_volume.sum() + dec_volume.sum())

    # Compute exponential moving averages
    adv_ema = pd.Series(adv_ratio).ewm(span=10, adjust=False).mean()
    vol_ema = pd.Series(adv_volume_ratio).ewm(span=10, adjust=False).mean()

    # Calculate breadth thrust indicator
    breadth_thrust = vol_ema / adv_ema

    # Compute 10-day exponential moving average
    breadth_thrust_ema = pd.Series(breadth_thrust).ewm(span=10, adjust=False).mean()

    return breadth_thrust_ema
def ta_mo_McclellanOscillator(df, n1=19, n2=39):
    """
    Calculates the McClellan Oscillator for a given stock dataframe.
    Args:
        df (pd.DataFrame): The stock dataframe.
        n1 (int): The short-term EMA period.
        n2 (int): The long-term EMA period.
    Returns:
        pd.DataFrame: The McClellan Oscillator values.
    """
    advancers = df['Advances'].rolling(n1).mean()
    decliners = df['Declines'].rolling(n1).mean()
    adv_diff = advancers - decliners
    mcclellan = adv_diff.ewm(span=n2).mean()
    return mcclellan

def ta_mo_ForecastOscillator(df, n=10):
    """Calculate the Forecast Oscillator for a given DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame containing stock data, including 'High' and 'Low' columns.
    n (int, optional): Number of periods to use for the oscillator calculation. Default is 10.

    Returns:
    pandas.Series: The Forecast Oscillator values.
    """
    hl_avg = (df['High'] + df['Low']) / 2
    forecast = hl_avg.shift(n).rolling(window=n, min_periods=1).mean()
    actual = hl_avg.rolling(window=n, min_periods=1).mean()
    return (forecast - actual) / actual


def ta_mo_ZeroCrossingRate(df):
    """
    Computes the zero crossing rate of a stock's closing price.

    Args:
    df (pd.DataFrame): Dataframe containing the stock data.

    Returns:
    float: Zero crossing rate.
    """

    prices = df['Close'].values
    zcr = ((prices[:-1] * prices[1:]) < 0).sum() / len(prices)

    return zcr
def ta_mo_ErgodicOscillator(df, period_long=20, period_short=5):
    """
    Computes the Ergodic Oscillator of a column in a dataframe.

    Parameters:
    df (pandas.DataFrame): Input dataframe.
    period_long (int): Long period for the Ergodic Oscillator.
    period_short (int): Short period for the Ergodic Oscillator.

    Returns:
    pandas.DataFrame: Dataframe containing the Ergodic Oscillator.
    """
    ema_long = df['close'].ewm(span=period_long, adjust=False).mean()
    ema_short = df['close'].ewm(span=period_short, adjust=False).mean()
    eo = ema_short / ema_long - 1
    eo_signal = eo.ewm(span=9, adjust=False).mean()
    return eo,eo_signal


def ta_mo_TwiggsMoneyFlow(df):
    """
    Calculates and returns Twiggs Money Flow indicator for a given dataframe

    Args:
    df: pandas DataFrame containing OHLC data

    Returns:
    pandas DataFrame containing Twiggs Money Flow indicator

    """
    twigs_mf = pd.Series(0.0, index=df.index)
    ad = ((2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']).cumsum()
    ad_diff = ad.diff()
    twigs_mf = ad_diff / (df['high'] - df['low'])
    return pd.DataFrame({'date': df['date'], 'twigs_mf': twigs_mf})
def ta_mo_PriceZoneOscillator(df, period=10):
    pzo = ((df['Close'] - df['Close'].rolling(window=period).mean()) / df['Close'].rolling(window=period).std())
    return pzo
def ta_mo_Qstick(data):
    diff = data['Close']-data['Open']
    return ta_ma_SMA(diff),ta_ma_EMA(diff)

def ta_mo_ROC(df, n=14):
    roc = 100 * df['Close'].pct_change(periods=n)
    return roc

def ta_mo_CenterOfGravity(df, period=10):
    weights = np.arange(1, period+1)
    numerator = np.sum(weights * df['Close'].rolling(window=period).mean())
    denominator = np.sum(weights)
    center_of_gravity = numerator / denominator
    return center_of_gravity
#-------------------------------------
def add_Momentum(df):
    df['ta_mo_stoch_K'],df['ta_mo_stoch_D'],\
    df['ta_mo_stoch_DS'],df['ta_mo_stoch_signal']=ta_mo_StochOscillator(df)

    df['ta_mo_CoppockCurve'] = ta_mo_CoppockCurve(df)

    df['ta_mo_CMO'] = ta_mo_CMO(df)
    df['ta_mo_HurstSpectralOscillator']=ta_HurstSpectralOscillator(df)
    df['ta_mo_UltimateOscillator']=ta_mo_UltimateOscillator(df)

    df['ta_mo_ConnorsRSI']=ta_mo_ConnorsRSI(df)
    df['ta_mo_KairiIndex']=ta_mo_KRI(df)

    df['ta_mo_RSI']=ta_mo_RSI(df)
    df['ta_mo_TSI']=ta_mo_TSI(df)

    df['ta_mo_RVI'],df['ta_mo_RVI_signal']=ta_mo_RVI(df)


    df['ta_mo_APO'],df['ta_mo_APO_hist'],df['ta_mo_APO_signal']=ta_mo_APO(df)
    df['ta_mo_PPO'],df['ta_mo_PPO_hist'],df['ta_mo_PPO_signal']=ta_mo_PPO(df)


    df['ta_mo_Qstick_SMA'],df['ta_mo_Qstick_EMA']=ta_mo_Qstick(df)

    df['ta_mo_ROC']=ta_mo_ROC(df)

    df['ta_mo_PMO'],df['ta_mo_PMO_hist'],df['ta_mo_PMO_signal']=ta_mo_PMO(df)
    df['ta_mo_WildersMovingAverage'] = ta_mo_WildersMovingAverage(df)

    df['ta_mo_SpecialK']=ta_mo_SpecialK(df)
    df['ta_mo_WilliamsR']=ta_mo_WilliamsR(df)

    #df['ta_mo_CMB'],df['ta_mo_CMB_signal']=ta_mo_CMB(df)
    #df = pd.concat([df,ta_mo_TTMSqueeze(df)],axis=1)
    return df



