import pandas as pd
from movingAverages import *
def ta_vol_VarianceRatios(stock_df):
    """
    Computes all of the variance ratios of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data

    Returns:
    pd.Series: A pandas series containing all of the variance ratios
    """
    # Compute the log returns
    log_returns = np.log(stock_df['Close']) - np.log(stock_df['Close'].shift(1))

    # Compute the variance ratios
    vr_2 = np.var(log_returns)
    vr_3 = np.sum((log_returns - np.mean(log_returns)) ** 3) / ((len(log_returns) - 1) * vr_2 ** (3 / 2))
    vr_4 = np.sum((log_returns - np.mean(log_returns)) ** 4) / ((len(log_returns) - 1) * vr_2 ** 2) - 3

    # Create a pandas series to store the variance ratios


    return vr_2,vr_3,vr_4
def ta_vol_ChoppinessIndex(stock_df, period=14):
    """
    Computes the Choppiness Index of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data
    period (int): The number of periods to use in the calculation (default is 14)

    Returns:
    pd.Series: A pandas series containing the Choppiness Index
    """
    # Compute the True Range and Average True Range
    stock_df['TR'] = np.maximum.reduce([stock_df['High'] - stock_df['Low'],
                                        abs(stock_df['High'] - stock_df['Close'].shift(1)),
                                        abs(stock_df['Low'] - stock_df['Close'].shift(1))])
    stock_df['ATR'] = stock_df['TR'].rolling(period).mean()

    # Compute the range over the period
    high_over_period = stock_df['High'].rolling(period).max()
    low_over_period = stock_df['Low'].rolling(period).min()
    range_over_period = high_over_period - low_over_period

    # Compute the Choppiness Index
    choppiness_index = 100 * np.log10(stock_df['ATR'] / range_over_period) / np.log10(period)

    return choppiness_index


def ta_vol_Returns(df,window=1):
    return df['Close'].pct_change(window).dropna()


def ta_vol_LogReturns(df,window=1):
    return np.log(df['Close'] / df['Close'].shift(window)).dropna()


def ta_vol_Volatility(data, window=1):
    returns = ta_vol_Returns(data)
    return returns.rolling(window).var().dropna()
def ta_vol_TP(data: pd.DataFrame):
    return (data['High'] + data['Low'] + data['Close']) / 3
def ta_vol_MP(df):
    mp = (df['High'] + df['Low']) / 2
    return mp

# -------------------Volatility indicators-----------------

# Bollinger Bands
def ta_vol_BollingerBands(data: pd.DataFrame, period=20):
    """
    Bollinger Bands: Bollinger Bands are a type of volatility indicator that consists of a simple moving average and two standard deviation lines plotted above and below the moving average. The distance between the bands and the moving average represents the stock's volatility, and traders use Bollinger Bands to identify potential buy and sell signals based on the stock's price action relative to the bands.
    :param data:
    :param period:
    :return:
    """
    sma = ta_ma_SMA(data, period)
    std = data['Close'].rolling(period).std()
    bollinger_up = sma + std * 2  # Calculate top band
    bollinger_down = sma - std * 2  # Calculate bottom band
    return bollinger_up, bollinger_down, sma


def ta_vol_BollingerBandWidth(data):
    a, b, c = ta_vol_BollingerBands(data=data)
    return a - b


# Keltner Channels
def ta_vol_KeltnerChannel(data: pd.DataFrame, period=20, k=2):
    """
    Keltner Channels: Keltner Channels are similar to Bollinger Bands, but instead of using standard deviation to calculate the channel width, Keltner Channels use the average true range (ATR) of the stock's price. The ATR is a measure of the stock's volatility, and traders use Keltner Channels to identify potential buy and sell signals based on the stock's price action relative to the channels.
    :param data:
    :param period:
    :param k:
    :return:
    """
    ema = ta_ma_EMA(data, period)
    keltner_up = ema + ta_vol_ATR(data, period) * k  # Calculate top band
    keltner_down = ema - ta_vol_ATR(data, period) * k  # Calculate bottom band
    return keltner_up, keltner_down, ema


def ta_vol_KeltnerChannelWidth(data):
    a, b, c = ta_vol_KeltnerChannel(data=data)
    return a - b



def ta_vol_ChaikinVolatility(df, n=1):
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    cv = clv.rolling(n).std() * np.sqrt(n)
    return cv

def ta_vol_TR(data):
    '''
    True Range
    The average true range (ATR) is a technical analysis indicator introduced by market technician J. Welles Wilder Jr. in his book New Concepts in Technical Trading Systems that measures market volatility by decomposing the entire range of an asset price for that period
    :param data:
    :return:
    '''
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_close, high_low, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range

def ta_vol_ATR(data, period=14):
    '''
    Average True Range
    The average true range (ATR) is a technical analysis indicator introduced by market technician J. Welles Wilder Jr. in his book New Concepts in Technical Trading Systems that measures market volatility by decomposing the entire range of an asset price for that period
    :param data:
    :param period:
    :return:
    '''
    return ta_vol_TR(data).rolling(period).mean()

def ta_vol_MassIndex(data,n=9,m=25):
    """
    The Mass Index is a technical indicator that uses the high-low range to identify trend reversals in the stock market. It measures the difference between the high and low prices and how it changes over time to identify price changes that may be occurring at the end of a trend.

    The formula for the Mass Index is:

    Single EMA = 9-period EMA of the high-low range
    Double EMA = 9-period EMA of the 9-period EMA of the high-low range
    EMA Ratio = Single EMA / Double EMA
    Mass Index = 25-period sum of the EMA Ratio
    :param data:
    :param n:
    :param m:
    :return:
    """
    high = data['High']
    low = data['Low']
    range_high_low = high - low
    single_ema = pd.Series(range_high_low.ewm(span=n, min_periods=n).mean())
    double_ema = pd.Series(single_ema.ewm(span=n, min_periods=n).mean())
    ema_ratio = single_ema / double_ema
    mass_index = pd.Series(ema_ratio.rolling(m).sum(), name='Mass Index_' + str(n) + '_' + str(m))
    return mass_index

'''
# True Range and Average True Range
def ta_vol_EfficiencyRatio(data):
    returns = get_returns(data)
    volatility = get_rolling_variance(data)
    print(len(returns))
    print(len(volatility))
    return [r / v for r, v in zip(returns, volatility)]
'''

def ta_vol_PercentB(data, n=20, mult=2):
    """Compute %B Indicator"""
    mid = data['Close'].rolling(n).mean()
    std = data['Close'].rolling(n).std()
    upper = mid + mult * std
    lower = mid - mult * std
    percent_b = (data['Close'] - lower) / (upper - lower)
    return percent_b

def ta_vol_PercentK(df, ema_period=20, atr_period=10, atr_multiplier=2):
    # Calculate EMA
    ema =ta_ma_EMA(df['Close'], window=ema_period)

    # Calculate ATR
    atr = ta_vol_ATR(df, atr_period)

    # Calculate upper and lower Keltner Channels
    upper_kc = ema + atr_multiplier * atr
    lower_kc = ema - atr_multiplier * atr

    # Calculate percent K
    percent_k = (df['Close'] - lower_kc) / (upper_kc - lower_kc)

    return percent_k
def ta_vol_UlcerIndex(df, n=14):
    df=df.copy()
    rs = df['Close'].pct_change()
    dd = (1 - df['Close'] / df['Close'].rolling(n).max()) * 100
    ui = np.sqrt((1 / n) * (dd ** 2).rolling(n).sum())
    df['Ulcer_Index'] = ui
    return df['Ulcer_Index']

def ta_vol_AberrationIndicator(df, n=10):
    """
    Computes the Aberration Indicator for a stock.

    Args:
    df (pd.DataFrame): Dataframe containing the stock data.
    n (int): Lookback period for the indicator.

    Returns:
    float: Aberration Indicator.
    """

    high_prices = df['High'].rolling(window=n).max()
    low_prices = df['Low'].rolling(window=n).min()
    mid_prices = (high_prices + low_prices) / 2
    ai = df['Close'] - mid_prices

    return ai.mean()


def ta_vol_FractalDimensionIndex(df, n=14):
    high, low = df['High'], df['Low']
    rs = np.log(high / low).cumsum()
    fdi = np.zeros(n-1)
    for i in range(n-1, len(high)):
        rescaled_range = (rs[i] - rs[i-n+1]) / n
        fdi = np.append(fdi, rescaled_range)
    return pd.Series(fdi, index=df.index)
def ta_vol_VQI(df, period=14, factor=2):
    TR = pd.Series(0.0, index=df.index)
    for i in range(1, len(df.index)):
        TR[i] = max(df['High'][i], df['Close'][i-1]) - min(df['Low'][i], df['Close'][i-1])
    ATR = TR.rolling(window=period).mean()
    std_dev = df['Close'].rolling(window=period).std()
    VQI = pd.Series(0.0, index=df.index)
    for i in range(period, len(df.index)):
        VQI[i] = (std_dev[i] / ATR[i]) * factor
    return VQI
def ta_vol_EfficiencyEatio(df, window=26):
    high = df['High']
    low = df['Low']
    close = df['Close']
    change = close.diff()
    volatility = (high - low).rolling(window).sum()
    return change.abs().rolling(window).sum() / volatility

def add_Volatility(df):
    #Maybe add moving average

    df['ta_vol_Returns_1']=ta_vol_Returns(df)
    df['ta_vol_Returns_5'] = ta_vol_Returns(df,5)
    df['ta_vol_Returns_10'] = ta_vol_Returns(df,10)
    df['ta_vol_Returns_30'] = ta_vol_Returns(df,30)
    df['ta_vol_Returns_90'] = ta_vol_Returns(df,90)

    df['ta_vol_LogReturns'] = ta_vol_LogReturns(df)
    df['ta_vol_LogReturns_5'] = ta_vol_LogReturns(df, 5)
    df['ta_vol_LogReturns_10'] = ta_vol_LogReturns(df, 10)
    df['ta_vol_LogReturns_30'] = ta_vol_LogReturns(df, 30)
    df['ta_vol_LogReturns_90'] = ta_vol_LogReturns(df, 90)

    df['ta_vol_Volatility_1'] = ta_vol_Volatility(df)
    df['ta_vol_Volatility_5'] = ta_vol_Volatility(df,5)
    df['ta_vol_Volatility_10'] = ta_vol_Volatility(df,10)
    df['ta_vol_Volatility_30'] = ta_vol_Volatility(df,30)
    df['ta_vol_Volatility_90'] = ta_vol_Volatility(df,90)


    df['ta_vol_TP'] = ta_vol_TP(df)

    df['ta_vol_TR'] = ta_vol_TR(df)
    df['ta_vol_ATR'] = ta_vol_ATR(df)

    df['ta_vol_BollingerBand_Upper'],df['ta_vol_BollingerBand_Lower'],df['ta_vol_BollingerBand_middle'] = ta_vol_BollingerBands(df)
    df['ta_vol_BollingerBand_Width'] = ta_vol_BollingerBandWidth(df)
    df['ta_vol_PercentB']=ta_vol_PercentB(df)



    df['ta_vol_KeltnerChannel_Upper'], df['ta_vol_KeltnerChannel_Lower'], df['ta_vol_KeltnerChannel_middle'] = ta_vol_KeltnerChannel(df)
    df['ta_vol_KeltnerChannel_Width'] = ta_vol_KeltnerChannelWidth(df)
    df['ta_vol_PercentK']=ta_vol_PercentK(df)

    df['ta_vol_ChaikinVolatility'] = ta_vol_ChaikinVolatility(df)
    df['ta_vol_UlcerIndex']=ta_vol_UlcerIndex(df)
    #df['ta_vol_'] = ta_vol_
    #df['ta_vol_'] = ta_vol_
    #df['ta_vol_'] = ta_vol_
    #df['ta_vol_'] = ta_vol_
    #df['ta_vol_'] = ta_vol_
    #df['ta_vol_'] = ta_vol_

    return df
