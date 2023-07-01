
# Moving Averages
# 'Moving Averages (MA) - A Moving Average is a line on a stock chart that represents the average price of a stock over a specified number of periods. The two most commonly used moving averages are the 50-day and 200-day moving averages. The 50-day MA is used to identify short-term trends, while the 200-day MA is used to identify long-term trends.'

# --------------------------Moving Averages--------------------------------------------
import numpy as np
import pandas as pd
import yfinance


def ta_ma_SmoothSMA(dataframe, window=14, smoothing_factor=.2):
    #print(dataframe.columns)
    try:
        sma = dataframe['Close'].rolling(window=window).mean()
    except:
        sma = dataframe.rolling(window=window).mean()
    # smoothed_sma = np.zeros_like(df.index)
    smoothed_sma = (sma * smoothing_factor) + (sma.shift(1) * (1 - smoothing_factor))
    return smoothed_sma
def ta_ma_DblSmoothEma(df, alpha1=.1,alpha2=.2, column='Close'):
    """
    Compute the doubly smoothed EMA for a given column in a dataframe.

    Parameters:
    df (pandas.DataFrame): the dataframe containing the data
    alpha (float): the smoothing factor (0 < alpha < 1)
    col_name (str): the name of the column to compute the EMA for

    Returns:
    pandas.Series: a series containing the doubly smoothed EMA values
    """
    ema1 = df[column].ewm(alpha=alpha1, adjust=False).mean()
    ema2 = ema1.ewm(alpha=alpha2, adjust=False).mean()
    return 2*ema1 - ema2


def ta_ma_TriplSmoothEMA(df, column='Close',alpha1=.1, alpha2=.15, alpha3=.2):
    """
    Compute the triply smoothed EMA for a given column in a dataframe.

    Parameters:
    df (pandas.DataFrame): the dataframe containing the data
    alpha (float): the smoothing factor (0 < alpha < 1)
    col_name (str): the name of the column to compute the EMA for

    Returns:
    pandas.Series: a series containing the triply smoothed EMA values
    """
    ema1 = df[column].ewm(alpha=alpha1, adjust=False).mean()
    ema2 = ema1.ewm(alpha=alpha2, adjust=False).mean()
    ema3 = ema2.ewm(alpha=alpha3, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3

def ta_ma_SMA(data, window=14):
    """
    The Simple Moving Average (SMA) is a type of moving average that gives equal weight to each price in the average. It is calculated by taking the sum of a specified number of days of a stock's closing price and dividing it by the number of days. The result is the average price over a specified period of time.
    :param data:
    :param window:
    :return:
    """
    if type(data) == pd.DataFrame:
        data = data['Close']
        sma = [np.NaN] * (window - 1)
        if window > len(data):
            return ValueError("Window must be smaller than the length of data")
        # sma = []
        for i in range(len(data) - window + 1):
            sma.append(sum(data.iloc[i:i + window]) / window)
        return sma
    else:
        sma = [np.NaN] * (window - 1)
        if window > len(data):
            return ValueError("Window must be smaller than the length of data")
        # sma = []
        for i in range(len(data) - window + 1):
            sma.append(sum(data[i:i + window]) / window)
        return sma


# Double check
def ta_ma_EMA(data, window=14):
    """
    The Exponential Moving Average (EMA) is a type of moving average that gives more weight to recent prices and less weight to older prices. It is calculated by applying a weighting factor to the previous day's EMA and to the current day's price. The weighting factor is based on the number of days in the average, with more recent prices given a higher weight.
    :param data:
    :param window:
    :return:
    """
    if type(data) == pd.DataFrame:
        data = data['Close']
    ema = [np.NaN] * (window - 1)
    if window > len(data):
        return ValueError("Window must be smaller than the length of data")
    # ema = []
    alpha = 2 / (window + 1)
    ema.append(sum(data[:window]) / window)
    for i in range(len(data) - window):
        ema.append(alpha * data[window + i] + (1 - alpha) * ema[-1])
    return ema


def ta_ma_KAMA(dataframe, n=10, pow1=2, pow2=30):
    change = abs(dataframe['Close'] - dataframe['Close'].shift(1))
    volatility = change.rolling(n).sum()
    er = change / volatility
    sc = ((er * (2 / (pow1 + 1) - 2 / (pow2 + 1)) + 2 / (pow2 + 1)) ** 2).rolling(n).sum()
    kama = np.zeros_like(dataframe['Close'])
    kama[n - 1] = dataframe['Close'].iloc[n - 1]
    for i in range(n, len(dataframe)):
        kama[i] = kama[i - 1] + sc[i] * (dataframe['Close'].iloc[i] - kama[i - 1])
    #dataframe['ta_ma_KAMA'] = kama
    return kama
    #return dataframe


def ta_ma_WMA(data, window=14):
    # print(data)
    # sma = [np.NaN] * window
    """
    The Weighted Moving Average (WMA) is a type of moving average that gives more weight to recent prices and less weight to older prices. The WMA calculation assigns a weighting factor to each price in the average, with the most recent price receiving the highest weighting factor and the oldest price receiving the lowest weighting factor.
    :param data:
    :param window:
    :return:
    """
    # print(data)
    weights = [i + 1 for i in range(window)]
    total_weights = sum(weights)
    # print(total_weights)
    # print(weights)
    # print(data)
    wma = [np.NaN] * window
    diff = len(data) - window
    for x in range(diff):

        try:
            weighted_data = [data[i + x] * weights[i] for i in range(window)]
        except:
            data = data['Close']
            weighted_data = [data[i + x] * weights[i] for i in range(window)]
            # print(weighted_data)

        wma.append(sum(weighted_data) / total_weights)
    return wma


def ta_ma_HMA(data, window=14):
    '''
    The Hull Moving Average (HMA) is a type of moving average that was developed by Alan Hull. It is designed to be a more responsive and smoother moving average than the traditional Simple Moving Average (SMA).
    The HMA is calculated using the weighted moving average (WMA) of a stock's price and then doubling the period of the WMA. The resulting average is then smoothed using another WMA with a period half the size of the original WMA.
    Traders often use the HMA as a trend-following indicator to identify potential buy and sell signals. The HMA can help traders identify changes in the trend of a stock's price, and it is considered to be a useful indicator for determining both short-term and long-term trends.
    :param data:
    :param window:
    :return:
    '''
    # data = data['Close']
    wma1 = ta_ma_WMA(data, window // 2)
    wma2 = ta_ma_WMA(wma1, window // 2)
    return wma2


def ta_ma_TRMA(data, window=14):
    """
    The Triangular Moving Average (TMA) is a type of moving average that is calculated by averaging the SMA and the WMA of a stock's price. The TMA is designed to provide a smoother average that is less sensitive to price fluctuations than the SMA, while still retaining the responsiveness of the WMA.
    Traders often use the TMA as a trend-following indicator to identify potential buy and sell signals. The TMA can help traders identify changes in the trend of a stock's price, and it is considered to be a useful indicator for determining both short-term and long-term trends.Traders often use the TMA as a trend-following indicator to identify potential buy and sell signals. The TMA can help traders identify changes in the trend of a stock's price, and it is considered to be a useful indicator for determining both short-term and long-term trends.
    :param data:
    :param window:
    :return:
    """
    sma = ta_ma_SMA(data, window)
    wma = ta_ma_WMA(data, window)
    sol = [(s + m) / 2 for s, m in zip(sma, wma)]
    return sol


def ta_ma_VMA(data, window=14):
    """
    The Variable Moving Average (VMA) is a type of moving average that adjusts its smoothing factor based on the volatility of the stock's price. The VMA calculation uses a smoothing factor that changes based on the stock's price volatility, resulting in a moving average that is more responsive in volatile markets and smoother in less volatile markets.
    :param data:
    :param window:
    :return:
    """
    data = data['Close']
    volatility = sum(abs(data[i + 1] - data[i]) for i in range(window - 1)) / window
    smoothing_factor = 2 / (window + 1) if volatility > 0.1 else 2 / (window + 1) + 0.5 * (volatility - 0.1)
    return ta_ma_SmoothSMA(data, window, smoothing_factor)

def ta_ma_McginleyDynamic(df, n=10):
    df = df.copy()
    md = df['Close'].iloc[0]
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        prev_md = md
        md += (price - prev_md) / (n * (price / prev_md) ** 4)
        df.loc[df.index[i], 'McGinley_Dynamic'] = md
    return df['McGinley_Dynamic']

def ta_ma_AdaptiveMA(df, n=10, fast=2, slow=30):
    high = df['High']
    low = df['Low']
    close = df['Close']
    smooth = pd.Series(index=df.index)
    smooth.iloc[0] = close.iloc[0]
    ER_num = abs(close - close.shift(n))
    ER_den = pd.Series(0.0, index=df.index)
    for i in range(1, n+1):
        ER_den += abs(close.shift(i) - close.shift(i+1))
    ER = ER_num / ER_den
    SC = ((ER * (2.0 / (fast + 1) - 2.0 / (slow + 1))) + 2 / (slow + 1)) ** 2.0
    for i in range(n, len(df)):
        smooth.iloc[i] = smooth.iloc[i-1] + SC.iloc[i] * (close.iloc[i] - smooth.iloc[i-1])
    return smooth
def ta_ma_McginleyDynamic(df, n=10):
    """
    Function to calculate the McGinley Dynamic Moving Average (MDMA) of a stock.

    Parameters:
    df (dataframe): Dataframe containing the stock's OHLC data
    n (int): Number of periods to consider for the MDMA

    Returns:
    dataframe: A new dataframe containing the MDMA values
    """

    # Calculate the initial value of the MDMA
    mdma = pd.Series(df['Close'].iloc[0], index=df.index[:n], name='MDMA')

    # Loop through the remaining values of the dataset and calculate the MDMA
    for i in range(n, len(df)):
        today_close = df['Close'].iloc[i]
        prev_close = mdma.iloc[i-1]
        prev_mdma = mdma.iloc[i-n:i].mean()

        # Calculate the MDMA
        mdma_today = prev_mdma + ((today_close - prev_mdma) / (n * (today_close / prev_close) ** 4))

        # Append the new value to the MDMA series
        mdma = mdma.append(pd.Series(mdma_today, index=[df.index[i]]))

    # Combine the MDMA series with the original OHLC data
    df = pd.concat([df, mdma], axis=1)

    return mdma
def ta_ma_ZeroLagExponentialMA(df, window=12):
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    zlema = 2 * ema - ema.ewm(span=window, adjust=False).mean()
    return zlema
'''
def wilders_ma(df):
    period = 14
    tr = pd.DataFrame(index=df.index)
    tr['h-l'] = abs(df['High'] - df['Low'])
    tr['h-pc'] = abs(df['High'] - df['Close'].shift())
    tr['l-pc'] = abs(df['Low'] - df['Close'].shift())
    tr['TR'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    tr = tr.drop(['h-l', 'h-pc', 'l-pc'], axis=1)
    wma = tr.iloc[:period, :]['TR'].sum() / period
    wmas = [wma]
    for i in range(period, len(df)):
        tr_val = tr.iloc[i]['TR']
        wma = (wma * (period - 1) + tr_val) / period
        wmas.append(wma)
    return pd.Series(wmas, index=df.index)
'''
def ta_ma_T3MA(df, period=14, vfactor=0.7):
    EMA1 = pd.Series(df['Close'].ewm(span=period).mean())
    EMA2 = pd.Series(EMA1.ewm(span=period).mean())
    EMA3 = pd.Series(EMA2.ewm(span=period).mean())
    C1 = -vfactor*vfactor*vfactor
    C2 = 3*vfactor*vfactor + 3*vfactor*vfactor*vfactor
    C3 = -6*vfactor*vfactor - 3*vfactor - 3*vfactor*vfactor*vfactor
    C4 = 1 + 3*vfactor + vfactor*vfactor*vfactor + 3*vfactor*vfactor
    T3 = pd.Series(0.0, index=df.index)
    for i in range(period-1, len(df.index)):
        T3[i] = C1*EMA3[i] + C2*EMA2[i] + C3*EMA1[i] + C4*df['Close'][i]
    return T3
def ta_ma_JurikMA(df, period=14, vfactor=0.7):
    alpha = 2 * np.pi / (period + 1)
    b = np.cos(alpha)
    a1 = -b
    c1 = 1 - b
    c2 = c1 / 2
    Jurik = pd.Series(0.0, index=df.index)
    for i in range(1, len(df.index)):
        Jurik[i] = c1 * (1 - a1 / 2) * df['Close'][i] + c2 * (1 + a1) * Jurik[i-1] - c1 * a1 * Jurik[i-2]
    return Jurik
def ta_ma_GuppyMultipleMA(df, short_periods=[3, 5, 8, 10, 12, 15], long_periods=[30, 35, 40, 45, 50, 60]):
    df=df.copy()
    for i, period in enumerate(short_periods):
        df[f'ShortEMA_{period}'] = df['Close'].ewm(span=period).mean()

    for i, period in enumerate(long_periods):
        df[f'LongEMA_{period}'] = df['Close'].ewm(span=period).mean()

    guppy_short = df[[f'ShortEMA_{period}' for period in short_periods]].transpose().mean()
    guppy_long = df[[f'LongEMA_{period}' for period in long_periods]].transpose().mean()

    return guppy_short, guppy_long

def ta_ma_FRAMA(df, window=10):
    # Define constants
    df = df.copy().reset_index()
    alpha = 2 / (window + 1)
    d1 = 0.5 * np.exp(-1.414 * 3 / window) - np.exp(-1.414 * 3 / (window * window))
    d2 = np.exp(-1.414 * 3 / window) - 2 * np.exp(-1.414 * 3 / (window * window))
    c1 = d1
    c2 = -d2
    c3 = 1 - c1 - c2

    # Calculate high and low fractals
    df['max'] = df['High'].rolling(window=window).max()
    df['min'] = df['Low'].rolling(window=window).min()

    # Calculate FRAMA
    df['diff'] = 0
    for i in range(1, len(df)):
        if df.loc[i, 'max'] == df.loc[i-1, 'max'] and df.loc[i, 'min'] == df.loc[i-1, 'min']:
            df.loc[i, 'diff'] = 0
        else:
            df.loc[i, 'diff'] = abs(df.loc[i, 'max'] - df.loc[i-1, 'max']) + abs(df.loc[i, 'min'] - df.loc[i-1, 'min'])
    df['ER'] = df['diff'] / (window * 2)

    # Initialize FRAMA
    df['FRAMA'] = df['Close']

    # Calculate FRAMA for each row
    for i in range(window, len(df)):
        sum_ER = 0
        for j in range(i-window, i):
            sum_ER += df.loc[j, 'ER']
        P = c1 * sum_ER + c2 * sum_ER * sum_ER + c3
        df.loc[i, 'FRAMA'] = P * df.loc[i, 'Close'] + (1 - P) * df.loc[i-window, 'FRAMA']

    df.drop(['max', 'min', 'diff', 'ER'], axis=1, inplace=True)

    return df['FRAMA']
def ta_ma_rainbowMA(df, periods=[8, 13, 21, 34, 55]):
    close = df['Close']

    rm = pd.concat([close.rolling(window=p).mean() for p in periods], axis=1)
    #return rm
    rm.columns = [f'ta_ma_RainbowMA_{p}' for p in periods]

    return rm

def ta_ma_ModifiedMa(df, window=9):
    """
    Computes the Modified Moving Average given a stock dataframe.
    Parameters:
        df (pandas.DataFrame): The stock dataframe.
        window (int): The window size.
    Returns:
        pandas.DataFrame: A dataframe with the Modified Moving Average.
    """
    mma = df["Close"].ewm(span=window, min_periods=window).mean()
    return mma
from scipy.fftpack import fft
def ta_ma_CenteredMa(df, window=9):
    """
    Computes the Centered Moving Average given a stock dataframe.
    Parameters:
        df (pandas.DataFrame): The stock dataframe.
        window (int): The window size.
    Returns:
        pandas.DataFrame: A dataframe with the Centered Moving Average.
    """
    cma = df["Close"].rolling(window=window, center=True).mean()
    return cma


def ta_ma_FourierTransformMa(df, window=14):
    close_fft = fft(df['Close'].values)
    ma = np.real(np.fft.ifft(np.concatenate([close_fft[:window], np.zeros(len(df)-window)])))
    return ma





def ta_ma_WildersMa(df, window=14):
    ma = df['Close'].rolling(window=window).mean()
    weights = np.arange(1, window+1)
    weights_sum = np.sum(weights)
    for i in range(window, len(df)):
        ma[i] = ma[i-1] + (df['Close'][i] - ma[i-1]) * (weights_sum - weights[-1]) / weights_sum
        weights = np.insert(weights[:-1], 0, window)
        weights_sum = np.sum(weights)
    return ma

def ta_ma_WildersSmoothMa(df, window=14):
    sma = df['Close'].rolling(window=window).mean()
    weights = np.arange(1, window+1)
    weights_sum = np.sum(weights)
    for i in range(window, len(df)):
        current_weight = window / (window - 1)
        sma[i] = sma[i-1] + (df['Close'][i] - sma[i-1]) * current_weight
        weights = np.insert(weights[:-1], 0, window)
        weights_sum = np.sum(weights)
        weight_avg = weights_sum / window
        weights = np.clip(weights - weight_avg, 0, np.inf)
        weights_sum = np.sum(weights)
    return sma

def ta_ma_GeometricMa(df, window=14):
    ma = np.power(df['Close'].rolling(window=window).apply(lambda x: np.prod(x)), 1/window)
    return ma
def ta_ma_AlligatorMa(df, jaw_period=13, teeth_period=8, lips_period=5):
    """
    Computes the Alligator Moving Average given a stock dataframe.
    Parameters:
        df (pandas.DataFrame): The stock dataframe.
        jaw_period (int): The jaw period.
        teeth_period (int): The teeth period.
        lips_period (int): The lips period.
    Returns:
        pandas.DataFrame: A dataframe with the Alligator Moving Average.
    """
    df_jaw = pd.DataFrame()
    df_teeth = pd.DataFrame()
    df_lips = pd.DataFrame()
    df_jaw["ta_ma_AlligatorMA_Jaw"] = df["Close"].rolling(window=jaw_period).mean().shift(jaw_period)
    df_teeth["ta_ma_AlligatorMa_Teeth"] = df["Close"].rolling(window=teeth_period).mean().shift(teeth_period)
    df_lips["ta_ma_AlligatorMa_Lips"] = df["Close"].rolling(window=lips_period).mean().shift(lips_period)
    return pd.concat([df_jaw, df_teeth, df_lips], axis=1)



def ta_ma_ParabolicSar_Down(df, high_col='High', low_col='Low', af=0.02, max_af=0.2, sar_col='SAR Down'):
    '''
    Computes the downwards SAR (Stop and Reverse) indicator for a given DataFrame of stock prices.

    Parameters:
    df (Pandas DataFrame): The DataFrame containing the high and low prices for each period.
    high_col (str): The name of the column containing the high prices. Default is 'High'.
    low_col (str): The name of the column containing the low prices. Default is 'Low'.
    af (float): The acceleration factor used in the calculation. Default is 0.02.
    max_af (float): The maximum acceleration factor used in the calculation. Default is 0.2.
    sar_col (str): The name of the new column to store the downwards SAR values. Default is 'SAR Down'.

    Returns:
    df (Pandas DataFrame): The original DataFrame with a new column containing the downwards SAR values.
    '''

    # Initialize the first SAR value as the highest high in the first period
    sar = [max(df[high_col][:2])]

    # Initialize the first acceleration factor as the default value
    af_current = af

    # Initialize the first extreme point as the lowest low in the first period
    extreme_point = min(df[low_col][:2])

    # Loop through the rest of the periods to compute the SAR values
    for i in range(1, len(df)):

        # If the previous SAR value is higher than the current low, set the extreme point to the current low
        if sar[-1] > df[low_col][i]:
            extreme_point = min(extreme_point, df[low_col][i])

            # If the current acceleration factor is less than the maximum, increase it by the acceleration factor
            if af_current < max_af:
                af_current += af

        # Otherwise, if the previous SAR value is lower than the current low, update the SAR value and reset the acceleration factor and extreme point
        else:
            sar.append(extreme_point - af_current * (extreme_point - sar[-1]))
            af_current = af
            extreme_point = min(df[low_col][i], sar[-1])

        # If the current high is higher than the current SAR value, update the extreme point
        if df[high_col][i] > sar[-1]:
            extreme_point = max(extreme_point, df[high_col][i])

    # Add the new SAR column to the original DataFrame and return it
    df[sar_col] = sar

    return df


def ta_ma_ParabolicSar_Up(data, acceleration=0.02, maximum=0.2):
    """
    The Parabolic SAR (Parabolic Stop and Reverse) is a technical indicator used in stock market analysis to determine the direction of an asset's price and potential reversal points. It is calculated using a set of calculations that plot points on a chart to represent the price direction and potential reversal points of a stock.
    The Parabolic SAR is plotted on a stock chart and is used to identify potential trend reversals. When the SAR is below the price, it indicates an uptrend, and when it is above the price, it indicates a downtrend. When the price crosses the SAR, it signals a potential trend reversal.
    One of the key strengths of the Parabolic SAR is its ability to adjust to changing market conditions, making it a useful tool for both long-term and short-term traders. By combining the Parabolic SAR with other technical indicators, traders can gain a more complete picture of the stock's price action and make more informed trading decisions.
    :param high:
    :param low:
    :param acceleration:
    :param maximum:
    :return:
    """
    high = data['High']
    low = data['Low']
    n = len(high)
    sar = [low[0]]
    ep = [high[0]]
    af = acceleration
    for i in range(1, n):
        if i == 1:
            sar.append(sar[0] + af * (high[0] - sar[0]))
        else:
            sar.append(sar[i - 1] + af * (ep[i - 1] - sar[i - 1]))
        ep.append(max(high[i], ep[i - 1]))
        af = min(af + acceleration, maximum)
        if high[i] > sar[i - 1]:
            sar[i] = ep[i - 1]
            af = acceleration
    return sar

def ta_ma_SSMA(df, period=14):
    """
    Calculates the Super Smoother Moving Average indicator.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame with a column ['Close']
    period (int): The number of periods for the moving average

    Returns:
    pandas.Series: A pandas Series with the Super Smoother Moving Average values
    """
    alpha = 2 / (period + 1)
    beta = 1 - alpha
    ssf = (df['Close'] + 2 * df['Close'].shift(1) + df['Close'].shift(2)) / 4
    sssf = pd.Series(index=df.index, dtype='float64')
    sssf[0] = df['Close'][0]
    for i in range(1, len(df)):
        sssf[i] = alpha * ssf[i] + beta * sssf[i - 1]
    return sssf
def ta_ma_LSMA(df, window=25):
    weights = np.arange(1, window + 1)
    denominator = np.sum(weights)
    lsma = df.rolling(window).apply(lambda x: np.dot(x, weights) / denominator, raw=True)
    return lsma
def ta_ma_ALMA(df, window=9, sigma=6, offset=.85):
    m = offset * (window - 1)
    s = window / sigma
    w = np.array([np.exp(-(i - m) ** 2 / (2 * s ** 2)) for i in range(window)])
    alma = ((df * w).sum() / w.sum()).rolling(window).mean()
    return alma
def ta_ma_MEDMA(df, window=10):
    medma = df.rolling(window).median()
    return medma
def ta_ma_ZLMA(df, window=14):
    zlma = 2 * df.rolling(window // 2).mean() - df.rolling(window).mean()
    return zlma
def ta_ma_DetrendedMA(df, window=20):
    detrended = df - df.rolling(window).mean()
    dma = detrended.rolling(window).mean()
    return dma
def ta_ma_VIDYA(df, window=9, alpha=.2):
    vol = df.diff().abs().rolling(window).mean()
    vidya = df.rolling(window).mean() + alpha * (df - df.rolling(window).mean()) / vol
    return vidya

def ta_ma_ChandeViDynamic(df, period=14, a_factor=.2):
    """
    Calculates the Chande's Variable Index Dynamic Average indicator.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame with a column ['Close']
    period (int): The number of periods for the moving average
    a_factor (float): The acceleration factor

    Returns:
    pandas.Series: A pandas Series with the Chande's Variable Index Dynamic Average values
    """
    diff = abs(df['Close'] - df['Close'].shift(1))
    direction = df['Close'] - df['Close'].shift(1)
    volatility = diff.rolling(period).sum()
    volatility[period:] = volatility[period:] + volatility[:-(period)].mean()
    vi = pd.Series(index=df.index, dtype='float64')
    vi[0] = df['Close'][0]
    for i in range(1, len(df)):
        if direction[i] > 0:
            vi[i] = vi[i - 1] + (a_factor / volatility[i]) * (df['Close'][i] - vi[i - 1])
        elif direction[i] < 0:
            vi[i] = vi[i - 1] - (a_factor / volatility[i]) * (vi[i - 1] - df['Close'][i])
        else:
            vi[i] = vi[i - 1]
    return vi

def ta_ma_HighLowMa(df, period=14):
    """
    Calculates the High Low Moving Average indicator.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame with columns ['Open', 'High', 'Low', 'Close']
    period (int): The number of periods for the moving average

    Returns:
    pandas.Series: A pandas Series with the High Low Moving Average values
    """
    hl = (df['High'] + df['Low']) / 2
    ma = hl.rolling(period).mean()
    return ma


def ta_ma_DetrendedMa(df, period=14):
    """
    Calculates the Detrended Moving Average indicator.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame with a column ['Close']
    period (int): The number of periods for the moving average

    Returns:
    pandas.Series: A pandas Series with the Detrended Moving Average values
    """
    ma = df['Close'].rolling(period).mean()
    dma = df['Close'] - ma
    return dma

from scipy.signal import butter, lfilter
def butterworth_filter(df, cutoff, fs=1, order=2):
    """
    Applies a 2-Pole Butterworth filter to a stock's closing price.

    Args:
    df (pd.DataFrame): Dataframe containing the stock data.
    cutoff (float): Cutoff frequency for the filter.
    fs (int): Sampling frequency of the data.
    order (int): Order of the filter.

    Returns:
    pd.DataFrame: Dataframe containing the filtered data.
    """

    # Design filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Filter data
    prices = df['Close'].values
    filtered_prices = lfilter(b, a, prices)

    # Create filtered dataframe
    filtered_df = df.copy()
    filtered_df['Close'] = filtered_prices

    return filtered_df
def ta_ma_TripleWeightedMA(df, column='Close', window=14):
    """
    Computes Triple Weighted Moving Average for a given column in a DataFrame.

    Args:
    df (DataFrame): Input DataFrame
    column (str): Name of the column for which to compute Triple Weighted Moving Average
    window (int): Rolling window size

    Returns:
    DataFrame: A new DataFrame with Triple Weighted Moving Average values added as a new column.
    """
    weights = [1, 2, 3]
    tma = df[column].rolling(window=window, center=False).apply(lambda x: np.dot(x, weights) / 6, raw=True)
    return df.assign(TMA=tma)

def ta_ma_RMA(df, window=14):
    rma = df.rolling(window).mean()
    return rma
def ta_ma_DWMA(df, window=14):
    dwma = (2 * df.rolling(window).mean() - df.rolling(2 * window).mean()).fillna(method='bfill')
    return dwma

def ta_ma_SuperSSMA(df, period=14):
    """
    Calculates the Super Smoother Moving Average indicator.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame with a column ['Close']
    period (int): The number of periods for the moving average

    Returns:
    pandas.Series: A pandas Series with the Super Smoother Moving Average values
    """
    alpha = 2 / (period + 1)
    beta = 1 - alpha
    ssf = (df['Close'] + 2 * df['Close'].shift(1) + df['Close'].shift(2)) / 4
    sssf = pd.Series(index=df.index, dtype='float64')
    sssf[0] = df['Close'][0]
    for i in range(1, len(df)):
        sssf[i] = alpha * ssf[i] + beta * sssf[i - 1]
    return sssf


def ta_ma_DisplacedMA(df, column='Close', displacement=20):
    """
    Computes the displaced moving average of a column in a dataframe.

    Parameters:
    df (pandas.DataFrame): Input dataframe.
    column (str): Column name for the input variable.
    displacement (int): Number of periods by which to displace the moving average.

    Returns:
    pandas.DataFrame: Dataframe containing the displaced moving average.
    """
    dma = df[column].rolling(window=20).mean().shift(displacement)
    return pd.DataFrame(dma, columns=['displaced_moving_average'])

def add_MovingAverages(df:pd.DataFrame):
    # FAST SLOW AND HIST
    df['ta_ma_SMA_14'] = ta_ma_SMA(df,14)
    df['ta_ma_SMA_25'] = ta_ma_SMA(df, 25)
    df['ta_ma_SMA_50'] = ta_ma_SMA(df, 50)
    df['ta_ma_SMA_100'] = ta_ma_SMA(df, 100)
    df['ta_ma_SMA_200'] = ta_ma_SMA(df, 200)

    df['ta_ma_EMA_12'] = ta_ma_EMA(df,12)
    df['ta_ma_EMA_26'] = ta_ma_EMA(df, 26)
    df['ta_ma_EMA_50'] = ta_ma_EMA(df, 50)
    df['ta_ma_EMA_100'] = ta_ma_EMA(df, 100)
    df['ta_ma_EMA_200'] = ta_ma_EMA(df, 200)


    df['ta_ma_WMA'] = ta_ma_WMA(df)
    df['ta_ma_HMA'] = ta_ma_HMA(df)
    df['ta_ma_TRMA'] = ta_ma_TRMA(df)
    df['ta_ma_VMA'] = ta_ma_VMA(df)
    df['ta_ma_KAMA'] = ta_ma_KAMA(df)



    df['ta_ma_SmoothSMA'] = ta_ma_SmoothSMA(df)
    df['ta_ma_DblSmoothEMA'] = ta_ma_DblSmoothEma(df)
    df['ta_ma_TrpleSmoothEMA']=ta_ma_TriplSmoothEMA(df)


    df['ta_ma_FRAMA'] = ta_ma_FRAMA(df)
    df['ta_ma_T3MA']=ta_ma_T3MA(df)
    df['ta_ma_JurikMA'] = ta_ma_JurikMA(df)
    df['ta_ma_GuppyMultipleMA_long'],df['ta_ma_GuppyMultipleMA_short'] = ta_ma_GuppyMultipleMA(df)

    #[8, 13, 21, 34, 55]

    # Rainbow MA
    #df = pd.concat([df,ta_ma_rainbowMA(df)],axis=1)
    df['ta_ma_RainbowMA']= ta_ma_rainbowMA(df).mean(axis=1)


    df['ta_ma_AdaptiveMA'] = ta_ma_AdaptiveMA(df)

    df['ta_ma_ZeroLagExponentialMA'] = ta_ma_ZeroLagExponentialMA(df)

    df['ta_ma_McginleyDynamic'] = ta_ma_McginleyDynamic(df)


    df['Upwards Parabolic Sar'] = ta_ma_ParabolicSar_Up(df)


    #df['Downwards Parabolic Sar'] = ta_ma_ParabolicSar_Down(df)


    df['ta_ma_ModifiedMa'] = ta_ma_ModifiedMa(df)
    df['ta_ma_FourierTransformMa']=ta_ma_FourierTransformMa(df)




    df['ta_ma_WildersMA']=ta_ma_WildersMa(df)
    df['ta_ma_WildersSmoothMa']=ta_ma_WildersSmoothMa(df)

    df['ta_ma_GeometricMa']=ta_ma_GeometricMa(df)
    df['ta_ma_CenteredMA']=ta_ma_CenteredMa(df)

    df = pd.concat([df, ta_ma_AlligatorMa(df)], axis=1)


    #df['ta_ma_ALMA'] = ta_ma_ALMA(df)
    #df['ta_ma_LSMA'] = ta_ma_LSMA(df)
    #df['ta_ma_ZLMA'] = ta_ma_ZLMA(df)
    #df['ta_ma_SSMA'] = ta_ma_SSMA(df)

    #df['ta_ma_DetrendedMA'] = ta_ma_DetrendedMa(df)
    #df['ta_ma_MedianMA'] = ta_ma_MEDMA(df)
    #df['ta_ma_HighLowMA'] = ta_ma_HighLowMa(df)
    #df['ta_ma_ChandeViDynamic'] = ta_ma_ChandeViDynamic(df)

    return df


df = yfinance.Ticker('AAPL').history(period='MAX')
print(df.columns)
df= add_MovingAverages(df)
#print(df['ta_ma_RainbowMA_8'])
print(df.dtypes)
import plotly.express  as px
fig=px.line(df)
fig.show()
