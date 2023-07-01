import numpy as np
import pandas as pd
import yfinance

from momentum import *
def ta_volume_vema(df, period=10, column="Volume"):
    return df[column].ewm(span=period, min_periods=0, adjust=False).mean()


def ta_volume_vsma(df, period=10, column="Volume"):
    return df[column].rolling(window=period, min_periods=0).mean()


def ta_volume_volumersi(df, period=14):
    """
    Computes the Volume Relative Strength Index (V-RSI) of a stock given a dataframe and a lookback period.
    """
    df = df.copy()
    delta = df['Volume'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['volumersi'] = 100 - (100 / (1 + rs))
    return df['volumersi']


# Volume Force Index (VFI)
def vfi(df, period=13):
    vfi = ((df['Close'] - df['Open']) / df['Open']) * df['Volume']
    vfi = vfi.rolling(window=period).sum()
    return vfi

# Volume flow indicator
def ta_volume_VolumeFlowIndicator(df, periods=21):
    df=df.copy()
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfv = mfm * df['Volume']
    cmfv = mfv.rolling(window=periods).sum()
    cvol = df['Volume'].rolling(window=periods).sum()
    vfi = cmfv / cvol
    df['VFI'] = vfi
    return df['VFI']


def ta_volume_OBV(dataframe):
    obv = []
    previous = 0
    dataframe=dataframe.copy()
    for index, row in dataframe.iterrows():
        if row['Close'] > previous:
            current_obv = obv[-1] + row['Volume'] if len(obv) > 0 else row['Volume']
        elif row['Close'] < previous:
            current_obv = obv[-1] - row['Volume'] if len(obv) > 0 else -row['Volume']
        else:
            current_obv = obv[-1]
        obv.append(current_obv)
        previous = row['Close']
    dataframe['OBV'] = obv
    return dataframe['OBV']



def ta_volume_NVI(dataframe, n=255):
    """
        Calculates the Negative Volume Index (NVI) of a stock dataframe.

        Parameters:
            dataframe (DataFrame): A Pandas DataFrame with stock prices and volumes.

        Returns:
            nvi (Series): A Pandas Series of NVI values for each row in the input DataFrame.
        """
    nvi = pd.Series(index=dataframe.index, dtype='float64')
    nvi.iloc[0] = 1000
    for i in range(1, len(dataframe)):
        if dataframe['Volume'].iloc[i] < dataframe['Volume'].iloc[i - 1]:
            nvi.iloc[i] = nvi.iloc[i - 1] + (dataframe['Close'].iloc[i] - dataframe['Close'].iloc[i - 1]) / \
                          dataframe['Close'].iloc[i - 1] * nvi.iloc[i - 1]
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]
    return nvi


# split into 2
def calculate_mass_and_force(dataframe, n1=9, n2=25):
    high = dataframe['High']
    low = dataframe['Low']
    close = dataframe['Close']
    volume = dataframe['Volume']

    # Calculate the single-period range for each day
    range = high - low

    # Calculate the Exponential Moving Average (EMA) of the range for the first period
    # Should be ewm?
    ema_range = range.rolling(n1).mean()

    # Calculate the Mass Index for each day
    ema_ratio = ema_range / ema_range.rolling(n1).mean()
    mass_index = ema_ratio.rolling(n1).sum()

    # Calculate the Force Index for each day
    force_index = range * volume
    force_ema = force_index.rolling(n2).mean()

    # Add the calculated values as new columns in the original DataFrame
    dataframe['Mass_Index'] = mass_index
    dataframe['Force_Index'] = force_ema

    return dataframe

def volume_by_price(df, n):
    """Computes the volume by price for a given period."""
    high = df['High'].rolling(n).max()
    low = df['Low'].rolling(n).min()
    price_range = high - low
    num_price_levels = 10
    price_levels = np.linspace(low.iloc[-1], high.iloc[-1], num_price_levels)
    volume_by_price = pd.DataFrame(index=price_levels, columns=['Volume'])
    for i in range(len(price_levels)):
        if i == 0:
            volume_by_price.loc[price_levels[i]] = df['Volume'][df['Close'] <= price_levels[i]].sum()
        elif i == num_price_levels - 1:
            volume_by_price.loc[price_levels[i]] = df['Volume'][df['Close'] > price_levels[i - 1]].sum()
        else:
            volume_by_price.loc[price_levels[i]] = df['Volume'][
                (df['Close'] > price_levels[i - 1]) & (df['Close'] <= price_levels[i])].sum()
    return volume_by_price



def ta_volume_BOP(dataframe):
    dataframe=dataframe.copy()
    bop = ((dataframe['Close'] - dataframe['Open']) / (dataframe['High'] - dataframe['Low'])) * dataframe['Volume']
    dataframe['BOP'] = bop
    return dataframe['BOP']

def calculate_vwap_vwma(dataframe, anchor_date=None, window_size=20):
    # Calculate VWAP
    dataframe['TP'] = (dataframe['High'] + dataframe['Low'] + dataframe['Close']) / 3
    dataframe['TPV'] = dataframe['TP'] * dataframe['Volume']
    dataframe['Cumulative TPV'] = dataframe['TPV'].cumsum()
    dataframe['Cumulative Volume'] = dataframe['Volume'].cumsum()
    dataframe['VWAP'] = dataframe['Cumulative TPV'] / dataframe['Cumulative Volume']

    # Calculate VMA
    dataframe['VWMA'] = dataframe['Volume'].rolling(window=window_size).mean()

    # Calculate Anchored VWAP
    if anchor_date:
        anchor_data = dataframe.loc[:anchor_date]
        anchor_tpv = (anchor_data['High'] + anchor_data['Low'] + anchor_data['Close']) / 3 * anchor_data['Volume']
        anchor_volume = anchor_data['Volume'].sum()
        anchor_vwap = anchor_tpv.sum() / anchor_volume

        dataframe['Anchored VWAP'] = anchor_vwap

    return dataframe.drop(columns=['Cumulative TPV', 'Cumulative Volume'])  # ,'TP', 'TPV',])

def ta_volume_RelativeVolume(df, n=50):
    vol = df['Volume'].rolling(n).sum()
    rv = df['Volume'] / vol
    return rv

#need to fix
def ta_volume_StochPVO(df, n1=10, n2=20, n3=9):
    apo,i,j = ta_mo_APO(df, n1, n2)
    ema1 = ta_ma_EMA(apo, n1)
    ema2 = ta_ma_EMA(apo, n2)
    pvo = [(e1-e2)/(e2*100) for e1,e2,in zip(ema1,ema2)]
    #pvo = [(e1 - e2) / e2 * 100 for e1,e2 in zip(ema1,ema2)]
    pvo_signal = ta_ma_EMA(pvo, n3)
    pvo_hist = [p - h for p,h in zip(pvo,pvo_signal)]
    return pvo, pvo_hist, pvo_signal

def ta_volume_PVO(df, n1=12, n2=26):
    pvo = (df['Volume'].rolling(n1).mean() - df['Volume'].rolling(n2).mean()) / df['Volume'].rolling(n2).mean() * 100
    return pvo
from src.Securities.Stocks.Technical.indicators.volatility import *
# FOR A/D
def ta_volume_MoneyFlowVolume(data: pd.DataFrame):
    return ta_vol_TP(data) * data['Volume']


def ta_volume_AD(data):
    return ta_volume_MoneyFlowVolume(data).expanding().sum()


def ta_volume_VolumeMA(df, window=14):
    """
    Volume Weighted Moving Average
    :param df:
    :param window:
    :return:
    """
    df = df.copy()
    vwma = (df['Close'] * df['Volume']).rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    # print(vwma)

    # rint(df)
    #print(vwma.values)
    return vwma.values


def ta_volume_VWAP(data, period=14):
    close = data['Close']
    volume = data['Volume']
    n = len(close)
    vwap = [np.NaN]*(period-1)
    for i in range(period - 1, n):
        sum_price = 0
        sum_volume = 0
        for j in range(i - period + 1, i + 1):
            sum_price += close[j] * volume[j]
            sum_volume += volume[j]
        vwap.append(sum_price / sum_volume)
    return vwap

def ta_volume_PVI(df):
    """Computes the positive volume index (PVI)."""
    pvi = pd.Series(index=df.index)
    pvi.iloc[0] = 1000
    for i in range(1, len(df)):
        if df['Volume'].iloc[i] > df['Volume'].iloc[i - 1]:
            pvi.iloc[i] = pvi.iloc[i - 1] + (df['Close'].iloc[i] - df['Close'].iloc[i - 1]) / df['Close'].iloc[i - 1] * \
                          pvi.iloc[i - 1]
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]
    return pvi

def ta_volume_ChaikinMoneyFlow(df, period=20):
    """
    Calculates the Chaikin Money Flow (CMF) for a given dataframe and period.
    The Chaikin Money Flow (CMF) is a technical indicator used in technical analysis to measure the amount of buying and selling pressure in a stock. It is calculated using the accumulation/distribution line and is used to determine whether a stock is under accumulation (buying pressure) or distribution (selling pressure). The CMF is designed to help traders identify potential buying opportunities in stocks that are showing signs of accumulation, and potential selling opportunities in stocks that are showing signs of distribution. The indicator is named after Marc Chaikin, who developed the original concept in the 1970s.
    Parameters:
    df (DataFrame): Dataframe containing the stock's price data.
    period (int): Lookback period for the CMF calculation. Default value is 20.

    Returns:
    Series: A series containing the CMF values for each date in the dataframe.
    """
    # Calculate typical price
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    # Calculate money flow volume
    money_flow_volume = typical_price * df['Volume']
    # Calculate the accumulation/distribution line
    adl = money_flow_volume.rolling(period).sum()
    # Calculate the CMF
    cmf = (adl / adl.rolling(period).sum()).dropna()
    return cmf
def ta_volume_MFI(df, n=14):
    """
    Compute the Money Flow Index (MFI) for a dataframe representing a stock's price.
    The Money Flow Index (MFI) is a technical indicator that uses both price and volume information to measure buying and selling pressure. It is commonly used in conjunction with other technical analysis tools to identify possible trend reversals, generate buy and sell signals, and to assess market momentum. The MFI is calculated as the ratio of the typical price multiplied by volume to the sum of the positive and negative money flows over a given period. A high MFI reading can indicate that buying pressure is increasing, while a low reading can indicate selling pressure is increasing.
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe representing the stock's price data.
        Should contain at least the following columns: "high", "low", "close", "volume"
    n : int, optional (default=14)
        The number of periods for the moving average

    Returns
    -------
    pandas.Series
        A series representing the MFI for each period in the input dataframe
    """
    df = df.copy()
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    mf_pos = np.where(tp > tp.shift(1), mf, 0)
    mf_neg = np.where(tp < tp.shift(1), mf, 0)
    mf_ratio = pd.Series(mf_pos).rolling(window=n).sum() / pd.Series(mf_neg).rolling(window=n).sum()
    mfi = 100 - (100 / (1 + mf_ratio))
    return mfi

def ta_volume_VQstick(data):
    diff = data['Close']-data['Open'] * data['Volume']
    return ta_ma_SMA(diff),ta_ma_EMA(diff)


def ta_volume_VPT(df):
    df = df.copy()
    vpt = [0]
    for i in range(1, len(df)):
        prev_close = df.iloc[i - 1]['Close']
        close = df.iloc[i]['Close']
        volume = df.iloc[i]['Volume']
        vpt_value = vpt[-1] + volume * ((close - prev_close) / prev_close)
        vpt.append(vpt_value)
    df['VPT'] = vpt
    return df['VPT']

def ta_volume_ForceIndex(df, n=14):
    fi = df['Volume'] * (df['Close'] - df['Close'].shift(1))
    return pd.Series(fi.rolling(n).sum(), name='FI')
def ta_volume_ChaikinADL(df):
    df = df.copy()
    mf_multiplier = 2 * ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_multiplier * df['Volume']
    adl = mf_volume.cumsum()
    df['Chaikin_ADL'] = adl
    return df['Chaikin_ADL']

def ta_volume_KVO(df, short_period=34, long_period=55, signal_period=13):
    df['vol_trend'] = np.where(df['close'] > df['close'].shift(1), df['volume'], -df['volume'])
    df['short_ema'] = df['vol_trend'].ewm(span=short_period, min_periods=short_period).mean()
    df['long_ema'] = df['vol_trend'].ewm(span=long_period, min_periods=long_period).mean()
    df['kvo'] = (df['short_ema'] - df['long_ema']) / df['long_ema'] * 100
    df['signal'] = df['kvo'].ewm(span=signal_period, min_periods=signal_period).mean()
    return df['kvo'],df['signal']


def ta_volume_DarvasBox(df, acceleration=0.02, max_box=4):
    high = df['High']
    low = df['Low']
    close = df['Close']

    boxes = []
    box_high = 0
    box_low = 0
    box_start = 0
    box_end = 0
    box_direction = 0
    box_count = 0
    box_size = 0
    current_box_size = 0

    for i in range(len(df)):
        if i == 0:
            boxes.append([0, 0])
            continue

        if box_direction == 0:
            box_size = close[i] * acceleration
            box_high = high[i]
            box_low = low[i]
            box_start = i
            box_direction = 1
            current_box_size = box_size
        elif box_direction == 1:
            if high[i] > box_high:
                box_high = high[i]
                current_box_size = box_size * (box_high - box_low) / box_size
            elif low[i] - box_size < box_low:
                box_end = i
                box_count += 1
                box_direction = -1
                boxes.append([box_start, box_end])
        else:
            if low[i] < box_low:
                box_low = low[i]
                current_box_size = box_size * (box_high - box_low) / box_size
            elif high[i] + box_size > box_high:
                box_end = i
                box_count += 1
                box_direction = 1
                boxes.append([box_start, box_end])

        if box_count == max_box:
            break

    return boxes

def ta_volume_ArmsIndex(stock_df, period=15):
    """
    Computes the Arms Index of a stock dataframe

    Parameters:
    stock_df (pd.DataFrame): A pandas dataframe containing the stock's price data
    period (int): The number of periods to use in the calculation (default is 15)

    Returns:
    pd.Series: A pandas series containing the Arms Index
    """
    # Compute the Up Volume and Down Volume
    stock_df['Up Volume'] = np.where(stock_df['Close'] > stock_df['Open'], stock_df['Volume'], 0)
    stock_df['Down Volume'] = np.where(stock_df['Close'] < stock_df['Open'], stock_df['Volume'], 0)

    # Compute the Ratio of Up Volume to Down Volume
    up_volume_sum = stock_df['Up Volume'].rolling(period).sum()
    down_volume_sum = stock_df['Down Volume'].rolling(period).sum()
    volume_ratio = up_volume_sum / down_volume_sum

    # Compute the Ratio of Advances to Declines
    advances = ((stock_df['Close'] - stock_df['Open']).rolling(period).sum() / period).apply(
        lambda x: 1 if x > 0 else 0)
    declines = ((stock_df['Open'] - stock_df['Close']).rolling(period).sum() / period).apply(
        lambda x: 1 if x > 0 else 0)
    advance_ratio = advances.rolling(period).sum() / declines.rolling(period).sum()

    # Compute the Arms Index
    arms_index = volume_ratio / advance_ratio

    return arms_index
def ta_volume_EMV(df,n=14):
    """
    The Ease of Movement (EMV) is a technical indicator that measures the relationship between price and volume to identify the ease or difficulty of a stock's price movement. It can help traders identify potential trend reversals, breakouts, or fakeouts.
    The formula for the EMV is:
    EMV = (High + Low) / 2 - (Prior High + Prior Low) / 2) / ((Volume / 100000000) / ((High - Low)))
    :param df:
    :return:
    """
    high = df['High']
    low = df['Low']
    prior_high = df['High'].shift(1)
    prior_low = df['Low'].shift(1)
    volume = df['Volume']
    emv = ((high + low) / 2 - (prior_high + prior_low) / 2) / ((volume / 100000000) / ((high - low)))
    emv_ma = pd.Series(emv.rolling(n).mean(), name='EMV_' + str(n))
    return emv, emv_ma

def add_Volume(df):
    df['ta_volume_VEMA_10'] = ta_volume_vema(df)
    df['ta_volume_VSMA_10'] = ta_volume_vsma(df)
    df['ta_volume_VolumeRSI'] =ta_volume_volumersi(df)

    df['ta_volume_OBV']=ta_volume_OBV(df)
    df['ta_volume_PVI'] = ta_volume_PVI(df)
    df['ta_volume_NVI']=ta_volume_NVI(df)
    df['ta_volume_VWAP']=ta_volume_VWAP(df)
    df['ta_volume_BOP']=ta_volume_BOP(df)

    df['ta_volume_PVO_fast'] = ta_volume_PVO(df)
    df['ta_volume_PVO_slow'] = ta_volume_PVO(df,26,50)

    df['ta_volume_StochPVO'],df['ta_volume_StochPVO_hist'],df['ta_volume_StochPVO_signal'] = ta_volume_StochPVO(df)

    df['ta_volume_MFI'] = ta_volume_MFI(df)
    df['ta_volume_MFV']=ta_volume_MoneyFlowVolume(df)
    df['ta_volume_AD']=ta_volume_AD(df)


    df['ta_volume_RelativeVolume_50']=ta_volume_RelativeVolume(df)

    df['ta_volume_ChaikinMoneyFlow']=ta_volume_ChaikinMoneyFlow(df)

    df['ta_volume_VolumeFlowIndicator']=ta_volume_VolumeFlowIndicator(df)

    df['ta_volume_VolumeMA'] = ta_volume_VolumeMA(df)

    df['ta_volume_VQstick_SMA'], df['ta_volume_VQstick_EMA'] = ta_mo_Qstick(df)

    df['ta_volume_ChaikinADL']=ta_volume_ChaikinADL(df)
    df['ta_volume_VPT']=ta_volume_VPT(df)
    df['ta_volume_EMV'],df['ta_volume_EMVMA']=ta_volume_EMV(df)

    df['ta_volume_ForceIndex']=ta_volume_ForceIndex(df)
    # Cannot Graph other ones
    cols = df.columns


    #df['ta_volume_VolumeByPrice_1']=volume_by_price(df,1)
    #df['ta_volume_VolumeByPrice_3'] = volume_by_price(df, 3)
    #df['ta_volume_VolumeByPrice_5'] = volume_by_price(df, 5)
    #df['ta_volume_VolumeByPrice_10'] = volume_by_price(df, 10)
    #df['ta_volume_VolumeByPrice_30'] = volume_by_price(df, 30)
    #df['ta_volume_VolumeByPrice_60'] = volume_by_price(df, 60)
    #df['ta_volume_VolumeByPrice_90'] = volume_by_price(df, 90)

    #ncols = [c for c in list(df.columns) if c not in cols]

    # For now
    return df#,cols,ncols

'''
df = yf.Ticker('AAPL').history(period='MAX')
print(df.columns)
df= add_volume(df)
print(df.dtypes)

for n in ncols:
    print(df[n].unique())
import plotly.express  as px
fig=px.line(df[cols])

import plotly.express  as px
fig=px.line(df)
fig.show()
'''
