import warnings

import pandas as pd
import tqdm

from Indicators.momentum import add_Momentum
from Indicators.volume import add_Volume
from Indicators.movingAverages import add_MovingAverages
from Indicators.volatility import add_Volatility
from Indicators.trend import add_Trends

import yfinance

warnings.filterwarnings("ignore")

def add_AllIndicators(df,vol=True):
    if vol:
        df = add_Volume(df)
    #print(df.columns)
    df = add_MovingAverages(df)
    #print(df.columns)
    df = add_Momentum(df)
    #print(df.columns)
    # When is the column High getting added, or why does it have 2 values ??
    # IS it coming from momentum?
    # Need to find
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = add_Trends(df)
    #print(df.columns)
    df = add_Volatility(df)
    #print(df.columns)
    #print(list(df.columns))
    #print(len(list(df.columns)))
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    a=list(df.columns)
    try:
        a.remove(0)
    except:
        pass
    df = df[a]
    df = df.select_dtypes(include=numerics)
    #import plotly.express  as px
    #fig = px.line(df)
    #fig.show()
    return df

t_stocks=[]
prc_file = r"E:\AWT\Finance\AWT(2.0)\MicroStocks\{}\yfin_prc_hst.csv"
file_ta = r"E:\AWT\Finance\AWT(2.0)\MicroStocks\{}\yfin_tech_hst.csv"
'''

from Constants.securityConstants import complete_stocks
def getStocks(start):
    if start == 23:
        lst = complete_stocks[start * len(complete_stocks) // 23:len(complete_stocks)]
    else:
        lst = complete_stocks[start * len(complete_stocks) // 23:len(complete_stocks) // 23 * (start + 1)]

    for c in tqdm.tqdm(lst):
        df =  pd.read_csv(prc_file.format(c),index_col=0)
        try:
            df = add_AllIndicators(df)
            df.to_csv(file_ta.format(c))
            t_stocks.append(c)
        except:
            try:
                df = add_AllIndicators(df,vol=False)
                df.to_csv(file_ta.format(c))
                t_stocks.append(c)
            except:

                continue
import threading

t0 = threading.Thread(target=getStocks, args=(0,))
t1 = threading.Thread(target=getStocks, args=(1,))
t2 = threading.Thread(target=getStocks, args=(2,))
t3 = threading.Thread(target=getStocks, args=(3,))
t4 = threading.Thread(target=getStocks, args=(4,))
t5 = threading.Thread(target=getStocks, args=(5,))
t6 = threading.Thread(target=getStocks, args=(6,))
t7 = threading.Thread(target=getStocks, args=(7,))
t8 = threading.Thread(target=getStocks, args=(8,))
t9 = threading.Thread(target=getStocks, args=(9,))
t10 = threading.Thread(target=getStocks, args=(10,))
t11 = threading.Thread(target=getStocks, args=(11,))
t12 = threading.Thread(target=getStocks, args=(12,))
t13 = threading.Thread(target=getStocks, args=(13,))
t14 = threading.Thread(target=getStocks, args=(14,))
t15 = threading.Thread(target=getStocks, args=(15,))
t16 = threading.Thread(target=getStocks, args=(16,))
t17 = threading.Thread(target=getStocks, args=(17,))
t18 = threading.Thread(target=getStocks, args=(18,))
t19 = threading.Thread(target=getStocks, args=(19,))
t20 = threading.Thread(target=getStocks, args=(20,))
t21 = threading.Thread(target=getStocks, args=(21,))
t22 = threading.Thread(target=getStocks, args=(22,))
t23 = threading.Thread(target=getStocks, args=(23,))

t0.start()
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()
t9.start()
t10.start()
t11.start()
t12.start()
t13.start()
t14.start()
t15.start()
t16.start()
t17.start()
t18.start()
t19.start()
t20.start()
t21.start()
t22.start()
t23.start()

t0.join()
t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()
t8.join()
t9.join()
t10.join()
t11.join()
t12.join()
t13.join()
t14.join()
t15.join()
t16.join()
t17.join()
t18.join()
t19.join()
t20.join()
t21.join()
t22.join()
t23.join()
'''
print(t_stocks)