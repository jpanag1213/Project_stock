import numpy as np
import Data
import SignalTester
import pandas as pd
import os
import configparser
from Utils import *
import Strategy
import time
def run(configfile):

    # dataPath = 'E:/data/stock/wind'
    # symbols = ['000001.SZ']
    # exchange = symbol.split('.')[1].lower()
    symbols, dataPath, dataReadType, tradingDays, outputpath, signal, lbwindow, lawindow,paraset= ConfigReader(configfile)
    for tradingDay in tradingDays:
        tradingDay = tradingDay.replace('-','')
        print('Processing tradingday = ', tradingDay)
        tradingSymbols = list(symbols[list(
            map(lambda symbol: CheckStockSuspend(symbol, dataPath, tradingDay),
                symbols.values.tolist()))])
        if len(tradingSymbols) == 0:
            continue
        data = Data.Data(dataPath, tradingSymbols, tradingDay,dataReadType= dataReadType, RAWDATA = 'True')

        for symbol in tradingSymbols[0]:
            quote = data.queueData
            print(quote)




    return 0

def ConfigReader(configFile):

    global dailyDataFile
    global dataReadType

    parser = configparser.ConfigParser()
    parser.read(configFile)

    symbolfile          = parser.get('DEFAULT', 'symbolfile')
    tradingDayFile      = parser.get('DEFAULT', 'tradingDayFile')
    dailyDataFile       = parser.get('DEFAULT', 'dailyDataFile')
    dataPathCsv         = parser.get('DEFAULT', 'dataPathCsv')
    dataPathGzip        = parser.get('DEFAULT', 'dataPathGzip')
    dataReadType        = parser.get('DEFAULT', 'dataReadType')
    mainFutureFile      = parser.get('DEFAULT', 'mainFutureFile')
    outputpath          = parser.get('DEFAULT', 'outputpath')

    signal              = parser.get('Signal','signal')
    lbwindow            = int(parser.get('Signal','lbwindow'))
    lawindow            = int(parser.get('Signal','lawindow'))
    startDate           = parser.get('Signal','startDate')
    endDate             = parser.get('Signal','endDate')
    paraset             = parser.get('Signal','paraset')

    paraset = paraset.split('-')
    print(paraset)
    tradingDays     = pd.read_csv(tradingDayFile, index_col=0, parse_dates=True)
    # main_future     = pd.read_csv(mainFutureFile,index_col=0,parse_dates=True)
    symbols = pd.read_csv(symbolfile,encoding='oem').loc[:,'secucode']

    # lastTradeDate = tradingDays.loc[tradingDays['date'] <= rundt, 'date'].iloc[-2]
    # nextTradeDate = tradingDays.loc[tradingDays['date'] >= rundt, 'date'].iloc[1]

    tradingDays = tradingDays.loc[(tradingDays['date'] >= startDate) & (tradingDays['date'] <= endDate), 'date']

    if dataReadType == 'gzip':
        dataPath = dataPathGzip
    elif dataReadType == 'csv':
        dataPath = dataPathCsv
    else:
        dataPath = ''

    return symbols, dataPath, dataReadType, tradingDays, outputpath, signal, lbwindow, lawindow,paraset

if __name__ == '__main__':
    """
    this script is the main function to see whether the signal is useful
    """
    # SummaryResult('./configs/signal_test.txt')
    # CalculatreHisData('./configs/signal_test.txt')

    t1 =  time.clock()
    run('./configs/signal_test.txt')
    t2 = time.clock()
    print(t2-t1)