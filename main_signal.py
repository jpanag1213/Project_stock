# -*- coding: utf-8 -*-
"""
Created on 2019-01-08

to test whether the signal is useful

@author: zhixiong

"""
import numpy as np
import Data
import SignalTester
import pandas as pd
import os
import configparser
from Utils import *
import Strategy

def run(configfile):

    # dataPath = 'E:/data/stock/wind'
    # symbols = ['000001.SZ']
    # exchange = symbol.split('.')[1].lower()
    symbols, dataPath, dataReadType, tradingDays, outputpath, signal, lbwindow, lawindow = ConfigReader(configfile)
    for tradingDay in tradingDays:
        tradingDay = tradingDay.replace('-','')
        print('Processing tradingday = ', tradingDay)
        tradingSymbols = list(symbols[list(
            map(lambda symbol: CheckStockSuspend(symbol, dataPath, tradingDay),
                symbols.values.tolist()))])
        if len(tradingSymbols) == 0:
            continue
        data = Data.Data(dataPath, tradingSymbols, tradingDay,dataReadType= dataReadType, RAWDATA = 'True')
        signalTester = SignalTester.SignalTester(data, dailyData=pd.DataFrame, tradeDate=tradingDay, symbol=tradingSymbols, dataSavePath=outputpath)
        # signalTester.CompareSectorAndStock(symbols[0], orderType='netMainOrderCashFlow')
        stsDf = list()
        strategyResult = list()
        for symbol in tradingSymbols:
            stsDf.append(signalTester.CheckSignal(symbol,signal,lbwindow,lawindow))
            quoteData = data.quoteData[symbol]
            strategy = Strategy.Strategy(symbol, round(1000000/quoteData['midp'].iloc[-1],-2), quoteData, signal, lbwindow, lawindow, 8, 'lawindow', outputpath = './strategy/' + tradingDay, stockType = 'low')
            strategy.SummaryStrategy()
            strategy.Plot()
            strategyResult.append(strategy.sts)


        pd.concat(stsDf,0).to_csv(outputpath+'./' + tradingDay + '.csv')
        pd.concat(strategyResult,0).to_csv('./strategy/' + tradingDay + '.csv')

    return 0

def CalculatreHisData(configfile):
    symbols, dataPath, dataReadType, tradingDays, outputpath, signal, lbwindow, lawindow = ConfigReader(configfile)
    stdDf = pd.DataFrame(columns=symbols,index=tradingDays)
    for tradingDay in tradingDays:
        tradeDate = tradingDay.replace('-','')
        print('Processing tradingday = ', tradeDate)
        tradingSymbols = list(symbols[list(
            map(lambda symbol: CheckStockSuspend(symbol, dataPath, tradeDate),
                symbols.values.tolist()))])
        data = Data.Data(dataPath, tradingSymbols, tradeDate,dataReadType= dataReadType, RAWDATA = 'True')
        signalTester = SignalTester.SignalTester(data, dailyData=pd.DataFrame, tradeDate=tradeDate, symbol=tradingSymbols, dataSavePath=outputpath)
        # signalTester.CompareSectorAndStock(symbols[0], orderType='netMainOrderCashFlow')
        stsDf = list()
        for symbol in tradingSymbols:

            signalTester.CalSignal(symbol,0,'obi',lbwindow)
            # df2save = pd.DataFrame(signalTester.allQuoteData[symbol][signal])
            # df2save.columns = [symbol]
            stds = signalTester.allQuoteData[symbol][signal].std()
            stdDf.loc[tradingDay,symbol] = stds

        stdDf.to_csv(outputpath + '/' + signal + '_std.csv')

    # emptyDf = pd.DataFrame({},index= tradingDays)
    # allSymbolResult = list()
    # for symbol in symbols:
    #
    #     stockResult = list()
    #     for tradingDay in tradingDays:
    #         tradingDay = tradingDay.replace('-', '')
    #         stsDf = dailyResult[tradingDay]
    #         if symbol in stsDf.columns:
    #             signalDf = stsDf[symbol]
    #             signalDf = pd.DataFrame(signalDf)
    #             signalDf.columns = [tradingDay]
    #             stockResult.append(stockResult)
    #     stockDf = pd.concat(stockResult,1)
    #     stds = pd.DataFrame(stockDf.std(1)).T
    #     allSymbolResult.append(stds)
    #
    # allSymbolResult.append(emptyDf)
    # pd.concat(allSymbolResult,0).to_csv('test.csv')

    return 0

def SummaryResult(configfile):
    symbols, dataPath, dataReadType, tradingDays, outputpath, signal, lbwindow, lawindow = ConfigReader(configfile)
    # stdDf = pd.DataFrame(columns=symbols, index=tradingDays)
    stsDfList = list()
    for tradingDay in tradingDays:
        tradingDay = tradingDay.replace('-', '')
        file2read = outputpath+'./' + tradingDay + '.csv'
        if os.path.exists(file2read):
            stsDf = pd.read_csv(outputpath+'./' + tradingDay + '.csv',index_col=0)
        else:
            continue
        stsDf.loc[:,'date'] = tradingDay
        stsDfList.append(stsDf)

    stsDf = pd.concat(stsDfList,0)
    summaryList = pd.DataFrame(columns = ['total_days','wr_days','pnl','average_times','wr'])
    for symbol in symbols:
        symbolResult = stsDf.loc[symbol,:]
        totay_days = symbolResult.shape[0]
        WR_days = (symbolResult.loc[:,'WR_excost'] > 0.55).sum()
        wr = (symbolResult['pnl']>0).sum() / totay_days
        pnl = symbolResult['pnl'].sum()
        average_times = symbolResult['times'].mean()
        summaryList.loc[symbol, 'total_days'] = totay_days
        summaryList.loc[symbol, 'wr_days'] = WR_days
        summaryList.loc[symbol, 'pnl'] = pnl
        summaryList.loc[symbol, 'average_times'] = average_times
        summaryList.loc[symbol, 'wr'] = wr

    summaryList.to_csv('summary.csv')


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

    return symbols, dataPath, dataReadType, tradingDays, outputpath, signal, lbwindow, lawindow

if __name__ == '__main__':
    """
    this script is the main function to see whether the signal is useful
    """
    # SummaryResult('./configs/signal_test.txt')
    # CalculatreHisData('./configs/signal_test.txt')
    run('./configs/signal_test.txt')