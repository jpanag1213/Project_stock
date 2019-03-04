# -*- coding: utf-8 -*-
"""
Created on 2019-03-24

#factor library
@author: jiaxiong

"""

import numpy as np
import Data
import pandas as pd
import os
import configparser
import time
from Utils import *
import matplotlib.pyplot as plt
import datetime

class SignalLibrary(object):

    def __init__(self, symbol, quoteData,signal,tradeData = None, outputpath = 'E://stats_test/'):
        self.symbol    = symbol
        self.allQuoteData = quoteData
        self.outputpath = outputpath
        self.tradeData =  tradeData
        self.signal = signal

    def getSignal(self):
        print(self.signal)
        signal = getattr(SignalLibrary,self.signal)
        #print(signal(self))
        return signal(self)


    def obi_demo(self):
        window = 20
        signal = self.signal
        symbol = self.symbol
        self.allQuoteData.loc[:, 'obi'] = np.log(self.allQuoteData.loc[:, 'bidVolume1']) - np.log(
            self.allQuoteData.loc[:, 'askVolume1'])

        self.allQuoteData.loc[:, 'obi1'] = np.log(self.allQuoteData .loc[:, 'bidVolume1'] +
                                                          self.allQuoteData .loc[:, 'bidVolume2']) - np.log(
            self.allQuoteData .loc[:, 'askVolume1'])

        self.allQuoteData .loc[:, 'obi2'] = np.log(self.allQuoteData .loc[:, 'bidVolume1']) - np.log(
            self.allQuoteData .loc[:, 'askVolume1'] +
            self.allQuoteData .loc[:, 'askVolume2'])
        # self.allQuoteData .loc[:, 'obi_' + str(window) + '_min'] = self.allQuoteData .loc[:,
        #                                                                   'obi'].rolling(window * 60).mean()
        self.allQuoteData .loc[:, 'obi_' + str(window) + '_min'] = self.allQuoteData .loc[:, 'obi'].diff(
            window)

        askPriceDiff = self.allQuoteData ['askPrice1'].diff()
        bidPriceDiff = self.allQuoteData ['bidPrice1'].diff()
        midPriceChange = self.allQuoteData ['midp'].diff()

        self.allQuoteData .loc[:, 'priceChange'] = 1
        self.allQuoteData .loc[midPriceChange == 0, 'priceChange'] = 0

        obi_change_list = list()
        last_obi = self.allQuoteData ['obi'].iloc[0]
        tick_count = 0
        row_count = 0
        for row in zip(self.allQuoteData ['priceChange'], self.allQuoteData ['obi']):
            priceStatus = row[0]
            obi = row[1]
            if (priceStatus == 1) or np.isnan(priceStatus):
                tick_count = 0
                last_obi = obi
            else:
                last_obi = self.allQuoteData ['obi'].iloc[row_count - tick_count]
                if tick_count <= window:
                    tick_count = tick_count + 1

            row_count = row_count + 1
            obi_change = obi - last_obi
            obi_change_list.append(obi_change)

        self.allQuoteData .loc[:, 'obi'] = obi_change_list
        positivePos = (self.allQuoteData ['obi2'] > 1)
        negativePos =  (self.allQuoteData ['obi1'] < -1)
        self.allQuoteData .loc[positivePos, signal + '_' + str(window) + '_min'] = 1
        self.allQuoteData .loc[negativePos, signal + '_' + str(window) + '_min'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), signal + '_' + str(window) + '_min'] = 0
        print(signal + '_' + str(window))
        return self.allQuoteData



    def signal1(self):

        return 0

    def signal2(self):
        signal = self.signal
        window = self.window
        symbol = self.symbol
        if signal == 'volumeRatio':
            last5days = self.GetLast5Days()
            last5volume = self.dailyData.loc[last5days, self.symbol]
            volumePerMin = last5volume.sum() / float(240 * 5)
            self.allQuoteData  .loc[:, 'curMinute'] = list(
                map(lambda targetTime: self.CalculateTimeDiff(targetTime) / float(60), self.allQuoteData  .index))
            self.allQuoteData  .loc[:, 'volumeRatio_' + str(window) + '_min'] = self.allQuoteData[
                                                                                          symbol].tradeVolume / \
                                                                                      self.allQuoteData  [
                                                                                          'curMinute'] / volumePerMin
        return self.allQuoteData













if __name__ == '__main__':
    dataPath = '//192.168.0.145/data/stock/wind'
    ## /sh201707d/sh_20170703
    tradeDate = '20190226'

    symbols = ['000001.SZ']
    # exchange = symbol.split('.')[1].lower()
    #print(dataPath)
    data = Data.Data(dataPath,symbols, tradeDate,'' ,dataReadType= 'gzip', RAWDATA = 'True')
    SignalLibrary   = SignalLibrary(symbols[0], data.quoteData,signal = 'obidemo')
    #SignalLibrary.getSignal()
    print(Data)




