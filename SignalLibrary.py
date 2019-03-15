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

    def __init__(self, symbol, quoteData,signal,tradeData = None,window = None, outputpath = 'E://stats_test/'):
        self.symbol    = symbol
        self.allQuoteData = quoteData
        self.outputpath = outputpath
        self.tradeData =  tradeData
        self.signal = signal
        self.window = window

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



    def signal_obi(self):

        return self.allQuoteData

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

    def obi_extreme(self):
        symbol = self.symbol
        signal = self.signal
        window = self.window
        midp = self.allQuoteData  .loc[:, 'midp']
        quotedata = self.allQuoteData  
        bid_Volume10 = (quotedata.loc[:, 'bidVolume1'] + quotedata.loc[:, 'bidVolume2'] + quotedata.loc[:,
                                                                                          'bidVolume3']) * 1 / 10
        ask_Volume10 = (quotedata.loc[:, 'askVolume1'] + quotedata.loc[:, 'askVolume2'] + quotedata.loc[:,
                                                                                          'askVolume3']) * 1 / 10
        bid_Volume10_2 = (quotedata.loc[:, 'bidVolume1'] + quotedata.loc[:, 'bidVolume2'])
        ask_Volume10_2 = (quotedata.loc[:, 'askVolume1'] + quotedata.loc[:, 'askVolume2'])
        bid_price = (bid_Volume10 < quotedata.loc[:, 'bidVolume1']) + 2 * (
                    (bid_Volume10 > quotedata.loc[:, 'bidVolume1']) & (bid_Volume10 < bid_Volume10_2))
        ask_price = (ask_Volume10 < quotedata.loc[:, 'askVolume1']) + 2 * (
                    (ask_Volume10 > quotedata.loc[:, 'askVolume1']) & (ask_Volume10 < ask_Volume10_2))
        quotedata.loc[:, 'bid_per10'] = quotedata.loc[:, 'bidPrice1']
        quotedata.loc[:, 'ask_per10'] = quotedata.loc[:, 'askPrice1']
        quotedata.loc[:, 'bid_vol10'] = quotedata.loc[:, 'bidVolume1']
        quotedata.loc[:, 'ask_vol10'] = quotedata.loc[:, 'askVolume1']

        quotedata.loc[bid_price == 2, 'bid_per10'] = quotedata.loc[bid_price == 2, 'bidPrice2']
        quotedata.loc[bid_price == 0, 'bid_per10'] = quotedata.loc[bid_price == 0, 'bidPrice3']
        quotedata.loc[ask_price == 2, 'ask_per10'] = quotedata.loc[ask_price == 2, 'askPrice2']
        quotedata.loc[ask_price == 0, 'ask_per10'] = quotedata.loc[ask_price == 0, 'askPrice3']
        quotedata.loc[quotedata.loc[:, 'ask_per10'] == 0, 'ask_per10'] = np.nan
        quotedata.loc[quotedata.loc[:, 'bid_per10'] == 0, 'bid_per10'] = np.nan
        quotedata.loc[bid_price == 2, 'bid_vol10'] = quotedata.loc[bid_price == 2, 'bidVolume2']
        quotedata.loc[bid_price == 0, 'bid_vol10'] = quotedata.loc[bid_price == 0, 'bidVolume3']
        quotedata.loc[ask_price == 2, 'ask_vol10'] = quotedata.loc[ask_price == 2, 'askVolume2']
        quotedata.loc[ask_price == 0, 'ask_vol10'] = quotedata.loc[ask_price == 0, 'askVolume3']

        midp_2 = (quotedata.loc[:, 'ask_per10'] * quotedata.loc[:, 'bid_vol10'] + quotedata.loc[:,
                                                                                  'bid_per10'] * quotedata.loc[:,
                                                                                                 'ask_vol10']) / (
                             quotedata.loc[:, 'bid_vol10'] + quotedata.loc[:, 'ask_vol10'])
        self.allQuoteData  .loc[:, 'midp_2'] = midp_2
        # midp = (quotedata.loc[:, 'askPrice1'] * quotedata.loc[:, 'bidVolume1'] + quotedata.loc[:, 'bidPrice1'] * quotedata.loc[:,'askVolume1']) / (quotedata.loc[:, 'bidVolume1'] + quotedata.loc[:, 'askVolume1'])
        mean_midp = midp_2.rolling(20).mean()
        Minute = 6
        ewm_midp = mean_midp.ewm(6 * 20).mean()

        fig, ax = plt.subplots(1, figsize=(20, 12), sharex=True)

        mean_midp_ = midp.rolling(20).mean()
        ewm_midp_ = mean_midp_.ewm(6 * 20).mean()

        not_point = list()
        kp1_point = list()
        kp2_point = list()
        kp_1 = 0
        kp_2 = 0
        id_1 = 0
        id_2 = 0
        count = 0
        std_ = mean_midp.ewm(6 * 20).std()
        # std = mean_midp.ewm(M2* T)
        STATE_test = list()
        kp_list = list()
        for row in zip(ewm_midp, std_):
            count = count + 1
            i = row[0]
            j = row[1]
            if i is not np.nan:

                if (kp_1 != 0) & (kp_2 != 0) & (kp_1 == kp_1) & (kp_2 == kp_2):

                    kp_diff = kp_1 - kp_2
                    if (kp_diff * (i - kp_2) < 0):

                        if ((abs(i - kp_2)) > 4 * j):
                            # print(id_2-id_1)
                            not_point.append(i)
                            kp_1 = kp_2
                            kp_2 = i
                            id_1 = id_2
                            id_2 = count
                            STATE_test.append(3)
                        else:
                            not_point.append(kp_2)
                            STATE_test.append(2)
                    else:
                        kp_2 = i
                        id_2 = count
                        not_point.append(kp_1)
                        STATE_test.append(1)

                else:
                    not_point.append(np.nan)
                    kp_1 = i
                    kp_2 = i
                    id_1 = count
                    id_2 = count
                    STATE_test.append(0)

                kp1_point.append(kp_1)
                kp2_point.append(kp_2)

            else:
                not_point.append(np.nan)
                kp1_point.append(np.nan)
                kp2_point.append(np.nan)
                STATE_test.append(np.nan)

        self.allQuoteData  .loc[:, 'ewm'] = ewm_midp_
        self.allQuoteData  .loc[:, 'filter_ewm'] = ewm_midp
        self.allQuoteData  .loc[:, 'not'] = not_point
        # self.allQuoteData  .loc[:,'kp_1'] = kp1_point
        # self.allQuoteData  .loc[:,'kp_2'] = kp2_point

        self.allQuoteData  .loc[:, 'std_'] = std_
        # self.allQuoteData  .loc[:,'std_'] = std_
        # self.allQuoteData  .loc[:,'state'] = STATE_test

        self.allQuoteData  .loc[:, 'upper_bound'] = self.allQuoteData  .loc[:, 'not'] + 3 * \
                                                          self.allQuoteData  .loc[:, 'std_']
        self.allQuoteData  .loc[:, 'lower_bound'] = self.allQuoteData  .loc[:, 'not'] - 3 * \
                                                          self.allQuoteData  .loc[:, 'std_']
        # negativePos = (self.allQuoteData  .loc[:,'ewm']> (self.allQuoteData  .loc[:,'not'] +3*self.allQuoteData  .loc[:,'std_']))&(self.allQuoteData  .loc[:,'ewm'].shift(-1) <(self.allQuoteData  .loc[:,'not'].shift(-1) + 3*self.allQuoteData  .loc[:,'std_'].shift(-1)))
        # negativePos = (self.allQuoteData  .loc[:,'ewm'].shift(1) > (self.allQuoteData  .loc[:,'not'].shift(1)  +3*self.allQuoteData  .loc[:,'std_'].shift(1) ))&(self.allQuoteData  .loc[:,'ewm'] <(self.allQuoteData  .loc[:,'not'] + 3*self.allQuoteData  .loc[:,'std_']))

        positivePos = (self.allQuoteData  .loc[:, 'ewm'].shift(1) < (
                    self.allQuoteData  .loc[:, 'not'].shift(1) + 3 * self.allQuoteData  .loc[:,
                                                                           'std_'].shift(1))) & (
                                  self.allQuoteData  .loc[:, 'ewm'] > (
                                      self.allQuoteData  .loc[:, 'not'] + 3 * self.allQuoteData  .loc[:,
                                                                                    'std_']))
        # positivePos = (self.allQuoteData  .loc[:,'ewm']< (self.allQuoteData  .loc[:,'not'] -3*self.allQuoteData  .loc[:,'std_']))&(self.allQuoteData  .loc[:,'ewm'].shift(-1)>(self.allQuoteData  .loc[:,'not'].shift(-1) - 3*self.allQuoteData  .loc[:,'std_'].shift(-1)))
        # positivePos = (self.allQuoteData  .loc[:,'ewm'].shift(1) < (self.allQuoteData  .loc[:,'not'].shift(1) -3*self.allQuoteData  .loc[:,'std_'].shift(1) ))&(self.allQuoteData  .loc[:,'ewm']>(self.allQuoteData  .loc[:,'not'] - 3*self.allQuoteData  .loc[:,'std_']))
        negativePos = (self.allQuoteData  .loc[:, 'ewm'].shift(1) > (
                    self.allQuoteData  .loc[:, 'not'].shift(1) - 3 * self.allQuoteData  .loc[:,
                                                                           'std_'].shift(1))) & (
                                  self.allQuoteData  .loc[:, 'ewm'] < (
                                      self.allQuoteData  .loc[:, 'not'] - 3 * self.allQuoteData  .loc[:,
                                                                                    'std_']))

        self.allQuoteData  .loc[positivePos, signal + '_' + str(window) + '_min'] = 1
        self.allQuoteData  .loc[negativePos, signal + '_' + str(window) + '_min'] = -1
        # self.allQuoteData  .to_csv(self.dataSavePath + './' + str(self.tradeDate.date()) + signal + ' ' + symbol + 'quote_order.csv')
        self.allQuoteData  .loc[(~positivePos) & (~negativePos), signal + '_' + str(window) + '_min'] = 0
        return self.allQuoteData  

    def obi_change(self):
        lb_window = 20
        change_window = 10
        diff_window = 3
        signal = self.signal
        askPriceDiff = self.allQuoteData ['askPrice1'].diff()
        bidPriceDiff = self.allQuoteData ['bidPrice1'].diff()
        midPriceChange = self.allQuoteData ['midp'].diff()
        self.allQuoteData.loc[:, 'obi'] = np.log(self.allQuoteData.loc[:, 'bidVolume1']) - np.log(
            self.allQuoteData.loc[:, 'askVolume1'])

        # self.allQuoteData.loc[:, 'obi1'] = np.log(self.allQuoteData .loc[:, 'bidVolume1'] +
        #                                                   self.allQuoteData .loc[:, 'bidVolume2']) - np.log(
        #     self.allQuoteData .loc[:, 'askVolume1'])
        #
        # self.allQuoteData .loc[:, 'obi2'] = np.log(self.allQuoteData .loc[:, 'bidVolume1']) - np.log(
        #     self.allQuoteData .loc[:, 'askVolume1'] +
        #     self.allQuoteData .loc[:, 'askVolume2'])
        self.allQuoteData.loc[:, 'obi_' + str(lb_window) + '_min'] = self.allQuoteData .loc[:, 'obi'].diff()
        self.allQuoteData.loc[:,'bid_volume_occupy'] = self.allQuoteData.loc[:,'bidVolume1']/self.allQuoteData.loc[:,list(map(lambda x:'bidVolume' + str(x),np.arange(1,11,1)))].sum(1)
        self.allQuoteData.loc[:,'ask_volume_occupy'] = self.allQuoteData.loc[:,'askVolume1']/self.allQuoteData.loc[:,list(map(lambda x:'askVolume' + str(x),np.arange(1,11,1)))].sum(1)
        self.allQuoteData.loc[:,'mid_quote_up'] = (midPriceChange > 0).shift(diff_window).rolling(change_window).sum()  # 用于记录过去tick涨的次数
        self.allQuoteData.loc[:,'mid_quote_down'] = (midPriceChange < 0).shift(diff_window).rolling(change_window).sum()  # 用于记录过去tick跌的次数


        self.allQuoteData .loc[:, 'priceChange'] = 1
        self.allQuoteData .loc[midPriceChange == 0, 'priceChange'] = 0

        obi_change_list = list()
        last_obi = self.allQuoteData ['obi'].iloc[0]
        tick_count = 0
        row_count = 0
        for row in zip(self.allQuoteData ['mid_quote_up'],self.allQuoteData ['mid_quote_down'], range(self.allQuoteData.shape[0])):
            priceStatus_up = row[0]
            priceStatus_down = row[1]
            row_count = row[2]
            if np.isnan(priceStatus_up) or (np.isnan(priceStatus_down)):
                obi_change = 0
            elif (priceStatus_up/change_window) >= 0.5:  # 意味着有上涨趋势，检查obi是否增加，且检查之前的挂单量是否符合预期
                # VOLUME_BIG = self.allQuoteData['ask_volume_occupy'].iloc[(row_count - diff_window):row_count].sum() > 0.1
                VOLUME_BIG = True
                # OBI_CHANGE_BIG = self.allQuoteData['obi_' + str(lb_window) + '_min'].iloc[(row_count - diff_window):row_count].sum() > 2
                OBI_CHANGE_BIG = self.allQuoteData['obi'].iloc[row_count] - self.allQuoteData['obi'].iloc[row_count - diff_window] > 4
                if VOLUME_BIG & OBI_CHANGE_BIG:
                    obi_change = 1
                else:
                    obi_change = 0
            elif (priceStatus_down/change_window) >= 0.5:
                # VOLUME_BIG = self.allQuoteData['bid_volume_occupy'].iloc[(row_count - diff_window):row_count].sum() > 0.1
                VOLUME_BIG = True
                OBI_CHANGE_BIG = self.allQuoteData['obi'].iloc[row_count] - self.allQuoteData['obi'].iloc[row_count - diff_window] < -4
                if VOLUME_BIG & OBI_CHANGE_BIG:
                    obi_change = -1
                else:
                    obi_change = 0

            else:
                obi_change = 0

            obi_change_list.append(obi_change)

        self.allQuoteData .loc[:, 'obi'] = obi_change_list
        positivePos = (self.allQuoteData ['obi'] >= 1)
        negativePos =  (self.allQuoteData ['obi'] <= -1)

        self.allQuoteData .loc[positivePos, signal + '_' + str(lb_window) + '_min'] = 1
        self.allQuoteData .loc[negativePos, signal + '_' + str(lb_window) + '_min'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), signal + '_' + str(lb_window) + '_min'] = 0
        print(signal + '_' + str(lb_window))
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




