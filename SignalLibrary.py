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
import stats
from numba import jit

class SignalLibrary(object):

    def __init__(self, symbol, quoteData,signal,tradeData = None,window = None, outputpath = 'E://stats_test/'):
        self.symbol    = symbol
        quoteData = quoteData[~quoteData.index.duplicated(keep='first')]
        self.allQuoteData = quoteData
        self.outputpath = outputpath
        self.tradeData =  tradeData
        self.signal = signal
        self.window = window
        self.Stats = stats.Stats(symbol,None,quoteData,tradeData)



    def getSignal(self):
        #print(self.signal)
        signal = getattr(SignalLibrary,self.signal)
        #print(signal(self))
        return signal(self)


    def obi_demo(self):
        window = 20
        windows = 2000
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
        atb_rate_list = list()
        ata_rate_list = list()
        last_obi = self.allQuoteData ['obi'].iloc[0]
        tick_count = 0
        row_count = 0
        av_count = np.nan
        bv_count = np.nan
        active_ask =  np.nan
        active_bid=  np.nan
        for row in zip(self.allQuoteData ['priceChange'], self.allQuoteData ['obi'], self.allQuoteData ['askVolume1'], self.allQuoteData ['bidVolume1']):
            priceStatus = row[0]
            obi = row[1]
            av = row[2]
            bv = row[3]

            if (priceStatus == 1) or np.isnan(priceStatus):
                tick_count = 0
                last_obi = obi
                av_count = av
                bv_count = bv
            else:
                active_ask = av - av_count

                active_bid = bv - bv_count
                #print(active_bid)
                last_obi = self.allQuoteData ['obi'].iloc[row_count - tick_count]
                if tick_count <= windows:
                    tick_count = tick_count + 1

            row_count = row_count + 1
            obi_change = obi - last_obi
            if (active_bid - active_ask)>0:
                ata_rate = active_bid / (active_bid - active_ask)
            else:
                ata_rate = 0
            if  (active_ask - active_bid) >0:
                atb_rate = float(active_ask )/ (active_ask - active_bid)
            else:
                atb_rate = 0
            #print(active_bid)
            #print((active_ask - active_bid))
            obi_change_list.append(obi_change)
            atb_rate_list.append(atb_rate)
            ata_rate_list.append(ata_rate)

        self.allQuoteData .loc[:, 'obi'] = obi_change_list
        self.allQuoteData .loc[:, 'atb_rate'] = atb_rate_list
        self.allQuoteData .loc[:, 'ata_rate'] = ata_rate_list
        positivePos = (self.allQuoteData ['obi'] > 2) &(self.allQuoteData .loc[:, 'atb_rate']>0.5)
        negativePos =  (self.allQuoteData ['obi'] < -2) &(self.allQuoteData .loc[:, 'ata_rate']>0.5)
        self.allQuoteData .loc[positivePos, signal + '_' + str(window) + '_min'] = 1
        self.allQuoteData .loc[negativePos, signal + '_' + str(window) + '_min'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), signal + '_' + str(window) + '_min'] = 0
        #print(signal + '_' + str(window))
        self.allQuoteData.to_csv(self.outputpath+ signal+'_.csv')

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
        #print(signal + '_' + str(lb_window))
        return self.allQuoteData



    def obi_test(self):
        ### 20190318
        ###下单点位全歪了
        ### 衡量了交易流的异常点。（其实就是通过std 和mean来衡量大单，相对于当前状态的）
        ### 计算了突破重要价格点所用的order量，这里重要价格点位弄的比较糟糕，因而下单点位不大对。
        window = self.window
        signal = self.signal
        symbol = self.symbol
        T =100
        self.allQuoteData = self.Stats.high_obi(symbol)
        tradeList = self.Stats.volume_imbalance_bar(symbol)
        tradeList.loc[:,'vol_imb'] = tradeList.loc[:,'abVolume'] - tradeList.loc[:,'asVolume']
        tradeList.loc[:,'vol_imb_mean'] = tradeList.loc[:,'vol_imb'].cumsum()
        tradeList.loc[:,'vol_imb_std'] = tradeList.loc[:,'vol_imb'].rolling(T).std()
        self.allQuoteData = pd.merge(left = self.allQuoteData,right = tradeList,left_index= True,right_index= True, how = 'outer')
        ##''
        self.allQuoteData.loc[:, 'obi'] = np.log(self.allQuoteData.loc[:, 'bidVolume1']) - np.log(
            self.allQuoteData.loc[:, 'askVolume1'])


        askPriceDiff = self.allQuoteData ['askPrice1'].diff()
        askVolumeDiff = self.allQuoteData ['askVolume1'].diff()
        bidPriceDiff = self.allQuoteData ['bidPrice1'].diff()
        bidVolumeDiff = self.allQuoteData ['bidVolume1'].diff()

        self.allQuoteData .loc[:, 'ask_negChange'] = 0
        self.allQuoteData .loc[askPriceDiff > 0 , 'ask_negChange'] = 1  ## 价格上升
        self.allQuoteData.loc[askPriceDiff < 0, 'ask_negChange'] = -1  ##价格下降


        self.allQuoteData .loc[:, 'bid_negChange'] = 0
        self.allQuoteData .loc[bidPriceDiff < 0 , 'bid_negChange'] = 1 ##价格下降
        self.allQuoteData .loc[bidPriceDiff > 0 , 'bid_negChange'] = -1 ##价格上升
        self.allQuoteData.loc[:, 'spread'] =  self.allQuoteData ['askPrice1'] -  self.allQuoteData ['bidPrice1']  ##价格下降
        active_bid_list = list()
        active_ask_list = list()
        atb_rate_list = list()
        ata_rate_list = list()
        bv_count_list = list()
        av_count_list= list()
        last_obi = self.allQuoteData ['obi'].iloc[0]
        tick_count = 0
        row_count = 0
        av_count = np.nan
        bv_count = np.nan
        active_ask = 0
        active_bid=  0
        as_cum = 0
        ab_cum = 0

        for row in zip(self.allQuoteData ['ask_negChange'], self.allQuoteData ['bid_negChange'], self.allQuoteData ['askVolume1'], self.allQuoteData ['bidVolume1'],askVolumeDiff,bidVolumeDiff,self.allQuoteData.loc[:, 'spread'],
                       self.allQuoteData['abVolume'],self.allQuoteData ['asVolume']):
            priceStatus_ask = row[0]
            priceStatus_bid = row[1]
            av              = row[2]
            bv              = row[3]
            av_diff         = row[4]
            bv_diff         = row[5]
            spread = row[6]
            abVolume = row[7]
            asVolume = row[8]
            if spread <0.015:
                if ((priceStatus_ask == 1)&(priceStatus_bid == -1)) or np.isnan(priceStatus_ask):
                    tick_count = 0

                    active_bid = active_bid +abVolume

                    #av_diff = NowTick - preTick
                elif (priceStatus_ask == -1) :
                    active_bid = 0
                else:
                    active_bid == active_bid +abVolume

                if ((priceStatus_ask == -1)&(priceStatus_bid == 1))or np.isnan(priceStatus_bid):

                    active_ask = active_ask + asVolume
                elif (priceStatus_bid == -1):
                    active_ask = 0
                else:
                    active_ask =active_ask + asVolume


                row_count = row_count + 1
                if (active_bid + active_ask)>0:
                    ata_rate = active_ask/ (active_bid + active_ask)
                else:
                    ata_rate = 0
                if  (active_ask + active_bid) >0:
                    atb_rate = float(active_bid )/ (active_ask + active_bid)
                else:
                    atb_rate = 0
            else:
                atb_rate = 0
                ata_rate = 0

            #print(active_bid)
            #print((active_ask - active_bid))
            active_ask_list.append(active_ask)
            active_bid_list.append(active_bid)
            atb_rate_list.append(atb_rate)
            ata_rate_list.append(ata_rate)
            av_count_list.append(av_count)
            bv_count_list.append(bv_count)
            if ((priceStatus_ask == 1)&(priceStatus_bid == -1)) or np.isnan(priceStatus_ask):
                active_ask = 0
                ab_cum = 0
                as_cum = 0
                bv_count = bv
                av_count = av
            if ((priceStatus_ask == -1)&(priceStatus_bid == 1))or np.isnan(priceStatus_bid):
                active_bid = 0
                ab_cum = 0
                as_cum = 0
                bv_count = bv
                av_count = av
        self.allQuoteData .loc[:, 'avtive_ask_list'] = active_ask_list
        self.allQuoteData .loc[:, 'avtive_bid_list'] = active_bid_list
        self.allQuoteData .loc[:, 'atb_rate'] = atb_rate_list
        self.allQuoteData .loc[:, 'ata_rate'] = ata_rate_list
        self.allQuoteData .loc[:, 'av_count_list'] = av_count_list
        self.allQuoteData .loc[:, 'bv_count_list'] = bv_count_list

        #self.allQuoteData.loc[:,'obt_rate_buy'] = self.allQuoteData .loc[:, 'avtive_bid_list'] / self.allQuoteData .loc[:, 'ab_cum_list']
        #self.allQuoteData.loc[:,'obt_rate_sell'] = self.allQuoteData .loc[:, 'avtive_ask_list'] / self.allQuoteData .loc[:, 'as_cum_list']
        #large_width
        positivePos =(self.allQuoteData .loc[:, 'large_width']==0.01)& (self.allQuoteData .loc[:, 'atb_rate']>0.5)&(self.allQuoteData .loc[:, 'avtive_bid_list']>50000)&(self.allQuoteData ['ask_negChange'] ==1)&(self.allQuoteData ['bar_label'] ==1)
        negativePos =  (self.allQuoteData .loc[:, 'large_width']==0.01)&(self.allQuoteData .loc[:, 'ata_rate']>0.5)&(self.allQuoteData .loc[:, 'avtive_ask_list'] >50000)&(self.allQuoteData ['bid_negChange'] ==1)&(self.allQuoteData ['bar_label'] ==-1)
        self.allQuoteData .loc[positivePos, signal + '_' + str(window) + '_min'] = 1
        self.allQuoteData .loc[negativePos, signal + '_' + str(window) + '_min'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), signal + '_' + str(window) + '_min'] = 0

        #print(signal + '_' + str(window))
        # self.allQuoteData.to_csv(self.outputpath+ signal+'_.csv')
        return self.allQuoteData


    def bolling(self):
        T =20
        symbol = self.symbol
        signal = self.signal
        window = self.window

        self.allQuoteData =self.Stats.high_obi(symbol)
        self.allQuoteData.loc[:,'upper_std'] = self.allQuoteData.loc[:,'large_ask'].rolling(T).std()
        self.allQuoteData.loc[:, 'upper'] = self.allQuoteData.loc[:, 'large_ask'].rolling(T).mean()
        self.allQuoteData.loc[:,'lower'] = self.allQuoteData.loc[:,'large_bid'].rolling(T).mean()
        self.allQuoteData.loc[:,'lower_std'] = self.allQuoteData.loc[:,'large_bid'].rolling(T).std()
        large_spread =  (self.allQuoteData.loc[:,'large_width'].isnull().any())
        positivePos =large_spread& (self.allQuoteData.loc[:,'midp']>self.allQuoteData.loc[:,'upper'].shift(1) )&(self.allQuoteData.loc[:,'midp'].shift(1)<self.allQuoteData.loc[:,'upper'].shift(1))
        negativePos = large_spread& ( self.allQuoteData.loc[:,'midp']<self.allQuoteData.loc[:,'lower'].shift(1))& ( self.allQuoteData.loc[:,'midp'].shift(1)>self.allQuoteData.loc[:,'lower'].shift(1))

        self.allQuoteData.loc[:,'midp']
        self.allQuoteData .loc[positivePos, signal + '_' + str(window) + '_min'] = 1
        self.allQuoteData .loc[negativePos, signal + '_' + str(window) + '_min'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), signal + '_' + str(window) + '_min'] = 0
        return self.allQuoteData


    def obi_org(self):
        signal = self.signal
        window =self.window
        symbol = self.symbol
        self.allQuoteData = self.Stats.high_obi(symbol)



        small_obi_bid = self.allQuoteData .loc[:, 'bidVolume1']<self.allQuoteData .loc[:, 'asVolume']
        small_obi_ask = self.allQuoteData .loc[:, 'askVolume1']<self.allQuoteData .loc[:, 'abVolume']

        self.allQuoteData.loc[small_obi_bid, 'bidVolume1'] = 1
        self.allQuoteData.loc[small_obi_ask, 'askVolume1'] = 1
        self.allQuoteData .loc[:, 'obi'] = np.log(self.allQuoteData .loc[:, 'bidVolume1']) - np.log(
            self.allQuoteData .loc[:, 'askVolume1'])
        large_obi = np.abs(self.allQuoteData.loc[:, 'obi']) > 2
        self.allQuoteData.loc[large_obi, 'obi'] = 0
        # self.allQuoteData .loc[:, 'obi_' + str(window) + '_min'] = self.allQuoteData .loc[:,
        #                                                                   'obi'].rolling(window * 60).mean()
        self.allQuoteData .loc[:, 'obi_' + str(window) + '_min'] = self.allQuoteData .loc[:, 'obi'].diff(
            window)
        # positivePos = self.allQuoteData ['obi_' + str(window) + '_min'] > 8
        # negativePos = self.allQuoteData ['obi_' + str(window) + '_min'] < -8
        # self.allQuoteData .loc[positivePos, 'obi_' + str(window) + '_min'] = 1
        # self.allQuoteData .loc[negativePos, 'obi_' + str(window) + '_min'] = -1
        # self.allQuoteData .loc[(~positivePos) & (~negativePos), 'obi_' + str(window) + '_min'] = 0'''''
        askPriceDiff = self.allQuoteData ['askPrice1'].diff()
        bidPriceDiff = self.allQuoteData ['bidPrice1'].diff()
        midPriceChange = self.allQuoteData ['midp'].diff()
    
        self.allQuoteData .loc[:, 'priceChange'] = 1
        self.allQuoteData .loc[midPriceChange == 0, 'priceChange'] = 0
    
        obi_change_list = list()
        last_obi = self.allQuoteData ['obi'].iloc[0]
        tick_count = 0
        row_count = 0
        bid_list = list()
        ask_list = list()
        for row in zip(self.allQuoteData ['priceChange'], self.allQuoteData ['obi'], self.allQuoteData ['large_bid'], self.allQuoteData ['large_ask'],self.allQuoteData ['bidPrice1'],self.allQuoteData ['askPrice1']):
            priceStatus = row[0]
            obi = row[1]
            large_bid = row[2]
            large_ask = row[3]
            askPrice1 = row[4]
            bidPrice1 = row[5]
            bid_list.append(large_bid)
            ask_list.append(large_ask)



            if (priceStatus == 1) or np.isnan(priceStatus):
                tick_count = 0
                last_obi = obi

            else:
                #if (bidPrice1 in bid_list) or (askPrice1 in ask_list) :
                last_obi = self.allQuoteData ['obi'].iloc[row_count - tick_count]
                if tick_count <= 50:
                    tick_count = tick_count + 1

            row_count = row_count + 1
            obi_change = obi - last_obi
            obi_change_list.append(obi_change)
    
        self.allQuoteData .loc[:, 'obi'] = obi_change_list
        large_wid = self.allQuoteData.loc[:,'large_width'] >0.05
        mid_window =( self.allQuoteData.loc[:,'large_ask']  +self.allQuoteData.loc[:,'large_bid'])/2
        large_bid_obi = (self.allQuoteData ['obi'].shift(2) > 3)
        large_ask_obi = (self.allQuoteData ['obi'].shift(2)<- 3)
        positivePos = large_bid_obi
        negativePos = large_ask_obi
        '''
        self.allQuoteData .loc[positivePos, 'obi_org_' + str(window) + '_min'] = 1
        self.allQuoteData .loc[negativePos, 'obi_org_' + str(window) + '_min'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), 'obi_org_' + str(window) + '_min'] = 0
       
        self.allQuoteData .loc[positivePos, 'kp'] = 1
        self.allQuoteData .loc[negativePos,  'kp'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), 'kp'] = 0
        self.allQuoteData = self.Stats.point_monitor(symbol,self.allQuoteData .loc[:,  'kp'] )
        

        positivePos = (self.allQuoteData ['midp_change'] >self.allQuoteData ['vm'] + 4*self.allQuoteData ['vs'] )
        negativePos = (self.allQuoteData ['midp_change']<self.allQuoteData ['vm'] - 4* self.allQuoteData ['vs'] )
        '''
        self.allQuoteData .loc[positivePos, 'obi_org_' + str(window) + '_min'] = 1
        self.allQuoteData .loc[negativePos, 'obi_org_' + str(window) + '_min'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), 'obi_org_' + str(window) + '_min'] = 0
        #negativePos = (self.allQuoteData ['obi'] < -2)
        # self.allQuoteData .loc[:,''] =
        # self.allQuoteData .loc[:, 'obi' + str(window) + '_min_sum'] = self.allQuoteData .loc[:,'obi'].rolling(window * 60).sum()
        # todo: 把几层obi当作一层看待，适合高价股？
        #print('Calculate obi here for symbol = ', symbol, 'with lbwindow = ', window)
        return  self.allQuoteData


    def ex_ob_test(self):
        symbol = self.symbol
        signal = self.signal
        window = self.window
        self.allQuoteData = self.Stats.PV_summary(symbol)

        #negativePos = (self.allQuoteData.loc[:, 'posChange']>2*self.allQuoteData.loc[:,'bid_loc'])&(self.allQuoteData.loc[:, 'marker'] == 1)
        #positivePos =  (self.allQuoteData.loc[:, 'negChange']<-2*self.allQuoteData.loc[:,'ask_loc'])&(self.allQuoteData.loc[:, 'marker'] ==-1)
        negativePos =(self.allQuoteData.loc[:,'bid_loc']==0)&(self.allQuoteData.loc[:, 'marker'] == 1)& (self.allQuoteData.loc[:,'ask_loc']!=0)#&(self.allQuoteData.loc[:, 'TOTALchange'] ==0)
        positivePos = (self.allQuoteData.loc[:,'ask_loc']==0)&(self.allQuoteData.loc[:, 'marker'] ==-1)& (self.allQuoteData.loc[:,'bid_loc']!=0)#&(self.allQuoteData.loc[:, 'TOTALchange'] ==0)
        pnSignal = list()
        count = 0
        '''
        for row in zip(positivePos,negativePos,(self.allQuoteData.loc[:, 'obi'] )):
            pos = row[0]
            neg = row[1]
            obi = row[2]
            SIGNAL = 0
            if pos:
                count = 100
            elif neg:
                count = -100

            if count > 0:
                if 1:
                    count = 0
                    SIGNAL = 1
                else:
                    count = count - (count>0)
                    SIGNAL = 0
            elif count < 0:
                if 1:
                    count = 0
                    SIGNAL = -1
                else:
                    count = count + (count<0)
                    SIGNAL = 0
            else:
                SIGNAL = 0
            pnSignal.append(SIGNAL)
        self.allQuoteData.loc[:,signal + '_' + str(window) + '_min'] = pnSignal
        #self.allQuoteData .loc[pPos, signal + '_' + str(window) + '_min'] = 1
        #self.allQuoteData .loc[nPos, signal +'_' + str(window) + '_min'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), signal +'_' + str(window) + '_min'] = 0
        '''
        self.allQuoteData .loc[positivePos, signal + '_' + str(window) + '_min'] = 1
        self.allQuoteData .loc[negativePos, signal + '_' + str(window) + '_min'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), signal + '_' + str(window) + '_min'] = 0
        return self.allQuoteData


    def ex_ob(self):
        window = self.window
        signal = self.signal
        # todo: revise the obi signal here
        ex_ob_ = list()
        ex_ob_.append(0)
        ex_ob_bid = list()
        ex_ob_bid.append(0)
        ex_ob_ask = list()
        ex_ob_ask.append(0)
        ex_ob_ahead = list()
        ex_ob_ahead.append(0)
        self.allQuoteData .loc[:, 'obi_'] = (self.allQuoteData .loc[:, 'bidVolume1'] +self.allQuoteData .loc[:, 'bidVolume2']
                                                    + self.allQuoteData .loc[:, 'bidVolume3']
                                                   )/(self.allQuoteData .loc[:, 'askVolume1'] + self.allQuoteData .loc[:, 'askVolume2']
                                                      + self.allQuoteData .loc[:, 'askVolume3'])

        lth = len(self.allQuoteData .loc[:, 'bidVolume1'])

        fac = []

        ap_list = ['askPrice1', 'askPrice2', 'askPrice3','askPrice4', 'askPrice5']
        bp_list = ['bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5']


        av_list = ['askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']
        bv_list = ['bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5']

        ap_list_ahead = ['askPrice1', 'askPrice2', 'askPrice3','askPrice4', 'askPrice5']
        bp_list_ahead = ['bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5']

        av_list_ahead = ['askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']
        bv_list_ahead = ['bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5']
        for i in range(1, lth):

            if (i == 1):
                Askp_array = np.array(())
                Askv_array = np.array(())
                bidp_array = np.array(())
                bidv_array = np.array(())
                pre_midp = (self.allQuoteData .bidPrice1.values[i] +self.allQuoteData .askPrice1.values[i]) / 2
            pre_ask = np.sum(Askv_array)
            pre_bid = np.sum(bidv_array)

            if (i > 1):

                bid_pos_ind = bidp_array <= self.allQuoteData .bidPrice1.values[i]
                ask_pos_ind = Askp_array >= self.allQuoteData .askPrice1.values[i]
                bidp_array = bidp_array[bid_pos_ind]
                Askp_array = Askp_array[ask_pos_ind]
                bidv_array = bidv_array[bid_pos_ind]
                Askv_array = Askv_array[ask_pos_ind]
            for bp, ap, bv, av in zip(bp_list, ap_list, bv_list, av_list):
                if (self.allQuoteData [bp][i] <= self.allQuoteData .bidPrice1.values[i]):
                    if (self.allQuoteData [bp][i] not in bidp_array):
                        bidp_array = np.append(bidp_array, self.allQuoteData [bp][i])
                        bidv_array = np.append(bidv_array, self.allQuoteData [bv][i])
                    else:
                        assert (len(np.where(bidp_array == self.allQuoteData [bp][i])[0]) == 1)
                        bidv_array[np.where(bidp_array == self.allQuoteData [bp][i])[0][0]] = self.allQuoteData [bv][i]

                if (self.allQuoteData [ap][i] >= self.allQuoteData .askPrice1.values[i]):
                    if (self.allQuoteData [ap][i] not in Askp_array):
                        Askp_array = np.append(Askp_array, self.allQuoteData [ap][i])
                        Askv_array = np.append(Askv_array, self.allQuoteData [av][i])
                    else:
                        assert (len(np.where(Askp_array == self.allQuoteData [ap][i])[0]) == 1)
                        Askv_array[np.where(Askp_array == self.allQuoteData [ap][i])[0][0]] = self.allQuoteData [av][i]

            Now_ask = np.sum(Askv_array)

            Now_bid = np.sum(bidv_array)

            midPriceChange = self.allQuoteData ['midp'].diff()

            self.allQuoteData .loc[:, 'priceChange'] = 1


            temp_back_ = (Now_bid - pre_bid) * - (Now_ask - pre_ask)


            ex_ob_ask.append((Now_ask - pre_ask))
            ex_ob_bid.append((Now_bid - pre_bid))


        self.allQuoteData .loc[:, 'obi_bid_diff'] = ex_ob_bid
        self.allQuoteData .loc[:, 'obi_ask_diff'] = ex_ob_ask
        self.allQuoteData.loc[:,'obi_fil_bid'] = self.allQuoteData .loc[:, 'obi_bid_diff'] .ewm(100).mean()
        self.allQuoteData.loc[:,'obi_fil_ask'] = self.allQuoteData .loc[:, 'obi_ask_diff'] .ewm(100).mean()
        posPoint = (self.allQuoteData ['obi_ask_diff']<- 150 *100 )&(self.allQuoteData.loc[:,'obi_fil_bid']>100)
        negPoint= (self.allQuoteData ['obi_bid_diff']<-150 *100 )&(self.allQuoteData.loc[:,'obi_fil_ask']>100)



        signalPos = list()
        count = 0
        waiting_count = 0
        spread = self.allQuoteData ['askPrice1'] - self.allQuoteData ['bidPrice1']
        for row in zip(posPoint,negPoint,self.allQuoteData ['obi_ask_diff'],self.allQuoteData ['obi_bid_diff'],spread):
            posP = row[0]
            negP = row[1]
            obi_ask = row[2]
            obi_bid = row[3]
            sp = row[4]
            if posP == True:
                count = 60
            elif negP == True:
                count = -60

            if count > 0:
                if obi_ask <- 300*100:
                    waiting_count =count
                    count = 0
                else:
                    count = count -(count>0)
            elif count <0:
                if obi_bid <-300*100:
                    waiting_count =count
                    count = 0
                else:
                    count = count +(count<0)


            if waiting_count>0:
                if sp<0.05:
                    signalPos.append(1)
                    waiting_count = 0
                else:
                    signalPos.append(0.5)
                    waiting_count = waiting_count - (waiting_count>0)
            elif waiting_count<0:
                if sp< 0.05:
                    signalPos.append(-1)
                    waiting_count = 0
                else:
                    signalPos.append(-0.5)
                    waiting_count = waiting_count + (waiting_count<0)
            else:
                signalPos.append(0)
        self.allQuoteData.loc[:,'sig'] = signalPos
        negativePos = self.allQuoteData.loc[:,'sig'] == -1
        positivePos = self.allQuoteData.loc[:,'sig'] == 1
        #pd.concat(q, 0).to_csv(outputpath + './' + tradingDay + '.csv')
        self.allQuoteData .loc[positivePos, signal + '_' + str(window) + '_min'] = 1
        self.allQuoteData .loc[negativePos, signal +'_' + str(window) + '_min'] = -1
        self.allQuoteData .loc[(~positivePos) & (~negativePos), signal +'_' + str(window) + '_min'] = 0
        # self.allQuoteData .loc[:,''] =
        # self.allQuoteData .loc[:, 'obi' + str(window) + '_min_sum'] = self.allQuoteData .loc[:,'obi'].rolling(window * 60).sum()
        # todo: 把几层obi当作一层看待，适合高价股？

        # print out the dataframe
        #q = self.allQuoteData
        #q.to_csv(self.dataSavePath + './' + str(self.tradeDate.date())+ signal+' '+symbol + '.csv')
        #print('Calculate obi here for symbol = ', symbol, 'with lbwindow = ', window)

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




