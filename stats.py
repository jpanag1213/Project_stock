# -*- coding: utf-8 -*-
"""
Created on 2019-02-23

to stats the order imbalance feature
#导出日线数据
@author: jiaxiong

"""

import numpy as np
import Data
import SignalTester
import pandas as pd
import os
import configparser
import time
from Utils import *
import matplotlib.pyplot as plt
import datetime

class Stats(object):

    def __init__(self, symbol, tradedate, quoteData,tradeData = None,futureData =None, outputpath = 'E://stats_test/'):
        self.symbol    = symbol
        self.tradeDate = tradedate
        if isinstance(quoteData,dict):
            self.quoteData = quoteData
            self.tradeData = tradeData
        else:
            self.quoteData = dict()
            self.quoteData[symbol] = quoteData
            self.tradeData = dict()
            self.tradeData[symbol] = tradeData

        self.outputpath = outputpath

        self.futureData = futureData
        print(os.path.exists(outputpath))
        if os.path.exists(outputpath) is False:
            os.makedirs(outputpath)
        if futureData == None:
            self.columns =[
                    'exchangeCode', 'exchangeTime', 'latest', 'pre_close', 'open', 'high', 'low', 'upper_limit',
                    'lower_limit', 'status', 'tradeNos', 'tradeVolume', 'totalTurnover',

                    'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5', 'bidPrice6', 'bidPrice7',
                    'bidPrice8', 'bidPrice9', 'bidPrice10',

                    'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5', 'askPrice6', 'askPrice7',
                    'askPrice8', 'askPrice9', 'askPrice10',

                    'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5', 'bidVolume6', 'bidVolume7',
                    'bidVolume8', 'bidVolume9', 'bidVolume10',

                    'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5', 'askVolume6', 'askVolume7',
                    'askVolume8', 'askVolume9', 'askVolume10',

                    'total_bid_qty', 'total_ask_qty', 'weighted_avg_bid_price', 'weighted_avg_ask_price',

                    'iopv', 'yield_to_maturity']

        else:
            self.columns =   [

                    'id', 'exchangeTime', 'latest', 'pre_close', 'settle','pre_settle','open', 'high', 'low','close', 'upper_limit',

                    'lower_limit', 'status', 'buy', 'tradeVolume', 'totalTurnover',

                    'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',

                    'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',

                    'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',

                    'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5',

                    'open_interest','pre_open_interest', 'open_qty','close_qty']





    def check_file(self,file,symbol = " "):

        file.to_csv(self.outputpath+str(self.tradeDate)+'_'+symbol + '_checkquote.csv')

        return 0

    def Evaluation_times(self,tick = 30):

        ##监测前30分钟的因子值。订单结构等。
        #定义：订单结构，因子设计等等。
        #input quotedata tradedata
        #output return？volatility？

        eva_duration = 20 * tick
        quoteData  = self.quoteData[self.symbol[0]]
        quoteData  = pd.concat([quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'),'%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 9:45:00'), '%Y%m%d %H:%M:%S'):],quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 13:00:00'),'%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 13:15:00'), '%Y%m%d %H:%M:%S'):]])
        columns = self.columns

        stats.check_file(quoteData)

        return 0


    def lastNday(self,tradingday,tradingDates):
        tradingDayFile ='./ref_data/TradingDay.csv'
        tradingDays = pd.read_csv(tradingDayFile)

    def opentime(self,symbol,closetime = ' 14:50:00'):
        quoteData = self.quoteData[symbol]
        quoteData.loc[:,'opentime'] = 0
        quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate +closetime), '%Y%m%d %H:%M:%S'),'opentime'] = 1
        quoteData.loc[
        datetime.datetime.strptime(str(self.tradeDate + closetime), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 14:57:00'), '%Y%m%d %H:%M:%S'), 'opentime'] =2
        #stats.check_file(quoteData)
        return quoteData

    def time_cut(self,symbol,closetime = ' 14:50:00'):
        quoteData = self.quoteData[symbol]
        quoteData.loc[:,'opentime'] = 0
        quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate +closetime), '%Y%m%d %H:%M:%S'),'opentime'] = 1
        quoteData = quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate +closetime), '%Y%m%d %H:%M:%S'),:]
        #stats.check_file(quoteData)
        return quoteData


    def quote_cut(self,quoteData,num = 10):
        price_column_ask= list()
        price_column_bid= list()
        volunme_column_ask = list()
        volunme_column_bid = list()
        for Nos in range(1,num+1):
            price_column_ask.append('askPrice'+ str(Nos))
            volunme_column_ask.append('askVolume' + str(Nos))

        for Nos in range(1,num+1):
            price_column_bid.append('bidPrice' + str(Nos))
            volunme_column_bid.append('bidVolume'+ str(Nos))
        return quoteData.loc[:,price_column_ask],quoteData.loc[:,volunme_column_ask],quoteData.loc[:,price_column_bid],quoteData.loc[:,volunme_column_bid]


    def responseFun(self,symbol):
        quoteData =  self.quoteData[symbol]
        quoteD = pd.DataFrame()

        return 0

    def spread_w(self):
        quotedata.loc[:,'bid_spread'] = quotedata.loc[:,'bidPrice1'] - quotedata.loc[:,'bidPrice10']


        quotedata.loc[:, 'bid_weight'] =( (quotedata.loc[:,'bidPrice1'] - quotedata.loc[:,'weighted_avg_bid_price'])**2*(quotedata.loc[:,'bidVolume1'])
                                            +(quotedata.loc[:, 'bidPrice2'] - quotedata.loc[:, 'weighted_avg_bid_price'])**2 * (quotedata.loc[:, 'bidVolume2'])
                                            +(quotedata.loc[:, 'bidPrice3'] - quotedata.loc[:, 'weighted_avg_bid_price'])**2 * (quotedata.loc[:, 'bidVolume3'])
                                            +(quotedata.loc[:, 'bidPrice4'] - quotedata.loc[:, 'weighted_avg_bid_price'])**2 * (quotedata.loc[:, 'bidVolume4'])
                                            +(quotedata.loc[:, 'bidPrice5'] - quotedata.loc[:, 'weighted_avg_bid_price'])**2 * (quotedata.loc[:, 'bidVolume5'])
                                            +(quotedata.loc[:, 'bidPrice6'] - quotedata.loc[:, 'weighted_avg_bid_price'])**2 * (quotedata.loc[:, 'bidVolume6'])
                                            +(quotedata.loc[:, 'bidPrice7'] - quotedata.loc[:, 'weighted_avg_bid_price'])**2 * (quotedata.loc[:, 'bidVolume7'])
                                            +(quotedata.loc[:, 'bidPrice8'] - quotedata.loc[:, 'weighted_avg_bid_price'])**2 * (quotedata.loc[:, 'bidVolume8'])
                                            +(quotedata.loc[:, 'bidPrice9'] - quotedata.loc[:, 'weighted_avg_bid_price'])**2 * (quotedata.loc[:, 'bidVolume9'])
                                            +(quotedata.loc[:, 'bidPrice10'] - quotedata.loc[:, 'weighted_avg_bid_price'])**2 * (quotedata.loc[:, 'bidVolume10']))/ (quotedata.loc[:, 'total_bid_qty'])

        quotedata.loc[:, 'ask_weight'] = ((quotedata.loc[:,'askPrice1'] - quotedata.loc[:,'weighted_avg_ask_price'])**2*(quotedata.loc[:,'askVolume1'])
                                            +(quotedata.loc[:, 'askPrice2'] - quotedata.loc[:, 'weighted_avg_ask_price'])**2 * (quotedata.loc[:, 'askVolume2'])
                                            +(quotedata.loc[:, 'askPrice3'] - quotedata.loc[:, 'weighted_avg_ask_price'])**2 * (quotedata.loc[:, 'askVolume3'])
                                            +(quotedata.loc[:, 'askPrice4'] - quotedata.loc[:, 'weighted_avg_ask_price'])**2 * (quotedata.loc[:, 'askVolume4'])
                                            +(quotedata.loc[:, 'askPrice5'] - quotedata.loc[:, 'weighted_avg_ask_price'])**2 * (quotedata.loc[:, 'askVolume5'])
                                            +(quotedata.loc[:, 'askPrice6'] - quotedata.loc[:, 'weighted_avg_ask_price'])**2 * (quotedata.loc[:, 'askVolume6'])
                                            +(quotedata.loc[:, 'askPrice7'] - quotedata.loc[:, 'weighted_avg_ask_price'])**2 * (quotedata.loc[:, 'askVolume7'])
                                            +(quotedata.loc[:, 'askPrice8'] - quotedata.loc[:, 'weighted_avg_ask_price'])**2 * (quotedata.loc[:, 'askVolume8'])
                                            +(quotedata.loc[:, 'askPrice9'] - quotedata.loc[:, 'weighted_avg_ask_price'])**2 * (quotedata.loc[:, 'askVolume9'])
                                            +(quotedata.loc[:, 'askPrice10'] - quotedata.loc[:, 'weighted_avg_ask_price'])**2 * (quotedata.loc[:, 'askVolume10']))/ (quotedata.loc[:, 'total_ask_qty'])
        quotedata.loc[:,'weight_ratio'] = quotedata.loc[:, 'bid_weight']/quotedata.loc[:, 'ask_weight']

        '''
        quotedata.loc[:,'ratio_mean'] = quotedata.loc[:,'weight_ratio'].ewm(10).mean()
        quotedata.loc[:,'ratio_std'] = quotedata.loc[:,'weight_ratio'].ewm(10).std()
        positivePos = quotedata.loc[:,'weight_ratio'] > (quotedata.loc[:,'ratio_mean'] + 2*quotedata.loc[:,'ratio_std'])
        negativePos = quotedata.loc[:,'weight_ratio'] < (quotedata.loc[:,'ratio_mean'] -2* quotedata.loc[:,'ratio_std'])
        quotedata.loc[:,'signal'] = 0
        quotedata.loc[positivePos,'signal'] = 1
        quotedata.loc[negativePos,'signal'] = -1
        '''
        quotedata.loc[:,'ask_spread'] = quotedata.loc[:,'askPrice1'] -quotedata.loc[:,'askPrice10']
        quotedata.loc[:,'spread']  =  quotedata.loc[:,'askPrice1'] -quotedata.loc[:,'bidPrice1']

        return 0


    def zaopan_stats(self,symbol):
        opendata = stats.time_cut(symbol,closetime = ' 09:30:06')

        base_price_ask,base_volume_ask = stats.rolling_dealer(opendata,num=10,name='ask')
        base_price_bid,base_volume_bid = stats.rolling_dealer(opendata,num=10,name='bid')

        quotedata = stats.time_cut(symbol, closetime=' 14:50:00')
        quotedata = stats.rolling_dealer(quotedata,num=5,name='ask',base_price = base_price_ask,base_qty= base_volume_ask)
        quotedata = stats.rolling_dealer(quotedata,num=5,name='bid',base_price = base_price_bid,base_qty= base_volume_bid)
        quotedata.loc[:,'ret_bid'] = np.log(quotedata.loc[:,'bid_leftPrice'] / quotedata.loc[:, 'pre_close'])
        quotedata.loc[:,'ret_ask'] = np.log(quotedata.loc[:,'ask_leftPrice'] / quotedata.loc[:, 'pre_close'])
        quotedata.loc[:, 'open_ret'] = np.log(quotedata.loc[:, 'open'] / quotedata.loc[:, 'pre_close'])
        stats.check_file(quotedata,symbol)

        return quotedata

    def rolling_dealer(self,quotedata,num = 10,name = 'ask',base_price = 0,base_qty = 0):
        total_qty_name = 'total_'+name+'_qty'
        avg_price_name = 'weighted_avg_'+name+'_price'
        temp = quotedata.loc[:, avg_price_name]*quotedata.loc[:, total_qty_name] - base_price *1*base_qty/2
        temp2 = quotedata.loc[:, total_qty_name] -1* base_qty/2
        for number in range(1,num+1):
            price_name = name+'Price'+str(number)

            Volume_name = name+'Volume'+str(number)
            temp = temp -quotedata.loc[:,price_name]*quotedata.loc[:,Volume_name]
            temp2 = temp2 -quotedata.loc[:,Volume_name]
        quotedata.loc[:,name+'_leftPrice'] = temp/temp2
        quotedata.loc[:, name + '_BasePrice'] = base_price
        quotedata.loc[:,name+'_leftVolume'] = temp2
        quotedata.loc[:,name+'_BaseVolume'] =base_qty*1/2
        if base_price ==0:
            return (temp/temp2).mean(),temp2.mean()
        else:
            return quotedata


    def cancel_order(self,symbol):
        quotedata = self.quoteData[symbol]
        tradeData = self.tradeData[symbol]

        quote_time = pd.to_datetime(quotedata.exchangeTime.values).values
        quotedata.loc[:,'tradeVolume'] =quotedata.loc[:, 'tradeVolume'].diff()
        quotedata.loc[:,'Turnover'] =quotedata.loc[:, 'totalTurnover'].diff()
        quotedata.index = pd.to_datetime(quotedata.loc[:, 'exchangeTime'].values, format='%Y-%m-%d %H:%M:%S')

        temp_1 = pd.to_datetime(tradeData.loc[:,' nTime'],  format='%Y-%m-%d %H:%M:%S.%f')
        qqq = temp_1[0].microsecond
        bid_order = tradeData.loc[:, ' nBSFlag'] == 'B'
        ask_order = tradeData.loc[:, ' nBSFlag'] == 'S'
        can_order = tradeData.loc[:, ' nBSFlag'] == ' '
        tradeData.loc[bid_order, 'numbs_flag'] = 1
        tradeData.loc[ask_order, 'numbs_flag'] = -1
        tradeData.loc[can_order, 'numbs_flag'] = 0
        cancel_order = tradeData.loc[:, ' nPrice'] == 0

        tradeData.loc[:, 'temp'] = tradeData.loc[:, ' nPrice']
        #tradeData.loc[pos, 'temp'] = np.nan
        tradeData.temp.fillna(method='ffill', inplace=True)
        lastrep = list(tradeData.temp.values[:-1])
        lastrep.insert(0, 0)
        lastrep = np.asarray(lastrep)
        tradeData_quote = pd.merge(quotedata.loc[:, ['bidPrice1', 'askPrice1','exchangeTime','tradeVolume','Turnover','tradeNos']],tradeData,  left_index=True,right_index=True, how='outer')
        tradeData_quote['bidPrice1'].fillna(method='ffill', inplace=True)
        tradeData_quote['askPrice1'].fillna(method='ffill', inplace=True)
        # tradeData_quote.to_csv(self.dataSavePath + './' + str(self.tradeDate.date()) + signal + ' ' + symbol + '.csv')
        ActiveBuy = (tradeData_quote.loc[:, 'numbs_flag'] == 1)
        ActiveSell = (tradeData_quote.loc[:, 'numbs_flag'] == -1)
        tradeData_quote.loc[ActiveBuy, 'abVolume'] = tradeData_quote.loc[ActiveBuy, ' nVolume']
        tradeData_quote.loc[ActiveSell, 'asVolume'] = tradeData_quote.loc[ActiveSell, ' nVolume']
        tradeData_quote.loc[ActiveBuy, 'abPrice'] = tradeData_quote.loc[ActiveBuy, ' nTurnover']
        tradeData_quote.loc[ActiveSell, 'asPrice'] = tradeData_quote.loc[ActiveSell, ' nTurnover']


        #stats.check_file(tradeData_quote)

        temp_quote_time = np.asarray(list(quote_time))
        Columns_ = ['abVolume', 'asVolume','abPrice', 'asPrice']

        resample_tradeData = tradeData_quote.loc[:, Columns_].resample('1S', label='right', closed='right').sum()

        resample_tradeData = resample_tradeData.cumsum()
        resample_tradeData = resample_tradeData.loc[temp_quote_time, :]
        r_tradeData = resample_tradeData.diff()
        r_tradeData.loc[:, 'abPrice'] = r_tradeData.loc[:, 'abPrice'] / r_tradeData.loc[:, 'abVolume']
        r_tradeData.loc[:, 'asPrice'] = r_tradeData.loc[:, 'asPrice']/ r_tradeData.loc[:, 'asVolume']
        r_tradeData.loc[:,'timecheck'] =quotedata.loc[:, 'exchangeTime']
        #stats.check_file(r_tradeData)

        #stats.check_file(r_tradeData)
        '''
        quote_order = pd.merge(self.quoteData[symbol].loc[:, ['midp', 'midp_10', 'spread']], r_tradeData, left_index=True,
                               right_index=True, how='left')
        # .loc[:,'midp'] =self.quoteData[symbol].loc[:,'midp']

        quote_order.to_csv(self.outputpath + './ quote_order.csv')
        
        # self.quoteData[symbol].loc[:, ['midp', 'bidVolume1', 'askVolume1']].to_csv(self.outputpath + './ quote_o.csv')
        '''
        return r_tradeData

    def plot(self):


        return 0


    def price_filter(self):
        symbol = self.symbol[0]
        midp = self.quoteData[symbol].loc[:, 'midp']
        quotedata = self.quoteData[symbol]
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
        self.quoteData[symbol].loc[:, 'midp_2'] = midp_2
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

        self.quoteData[symbol].loc[:, 'ewm'] = ewm_midp_
        self.quoteData[symbol].loc[:, 'filter_ewm'] = ewm_midp
        self.quoteData[symbol].loc[:, 'not'] = not_point
        # self.quoteData[symbol].loc[:,'kp_1'] = kp1_point
        # self.quoteData[symbol].loc[:,'kp_2'] = kp2_point

        self.quoteData[symbol].loc[:, 'std_'] = std_
        # self.quoteData[symbol].loc[:,'std_'] = std_
        # self.quoteData[symbol].loc[:,'state'] = STATE_test

        self.quoteData[symbol].loc[:, 'upper_bound'] = self.quoteData[symbol].loc[:, 'not'] + 3 * \
                                                          self.quoteData[symbol].loc[:, 'std_']
        self.quoteData[symbol].loc[:, 'lower_bound'] = self.quoteData[symbol].loc[:, 'not'] - 3 * \
                                                          self.quoteData[symbol].loc[:, 'std_']
        # negativePos = (self.quoteData[symbol].loc[:,'ewm']> (self.quoteData[symbol].loc[:,'not'] +3*self.quoteData[symbol].loc[:,'std_']))&(self.quoteData[symbol].loc[:,'ewm'].shift(-1) <(self.quoteData[symbol].loc[:,'not'].shift(-1) + 3*self.quoteData[symbol].loc[:,'std_'].shift(-1)))
        # negativePos = (self.quoteData[symbol].loc[:,'ewm'].shift(1) > (self.quoteData[symbol].loc[:,'not'].shift(1)  +3*self.quoteData[symbol].loc[:,'std_'].shift(1) ))&(self.quoteData[symbol].loc[:,'ewm'] <(self.quoteData[symbol].loc[:,'not'] + 3*self.quoteData[symbol].loc[:,'std_']))

        positivePos = (self.quoteData[symbol].loc[:, 'ewm'].shift(1) < (
                    self.quoteData[symbol].loc[:, 'not'].shift(1) + 3 * self.quoteData[symbol].loc[:,
                                                                           'std_'].shift(1))) & (
                                  self.quoteData[symbol].loc[:, 'ewm'] > (
                                      self.quoteData[symbol].loc[:, 'not'] + 3 * self.quoteData[symbol].loc[:,
                                                                                    'std_']))
        # positivePos = (self.quoteData[symbol].loc[:,'ewm']< (self.quoteData[symbol].loc[:,'not'] -3*self.quoteData[symbol].loc[:,'std_']))&(self.quoteData[symbol].loc[:,'ewm'].shift(-1)>(self.quoteData[symbol].loc[:,'not'].shift(-1) - 3*self.quoteData[symbol].loc[:,'std_'].shift(-1)))
        # positivePos = (self.quoteData[symbol].loc[:,'ewm'].shift(1) < (self.quoteData[symbol].loc[:,'not'].shift(1) -3*self.quoteData[symbol].loc[:,'std_'].shift(1) ))&(self.quoteData[symbol].loc[:,'ewm']>(self.quoteData[symbol].loc[:,'not'] - 3*self.quoteData[symbol].loc[:,'std_']))
        negativePos = (self.quoteData[symbol].loc[:, 'ewm'].shift(1) > (
                    self.quoteData[symbol].loc[:, 'not'].shift(1) - 3 * self.quoteData[symbol].loc[:,
                                                                           'std_'].shift(1))) & (
                                  self.quoteData[symbol].loc[:, 'ewm'] < (
                                      self.quoteData[symbol].loc[:, 'not'] - 3 * self.quoteData[symbol].loc[:,
                                                                                    'std_']))

        '''
        y_value = list(midp.iloc[:])
        yvalue  =list(ewm_midp)
        yvalue_3 = list(ewm_midp_)

        ax.plot(yvalue,label  = '1')
        ax.plot(y_value,label = '2')
        ax.plot(not_point, marker='^', c='red')

        #plt.savefig(self.dataSavePath + '/'+ str(self.tradeDate.date())  +symbol +signal+ '.jpg')
        '''

        quotedata = self.quoteData[symbol]
        stats.check_file(quotedata,symbol)

        return 0



    def price_filter(self):
        symbol = self.symbol[0]
        midp = self.quoteData[symbol].loc[:, 'midp']
        quotedata = self.quoteData[symbol]
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
        self.quoteData[symbol].loc[:, 'midp_2'] = midp_2
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

        self.quoteData[symbol].loc[:, 'ewm'] = ewm_midp_
        self.quoteData[symbol].loc[:, 'filter_ewm'] = ewm_midp
        self.quoteData[symbol].loc[:, 'not'] = not_point
        # self.quoteData[symbol].loc[:,'kp_1'] = kp1_point
        # self.quoteData[symbol].loc[:,'kp_2'] = kp2_point

        self.quoteData[symbol].loc[:, 'std_'] = std_
        # self.quoteData[symbol].loc[:,'std_'] = std_
        # self.quoteData[symbol].loc[:,'state'] = STATE_test

        self.quoteData[symbol].loc[:, 'upper_bound'] = self.quoteData[symbol].loc[:, 'not'] + 3 * \
                                                          self.quoteData[symbol].loc[:, 'std_']
        self.quoteData[symbol].loc[:, 'lower_bound'] = self.quoteData[symbol].loc[:, 'not'] - 3 * \
                                                          self.quoteData[symbol].loc[:, 'std_']
        # negativePos = (self.quoteData[symbol].loc[:,'ewm']> (self.quoteData[symbol].loc[:,'not'] +3*self.quoteData[symbol].loc[:,'std_']))&(self.quoteData[symbol].loc[:,'ewm'].shift(-1) <(self.quoteData[symbol].loc[:,'not'].shift(-1) + 3*self.quoteData[symbol].loc[:,'std_'].shift(-1)))
        # negativePos = (self.quoteData[symbol].loc[:,'ewm'].shift(1) > (self.quoteData[symbol].loc[:,'not'].shift(1)  +3*self.quoteData[symbol].loc[:,'std_'].shift(1) ))&(self.quoteData[symbol].loc[:,'ewm'] <(self.quoteData[symbol].loc[:,'not'] + 3*self.quoteData[symbol].loc[:,'std_']))

        positivePos = (self.quoteData[symbol].loc[:, 'ewm'].shift(1) < (
                    self.quoteData[symbol].loc[:, 'not'].shift(1) + 3 * self.quoteData[symbol].loc[:,
                                                                           'std_'].shift(1))) & (
                                  self.quoteData[symbol].loc[:, 'ewm'] > (
                                      self.quoteData[symbol].loc[:, 'not'] + 3 * self.quoteData[symbol].loc[:,
                                                                                    'std_']))
        # positivePos = (self.quoteData[symbol].loc[:,'ewm']< (self.quoteData[symbol].loc[:,'not'] -3*self.quoteData[symbol].loc[:,'std_']))&(self.quoteData[symbol].loc[:,'ewm'].shift(-1)>(self.quoteData[symbol].loc[:,'not'].shift(-1) - 3*self.quoteData[symbol].loc[:,'std_'].shift(-1)))
        # positivePos = (self.quoteData[symbol].loc[:,'ewm'].shift(1) < (self.quoteData[symbol].loc[:,'not'].shift(1) -3*self.quoteData[symbol].loc[:,'std_'].shift(1) ))&(self.quoteData[symbol].loc[:,'ewm']>(self.quoteData[symbol].loc[:,'not'] - 3*self.quoteData[symbol].loc[:,'std_']))
        negativePos = (self.quoteData[symbol].loc[:, 'ewm'].shift(1) > (
                    self.quoteData[symbol].loc[:, 'not'].shift(1) - 3 * self.quoteData[symbol].loc[:,
                                                                           'std_'].shift(1))) & (
                                  self.quoteData[symbol].loc[:, 'ewm'] < (
                                      self.quoteData[symbol].loc[:, 'not'] - 3 * self.quoteData[symbol].loc[:,
                                                                                    'std_']))

        '''
        y_value = list(midp.iloc[:])
        yvalue  =list(ewm_midp)
        yvalue_3 = list(ewm_midp_)

        ax.plot(yvalue,label  = '1')
        ax.plot(y_value,label = '2')
        ax.plot(not_point, marker='^', c='red')

        #plt.savefig(self.dataSavePath + '/'+ str(self.tradeDate.date())  +symbol +signal+ '.jpg')
        '''

        quotedata = self.quoteData[symbol]
        stats.check_file(quotedata,symbol)

        return 0


    def obi_fixedprice(self,symbol):

        self.quoteData[symbol].loc[:, 'obi'] = np.log(self.quoteData[symbol].loc[:, 'bidVolume1']+1) - np.log(
            self.quoteData[symbol].loc[:, 'askVolume1']+1)
        # self.quoteData[symbol].loc[:, 'obi_' + str(window) + '_min'] = self.quoteData[symbol].loc[:,
        #                                                                   'obi'].rolling(window * 60).mean()
        self.quoteData[symbol].loc[:, 'obi_' + '_min'] = self.quoteData[symbol].loc[:, 'obi'].diff()

        askPriceDiff = self.quoteData[symbol]['askPrice1'].diff()
        bidPriceDiff = self.quoteData[symbol]['bidPrice1'].diff()
        midPriceChange = self.quoteData[symbol]['midp'].diff()
        self.quoteData[symbol].loc[:, 'priceChange'] = 1
        self.quoteData[symbol].loc[midPriceChange == 0, 'priceChange'] = 0

        obi_change_list = list()
        last_obi = self.quoteData[symbol]['obi'].iloc[0]
        tick_count = 0
        row_count = 0
        for row in zip(self.quoteData[symbol]['priceChange'], self.quoteData[symbol]['obi']):
            priceStatus = row[0]
            obi = row[1]
            if (priceStatus == 1) or np.isnan(priceStatus):
                tick_count = 0
                last_obi = obi
            else:
                last_obi = self.quoteData[symbol]['obi'].iloc[row_count - tick_count]
                tick_count = tick_count + 1

            row_count = row_count + 1
            obi_change = obi - last_obi
            obi_change_list.append(obi_change)

        self.quoteData[symbol].loc[:, 'obi'] = obi_change_list
        positivePos = self.quoteData[symbol]['obi'] > 6
        negativePos = self.quoteData[symbol]['obi'] < -6
        self.quoteData[symbol].loc[positivePos, 'obi_'+ '_min'] = 1
        self.quoteData[symbol].loc[negativePos, 'obi_'+ '_min'] = -1
        self.quoteData[symbol].loc[(~positivePos) & (~negativePos), 'obi_' + '_min'] = 0
        # self.quoteData[symbol].loc[:,''] =
        # self.quoteData[symbol].loc[:, 'obi' + str(window) + '_min_sum'] = self.quoteData[symbol].loc[:,'obi'].rolling(window * 60).sum()
        # todo: 把几层obi当作一层看待，适合高价股？
        return self.quoteData[symbol]


    def price_volume(self,symbol):
        quotedata = stats.time_cut(symbol)
        askPrice1 = quotedata.loc[:,'askPrice2']
        bidPrice1 = quotedata.loc[:,'bidPrice2']
        askVolume1 = quotedata.loc[:,'askVolume2']
        bidVolume1 = quotedata.loc[:,'bidVolume2']
        Time = quotedata.loc[:, 'exchangeTime']
        MAX_PRICE_VOLUME_ASK = dict()
        MAX_PRICE_VOLUME_BID = dict()
        for row in zip(askPrice1,bidPrice1,askVolume1,bidVolume1,Time):
            ap1  = row[0]
            bp1  = row[1]
            av1  = row[2]
            bv1  = row[3]
            time = row[4]
            key_ask = list(MAX_PRICE_VOLUME_ASK.keys())
            key_bid = list(MAX_PRICE_VOLUME_BID.keys())
            if ap1 in key_ask:

                MAX_PRICE_VOLUME_ASK[ap1][1] =(av1 -MAX_PRICE_VOLUME_ASK[ap1][0])
                MAX_PRICE_VOLUME_ASK[ap1][0] = av1
                MAX_PRICE_VOLUME_ASK[ap1][2] = MAX_PRICE_VOLUME_ASK[ap1][1]/(av1+1)
                MAX_PRICE_VOLUME_ASK[ap1][4] = MAX_PRICE_VOLUME_ASK[ap1][4] + 1
                if MAX_PRICE_VOLUME_ASK[ap1][2] > 0.5:
                    MAX_PRICE_VOLUME_ASK[ap1][3] = MAX_PRICE_VOLUME_ASK[ap1][3]+1
                    MAX_PRICE_VOLUME_ASK[ap1][5] = time
            else:
                MAX_PRICE_VOLUME_ASK[ap1] = [av1,0,0,0,0,time]

            if bp1 in key_bid:
                MAX_PRICE_VOLUME_BID[bp1][0] = max(bv1,MAX_PRICE_VOLUME_BID[bp1][0])
                if MAX_PRICE_VOLUME_BID[bp1][0] == bv1:
                    MAX_PRICE_VOLUME_BID[bp1][1] = time
            else:
                MAX_PRICE_VOLUME_BID[bp1] = [bv1 ,time]

            if bp1 in key_bid:

                MAX_PRICE_VOLUME_BID[bp1][1] =(bv1 -MAX_PRICE_VOLUME_BID[bp1][0])
                MAX_PRICE_VOLUME_BID[bp1][0] = bv1
                MAX_PRICE_VOLUME_BID[bp1][2] = MAX_PRICE_VOLUME_BID[bp1][1]/(bv1+1)
                MAX_PRICE_VOLUME_BID[bp1][4] = MAX_PRICE_VOLUME_BID[bp1][4] + 1
                if MAX_PRICE_VOLUME_BID[bp1][2] > 0.5:
                    MAX_PRICE_VOLUME_BID[bp1][3] = MAX_PRICE_VOLUME_BID[bp1][3]+1
                    MAX_PRICE_VOLUME_ASK[bp1][5] = time
            else:
                MAX_PRICE_VOLUME_BID[bp1] = [av1,0,0,0,0,time]
        ask_df = pd.DataFrame.from_dict(MAX_PRICE_VOLUME_ASK,orient='index',columns = ['volume_ask','diff_v_ask','diff_r_ask','diff_t_ask','diff_l_ask','time_ask'])
        bid_df = pd.DataFrame.from_dict(MAX_PRICE_VOLUME_BID,orient='index',columns =  ['volume_bid','diff_v_bid','diff_r_bid','diff_t_bid','diff_l_bid','time_bid'])
        df = pd.merge(ask_df,bid_df,left_index=True,right_index= True,how = 'outer')
        return df

    def large_order(self,symbol,price = 5.9, closetime=' 14:50:00'):

        window = 50
        quotedata = stats.time_cut(symbol,closetime=closetime )
        Price_form = dict()

        price_ask,volume_ask,price_bid,volume_bid = stats.quote_cut(quotedata)
        quote_index = quotedata.index
        price_quote = price_ask.loc[:,:] ==price
        ask_cum_quote = 11 - pd.DataFrame(price_quote.cumsum(axis=1).sum(axis = 1),columns = ['location'])
        ask_cum_quote.loc[ ask_cum_quote.loc[:,'location'] == 11,'location'] = 0
        volume_ask.columns = price_ask.columns
        price_quote = pd.DataFrame(price_quote,dtype=int)

        vol_ask =  price_quote*volume_ask
        vol_ask = vol_ask.sum(axis = 1)
        vol_ask = pd.DataFrame(vol_ask,columns = ['order'])

        price_quote = price_bid.loc[:,:] ==price
        volume_bid.columns = price_bid.columns
        bid_cum_quote = -(11 - pd.DataFrame(price_quote.cumsum(axis=1).sum(axis=1), columns=['location']))
        bid_cum_quote.loc[bid_cum_quote.loc[:, 'location'] == -11, 'location'] = 0
        price_quote = pd.DataFrame(price_quote,dtype=int)

        vol_bid =  price_quote*volume_bid
        vol_bid = -1*vol_bid.sum(axis = 1)
        vol_bid = pd.DataFrame(vol_bid,columns = ['order'])

        vol = vol_ask+vol_bid


        inorderbook = vol.loc[:,'order']==0
        vol.loc[inorderbook,:] = np.nan
        vol.fillna(method='ffill', inplace=True)
        vol_diff = vol.loc[:,'order'].diff()
        vol.loc[:, 'order_diff'] = vol_diff
        vol.loc[:, 'location'] = bid_cum_quote+ ask_cum_quote
        tv = quotedata.loc[:,'tradeVolume'].diff()
        #vol.loc[:,'tv'] = tv
        tradedata_price= stats.tradeorder(symbol,quotedata,price = price)
        vol.loc[:,'nVolume'] = tradedata_price.loc[:,' nVolume']
        vol.loc[:, 'nVolume'] = vol.loc[:,'nVolume'].fillna(0)
        vol.loc[:,'pre_order'] = vol.loc[:,'order']-vol.loc[:, 'order_diff']
        tempPositive = vol.loc[:,'pre_order'] >0
        tempNegative = vol.loc[:,'pre_order'] <0
        vol.loc[:,'temp'] = 0
        vol.loc[tempPositive,'temp'] = vol.loc[tempPositive, 'order_diff']
        vol.loc[tempNegative,'temp'] = -1*vol.loc[tempNegative, 'order_diff']
        vol.loc[:,'add_cancel'] =vol.loc[:,'temp'] +vol.loc[:,'nVolume']
        vol.loc[:,'add_cancel_abs'] =vol.loc[:,'add_cancel'].abs()
        diff_where = np.sign(vol.loc[:, 'order']).diff() !=0
        last_time = (list(vol.loc[diff_where,:].index)[-1])
        return max(vol.loc[last_time:,'add_cancel_abs'])


    def tradeorder(self,symbol,quotedata,price = 6.25):
        tradeData = self.tradeData[symbol]

        #print(tradeData.columns)
        price_loc = tradeData.loc[:,' nPrice'] == price
        tradeData_loc = tradeData.loc[price_loc,:]
        temp_quote_time = quotedata.index

        tradeData_ = tradeData_loc.resample('1S').sum()
        tradeData_ = tradeData_.cumsum()
        #print(tradeData_)
        #stats.check_file(pd.DataFrame(tradeData_))
        try:
            tradeData_ = tradeData_.loc[temp_quote_time, :]
            r_tradeData = tradeData_.diff()

            return r_tradeData
        except:
            print('AError')
            return tradeData_



    def total_price(self,symbol,closetime):
        quotedata = stats.time_cut(symbol,closetime = closetime)
        price_ask,volume_ask,price_bid,volume_bid = stats.quote_cut(quotedata)
        #temp = pd.Series(price_ask).value_count()
        price_list = (pd.Series(price_bid.loc[:,'bidPrice1']).unique())

        return price_list

    def high_obi(self,symbol,closetime):
        quotedata = stats.time_cut(symbol,closetime = closetime)

        quotedata.loc[:,'obi'] = np.log(quotedata.loc[:,'askVolume1'] /quotedata.loc[:,'bidVolume1'])
        quotedata.loc[:, 'tv'] = quotedata.loc[:,'tradeVolume'].diff(5)
        large_bid = quotedata.loc[:,'bidVolume1'] > quotedata.loc[:, 'tv']
        large_ask = quotedata.loc[:, 'askVolume1'] > quotedata.loc[:, 'tv']
        volume_bid, volume_ask,bid_loc,ask_loc = stats.large_order_count(quotedata,quotedata.loc[:, 'tv'])


        quotedata.loc[:,'large_bid'] =  volume_bid
        quotedata.loc[:,'large_ask'] = volume_ask
        #quotedata.loc[large_ask,'large_ask'] =  quotedata.loc[large_ask,'askPrice1']
        #quotedata.loc[:,'large_bid'].fillna(method='ffill', inplace=True)
        #quotedata.loc[:,'large_ask'].fillna(method='ffill', inplace=True)
        quotedata.loc[:,'large_width'] = quotedata.loc[:,'large_ask'] - quotedata.loc[:,'large_bid']
        bid_max = list()
        ask_max = list()
        for row in zip(quotedata.index ,volume_bid,volume_ask):
            times = str(row[0])[10:]
            print(times)
            bp = row[1]
            ap = row[2]
            bid_max.append(stats.large_order(symbol,bp,times))
            ask_max.append(stats.large_order(symbol,ap,times))
        quotedata.loc[:,'bid_max'] = bid_max
        quotedata.loc[:,'ask_max'] = ask_max
        quotedata.loc[:,'bid_loc'] = bid_loc
        quotedata.loc[:,'ask_loc'] = ask_loc

        return quotedata

    def large_order_count(self,quotedata,large_margin,num = 10):
        price_ask, volume_ask, price_bid, volume_bid = stats.quote_cut(quotedata)
        bid_ = (volume_bid).apply(lambda  x : x - np.asarray(large_margin))> 0
        bid_ = pd.DataFrame(bid_,dtype= int)
        price_bid.columns = bid_.columns
        bid_ = price_bid *bid_
        volume_zero = bid_ ==0
        bid_[volume_zero] = np.nan
        bid_ = bid_.max(axis =1)



        bid_loc = (price_bid).apply(lambda  x : x == np.asarray(bid_))
        bid_loc = pd.DataFrame(bid_loc,columns=volume_bid.columns,dtype= int)
        bid_loc =( bid_loc * volume_bid ).sum(axis=1)

        ask_ = (volume_ask).apply(lambda  x : x - np.asarray(large_margin))> 0
        ask_ = pd.DataFrame(ask_,dtype= int)
        #volume_loc = (volume_bid).apply(lambda x: x == np.asarray(volume_bid))
        price_ask.columns = ask_.columns
        ask_ = price_ask *ask_
        volume_zero = ask_ ==0
        ask_[volume_zero] = np.nan
        ask_ = ask_.min(axis =1)

        ask_loc = (price_ask).apply(lambda  x : x == np.asarray(ask_))
        ask_loc = pd.DataFrame(ask_loc,columns=volume_ask.columns,dtype= int)
        ask_loc =( ask_loc * volume_ask ).sum(axis=1)
        return bid_,ask_,bid_loc,ask_loc

    def temp(self):

        price_list = stats.total_price(symbol, closetime=' 14:57:00')
        price_rate_list = list()
        price_abs = list()
        price_sum = list()
        # price_list = [6.69]
        for price in price_list:
            # print(price)
            price_situation = stats.large_order(symbol, price, closetime=' 14:57:00')
            price_situation.loc[price_situation.loc[:, 'location'] == 0, 'add_cancel_abs'] = np.nan
            price_situation.loc[price_situation.loc[:, 'location'] == 0, 'add_cancel'] = np.nan
            price_rate = (price_situation.loc[:, 'add_cancel_abs'].mean() / (
                price_situation.loc[:, 'add_cancel_abs'].std()))
            price_rate_list.append(price_rate)
            price_abs.append(list(price_situation.loc[:, 'order'])[-1])
            price_sum.append(list(price_situation.loc[:, 'location'])[-1])

        q.loc[:, 'price'] = price_list
        q.loc[:, 'price_rate'] = price_rate_list
        q.loc[:, 'price_abs'] = price_abs
        q.loc[:, 'price_sum'] = price_sum
        stats.check_file(price_situation)
        return 0



if __name__ == '__main__':
    """
    test the class
    """
    # data = Data('E:/personalfiles/to_zhixiong/to_zhixiong/level2_data_with_factor_added','600030.SH','20170516')
    dataPath = '//192.168.0.145/data/stock/wind'
    ## /sh201707d/sh_20170703
    tradeDate = '20190314'
    symbols_path  = 'D:/SignalTest/SignalTest/ref_data/sh50.csv'
    symbol_list = pd.read_csv(symbols_path)

    symbols = ['601766.SH']


    data = Data.Data(dataPath,symbols, tradeDate,'' ,dataReadType= 'gzip', RAWDATA = 'True')
    stats   = Stats(symbols,tradeDate,data.quoteData,data.tradeData)
    #print(data.tradeData[symbols[0]])
    q = pd.DataFrame()
    for symbol in symbols:
        print(symbol)
        #quotedata = stats.zaopan_stats(symbol)
        #stats.cancel_order(symbol)
        #stats.price_filter()
        stats.cancel_order(symbol)
        #price_situation = stats.high_obi(symbol,' 14:55:00')
        #stats.check_file(price_situation)
    print('Test end')
