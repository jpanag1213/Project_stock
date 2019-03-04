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
        self.quoteData = quoteData
        self.outputpath = outputpath
        self.tradeData =  tradeData
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


    def plot(self):


        return 0


    def Filter(self):


        return 0



    def check_file(self,file):

        file.to_csv(self.outputpath + 'checkquote.csv')

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

    def factor(self):
        quotedata = self.quoteData


        return 0



    def price_change_volume(self,symbol):
        quotedata = self.quoteData[symbol]
        Columns = ['bidPrice1','bidPrice2','bidPrice3','askPrice1','askPrice2','askPrice3','bidVolume1','bidVolume2','bidVolume3','askVolume1','askVolume2','askVolume3']
        Quotedata = quotedata.loc[:,Columns]
        Quotedata.loc[:, 'midp'] = quotedata.loc[:, 'midp']
        Quotedata.loc[:, 'tradeNos'] = quotedata.loc[:, 'tradeNos']
        price_diff = ( Quotedata.loc[:,'midp'].diff() ) !=0
        Quotedata.loc[:, 'diff'] = 0

        Quotedata.loc[price_diff, 'diff'] = 1
        Quotedata.loc[:, 'tradeNos_diff '] = Quotedata.loc[:, 'tradeNos'].diff()
        Quotedata.loc[:, 'bidvolume_diff'] = Quotedata.loc[:,'bidVolume1'].diff()
        Quotedata.loc[:, 'askvolume_diff'] = Quotedata.loc[:,'askVolume1'].diff()
        max_bid_positive = list()
        max_ask_positive = list()
        max_bid_negative = list()
        max_ask_negative = list()

        for row in zip(Quotedata.loc[:,'diff'],Quotedata.loc[:,'bidvolume_diff'],Quotedata.loc[:,'askvolume_diff'],Quotedata.loc[:, 'tradeNos_diff '] ):
            pc = row[0]
            bidv = row[1]
            askv = row[2]
            numtrade= row[3]
            if pc !=0:
                max_bid_positive.append(0)
                max_ask_positive.append(0)
                max_bid_negative.append(0)
                max_ask_negative.append(0)
                mbp = 0
                map = 0
                mbn = 0
                man = 0
            else:
                if numtrade == 0:
                    numtrade = 1
                mbp = max(mbp,bidv/numtrade)
                map = max(map,askv/numtrade)
                mbn = min(mbn,bidv/numtrade)
                man = min(man,askv/numtrade)
                max_bid_positive.append(mbp)
                max_ask_positive.append(map)
                max_bid_negative.append(mbn)
                max_ask_negative.append(man)


        Quotedata.loc[:, 'max_bid_positive'] =max_bid_positive
        Quotedata.loc[:, 'max_ask_positive'] =max_ask_positive
        Quotedata.loc[:, 'max_bid_negative'] =max_bid_negative
        Quotedata.loc[:, 'max_ask_negative'] =max_ask_negative


    # todo: revise the obi signal here
        Quotedata.loc[:, 'obi'] = np.log(Quotedata.loc[:, 'bidVolume1']) - np.log(
            Quotedata.loc[:, 'askVolume1'])
        # Quotedata.loc[:, 'obi_' + str(window) + '_min'] = Quotedata.loc[:,
        #                                                                   'obi'].rolling(window * 60).mean()




        obi_change_list = list()
        last_obi = Quotedata.loc[:,'obi'].iloc[0]
        tick_count = 0
        row_count = 0
        for row in zip(Quotedata.loc[:, 'diff'], Quotedata.loc[:,'obi']):
            priceStatus = row[0]
            obi = row[1]
            if (priceStatus == 1) or np.isnan(priceStatus):
                tick_count = 0
                last_obi = obi
            else:
                last_obi = Quotedata.loc[:,'obi'].iloc[row_count - tick_count]

            tick_count = tick_count + 1
            row_count = row_count + 1
            obi_change = obi - last_obi
            obi_change_list.append(obi_change)

        Quotedata.loc[:, 'obichange'] = obi_change_list

        Quotedata.to_csv(self.outputpath +symbol+ '_quotedata.csv')

    def obi(self, symbol):
        quotedata = self.quoteData[symbol]
        Columns = ['bidPrice1','bidPrice2','askPrice1','askPrice2','bidVolume1','bidVolume2','askVolume1','askVolume2']
        Quotedata = quotedata.loc[:,Columns]
        Quotedata.loc[:, 'midp'] = quotedata.loc[:, 'midp']
        Quotedata.loc[:, 'bid_obi'] = np.log(Quotedata.loc[:, 'bidVolume1']+Quotedata.loc[:, 'bidVolume2']) - np.log( Quotedata.loc[:, 'askVolume1'])
        Quotedata.loc[:, 'ask_obi'] = np.log(Quotedata.loc[:, 'bidVolume1']) - np.log( Quotedata.loc[:, 'askVolume1']+Quotedata.loc[:, 'askVolume2'])
        Quotedata.loc[:, 'obi']     = np.log(Quotedata.loc[:, 'bidVolume1']) - np.log(Quotedata.loc[:, 'askVolume1'])
        price = Quotedata.loc[:,'midp'].iloc[0]
        node_price  = list()
        bid_obi_list = list()
        ask_obi_list = list()
        for row in zip(Quotedata.loc[:, 'bid_obi'],Quotedata.loc[:, 'ask_obi'],Quotedata.loc[:, 'midp'],Quotedata.loc[:, 'obi']):
            bid_obi = row[0]
            ask_obi = row[1]
            pric    = row[2]
            obi     = row[3]
            if pric  ==  price:
                bid_obi_list.append(ask_obi)
                ask_obi_list.append(bid_obi)
            elif ((pric-price) < 0.006)&((pric-price) > 0.004):
                bid_obi_list.append(obi)
                ask_obi_list.append(np.nan)
            elif ((pric-price) < -0.004)&((pric-price) > -0.006):
                bid_obi_list.append(np.nan)
                ask_obi_list.append(obi)

            elif  ((pric-price) < 0.011)&((pric-price) > 0.009):
                bid_obi_list.append(bid_obi)
                ask_obi_list.append(np.nan)
            elif  ((pric-price) >- 0.011)&((pric-price) <- 0.009):
                bid_obi_list.append(np.nan)
                ask_obi_list.append(ask_obi)
            elif ((pric-price) >0.011):
                bid_obi_list.append(ask_obi)
                ask_obi_list.append(bid_obi)
                price =pric
            elif ((pric-price)<-0.011):
                bid_obi_list.append(ask_obi)
                ask_obi_list.append(bid_obi)
                price = pric
            else:
                bid_obi_list.append(ask_obi)
                ask_obi_list.append(bid_obi)
            node_price.append(price)
        Quotedata.loc[:, 'BID_o'] = bid_obi_list
        Quotedata.loc[:, 'ASK_o'] = ask_obi_list
        Quotedata.loc[:, 'node_price'] = node_price


        node_diff = Quotedata.loc[:, 'node_price'].diff()
        price_change =abs (node_diff )>0.011
        Quotedata.loc[:, 'node_price_change'] = 0
        Quotedata.loc[price_change, 'node_price_change'] = 1




        # Quotedata.loc[:, 'obi_' + str(window) + '_min'] = Quotedata.loc[:,
        #                                                                 'obi'].rolling(window * 60).mean()

        obi_bid_change = list()
        obi_ask_change = list()
        last_obi = Quotedata.loc[:,'obi'].iloc[0]
        last_obi_bid = Quotedata.loc[:,'BID_o'].iloc[0]
        last_obi_ask = Quotedata.loc[:,'ASK_o'].iloc[0]
        tick_count = 0
        row_count = 0
        for row in zip(Quotedata.loc[:, 'node_price_change'],Quotedata.loc[:, 'ASK_o'],Quotedata.loc[:, 'BID_o']):
            priceStatus = row[0]
            obi_a = row[1]
            obi_b = row[2]
            if (priceStatus == 1) or np.isnan(priceStatus):
                tick_count = 0
                last_obi_bid = obi_b
                last_obi_ask = obi_a
            else:
                last_obi_bid = Quotedata.loc[:,'BID_o'].iloc[row_count - tick_count]
                last_obi_ask = Quotedata.loc[:,'ASK_o'].iloc[row_count - tick_count]

            tick_count = tick_count + 1
            row_count = row_count + 1
            obi_change_b = obi_b - last_obi_bid
            obi_change_a = obi_a - last_obi_ask

            obi_bid_change.append(obi_change_b)
            obi_ask_change.append(obi_change_a)

        Quotedata.loc[:, 'obi_bid_change'] = obi_bid_change
        Quotedata.loc[:, 'obi_ask_change'] = obi_ask_change

        Quotedata.to_csv(self.outputpath +symbol+ '_quotedata.csv')


    def distribution(self,symbol):
        quotedata = self.quoteData[symbol]
        obi = np.log(quotedata.loc[:,'askVolume1'] / quotedata.loc[:,'bidVolume1'])
        #print(obi)
        quotedata.loc[:,'obi'] = obi

        Quotedata = pd.DataFrame()
        Columns = ['bidPrice1','bidPrice2','bidPrice3','askPrice1','askPrice2','askPrice3','bidVolume1','bidVolume2','bidVolume3','askVolume1','askVolume2','askVolume3']
        Quotedata = quotedata.loc[:,Columns]
        Quotedata.loc[:, 'obi'] = obi
        Quotedata.loc[:, 'ewm_bv1'] = Quotedata.loc[:,'bidVolume1'].ewm(30).mean()
        Quotedata.loc[:, 'ewm_av1'] = Quotedata.loc[:, 'askVolume1'].ewm(30).mean()
        Quotedata.loc[:, 'ewm_obi'] = np.log(Quotedata.loc[:,'ewm_av1'] / Quotedata.loc[:,'ewm_bv1'])
        Quotedata.loc[:, 'ewm_std'] = Quotedata.loc[:, 'ewm_obi'].ewm(30).std()
        Quotedata.loc[:, 'ewm_mean'] = Quotedata.loc[:, 'ewm_obi'].ewm(30).mean()
        upper = Quotedata.loc[:, 'ewm_obi']  >  Quotedata.loc[:, 'ewm_mean'] +  2*Quotedata.loc[:, 'ewm_std']
        lower = Quotedata.loc[:, 'ewm_obi']  <  Quotedata.loc[:, 'ewm_mean'] -  2*Quotedata.loc[:, 'ewm_std']
        Quotedata.loc[:,'over_obi'] = 0
        Quotedata.loc[upper,'over_obi'] = 1
        Quotedata.loc[lower,'over_obi'] = -1
        Quotedata.loc[:,'midp'] = quotedata.loc[:,'midp']
        price_diff = ( Quotedata.loc[:,'midp'].diff() ) !=0
        Quotedata.loc[:, 'diff'] = price_diff
        print(self.outputpath  +  'quotedata.csv')
        Quotedata.to_csv(self.outputpath  +  'quotedata.csv')


        return 0
    def data_link(self,symbol):
        quotedata = self.quoteData[symbol]

        print(1)
        return 0

    def vol2diffop(self, symbol):
        stats_ = pd.DataFrame()
        open_interest = self.futureData[symbol].loc[:,'open_interest']
        int_diff = open_interest.diff(240)
        midp = self.futureData[symbol].loc[:,'open_interest']
        midp_std = midp.rolling(240).std()
        midp_std_diff = midp_std .diff(240)
        stats_.loc[:,'midp_std'] = midp_std_diff
        stats_.loc[:,'int_diff'] = int_diff

        print(stats_)
        stats_.to_csv(self.outputpath  + 'stats_vol.csv')
        print(self.outputpath  + 'stats_vol.csv')

    '''
    Test：统计的内容
    尝试以下两个现象
    1.突破当前盘口和突破下一个盘口所用的vol之比 价格粘稠性
    2.
    
    
    '''
    def volatility(self,symbol,window, type = 'std'):
        price = self.quoteData[symbol].loc[:,'midp']

        if type == 'std':
            std_ = price.rolling(window).std()
        elif type == 'quote_volatility':
            bid_Volume10 =  (self.quoteData[symbol].loc[:,'bidVolume1']+self.quoteData[symbol].loc[:,'bidVolume2']+self.quoteData[symbol].loc[:,'bidVolume3'])* 1 / 10
            ask_Volume10 =  (self.quoteData[symbol].loc[:,'askVolume1']+self.quoteData[symbol].loc[:,'askVolume2']+self.quoteData[symbol].loc[:,'askVolume3'])* 1 / 10
            bid_Volume10_2=   (self.quoteData[symbol].loc[:, 'bidVolume1'] + self.quoteData[symbol].loc[:, 'bidVolume2'])
            ask_Volume10_2 =   (self.quoteData[symbol].loc[:, 'askVolume1'] + self.quoteData[symbol].loc[:, 'askVolume2'])
            bid_price = (bid_Volume10 < self.quoteData[symbol].loc[:,'bidVolume1'] ) +   2 * ((bid_Volume10 > self.quoteData[symbol].loc[:,'bidVolume1'] ) &  (bid_Volume10 < bid_Volume10_2 ) )
            ask_price = (ask_Volume10 < self.quoteData[symbol].loc[:,'askVolume1'] ) +   2 * ((ask_Volume10 > self.quoteData[symbol].loc[:,'askVolume1'] ) &  (ask_Volume10 < ask_Volume10_2 ) )
            self.quoteData[symbol].loc[:, 'bid_per10'] = self.quoteData[symbol].loc[:, 'bidPrice1']
            self.quoteData[symbol].loc[:, 'ask_per10'] = self.quoteData[symbol].loc[:, 'askPrice1']
            self.quoteData[symbol].loc[:, 'bid_vol10'] = self.quoteData[symbol].loc[:, 'bidVolume1']
            self.quoteData[symbol].loc[:, 'ask_vol10'] = self.quoteData[symbol].loc[:, 'askVolume1']

            self.quoteData[symbol].loc[bid_price == 2, 'bid_per10'] = self.quoteData[symbol].loc[bid_price == 2, 'bidPrice2']
            self.quoteData[symbol].loc[bid_price == 0, 'bid_per10'] = self.quoteData[symbol].loc[bid_price == 0, 'bidPrice3']
            self.quoteData[symbol].loc[ask_price == 2, 'ask_per10'] = self.quoteData[symbol].loc[ask_price == 2, 'askPrice2']
            self.quoteData[symbol].loc[ask_price == 0, 'ask_per10'] = self.quoteData[symbol].loc[ask_price == 0, 'askPrice3']
            self.quoteData[symbol].loc[self.quoteData[symbol].loc[:, 'ask_per10']==0, 'ask_per10'] =np.nan
            self.quoteData[symbol].loc[:,'spread'] = self.quoteData[symbol].loc[:, 'ask_per10']  - self.quoteData[symbol].loc[:, 'bid_per10']
            self.quoteData[symbol].loc[:, 'midp_10'] = (self.quoteData[symbol].loc[:, 'ask_per10'] +  self.quoteData[symbol].loc[:,'bid_per10'] )/2

            quote_time = pd.to_datetime(self.quoteData[symbol].exchangeTime.values).values
            standard_start = quote_time[0] - 3 * 1000000000
            # np.datetime64('2018-07-31T09:30:06.000000000')
            tradeData = self.tradeData[symbol]
            # print(tradeData.columns)
            # print(tradeData.loc[:,' nBSFlag'])

            bid_order = tradeData.loc[:, ' nBSFlag'] == 'B'
            ask_order = tradeData.loc[:, ' nBSFlag'] == 'S'
            can_order = tradeData.loc[:, ' nBSFlag'] == ' '
            tradeData.loc[bid_order, 'numbs_flag'] = 1
            tradeData.loc[ask_order, 'numbs_flag'] = -1
            tradeData.loc[can_order, 'numbs_flag'] = 0

            pos = tradeData.loc[:, ' nPrice'] == 0

            tradeData.loc[:, 'temp'] = tradeData.loc[:, ' nPrice']
            tradeData.loc[pos, 'temp'] = np.nan
            tradeData.temp.fillna(method='ffill', inplace=True)
            lastrep = list(tradeData.temp.values[:-1])
            lastrep.insert(0, 0)
            lastrep = np.asarray(lastrep)
            tradeData_quote = pd.merge(tradeData, self.quoteData[symbol].loc[:,
                                                  ['bid_per10', 'ask_per10', 'bid_vol10', 'ask_vol10']], left_index=True,
                                       right_index=True, how='outer')
            tradeData_quote['bid_per10'].fillna(method='ffill', inplace=True)
            tradeData_quote['ask_per10'].fillna(method='ffill', inplace=True)
            # tradeData_quote.to_csv(self.dataSavePath + './' + str(self.tradeDate.date()) + signal + ' ' + symbol + '.csv')
            ActiveBuy = (tradeData_quote.loc[:, 'numbs_flag'] == 1) & (
                        tradeData_quote.loc[:, ' nPrice'] == tradeData_quote.loc[:, 'ask_per10'])
            ActiveSell = (tradeData_quote.loc[:, 'numbs_flag'] == -1) & (
                        tradeData_quote.loc[:, ' nPrice'] == tradeData_quote.loc[:, 'bid_per10'])
            OverBuy = (tradeData_quote.loc[:, 'numbs_flag'] == 1) & (
                        tradeData_quote.loc[:, ' nPrice'] > tradeData_quote.loc[:, 'ask_per10'])
            OverSell = (tradeData_quote.loc[:, 'numbs_flag'] == -1) & (
                        tradeData_quote.loc[:, ' nPrice'] < tradeData_quote.loc[:, 'bid_per10'])
            PassiveBuy = (tradeData_quote.loc[:, 'numbs_flag'] == 1) & (
                        tradeData_quote.loc[:, ' nPrice']  < tradeData_quote.loc[:, 'ask_per10'])
            PassiveSell = (tradeData_quote.loc[:, 'numbs_flag'] == -1) & (
                        tradeData_quote.loc[:, ' nPrice'] > tradeData_quote.loc[:, 'bid_per10'])
            tradeData_quote.loc[ActiveBuy, 'ActiveBuy'] = tradeData_quote.loc[ActiveBuy, ' nVolume']
            tradeData_quote.loc[ActiveSell, 'ActiveSell'] = tradeData_quote.loc[ActiveSell, ' nVolume']
            tradeData_quote.loc[OverBuy, 'OverBuy'] = tradeData_quote.loc[OverBuy, ' nVolume']
            tradeData_quote.loc[OverSell, 'OverSell'] = tradeData_quote.loc[OverSell, ' nVolume']
            tradeData_quote.loc[PassiveBuy, 'PassiveBuy'] = tradeData_quote.loc[PassiveBuy, ' nVolume']
            tradeData_quote.loc[PassiveSell, 'PassiveSell'] = tradeData_quote.loc[PassiveSell, ' nVolume']

            kk = list(quote_time)
            kk.insert(0, standard_start)
            temp_quote_time = np.asarray(kk)
            # resample_tradeData = resample_tradeData.loc[temp_quote_time]
            Columns_ = ['ActiveBuy', 'ActiveSell', 'OverBuy', 'OverSell', 'PassiveBuy', 'PassiveSell']
            resample_tradeData = tradeData_quote.loc[:, Columns_].resample('1S', label='right', closed='right').sum()
            resample_tradeData = resample_tradeData.cumsum()
            resample_tradeData = resample_tradeData.loc[temp_quote_time, :]
            r_tradeData = resample_tradeData.diff()
            quote_order = pd.merge(self.quoteData[symbol].loc[:,['midp','midp_10','spread']], r_tradeData, left_index=True, right_index=True,how='left')
            #.loc[:,'midp'] =self.quoteData[symbol].loc[:,'midp']

            quote_order.to_csv(self.outputpath+ './ quote_order.csv')
            #self.quoteData[symbol].loc[:, ['midp', 'bidVolume1', 'askVolume1']].to_csv(self.outputpath + './ quote_o.csv')

    def ox_ob(self):
            # todo: revise the obi signal here
            ex_ob_ = list()
            ex_ob_.append(0)
            ex_ob_bid = list()
            ex_ob_bid.append(0)
            ex_ob_ask = list()
            ex_ob_ask.append(0)
            ex_ob_ahead = list()
            ex_ob_ahead.append(0)
            self.allQuoteData[symbol].loc[:, 'obi_'] = (self.allQuoteData[symbol].loc[:, 'bidVolume1'] + self.allQuoteData[symbol].loc[:,'bidVolume2']
                                                        + self.allQuoteData[symbol].loc[:, 'bidVolume3']
                                                        ) / (self.allQuoteData[symbol].loc[:, 'askVolume1'] +
                                                             self.allQuoteData[symbol].loc[:, 'askVolume2']
                                                             + self.allQuoteData[symbol].loc[:, 'askVolume3'])

            lth = len(self.allQuoteData[symbol].loc[:, 'bidVolume1'])

            fac = []
            # ap_list = ['askPrice1', 'askPrice2', 'askPrice3']#, 'askPrice4', 'askPrice5']
            # bp_list = ['bidPrice1', 'bidPrice2', 'bidPrice3']#, 'bidPrice4', 'bidPrice5']

            ap_list = ['askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5']
            bp_list = ['bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5']

            # av_list = ['askVolume1', 'askVolume2', 'askVolume3']#, 'askVolume4', 'askVolume5']
            # bv_list = ['bidVolume1', 'bidVolume2', 'bidVolume3']#, 'bidVolume4', 'bidVolume5']

            av_list = ['askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']
            bv_list = ['bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5']

            ap_list_ahead = ['askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5']
            bp_list_ahead = ['bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5']

            av_list_ahead = ['askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']
            bv_list_ahead = ['bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5']
            for i in range(1, lth):

                if (i == 1):
                    Askp_array = np.array(())
                    Askv_array = np.array(())
                    bidp_array = np.array(())
                    bidv_array = np.array(())
                    pre_midp = (self.allQuoteData[symbol].bidPrice1.values[i] + self.allQuoteData[symbol].askPrice1.values[
                        i]) / 2
                pre_ask = np.sum(Askv_array)
                pre_bid = np.sum(bidv_array)

                if (i > 1):
                    bid_pos_ind = bidp_array <= self.allQuoteData[symbol].bidPrice1.values[i]
                    ask_pos_ind = Askp_array >= self.allQuoteData[symbol].askPrice1.values[i]
                    bidp_array = bidp_array[bid_pos_ind]
                    Askp_array = Askp_array[ask_pos_ind]
                    bidv_array = bidv_array[bid_pos_ind]
                    Askv_array = Askv_array[ask_pos_ind]
                for bp, ap, bv, av in zip(bp_list, ap_list, bv_list, av_list):
                    if (self.allQuoteData[symbol][bp][i] <= self.allQuoteData[symbol].bidPrice1.values[i]):
                        if (self.allQuoteData[symbol][bp][i] not in bidp_array):
                            bidp_array = np.append(bidp_array, self.allQuoteData[symbol][bp][i])
                            bidv_array = np.append(bidv_array, self.allQuoteData[symbol][bv][i])
                        else:
                            assert (len(np.where(bidp_array == self.allQuoteData[symbol][bp][i])[0]) == 1)
                            bidv_array[np.where(bidp_array == self.allQuoteData[symbol][bp][i])[0][0]] = \
                            self.allQuoteData[symbol][bv][i]

                    if (self.allQuoteData[symbol][ap][i] >= self.allQuoteData[symbol].askPrice1.values[i]):
                        if (self.allQuoteData[symbol][ap][i] not in Askp_array):
                            Askp_array = np.append(Askp_array, self.allQuoteData[symbol][ap][i])
                            Askv_array = np.append(Askv_array, self.allQuoteData[symbol][av][i])
                        else:
                            assert (len(np.where(Askp_array == self.allQuoteData[symbol][ap][i])[0]) == 1)
                            Askv_array[np.where(Askp_array == self.allQuoteData[symbol][ap][i])[0][0]] = \
                            self.allQuoteData[symbol][av][i]

                Now_ask = np.sum(Askv_array)

                Now_bid = np.sum(bidv_array)

                midPriceChange = self.allQuoteData[symbol]['midp'].diff()

                self.allQuoteData[symbol].loc[:, 'priceChange'] = 1

                temp_back_ = (Now_bid - pre_bid) * - (Now_ask - pre_ask)

                if (i > 200) & ((self.allQuoteData[symbol].bidPrice1.values[i] + self.allQuoteData[symbol].askPrice1.values[
                    i]) / 2 == pre_midp):

                    ex_ob_.append((Now_bid - pre_bid) - (Now_ask - pre_ask))
                else:
                    ex_ob_.append(0.0)
                pre_midp = (self.allQuoteData[symbol].bidPrice1.values[i] + self.allQuoteData[symbol].askPrice1.values[
                    i]) / 2
                ex_ob_ask.append((Now_ask - pre_ask))
                ex_ob_bid.append((Now_bid - pre_bid))

            self.allQuoteData[symbol].loc[:, 'obi_diff'] = ex_ob_
            self.allQuoteData[symbol].loc[:, 'obi_bid_diff'] = ex_ob_bid
            self.allQuoteData[symbol].loc[:, 'obi_ask_diff'] = ex_ob_ask
            positivePos = (self.allQuoteData[symbol]['obi_bid_diff'] > 10) & (self.allQuoteData[symbol]['obi_'] > 1.1)
            negativePos = (self.allQuoteData[symbol]['obi_ask_diff'] > 10) & (self.allQuoteData[symbol]['obi_'] < 1 / 1.1)

            # pd.concat(q, 0).to_csv(outputpath + './' + tradingDay + '.csv')
            self.allQuoteData[symbol].loc[positivePos, signal + '_' + str(window) + '_min'] = 1
            self.allQuoteData[symbol].loc[negativePos, signal + '_' + str(window) + '_min'] = -1
            self.allQuoteData[symbol].loc[(~positivePos) & (~negativePos), signal + '_' + str(window) + '_min'] = 0
            # self.allQuoteData[symbol].loc[:,''] =
            # self.allQuoteData[symbol].loc[:, 'obi' + str(window) + '_min_sum'] = self.allQuoteData[symbol].loc[:,'obi'].rolling(window * 60).sum()
            # todo: 把几层obi当作一层看待，适合高价股？

            # print out the dataframe
            q = self.allQuoteData[symbol]
            q.to_csv(self.dataSavePath + './' + str(self.tradeDate.date()) + signal + ' ' + symbol + '.csv')
            print('Calculate obi here for symbol = ', symbol, 'with lbwindow = ', window)
    def lastNday(self,tradingday,tradingDates):
        tradingDayFile ='./ref_data/TradingDay.csv'
        tradingDays = pd.read_csv(tradingDayFile)

    def opentime(self,symbol):
        quoteData = self.quoteData[symbol]
        quoteData.loc[:,'opentime'] = 0
        quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 14:50:00'), '%Y%m%d %H:%M:%S'),'opentime'] = 1
        quoteData.loc[
        datetime.datetime.strptime(str(self.tradeDate + ' 14:50:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 14:57:00'), '%Y%m%d %H:%M:%S'), 'opentime'] =2
        stats.check_file(quoteData)
        return 0

    def responseFun(self,symbol):
        quoteData =  self.quoteData[symbol]
        quoteD = pd.DataFrame()



if __name__ == '__main__':
    """
    test the class
    """
    # data = Data('E:/personalfiles/to_zhixiong/to_zhixiong/level2_data_with_factor_added','600030.SH','20170516')
    dataPath = '//192.168.0.145/data/stock/wind'
    ## /sh201707d/sh_20170703
    tradeDate = '20190226'

    symbols = ['000001.SZ']
    # exchange = symbol.split('.')[1].lower()
    #print(dataPath)
    data = Data.Data(dataPath,symbols, tradeDate,'' ,dataReadType= 'gzip', RAWDATA = 'True')
    stats   = Stats(symbols,tradeDate,data.quoteData,data.tradeData)
    stats.opentime(symbols[0])
    #print(data.quoteData[symbols[0]].loc[:,'midp'])
    #stats.vol2diffop(symbols[0])
    #stats.volatility(symbols[0],20,type = 'quote_volatility')
    #stats.Evaluation_times()
    # signalTester.CompareSectorAndStock(symbols[0], orderType='netMainOrderCashFlow')
    #stats.distribution(symbols[0])


    print('Test end')
