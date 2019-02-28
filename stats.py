# -*- coding: utf-8 -*-
"""
Created on 2019-02-23

to stats the order imbalance feature

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

class Stats(object):

    def __init__(self, symbol, quoteData,futureData =None, outputpath = 'E://stats_test/'):
        self.symbol    = symbol
        self.quoteData = quoteData
        self.outputpath = outputpath
        self.futureData = futureData
        print(os.path.exists(outputpath))
        if os.path.exists(outputpath) is False:
            os.makedirs(outputpath)

    def plot(self):


        return 0


    def Filter(self):


        return 0


    def Evaluation(self):


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



if __name__ == '__main__':
    """
    test the class
    """
    # data = Data('E:/personalfiles/to_zhixiong/to_zhixiong/level2_data_with_factor_added','600030.SH','20170516')
    dataPath = '//192.168.0.145/data/stock/wind'
    ## /sh201707d/sh_20170703
    tradeDate = '20190226'

    symbols = ['IC.CF']
    # exchange = symbol.split('.')[1].lower()
    #print(dataPath)
    data = Data.Data(dataPath, '', tradeDate,futureSymbols = symbols ,dataReadType= 'gzip', RAWDATA = 'True')
    stats   = Stats(symbols,data.quoteData,data.futureData)

    #print(data.futureData[symbols[0]].loc[:,'midp'])
    stats.vol2diffop(symbols[0])
    # signalTester.CompareSectorAndStock(symbols[0], orderType='netMainOrderCashFlow')
    #stats.distribution(symbols[0])


    print('Test end')
