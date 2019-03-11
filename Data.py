# -*- coding: utf-8 -*-
"""
Created on 2017-09-08

@author: zhixiong

use: to load the tick data from the source and process the data into the struct we want
"""
import pandas as pd
import numpy as np
import datetime
import os
# from multiprocessing.dummy import Pool as ThreadPool  # IO密集型使用dummy，CPU密集型任务选择multiprocessing
# from multiprocessing.dummy import Process  # 使用多进程而非多线程
# from multiprocessing import Pool as ThreadPool  # IO密集型使用dummy，CPU密集型任务选择multiprocessing
# from itertools import repeat  # repeat  用于多线程时传入其他固定参数
# import time  # 测试时间

class Data(object):

    def __init__(self, dataPath, symbol, tradeDate,futureSymbols = '', indexmicode='',freq = '1S',dataReadType = 'csv',RAWDATA = 'False'):
        """

        :param dataPath: data store path
        :param symbol: stock symbol
        :param tradeDate: trading day
        :param indexmicode: index code, such as 000300.SH or 000016.SH
        """
        self.dataPath       = dataPath
        ##
        # self.dataPath = 'C:/Users/lightsz04/Documents/StockTickData/sh201707d/sh_20170703'
        self.symbol         = symbol
        self.indexmiccode   = indexmicode
        self.tradeDate      = tradeDate
        self.quoteqa        = list()
        self.tradeqa        = list()
        self.futureSymbol   = futureSymbols
        self.freq           = freq
        self.dataReadType   = dataReadType
        self.RAWDATA        = RAWDATA
        # self.hdf_buffer     = hdf_buffer.HdfBuffer('TickData_' + freq)
        if dataReadType == 'csv':
            self.priceMultiplier = 10000
            self.volumeMultiplier = 1
        elif dataReadType == 'gzip':
            self.priceMultiplier = 1
            self.volumeMultiplier = 1

        ## data part.
        self.quoteData  = self.StructQuoteData()
        # self.quoteData = self.ReadQuoteData()
        # self.symbolQuoteData = self.quoteData[self.quoteData.loc[:,'windcode'] == symbol]
        # self.tradeData  = pd.DataFrame()
        #self.tradeData = self.StructTradeData()
        # self.orderData  = pd.DataFrame()
        # self.orderData  = self.StructOrderData()
        # self.indexData = pd.DataFrame()
        if indexmicode == '':
            self.indexData  = pd.DataFrame()
        else:
            self.indexData  = self.StructIndexData()
        # self.indexData = self.quoteData[self.quoteData.loc[:,'windcode'] == indexmicode]
        self.queueData  = pd.DataFrame()
        if self.futureSymbol is not '':
            self.futureData = self.StructFutureData()
        else:
            self.futureData = pd.DataFrame()
        self.etfData    = pd.DataFrame()


    """ the following function will be used to generate data"""

    # TODO: split the data from current data structure and extract the symbols we need.

    def ReadSingleQuoteData(self,dataPath,symbol,tradeDate,type = 'stock'):
        """
        used to construct the quote data with factors
        :return:quoteData with factors, like
        """
        # quoteData = pd.read_csv(self.dataPath + '/' + self.symbol + '_' + self.tradeDate + '_quoteqa.csv')
        if type is 'stock':
            # exchange = symbol.split('.')[1].lower()
            # dataPath = dataPath + '/' + exchange + tradeDate[:6] + 'd' + '/' + exchange + '_' + tradeDate
            # fileName = dataPath + '/' + symbol.split('.')[0] + '_' + tradeDate + '.csv'
            fileName = dataPath + '/' + symbol + '_' + tradeDate + '_quote.csv'
            # colnames: 市场代码,证券代码,时间(yyyy-mm-dd),最新,成交笔数,成交额,成交量,方向,买一价,买二价,买三价,买四价,买五价,卖一价,卖二价,卖三价,卖四价,卖五价,买一量,买二量,买三量,买四量,买五量,卖一量,卖二量,卖三量,卖四量,卖五量
            # msgtime, msgsource, msgtype, msgid, nanotime, time, nTime, szWindCode, szCode, nActionDay, nTradingDay, nStatus, nPreClose, nOpen, nHigh, nLow, nMatch, nNumTrades, iVolume, iTurnover, nTotalBidVol, nTotalAskVol, nWeightedAvgBidPrice, nWeightedAvgAskPrice, nIOPV, nYieldToMaturity, nHighLimited, nLowLimited, chPrefix
            # askprice01, askvolume01, askprice02, askvolume02, askprice03, askvolume03, askprice04, askvolume04, askprice05, askvolume05, askprice06, askvolume06, askprice07, askvolume07, askprice08, askvolume08, askprice09, askvolume09, askprice10, askvolume10, bidprice01, bidvolume01, bidprice02, bidvolume02, bidprice03, bidvolume03, bidprice04, bidvolume04, bidprice05, bidvolume05, bidprice06, bidvolume06, bidprice07, bidvolume07, bidprice08, bidvolume08, bidprice09, bidvolume09, bidprice10, bidvolume10
            columns = ['exchangeCode', 'micCode', 'exchangeTime', 'latest', 'tradeNos',
                       'totalTurnover', 'tradeVolume', 'side',
                       'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
                       'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',
                       'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',
                       'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']
        elif type is 'index':
            exchange = symbol.split('.')[1].lower()
            dataPath = dataPath + '/index_tick_' + exchange + tradeDate[:6] + 'd' + '/' + exchange + '_' + tradeDate
            fileName = dataPath + '/' + symbol.split('.')[0] + '_' + tradeDate + '.csv'
            # colnames: 市场代码,证券代码,时间(yyyy-mm-dd),最新,成交笔数,成交额,成交量,方向,买一价,买二价,买三价,卖一价,卖二价,卖三价,买一量,买二量,买三量,卖一量,卖二量,卖三量
            columns = ['exchangeCode', 'micCode', 'exchangeTime', 'latest', 'tradeNos',
                       'totalTurnover', 'tradeVolume', 'side',
                       'bidPrice1', 'bidPrice2', 'bidPrice3',
                       'askPrice1', 'askPrice2', 'askPrice3',
                       'bidVolume1', 'bidVolume2', 'bidVolume3',
                       'askVolume1', 'askVolume2', 'askVolume3']
        elif ('IH' in type) or ('IC' in type) or ('IF' in type):
            dataPath = dataPath + '/sf5_c' + tradeDate[:6] + 'd' + '/' + tradeDate
            files = os.listdir(dataPath)
            contractDate = '1711'
            for file in files:
                if type + contractDate in file:
                    break
            fileName = dataPath + '/' + file
            # fileName = dataPath + '/' + symbol.split('.')[0] + '_' + tradeDate + '.csv'
            # 市场代码,合约代码,时间,最新,持仓,增仓,成交额,成交量,开仓,平仓,成交类型,方向,买一价,买二价,买三价,买四价,买五价,卖一价,卖二价,卖三价,卖四价,卖五价,买一量,买二量,买三量,买四量,买五量,卖一量,卖二量,卖三量,卖四量,卖五量

            columns = ['exchangeCode', 'micCode', 'exchangeTime', 'latest', 'oi','add_oi',
                       'totalTurnover', 'tradeVolume', 'openPosition', 'closePosition', 'transactionType', 'side',
                       'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
                       'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',
                       'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',
                       'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']
        elif type is 'ETF':
            exchange = symbol.split('.')[1].lower()
            dataPath = dataPath + '/' + exchange + tradeDate[:6] + 'd' + '/' + exchange + '_' + tradeDate
            fileName = dataPath + '/' + symbol.split('.')[0] + '_' + tradeDate + '.csv'
            # fileName = dataPath + '/' + symbol.split('.')[0] + '_' + tradeDate + '.csv'
            # 市场代码,合约代码,时间,最新,持仓,增仓,成交额,成交量,开仓,平仓,成交类型,方向,买一价,买二价,买三价,买四价,买五价,卖一价,卖二价,卖三价,卖四价,卖五价,买一量,买二量,买三量,买四量,买五量,卖一量,卖二量,卖三量,卖四量,卖五量

            columns = ['exchangeCode', 'micCode', 'exchangeTime', 'latest', 'oi','add_oi',
                       'tradeVolume', 'totalTurnover','openPosition','closePosition','transactionType' ,'side',
                       'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
                       'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',
                       'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',
                       'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']

        else:
            print('Type is incorrect')
        # fileName = dataPath + '/' + symbol.split('.')[0] + '_' + tradeDate + '.csv'
        if os.path.exists(fileName) is False:
            return None
        quoteData = pd.read_csv(fileName,skiprows = 1,header=None,encoding= 'oem') ## NOTICE: need to check the file encoding before reading the file
        quoteData.columns = columns
        # quoteData.index = map(lambda stime: datetime.datetime.strptime(str(stime)[:19], '%Y-%m-%d %H:%M:%S'), # :19 due to there might be the miro seconds.
        #                       quoteData.loc[:, 'exchangeTime'].values)
        # quoteData.index = pd.to_datetime(pd.Series(map(lambda stime: str(stime)[:19], # :19 due to there might be the miro seconds.
        #                       quoteData.loc[:, 'exchangeTime'].values)), format = '%Y-%m-%d %H:%M:%S')
        quoteData.index = pd.to_datetime(quoteData.loc[:, 'exchangeTime'].values, format = '%Y-%m-%d %H:%M:%S')
        ## filter the error data and extract the columns we need
        # columnsToRemove = [u'msgtime', u' msgsource', u' msgtype', u' msgid', u' nanotime',
        #                    u' time', u' nTime', u' szWindCode', u' szCode', u' nActionDay', u' nTradingDay']
        # self.quoteqa = data.columns[70:]
        # data = data.loc[:, data.columns - columnsToRemove]
        # quoteData['midp'] = (quoteData.loc[:,'bidPrice1'] + quoteData.loc[:,'askPrice1'])/2 # wrong!! :: need to consider the uplimit or downlimit price
        bidPrice1 = quoteData.loc[:, 'bidPrice1']
        bidPrice1.loc[bidPrice1.loc[:] == 0] = quoteData.loc[bidPrice1.loc[:] == 0, 'askPrice1']
        askPrice1 = quoteData.loc[:, 'askPrice1']
        askPrice1.loc[askPrice1.loc[:] == 0] = quoteData.loc[askPrice1.loc[:] == 0, 'bidPrice1']
        quoteData['midp'] = (askPrice1 + bidPrice1)/2 # wrong!! :: need to consider the uplimit or downlimit price
        quoteData.loc[quoteData.loc[:,'midp'] == 0,'midp'] = np.nan
        # data = data[data.loc[:, ' midp'] > 0]
        # quoteData = quoteData[quoteData.loc[:, 'latest'] > 0] # filter the error data
        quoteData = pd.concat([quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:15:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 11:30:00'), '%Y%m%d %H:%M:%S'):],
                              quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 13:00:00'),
                                                                       '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
                                  str(self.tradeDate + ' 15:01:00'), '%Y%m%d %H:%M:%S'):]])
        ## expand the data to second data.
        # secondTempData = pd.concat([pd.DataFrame(index=pd.date_range(self.tradeDate + " 9:25:00", self.tradeDate + " 11:30:00", freq="1S"), columns=quoteData.columns),
        #                              pd.DataFrame(index=pd.date_range(self.tradeDate + " 13:00:00", self.tradeDate + " 15:01:00", freq="1S"),
        #                                           columns=quoteData.columns)]) ## s means second
        #
        # secondTempData.loc[quoteData.index,:] = quoteData  ## TODO: the future time stamp need to convert into second instead of micro second.
        # secondData = secondTempData.fillna(method = 'ffill')
        # secondData = secondData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:25:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 15:01:00'), '%Y%m%d %H:%M:%S'),:]
        # return quoteData
        return quoteData

    def ReadSingleQuoteDataLevel2(self,dataPath,symbol,tradeDate,type = 'stock'):
        """
        used to construct the quote data with factors
        :return:quoteData with factors, like
        """
        # quoteData = pd.read_csv(self.dataPath + '/' + self.symbol + '_' + self.tradeDate + '_quoteqa.csv')
        # if type == 'stock':
        #     fileName = dataPath + '/' + symbol + '_' + tradeDate + '_quote.csv'
        #     # msgtime, msgsource, msgtype, msgid, nanotime, time, nTime, szWindCode, szCode, nActionDay, nTradingDay, nStatus, nPreClose, nOpen, nHigh, nLow, nMatch, nNumTrades, iVolume, iTurnover, nTotalBidVol, nTotalAskVol, nWeightedAvgBidPrice, nWeightedAvgAskPrice, nIOPV, nYieldToMaturity, nHighLimited, nLowLimited, chPrefix
        #     # askprice01, askvolume01, askprice02, askvolume02, askprice03, askvolume03, askprice04, askvolume04, askprice05, askvolume05, askprice06, askvolume06, askprice07, askvolume07, askprice08, askvolume08, askprice09, askvolume09, askprice10, askvolume10, bidprice01, bidvolume01, bidprice02, bidvolume02, bidprice03, bidvolume03, bidprice04, bidvolume04, bidprice05, bidvolume05, bidprice06, bidvolume06, bidprice07, bidvolume07, bidprice08, bidvolume08, bidprice09, bidvolume09, bidprice10, bidvolume10
        #     columns = ['msgtime', 'msgsource', 'msgtype', 'msgid', 'nanotime', 'time','exchangeTime', 'micCode', 'szCode', 'nActionDay', 'nTradingDay', 'nStatus', 'nPreclose', 'nOpen', 'nHigh','nLow',
        #                'latest', 'tradeNos','tradeVolume','totalTurnover',  'nTotalBidVol', 'nTotalAskVol', 'nWeightedAvgBidPrice', 'nIOPV', 'nYieldToMaturity', 'nHighLimited', 'nLowLimited', 'chPrefix',
        #                'askPrice1',  'askVolume1', 'askPrice2','askVolume2', 'askPrice3',  'askVolume3','askPrice4', 'askVolume4', 'askPrice5','askVolume5',
        #                'askPrice6', 'askVolume6', 'askPrice7', 'askVolume7', 'askPrice8', 'askVolume8', 'askPrice9', 'askVolume9','askPrice10','askVolume10',
        #                'bidPrice1','bidVolume1', 'bidPrice2','bidVolume2', 'bidPrice3','bidVolume3', 'bidPrice4', 'bidVolume4', 'bidPrice5', 'bidVolume5',
        #                'bidPrice6',  'bidVolume6','bidPrice7','bidVolume7', 'bidPrice8','bidVolume8', 'bidPrice9', 'bidVolume9','bidPrice10', 'bidVolume10'
        #                  ]
        # elif type == 'future':
        #     fileName = dataPath + '/' + symbol + '.CF_' + tradeDate + '_future.csv'
        #     columns = ['msgtime', 'msgsource', 'msgtype', 'msgid', 'nanotime', 'time','exchangeTime', 'micCode', 'szCode', 'nActionDay', 'nTradingDay', 'nStatus', 'tradeVolume','totalTurnover',
        #                'iOpenInterest', 'iPreOpenInterest','nPreClose','nPreSettlePrice','nOpen',  'nHigh', 'nLow', 'nMatch', 'latest', 'nSettlePrice', 'nHighLimited','nLowLimited', 'nPreDelta', 'nCurrDelta',
        #                'askPrice1',  'askVolume1', 'askPrice2','askVolume2', 'askPrice3',  'askVolume3','askPrice4', 'askVolume4', 'askPrice5','askVolume5',
        #                'bidPrice1','bidVolume1', 'bidPrice2','bidVolume2', 'bidPrice3','bidVolume3', 'bidPrice4', 'bidVolume4', 'bidPrice5', 'bidVolume5'
        #                ]
        #
        # else:
        #     raise('Error Type ', type)
        # if os.path.exists(fileName) is False:
        #     print('file',fileName,' Not exist')
        #     return None
        # quoteData = pd.read_csv(fileName,skiprows = 1,header=None,encoding= 'oem') ## NOTICE: need to check the file encoding before reading the file
        # if quoteData.shape[0] <= 10:
        #     return None
        # quoteData.columns = columns

        # quoteData.index = pd.to_datetime(quoteData.loc[:, 'exchangeTime'].values, format = '%Y-%m-%d %H:%M:%S')
        # quoteData.index = pd.to_datetime(pd.Series(map(lambda stime: self.tradeDate + str(stime)[:-3], quoteData.loc[:, 'exchangeTime'])),
        #                format='%Y%m%d%H%M%S')
        ## filter the error data and extract the columns we need
        """  Read data from file"""
        quoteData = self.ReadFile(dataPath,symbol=symbol,tradeDate = tradeDate,dataType=type)
        if quoteData is None:
            return None
        quoteData.loc[:,'tradeVolume'] = quoteData.loc[:,'tradeVolume'] * self.volumeMultiplier
        """  convert the data into the type we need"""
        quoteData.loc[quoteData.loc[:, 'bidPrice1'] == 0, 'bidPrice1'] = list(quoteData.loc[quoteData.loc[:, 'bidPrice1'] == 0, 'askPrice1'])
        bidPrice1 = quoteData.loc[:, 'bidPrice1']/self.priceMultiplier
        quoteData.loc[quoteData.loc[:, 'askPrice1'] == 0, 'askPrice1'] = list(quoteData.loc[quoteData.loc[:, 'askPrice1'] == 0, 'bidPrice1'])
        askPrice1 = quoteData.loc[:, 'askPrice1']/self.priceMultiplier
        quoteData = quoteData.assign(midp=((bidPrice1 + askPrice1) / 2).values)
        quoteData.loc[quoteData.loc[:,'midp'] == 0,'midp'] = np.nan
        quoteData = quoteData.loc[quoteData.index > datetime.datetime.strptime(str(self.tradeDate + ' 09:15:00'), '%Y%m%d %H:%M:%S'),:]
        quoteData = quoteData.sort_index()
        # quoteData = quoteData.iloc[2:-2,:]
        quoteData = quoteData.iloc[3:-3,:]
        quoteData = pd.concat([quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:15:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 11:30:00'), '%Y%m%d %H:%M:%S'):],
                              quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 13:00:00'),
                                                                       '%Y%m%d %H:%M:%S'):]])
        if self.RAWDATA == True:
            return quoteData
        # data = data[data.loc[:, ' midp'] > 0]
        # quoteData = quoteData[quoteData.loc[:, 'latest'] > 0] # filter the error data
        # quoteData = pd.concat([quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:25:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 11:30:00'), '%Y%m%d %H:%M:%S'):],
        #                       quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 13:00:00'),
        #                                                                '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
        #                           str(self.tradeDate + ' 15:01:00'), '%Y%m%d %H:%M:%S'):]])
        # ## expand the data to second data.
        # secondTempData = pd.concat([pd.DataFrame(index=pd.date_range(self.tradeDate + " 9:25:00", self.tradeDate + " 11:30:00", freq='1S'), columns=quoteData.columns),
        #                              pd.DataFrame(index=pd.date_range(self.tradeDate + " 13:00:00", self.tradeDate + " 15:01:00", freq='1S'),
        #                                           columns=quoteData.columns)]) ## s means second
        #
        # secondTempData.loc[quoteData.index,:] = quoteData  ## TODO: the future time stamp need to convert into second instead of micro second.
        # secondData = secondTempData.fillna(method = 'ffill')
        # secondData = secondData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 15:00:00'), '%Y%m%d %H:%M:%S'),:]
        if self.freq != '1S':
            secondData = quoteData
            secondData = secondData.resample(self.freq,label = 'right').last()
            secondData = secondData.fillna(method = 'ffill')
            secondData = pd.concat([secondData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 11:30:00'), '%Y%m%d %H:%M:%S'),:],
                                    secondData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 13:00:01'),
                                                                              '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
                                        str(self.tradeDate + ' 15:00:00'), '%Y%m%d %H:%M:%S'), :]])
            quoteData = secondData
        # return quoteData
        return quoteData


    def ReadSingleTradeData(self,dataPath,symbol,tradeDate,type = 'stock'):
        """
        used to construct the quote data with factors

        :param dataPath: path with data
        :param symbol:  stock code
        :param tradeDate: trading day
        :return: tradeData with time as index and columns with price and volume
        """

        """ Read data from file"""
        tradeData = self.ReadFile(dataPath,symbol,tradeDate,dataType=type)
        #print(dataPath)
        tradeData.loc[:, ' nVolume'] = tradeData.loc[:, ' nVolume'] * self.volumeMultiplier
        """ Convert the data into the type we need"""
        sampleFrq = '1S' # aggregate data in one second
        tradeData.loc[:, ' nTurnover'] = tradeData.loc[:, ' nPrice'] * tradeData.loc[:, ' nVolume'] / float(self.priceMultiplier)
        if self.RAWDATA == True:
            return tradeData
        # tradeData = self.SortDataByOrder(tradeData)  # sort the data by order info.
        # # BuySample  = tradeData.loc[tradeData.loc[:, ' nBSFlag'] == 'B', :]
        # # SellSample = tradeData.loc[tradeData.loc[:, ' nBSFlag'] == 'S', :]
        # # Cancel     = tradeData.loc[tradeData.loc[:, ' chFunctionCode'] == 'C', :]
        # BuySample = tradeData.loc[tradeData.loc[:, 'orderDirection'] == 'B', :]   # 需要考虑到涨跌停的情况，一字板情况下可能没有买或者卖
        # SellSample = tradeData.loc[tradeData.loc[:, 'orderDirection'] == 'S', :]  # 需要考虑到涨跌停的情况
        # # Cancel = tradeData.loc[tradeData.loc[:, ' chFunctionCode'] == 'C', :]
        # # start = time.time()
        # buyCounts,      buyVwap,        buyVolume,      buyTurnover, buysuperOrderVolume, buysuperOrderTurn, buybigOrderVolume, buybigOrderTurn, buymiddleOrderVolume, buymiddleOrderTurn, buysmallOrderVolume, buysmallOrderTurn, buymainOrderVolume, buymainOrderTurn     = self.AggregateData(BuySample, sampleFrq, 'trade')
        # sellCounts,     sellVwap,       sellVolume,     sellTurnover, sellsuperOrderVolume, sellsuperOrderTurn, sellbigOrderVolume, sellbigOrderTurn, sellmiddleOrderVolume, sellmiddleOrderTurn, sellsmallOrderVolume,  sellsmallOrderTurn, sellmainOrderVolume, sellmainOrderTurn    = self.AggregateData(SellSample, sampleFrq, 'trade')
        # # end = time.time()
        # # print(end - start,'s')
        # # cancelCounts,   cancelVolume                    = self.AggregateData(Cancel, sampleFrq, 'cancel')
        # resampledData = pd.concat([buyVwap, buyVolume, buyCounts, buyTurnover, sellVwap, sellVolume, sellCounts,
        #                            sellTurnover,
        #                            # cancelCounts, cancelVolume,
        #                            buysuperOrderVolume, buysuperOrderTurn, buybigOrderVolume, buybigOrderTurn, buymiddleOrderVolume,
        #                            buymiddleOrderTurn, buysmallOrderVolume, buysmallOrderTurn, buymainOrderVolume, buymainOrderTurn,
        #                            sellsuperOrderVolume, sellsuperOrderTurn, sellbigOrderVolume,
        #                            sellbigOrderTurn, sellmiddleOrderVolume, sellmiddleOrderTurn, sellsmallOrderVolume,
        #                            sellsmallOrderTurn, sellmainOrderVolume, sellmainOrderTurn], 1)
        # resampledData.columns = ['buyVwap', 'buyVolume', 'buyCounts', 'buyTurnover', 'sellVwap', 'sellVolume', 'sellCounts',
        #                          'sellTurnover',
        #                          # 'cancelCounts', 'cancelVolume',
        #                          'buysuperOrderVolume', 'buysuperOrderTurn', 'buybigOrderVolume', 'buybigOrderTurn',
        #                          'buymiddleOrderVolume',
        #                          'buymiddleOrderTurn', 'buysmallOrderVolume', 'buysmallOrderTurn', 'buymainOrderVolume',
        #                          'buymainOrderTurn',
        #                          'sellsuperOrderVolume', 'sellsuperOrderTurn', 'sellbigOrderVolume',
        #                          'sellbigOrderTurn', 'sellmiddleOrderVolume', 'sellmiddleOrderTurn', 'sellsmallOrderVolume',
        #                          'sellsmallOrderTurn', 'sellmainOrderVolume', 'sellmainOrderTurn']
        # resampledData.loc[:, ['buyVolume',  'buyTurnover',  'sellVolume',  'sellTurnover',
        #                      'buysuperOrderVolume', 'buysuperOrderTurn', 'buybigOrderVolume', 'buybigOrderTurn',
        #                     'buymiddleOrderVolume', 'buymiddleOrderTurn', 'buysmallOrderVolume', 'buysmallOrderTurn',
        #                       'buymainOrderVolume', 'buymainOrderTurn', 'sellsuperOrderVolume',
        #                       'sellsuperOrderTurn', 'sellbigOrderVolume','sellbigOrderTurn', 'sellmiddleOrderVolume',
        #                       'sellmiddleOrderTurn', 'sellsmallOrderVolume', 'sellsmallOrderTurn', 'sellmainOrderVolume', 'sellmainOrderTurn']] \
        #     = resampledData.loc[:, ['buyVolume',  'buyTurnover',  'sellVolume',  'sellTurnover',
        #                      'buysuperOrderVolume', 'buysuperOrderTurn', 'buybigOrderVolume', 'buybigOrderTurn',
        #                     'buymiddleOrderVolume', 'buymiddleOrderTurn', 'buysmallOrderVolume', 'buysmallOrderTurn',
        #                       'buymainOrderVolume', 'buymainOrderTurn', 'sellsuperOrderVolume',
        #                       'sellsuperOrderTurn', 'sellbigOrderVolume','sellbigOrderTurn', 'sellmiddleOrderVolume',
        #                       'sellmiddleOrderTurn', 'sellsmallOrderVolume', 'sellsmallOrderTurn', 'sellmainOrderVolume', 'sellmainOrderTurn']].fillna(0)
        # # Calculate total turnover and vwap here.
        # resampledData.loc[:, 'totalVolume']             = resampledData.loc[:, 'buyVolume'] + resampledData.loc[:, 'sellVolume']
        # resampledData.loc[:, 'totalTurnover']           = resampledData.loc[:, 'buyTurnover'] + resampledData.loc[:, 'sellTurnover']
        # resampledData.loc[:, 'netCashFlow']             = resampledData.loc[:, 'buyTurnover'] - resampledData.loc[:, 'sellTurnover']
        # resampledData.loc[:, 'netSuperOrderCashFlow']   = resampledData.loc[:, 'buysuperOrderTurn'] - resampledData.loc[:, 'sellsuperOrderTurn']
        # resampledData.loc[:, 'netBigOrderCashFlow']     = resampledData.loc[:, 'buybigOrderTurn'] - resampledData.loc[:, 'sellbigOrderTurn']
        # resampledData.loc[:, 'netMiddleOrderCashFlow']  = resampledData.loc[:, 'buymiddleOrderTurn'] - resampledData.loc[:, 'sellmiddleOrderTurn']
        # resampledData.loc[:, 'netSmallOrderCashFlow']   = resampledData.loc[:, 'buysmallOrderTurn'] - resampledData.loc[:, 'sellsmallOrderTurn']
        # resampledData.loc[:, 'netMainOrderCashFlow']    = resampledData.loc[:, 'buymainOrderTurn'] - resampledData.loc[:, 'sellmainOrderTurn']
        # resampledData.loc[:, 'vwap']                    = resampledData.loc[:, 'totalTurnover'] / resampledData.loc[:, 'totalVolume']
        #
        # # create second sample data
        # resampledData = pd.concat([resampledData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:25:00'),
        #                                                                 '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
        #     str(self.tradeDate + ' 11:30:00'), '%Y%m%d %H:%M:%S'):],
        #                            resampledData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 13:00:00'),
        #                                                                 '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
        #                            str(self.tradeDate + ' 15:01:00'), '%Y%m%d %H:%M:%S'):]])
        # # expand the data to second data.
        # secondTempData = pd.concat([pd.DataFrame(index=pd.date_range(self.tradeDate + " 9:25:00", self.tradeDate + " 11:30:00", freq= sampleFrq), columns=resampledData.columns),
        #                             pd.DataFrame(index=pd.date_range(self.tradeDate + " 13:00:00", self.tradeDate + " 15:01:00", freq= sampleFrq),
        #                                           columns=resampledData.columns)]) ## s means second
        #
        # secondTempData.loc[resampledData.index,:] = resampledData
        # secondTempData.loc[:, ['buyVwap','sellVwap','vwap']] = secondTempData.loc[:, ['buyVwap', 'sellVwap', 'vwap']].fillna(method = 'ffill')
        # secondData = secondTempData.fillna(0)
        # # before the 9:30:00, we should get the latest record for this period. In this case, we could use the cumsum due
        # # to that the trades before 93000 must on the same time
        # secondData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:29:59'), '%Y%m%d %H:%M:%S'),[column for column in secondData.columns if 'vwap' not in column.lower()]] = secondData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:25:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 09:29:59'), '%Y%m%d %H:%M:%S'), :].cumsum().iloc[-1,:]
        # secondData = secondData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 15:00:00'), '%Y%m%d %H:%M:%S'), :]
        # return secondData
        return tradeData

    # def AggregateData(self,sampleData,freq,type = 'trade'):
    #     """
    #
    #     :param sampleData: the data that we want to aggregate by given freq
    #     :param freq: such as '1s', '1T', '1H'
    #     :return: the resampling data with columns:last or vwap, volume, turnover,.
    #     """
    #     """
    #         加入以下大单和超大单的计算
    #         超大单：大于等于50万股或者100万元的成交单
    #         大单：大于等于10万股或者20万元且小于50万股和100万元的成交单
    #         中单：大于等于2万股或者4万元且小于10万股和20万元的成交单
    #         小单：小于2万股和4万元的成交单
    #         主力流入：超大单+大单买入成交额之和
    #         主力流出：超大单+大单卖出成交额之和
    #         主力净流入：主力流入-主力流出
    #         净额：流入-流出
    #         净占比：（流入-流出）/ 总成交额
    #     """
    #     if type is 'trade':
    #         # start = time.time()
    #         sampleData.loc[:, 'superOrderVolume'] = 0
    #         sampleData.loc[:, 'superOrderTurn'] = 0
    #         # CritVol1 = sampleData.loc[:, 'activeVolume'] > 500000
    #         # CritVol2 = sampleData.loc[:, 'activeVolume'] >= 100000
    #         # CritVol3 = sampleData.loc[:, 'activeVolume'] >= 20000
    #         # CritTur1 = sampleData.loc[:, 'activeTurnover'] > 1000000
    #         # CritTur2 = sampleData.loc[:, 'activeTurnover'] >= 200000
    #         # CritTur3 = sampleData.loc[:, 'activeTurnover'] >= 40000
    #
    #         superOrderIndex = (sampleData.loc[:, 'activeVolume'] > 500000) | (sampleData.loc[:, 'activeTurnover'] > 1000000)
    #         # superOrderIndex = (CritVol1) | (CritTur1)
    #         sampleData.loc[superOrderIndex, 'superOrderVolume'] = sampleData.loc[superOrderIndex, 'activeVolume']
    #         sampleData.loc[superOrderIndex, 'superOrderTurn'] = sampleData.loc[superOrderIndex, 'activeTurnover']
    #         # end = time.time()
    #         # print(end-start,'s')
    #         # start = time.time()
    #         sampleData.loc[:, 'bigOrderVolume'] = 0
    #         sampleData.loc[:, 'bigOrderTurn'] = 0
    #         bigOrderIndex = ((sampleData.loc[:, 'activeVolume'] >= 100000) | (sampleData.loc[:, 'activeTurnover'] >= 200000)) & (sampleData.loc[:, 'activeVolume'] < 500000) & (sampleData.loc[:, 'activeTurnover'] < 1000000)
    #         # bigOrderIndex = ((CritVol2) | (CritTur2)) & (~CritVol1) & (~CritTur1)
    #         sampleData.loc[bigOrderIndex, 'bigOrderVolume'] = sampleData.loc[bigOrderIndex, 'activeVolume']
    #         sampleData.loc[bigOrderIndex, 'bigOrderTurn'] = sampleData.loc[bigOrderIndex, 'activeTurnover']
    #         # end = time.time()
    #         # print(end-start,'s')
    #         # start = time.time()
    #         sampleData.loc[:, 'middleOrderVolume'] = 0
    #         sampleData.loc[:, 'middleOrderTurn'] = 0
    #         middleOrderIndex = ((sampleData.loc[:, 'activeVolume'] >= 20000) | (sampleData.loc[:, 'activeTurnover'] >= 40000)) & ((sampleData.loc[:, 'activeVolume'] < 100000) & (sampleData.loc[:, 'activeTurnover'] < 200000))
    #         # middleOrderIndex = ((CritVol3) | (CritTur3)) & ((~CritVol2) & (~CritTur2))
    #         sampleData.loc[middleOrderIndex, 'middleOrderVolume'] = sampleData.loc[middleOrderIndex, 'activeVolume']
    #         sampleData.loc[middleOrderIndex, 'middleOrderTurn'] = sampleData.loc[middleOrderIndex, 'activeTurnover']
    #         # end = time.time()
    #         # print(end-start,'s')
    #         # start = time.time()
    #         sampleData.loc[:, 'smallOrderVolume'] = 0
    #         sampleData.loc[:, 'smallOrderTurn'] = 0
    #         smallOrderIndex  = (sampleData.loc[:, 'activeVolume'] < 20000) & (sampleData.loc[:, 'activeTurnover'] < 40000)
    #         # smallOrderIndex  = (~CritVol3) & (~CritTur3)
    #         sampleData.loc[smallOrderIndex, 'smallOrderVolume'] = sampleData.loc[smallOrderIndex , 'activeVolume']
    #         sampleData.loc[smallOrderIndex, 'smallOrderTurn'] = sampleData.loc[smallOrderIndex, 'activeTurnover']
    #         # end = time.time()
    #         # print(end-start,'s')
    #         # start = time.time()
    #         sampleData.loc[:, 'mainOrderVolume'] = sampleData.loc[:, 'superOrderVolume'] + sampleData.loc[:, 'bigOrderVolume']
    #         sampleData.loc[:, 'mainOrderTurn'] = sampleData.loc[:, 'superOrderTurn'] + sampleData.loc[:, 'bigOrderTurn']
    #         # end = time.time()
    #         # print(end-start,'s')
    #         resample = sampleData.resample(freq,label='right')
    #         tradeSample = resample.sum()
    #         countSample = resample.count()
    #
    #         # volume
    #         # volume              = tradeSample.loc[:, ' nVolume']
    #         volume              = tradeSample.loc[:, 'activeVolume']
    #         superOrderVolume    = tradeSample.loc[:, 'superOrderVolume']
    #         bigOrderVolume      = tradeSample.loc[:, 'bigOrderVolume']
    #         middleOrderVolume   = tradeSample.loc[:, 'middleOrderVolume']
    #         smallOrderVolume    = tradeSample.loc[:, 'smallOrderVolume']
    #         mainOrderVolume     = tradeSample.loc[:, 'mainOrderVolume']
    #
    #         # turnover
    #         # turnover        = tradeSample.loc[:, ' nTurnover']  # turnover should be revised due to the wrong data, 原因：股数非100整数倍，交易所发布数据舍去了小数点
    #         turnover        = tradeSample.loc[:, 'activeTurnover']
    #         superOrderTurn  = tradeSample.loc[:, 'superOrderTurn']
    #         bigOrderTurn    = tradeSample.loc[:, 'bigOrderTurn']
    #         middleOrderTurn = tradeSample.loc[:, 'middleOrderTurn']
    #         smallOrderTurn  = tradeSample.loc[:, 'smallOrderTurn']
    #         mainOrderTurn   = tradeSample.loc[:, 'mainOrderTurn']
    #         counts = countSample.loc[:, 'orderDirection']
    #         vwap = turnover/volume
    #         return counts, vwap, volume, turnover, superOrderVolume, superOrderTurn, bigOrderVolume, bigOrderTurn, middleOrderVolume, middleOrderTurn, smallOrderVolume,  smallOrderTurn, mainOrderVolume, mainOrderTurn
    #         # return tradeSample
    #     elif type is 'cancel':
    #         #  cancel order situation
    #         # counts = countSample.loc[:, 'chFunction']
    #         sampleData.loc[:, 'activeTurnover'] = sampleData.loc[:, ' nPrice'] * sampleData.loc[:, 'activeVolume'] / float(
    #             10000)  # /10000 due to the price adjust
    #         resample = sampleData.resample(freq, label='right')
    #         tradeSample = resample.sum()
    #         countSample = resample.count()
    #         counts = countSample.loc[:, 'activeVolume']
    #         # volume
    #         volume = tradeSample.loc[:, 'activeVolume']
    #         return counts, volume
    #     else:
    #         raise('Error type to resample : ' + type)
    #     return None

    def AggregateData(self,sampleData,freq,type = 'trade'):
        """

        :param sampleData: the data that we want to aggregate by given freq
        :param freq: such as '1s', '1T', '1H'
        :return: the resampling data with columns:last or vwap, volume, turnover,.
        """
        """ 
            加入以下大单和超大单的计算
            超大单：大于等于50万股或者100万元的成交单
            大单：大于等于10万股或者20万元且小于50万股和100万元的成交单
            中单：大于等于2万股或者4万元且小于10万股和20万元的成交单
            小单：小于2万股和4万元的成交单
            主力流入：超大单+大单买入成交额之和
            主力流出：超大单+大单卖出成交额之和
            主力净流入：主力流入-主力流出
            净额：流入-流出
            净占比：（流入-流出）/ 总成交额
        """
        # start = time.time()
        if sampleData.shape[0] == 0:
            # return None,None,None,None,None,None,None,None,None,None,None,None,None,None
            sampleData.at[pd.to_datetime(self.tradeDate + '093000000', format='%Y%m%d%H%M%S%f'),:] = 0
        volSortedSample = sampleData.sort_values('activeVolume')  # sort the order in order to find out the quicker search.
        # turSortedSample = sampleData.sort_values('activeTurnover')
        CritVol1 = volSortedSample['activeVolume'].searchsorted(500000, 'left')[0]  # 假设此处交易额和交易量都是递增的，即使交易额不是严格递增，误差范围也可以忽略不计。
        CritVol2 = volSortedSample['activeVolume'].searchsorted(100000, 'left')[0]
        CritVol3 = volSortedSample['activeVolume'].searchsorted(20000, 'left')[0]
        CritTur1 = volSortedSample['activeTurnover'].searchsorted(1000000,'left')[0]
        CritTur2 = volSortedSample['activeTurnover'].searchsorted(200000,'left')[0]
        CritTur3 = volSortedSample['activeTurnover'].searchsorted(40000,'left')[0]

        volSortedSample.loc[:, 'superOrderVolume'] = 0
        volSortedSample.loc[:, 'superOrderTurn'] = 0
        superOrderIndex = min(CritVol1,CritTur1)
        # volSortedSample.loc[volSortedSample.index[superOrderIndex]:, 'superOrderVolume'] = volSortedSample.loc[volSortedSample.index[superOrderIndex]:, 'activeVolume']
        # volSortedSample.loc[volSortedSample.index[superOrderIndex]:, 'superOrderTurn'] = volSortedSample.loc[volSortedSample.index[superOrderIndex]:, 'activeTurnover']
        volSortedSample.ix[superOrderIndex:, 'superOrderVolume'] = volSortedSample.ix[superOrderIndex:, 'activeVolume']
        volSortedSample.ix[superOrderIndex:, 'superOrderTurn'] = volSortedSample.ix[superOrderIndex:, 'activeTurnover']

        volSortedSample.loc[:, 'bigOrderVolume'] = 0
        volSortedSample.loc[:, 'bigOrderTurn'] = 0
        bigOrderIndexFirst = min(CritTur2,CritVol2)
        bigOrderIndexLast  = superOrderIndex
        # volSortedSample.loc[volSortedSample.index[bigOrderIndexFirst]:volSortedSample.index[bigOrderIndexLast], 'bigOrderVolume'] = volSortedSample.loc[volSortedSample.index[bigOrderIndexFirst]:volSortedSample.index[bigOrderIndexLast], 'activeVolume']
        # volSortedSample.loc[volSortedSample.index[bigOrderIndexFirst]:volSortedSample.index[bigOrderIndexLast], 'bigOrderTurn'] = volSortedSample.loc[volSortedSample.index[bigOrderIndexFirst]:volSortedSample.index[bigOrderIndexLast], 'activeTurnover']
        volSortedSample.ix[bigOrderIndexFirst:bigOrderIndexLast, 'bigOrderVolume'] = volSortedSample.ix[bigOrderIndexFirst:bigOrderIndexLast, 'activeVolume']
        volSortedSample.ix[bigOrderIndexFirst:bigOrderIndexLast, 'bigOrderTurn'] = volSortedSample.ix[bigOrderIndexFirst:bigOrderIndexLast, 'activeTurnover']

        volSortedSample.loc[:, 'middleOrderVolume'] = 0
        volSortedSample.loc[:, 'middleOrderTurn'] = 0
        middleOrderIndexFirst = min(CritTur3,CritVol3)
        middleOrderIndexLast  = bigOrderIndexFirst
        # volSortedSample.loc[volSortedSample.index[middleOrderIndexFirst]:volSortedSample.index[middleOrderIndexLast], 'middleOrderVolume'] = volSortedSample.loc[volSortedSample.index[middleOrderIndexFirst]:volSortedSample.index[middleOrderIndexLast], 'activeVolume']
        # volSortedSample.loc[volSortedSample.index[middleOrderIndexFirst]:volSortedSample.index[middleOrderIndexLast], 'middleOrderTurn'] = volSortedSample.loc[volSortedSample.index[middleOrderIndexFirst]:volSortedSample.index[middleOrderIndexLast], 'activeTurnover']
        volSortedSample.ix[middleOrderIndexFirst:middleOrderIndexLast, 'middleOrderVolume'] = volSortedSample.ix[middleOrderIndexFirst:middleOrderIndexLast, 'activeVolume']
        volSortedSample.ix[middleOrderIndexFirst:middleOrderIndexLast, 'middleOrderTurn'] = volSortedSample.ix[middleOrderIndexFirst:middleOrderIndexLast, 'activeTurnover']

        volSortedSample.loc[:, 'smallOrderVolume'] = 0
        volSortedSample.loc[:, 'smallOrderTurn'] = 0
        smallOrderIndex  = middleOrderIndexFirst
        # volSortedSample.loc[:volSortedSample.index[smallOrderIndex], 'smallOrderVolume'] = volSortedSample.loc[:volSortedSample.index[smallOrderIndex] , 'activeVolume']
        # volSortedSample.loc[:volSortedSample.index[smallOrderIndex], 'smallOrderTurn'] = volSortedSample.loc[:volSortedSample.index[smallOrderIndex], 'activeTurnover']
        volSortedSample.ix[:smallOrderIndex, 'smallOrderVolume'] = volSortedSample.ix[:smallOrderIndex , 'activeVolume']
        volSortedSample.ix[:smallOrderIndex, 'smallOrderTurn'] = volSortedSample.ix[:smallOrderIndex, 'activeTurnover']

        volSortedSample.loc[:, 'mainOrderVolume'] = volSortedSample.loc[:, 'superOrderVolume'] + volSortedSample.loc[:, 'bigOrderVolume']
        volSortedSample.loc[:, 'mainOrderTurn'] = volSortedSample.loc[:, 'superOrderTurn'] + volSortedSample.loc[:, 'bigOrderTurn']

        sampleData = volSortedSample.sort_index()
        resample = sampleData.resample(freq,label='right')
        tradeSample = resample.sum()
        countSample = resample.count()

        # volume
        # volume              = tradeSample.loc[:, ' nVolume']
        volume              = tradeSample.loc[:, 'activeVolume']
        superOrderVolume    = tradeSample.loc[:, 'superOrderVolume']
        bigOrderVolume      = tradeSample.loc[:, 'bigOrderVolume']
        middleOrderVolume   = tradeSample.loc[:, 'middleOrderVolume']
        smallOrderVolume    = tradeSample.loc[:, 'smallOrderVolume']
        mainOrderVolume     = tradeSample.loc[:, 'mainOrderVolume']

        # turnover
        # turnover        = tradeSample.loc[:, ' nTurnover']  # turnover should be revised due to the wrong data, 原因：股数非100整数倍，交易所发布数据舍去了小数点
        turnover        = tradeSample.loc[:, 'activeTurnover']
        superOrderTurn  = tradeSample.loc[:, 'superOrderTurn']
        bigOrderTurn    = tradeSample.loc[:, 'bigOrderTurn']
        middleOrderTurn = tradeSample.loc[:, 'middleOrderTurn']
        smallOrderTurn  = tradeSample.loc[:, 'smallOrderTurn']
        mainOrderTurn   = tradeSample.loc[:, 'mainOrderTurn']
        counts = countSample.loc[:, 'orderDirection']
        vwap = turnover/volume
        return counts, vwap, volume, turnover, superOrderVolume, superOrderTurn, bigOrderVolume, bigOrderTurn, middleOrderVolume, middleOrderTurn, smallOrderVolume,  smallOrderTurn, mainOrderVolume, mainOrderTurn
        # return tradeSample


    def SortDataByOrder(self,sampleData):
        """
        Used to aggregate data by order info( bid order and ask order)
        :param sampleData: the tick data
        :return: the resampling data with columns:last or vwap, volume, turnover,.
        """
        """ 
            加入通过单号合成信息：
            1. 非cancel的数据
            2. 一个order只有一条的数据作为单独的一条数据，不需要再合并。
            3. 一个order有多条数据但是只有一个方向的数据，将同order数据的交易量合并，如果只在一档成交，则可得价格，如果
               在不同档成交，即表示该单的价格较高，主动交易意愿很强，计算vwap是否足够？
            4. 一个order有多条数据同时有b跟s的方向的数据，需要考虑如何合并？如平安银行20170419当天bidorder里8419284这个单，
               发了149800的单，分成48次不同的B和S成交，应该如何去融合这个信息量？意图是发一个超大单去cross当前的8.92的价位
               并将股票价格维持在>=8.92，在板块资金流方面有何帮助？
            5. 一个order有多条数据同时有b跟s还有c方向的数据，是否需要把c的量加进来？
            方法：
            1. 针对b方向，获取bid orders，针对s方向，获取ask orders
            2. group by orders and side, sum the quantity to get the order volume and total turnover. Then get vwap for 
                every order
            3. 整合得到以下信息：主动报单方向，主动报单量，主动成交量，被动成交量，撤单量，主动报单金额，主动成交金额，被动成交金额
                                 撤单金额，主动报单价格
        """
        sortBidOrder = self.ProcessOrderInfo(sampleData, orderType=' nBidOrder')
        sortAskOrder = self.ProcessOrderInfo(sampleData, orderType=' nAskOrder')
        sortOrderDf = pd.concat([sortAskOrder, sortBidOrder], 0)
        sortOrderDf = sortOrderDf.sort_index()
        return sortOrderDf

    def ProcessOrderInfo(self, sampleData, orderType = ' nBidOrder'):
        """
        This function is used to aggregate the orderInfo
        :param orderInfo: the groupby object which is group by the bidorder or askorder
        :return:整合得到以下信息：主动报单方向，主动报单量，主动成交量，被动成交量，撤单量，主动报单金额，主动成交金额，被动成交金额
                                 撤单金额，主动报单价格
        """
        # activeBuy     = sampleData.groupby([orderType, ' nBSFlag'])
        # activeBuyTime = activeBuy.first().loc[:, [' nTime']]
        # activeBuyPrice= activeBuy.last().loc[:, [' nPrice']]
        # activeBuy     = pd.concat([activeBuy.sum().loc[:, [' nVolume', ' nTurnover']], activeBuyTime, activeBuyPrice], 1)  # here, sort by level = 2 due to that level = 2 is the time index level, first two levels is order and bs flag
        activeBuy     = sampleData.groupby([orderType, ' nBSFlag']).agg({' nVolume': 'sum', ' nTurnover': 'sum', ' nTime': 'first', ' nPrice': 'last'})  # use agg can apply different type of
        if orderType == ' nBidOrder':
            orderDirection = 'B'
            otherSideDirection = 'S'
        else:
            orderDirection = 'S'
            otherSideDirection = 'B'

        # start = time.time()
        # activeBuy     = activeBuy.sort_values(' nTime')
        # activeBuy     = activeBuy.reset_index()
        # activeBuy.index = pd.to_datetime(pd.Series(map(lambda stime: self.tradeDate + str(stime),
        #                            activeBuy.loc[:, ' nTime'])), format='%Y%m%d%H%M%S%f')

        # activeBuy.index = list(map(lambda stime: datetime.datetime.strptime(self.tradeDate + str(stime), '%Y%m%d%H%M%S%f'),
        #                            activeBuy.loc[:, ' nTime']))


        # activeBuyB = activeBuy.loc[activeBuy.loc[:, ' nBSFlag'] == orderDirection, [orderType, ' nPrice', ' nVolume', ' nTurnover']]  # which is the part of active buying
        # activeBuyB.columns = ['order', 'auctionPrice', 'activeVolume', 'activeTurnover']
        # activeBuyS = activeBuy.loc[activeBuy.loc[:, ' nBSFlag'] == otherSideDirection, [orderType, ' nPrice', ' nVolume', ' nTurnover']]  # which is the part of active buying
        # activeBuyS.columns = ['order', 'tradePrice', 'passiveVolume', 'passiveTurnover']
        # activeBuyC = activeBuy.loc[activeBuy.loc[:, ' nBSFlag'] == ' ', [orderType, ' nVolume']]  # which is the part of active buying and cancel
        # activeBuyC.columns = ['order', 'cancelVolume']
        # activeBuy = pd.merge(activeBuyB, activeBuyS, on='order', sort=False, how='left')
        # activeBuy = pd.merge(activeBuy, activeBuyC, on='order', sort=False, how='left')
        # activeBuy.index = activeBuyB.index
        # end = time.time()
        # start = time.time()
        activeBuyB = activeBuy.iloc[activeBuy.index.get_level_values(' nBSFlag') == orderDirection]
        if activeBuyB.shape[0] == 0:
            return None
        activeBuyB.columns = ['activeVolume', 'activeTurnover', ' nTime', 'auctionPrice']
        activeBuyS = activeBuy.iloc[activeBuy.index.get_level_values(' nBSFlag') == otherSideDirection]
        activeBuyS.columns = ['passiveVolume', 'passiveTurnover', ' nTime', 'tradePrice']
        activeBuyC = activeBuy.iloc[activeBuy.index.get_level_values(' nBSFlag') == ' '].loc[:, [' nVolume', ' nTime']]
        activeBuyC.columns = ['cancelVolume', ' nTime']
        activeBuy = pd.merge(activeBuyB.reset_index(), activeBuyS.loc[:,['passiveVolume', 'passiveTurnover', 'tradePrice']].reset_index(), on=orderType, sort=False, how='left')
        activeBuy = pd.merge(activeBuy, activeBuyC.loc[:, 'cancelVolume'].reset_index(), on=orderType, sort=False, how='left')
        if self.dataReadType == 'csv':
            activeBuy.index = pd.to_datetime(pd.Series(map(lambda stime: self.tradeDate + str(stime),
                                                       activeBuy.loc[:, ' nTime'])), format='%Y%m%d%H%M%S%f')
        elif self.dataReadType == 'gzip':
            activeBuy.index = pd.to_datetime(activeBuy.loc[:, ' nTime'])
        else:
            raise('Error data read type',self.dataReadType)
        # end = time.time()
        # print(end - start,' s')
        activeBuy = activeBuy.rename(columns = {orderType:'order'})
        activeBuy = activeBuy.loc[:, ['order', 'auctionVolume', 'auctionPrice', 'auctionTurnover',
                                 'activeVolume', 'activeTurnover', 'passiveVolume', 'passiveTurnover', 'cancelVolume']].fillna(0)
        activeBuy.loc[:, 'auctionVolume'] = activeBuy.loc[:, 'activeVolume'] + activeBuy.loc[:, 'passiveVolume'] + activeBuy.loc[:, 'cancelVolume']
        activeBuy.loc[:, 'auctionTurnover'] = activeBuy.loc[:, 'auctionPrice'] * activeBuy.loc[:, 'auctionVolume'] / float(self.priceMultiplier)
        activeBuy.loc[:, 'orderDirection'] = orderDirection
        return activeBuy.loc[:, ['order', 'orderDirection', 'auctionVolume', 'auctionPrice', 'auctionTurnover',
                                 'activeVolume', 'activeTurnover', 'passiveVolume', 'passiveTurnover', 'cancelVolume']]

    def StructQuoteData(self,type = 'file'):
        quoteDataQueue = {}
        if type == 'cpp':
            allData = self.ReadDataFromCpp('Quote')
        if len(self.symbol) == 0:
            return {}
        else:
            for symbol in self.symbol:
                if type == 'cpp':
                    singleSymbolData = allData.loc[symbol,:]
                else:
                    'test'
                # print(symbol)
                    # singleSymbolData = pd.read_csv('')  # read  data from single file. TODO:将读取数据和处理数据分离，方便模块化处理
                # quoteDataQueue[symbol] = self.ReadSingleQuoteData(self.dataPath, symbol, self.tradeDate)
                # print(symbol)
                # if symbol == '300100.SZ':
                #     print(symbol)
                # data_key = 'quote_' + symbol.replace('.','') + '_' + self.tradeDate.replace('-','')
                # quoteData = self.hdf_buffer.read(data_key)
                # if quoteData.empty:
                #     quoteData = self.ReadSingleQuoteDataLevel2(self.dataPath, symbol, self.tradeDate,'Quote')
                #     quoteDataQueue[symbol] = quoteData
                    # quoteData.to_csv('./ref_data/hdfs_store/' + self.tradeDate + '.csv')
                    # self.hdf_buffer.write(data_key,quoteData)
                # else:
                #     quoteDataQueue[symbol] = quoteData
                quoteData = self.ReadSingleQuoteDataLevel2(self.dataPath, symbol, self.tradeDate, 'Quote')
                quoteDataQueue[symbol] = quoteData
            return quoteDataQueue

    def StructTradeData(self):
        """
        used to construct the trade data with factors
        :return: tradeData with factors, like index(time), factor, ...
        """

        tradeDataQueue = {}
        if len(self.symbol) == 0:
            return {}
        else:
            for symbol in self.symbol:
                # print(symbol)
                # data_key = 'trade_' + symbol.replace('.','') + '_' + self.tradeDate.replace('-','')
                # tradeData = self.hdf_buffer.read(data_key)
                # if tradeData.empty:

                tradeDataQueue[symbol] = self.ReadSingleTradeData(self.dataPath,symbol,self.tradeDate,'Trade')
                    # self.hdf_buffer.write(data_key,tradeData)
                # else:
                #     tradeDataQueue[symbol] = tradeData
            return tradeDataQueue

    def StructOrderData(self):
        """
        用于构建order数据
        :return:
        """
        orderData = {}
        for symbol in self.symbol:
            exchange = symbol.split('.')[1]
            if exchange == 'SH':# 因为上交所没有order信息
                continue
            else:
                # print(symbol)
                # data_key = 'order_' + symbol.replace('.','') + '_' + self.tradeDate.replace('-','')
                # tradeData = self.hdf_buffer.read(data_key)
                # if tradeData.empty:
                orderData[symbol] = self.ReadOrderData(self.dataPath,symbol,self.tradeDate,type= 'order')
                #     self.hdf_buffer.write(data_key,tradeData)
                # else:
                #     orderData[symbol] = tradeData

        return orderData


    def ReadOrderData(self,dataPath,symbol,tradeDate,type = 'order'):
        """  Read data from file"""
        quoteData = self.ReadFile(dataPath,symbol=symbol,tradeDate = tradeDate,dataType="Order",symbolType=type)
        quoteData = quoteData.sort_index()
        # quoteData = quoteData.iloc[2:-2,:]
        quoteData = quoteData.iloc[:-3,:]
        quoteData = pd.concat([quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:15:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 11:30:00'), '%Y%m%d %H:%M:%S'):],
                              quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 13:00:00'),
                                                                       '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
                                  str(self.tradeDate + ' 15:01:00'), '%Y%m%d %H:%M:%S'):]])

        return quoteData

    def StructQueueData(self):
        """

        :return:
        """

    def ReadFile(self,dataPath,symbol,tradeDate,dataType = 'Quote',symbolType = 'stock'):
        """
        将读取文件的步骤都写在这里，分为quote,trade,etf,future，并重新命名column name，在这里将输出的格式统一
        :param dataPath:  str，数据存储路径
        :param symbol:  str，标的代码
        :param tradeDate: str，交易日期
        :param dataType: str，quote还是trade，或者order
        :param symbolType: str，标的类型
        :return: dataframe，返回读取出来的数据
        """
        if self.dataReadType == 'csv':
            if dataType == 'Quote':
                if symbolType == 'stock':
                    fileName = dataPath + '/' + tradeDate + '_wind/I/' + symbol + '_' + tradeDate + '_quote.csv'
                    # msgtime, msgsource, msgtype, msgid, nanotime, time, nTime, szWindCode, szCode, nActionDay, nTradingDay, nStatus, nPreClose, nOpen, nHigh, nLow, nMatch, nNumTrades, iVolume, iTurnover, nTotalBidVol, nTotalAskVol, nWeightedAvgBidPrice, nWeightedAvgAskPrice, nIOPV, nYieldToMaturity, nHighLimited, nLowLimited, chPrefix
                    # askprice01, askvolume01, askprice02, askvolume02, askprice03, askvolume03, askprice04, askvolume04, askprice05, askvolume05, askprice06, askvolume06, askprice07, askvolume07, askprice08, askvolume08, askprice09, askvolume09, askprice10, askvolume10, bidprice01, bidvolume01, bidprice02, bidvolume02, bidprice03, bidvolume03, bidprice04, bidvolume04, bidprice05, bidvolume05, bidprice06, bidvolume06, bidprice07, bidvolume07, bidprice08, bidvolume08, bidprice09, bidvolume09, bidprice10, bidvolume10
                    columns = ['msgtime', 'msgsource', 'msgtype', 'msgid', 'nanotime', 'time', 'exchangeTime', 'micCode',
                               'szCode', 'nActionDay', 'nTradingDay', 'nStatus', 'nPreclose', 'nOpen', 'nHigh', 'nLow',
                               'latest', 'tradeNos', 'tradeVolume', 'totalTurnover', 'nTotalBidVol', 'nTotalAskVol',
                               'nWeightedAvgBidPrice', 'nIOPV', 'nYieldToMaturity', 'nHighLimited', 'nLowLimited',
                               'chPrefix',
                               'askPrice1', 'askVolume1', 'askPrice2', 'askVolume2', 'askPrice3', 'askVolume3', 'askPrice4',
                               'askVolume4', 'askPrice5', 'askVolume5',
                               'askPrice6', 'askVolume6', 'askPrice7', 'askVolume7', 'askPrice8', 'askVolume8', 'askPrice9',
                               'askVolume9', 'askPrice10', 'askVolume10',
                               'bidPrice1', 'bidVolume1', 'bidPrice2', 'bidVolume2', 'bidPrice3', 'bidVolume3', 'bidPrice4',
                               'bidVolume4', 'bidPrice5', 'bidVolume5',
                               'bidPrice6', 'bidVolume6', 'bidPrice7', 'bidVolume7', 'bidPrice8', 'bidVolume8', 'bidPrice9',
                               'bidVolume9', 'bidPrice10', 'bidVolume10'
                               ]
                elif symbolType == 'future':
                    fileName = dataPath + '/' + symbol + '_' + tradeDate + '_future.csv'
                    columns = ['msgtime', 'msgsource', 'msgtype', 'msgid', 'nanotime', 'time','exchangeTime', 'micCode', 'szCode', 'nActionDay', 'nTradingDay', 'nStatus', 'tradeVolume','totalTurnover',
                               'iOpenInterest', 'iPreOpenInterest','nPreClose','nPreSettlePrice','nOpen',  'nHigh', 'nLow', 'nMatch', 'latest', 'nSettlePrice', 'nHighLimited','nLowLimited', 'nPreDelta', 'nCurrDelta',
                               'askPrice1',  'askVolume1', 'askPrice2','askVolume2', 'askPrice3',  'askVolume3','askPrice4', 'askVolume4', 'askPrice5','askVolume5',
                               'bidPrice1','bidVolume1', 'bidPrice2','bidVolume2', 'bidPrice3','bidVolume3', 'bidPrice4', 'bidVolume4', 'bidPrice5', 'bidVolume5']



            elif dataType == 'Trade':
                fileName = dataPath + '/' + symbol + '_' + tradeDate + '_trade.csv'
            elif dataType == 'Queue':

                return None
            elif dataType == 'Order':
                return None
            elif dataType == 'Index':
                fileName = dataPath + '/' + symbol + '_' + tradeDate + '_index.csv'
                columns = ['msgtime', 'msgsource', 'msgtype', 'msgid', 'nanotime', 'time', 'exchangeTime', 'micCode',
                           'szCode', 'nActionDay', 'nTradingDay', 'nOpenIndex', 'nHighIndex', 'nLowIndex',
                           'latest', 'nPreClose', 'iTotalVolume', 'iTurnover']
            else:
                raise ('Wrong data type.')

            if os.path.exists(fileName) is False:
                return None
            # print(fileName)
            output = pd.read_csv(fileName,skiprows = 1,header=None,encoding= 'oem')
            output.columns = columns
            if dataType == 'Quote':
                output.index = pd.to_datetime(pd.Series(map(lambda stime: self.tradeDate + str(stime)[:-3], output.loc[:, 'exchangeTime'])),
                                              format='%Y%m%d%H%M%S')
            elif dataType == 'Trade':
                output.index = pd.to_datetime(pd.Series(map(lambda stime:self.tradeDate + str(stime), output.loc[:, ' nTime'])),
                               format='%Y%m%d%H%M%S%f')
            elif dataType == 'Index':
                output.index = pd.to_datetime(pd.Series(map(lambda stime:self.tradeDate + str(stime), output.loc[:, 'exchangeTime'])),
                               format='%Y%m%d%H%M%S%f')



        elif self.dataReadType == 'gzip':

            exchange = symbol.split('.')[1].upper()

            miccode = symbol.split('.')[0]

            if dataType == 'Quote':

                # dataPath = dataPath + '/wd_quote/tick' + tradeDate[:4] + 'd/' + exchange + tradeDate[:6] + 'd' + '/' + exchange + '_' + tradeDate

                dataPath = dataPath + '/SecTick/' + exchange + '/' + tradeDate[:6] + '/' + tradeDate + '/'

                fileName = dataPath + '/' + miccode + '_' + tradeDate + '.csv.gz'

                # colnames: 市场代码,证券代码,时间(yyyy-mm-dd),最新,成交笔数,成交额,成交量,方向,买一价,买二价,买三价,买四价,买五价,卖一价,卖二价,卖三价,卖四价,卖五价,买一量,买二量,买三量,买四量,买五量,卖一量,卖二量,卖三量,卖四量,卖五量

                # columns = ['exchangeCode', 'micCode', 'exchangeTime', 'latest', 'tradeNos',

                #            'totalTurnover', 'tradeVolume', 'side',

                #            'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',

                #            'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',

                #            'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',

                #            'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']

                columns = [

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

                dateFormat = '%Y-%m-%d %H:%M:%S'

                timeColumn = 'exchangeTime'

            elif dataType == 'Trade':

                # dataPath = dataPath + '/wd_trade/tick' + tradeDate[:4] + 'd/' + exchange + tradeDate[:6] + 'd' + '/' + exchange + '_' + tradeDate

                dataPath = dataPath + '/Transaction/' + exchange + '/' + tradeDate[:6] + '/' + tradeDate + '/'

                fileName = dataPath + '/' + miccode + '_' + tradeDate + '.csv.gz'

                columns = ['exchangeCode', ' nTime', ' nPrice', ' nVolume',

                           ' nTurnover', ' nBSFlag', ' nSeqnum', ' nBidOrder', ' nAskOrder', ' nOrderKind',
                           ' nChFunctionCode']

                dateFormat = '%Y%m%d% H%M%S%f'

                timeColumn = ' nTime'

            elif dataType == 'Order':

                # dataPath = dataPath + '/wd_order/tick' + tradeDate[:4] + 'd/' + exchange + tradeDate[

                #                                                                            :6] + 'd' + '/' + exchange + '_' + tradeDate

                dataPath = dataPath + '/MarketOrder/' + exchange + '/' + tradeDate[:6] + '/' + tradeDate + '/'

                fileName = dataPath + '/' + miccode + '_' + tradeDate + '.csv.gz'

                columns = ['inst', 'time', 'order_id',

                           'price', 'qty', 'order_kind', 'function_code']

                timeColumn = 'time'

            elif dataType == 'Future':

                # dataPath = dataPath + '/wd_order/tick' + tradeDate[:4] + 'd/' + exchange + tradeDate[

                #                                                                            :6] + 'd' + '/' + exchange + '_' + tradeDate
                ##exchange CF
                ##midcode _IC
                dataPath = dataPath + '/FutTick/' + exchange + '/' + tradeDate[:6] + '/' + tradeDate + '/'

                fileName = dataPath + '/' + miccode + '_' + tradeDate + '.csv.gz'

                columns = [

                    'id', 'exchangeTime', 'latest', 'pre_close', 'settle','pre_settle','open', 'high', 'low','close', 'upper_limit',

                    'lower_limit', 'status', 'buy', 'tradeVolume', 'totalTurnover',

                    'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',

                    'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',

                    'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',

                    'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5',



                    # 'open_interest', 'open_qty', 'close_qty']

                    'open_interest','pre_open_interest', 'open_qty','close_qty']



                timeColumn = 'exchangeTime'

            elif dataType == 'Index':

                # dataPath = dataPath + '/wd_order/tick' + tradeDate[:4] + 'd/' + exchange + tradeDate[

                #                                                                            :6] + 'd' + '/' + exchange + '_' + tradeDate

                dataPath = dataPath + '/IndexTick/' + exchange + '/' + tradeDate[:6] + '/' + tradeDate + '/'

                fileName = dataPath + '/' + miccode + '_' + tradeDate + '.csv.gz'

                columns = ['id', 'time', 'latest', 'pre_close', 'open', 'high', 'low', 'trade_qty', 'turnover']

                timeColumn = 'time'

            if os.path.exists(fileName) is False:
                return None

            output = pd.read_csv(fileName, encoding='oem',compression='gzip')  ## NOTICE: oem in case of 中文
            output.columns = columns
            # output.index = pd.to_datetime(output.loc[:,'exchangeTime'],dateFormat)
            output.index = pd.to_datetime(output.loc[:,timeColumn])
        else:
            print('Error data reading type')
            return None

        return output


    # older version to read the data from csv with level2 data.
    def ReadQuoteData(self):
        """

        :return: all quote data for one day
        """
        fileName = self.dataPath + '/' + self.tradeDate + '/MarketData.csv'
        quoteData = pd.read_csv(fileName)
        # 日期	 本地时间	 服务器时间	 交易所时间	 万得代码	 原始代码	 业务发生日(自然日)	 交易日	状态	前收	开盘价	最高价	最低价	最新价	申卖价	申卖量	申买价	申买量	成交笔数	成交总量
        # 成交总金额	委托买入总量	委托卖出总量	加权平均委买价格	加权平均委卖价格	IOPV净值估值	到期收益率	涨停价	跌停价	证券信息前缀	市盈率1	市盈率2	升跌2（对比上一笔）

        quoteData.columns = ['date', 'localTime', 'serverTime', 'exchangeTime', 'windcode', 'exchangeCode',
                             'businessDate', 'tradeDate', 'status', 'preClose', 'open', 'high', 'low', 'latest',
                             'askPrice', 'askVolume', 'bidPrice', 'bidVolume', 'tradeNos', 'tradeVolume', \
                             'totalTurnover', 'entrustBuyVolume', 'entrustSellVolume', 'wtdAvgBuyPrice',
                             'wtdAvgSellPrice', 'IPONetWorth', 'YTD', 'upperLimitPrice', 'downLimitPrice', 'address',
                             'pe1', 'pe2', 'change']
        quoteData = quoteData[quoteData.loc[:, 'exchangeTime'] >= 92500000]
        quoteData.index = map(lambda stime: datetime.datetime.strptime(self.tradeDate + str(stime), '%Y%m%d%H%M%S%f'),
                              quoteData.loc[:, 'exchangeTime'].values)
        columnsToSave = ['windcode', 'preClose', 'open', 'high', 'low', 'latest', 'askPrice', 'askVolume', 'bidPrice',
                         'bidVolume', 'tradeVolume', 'totalTurnover']
        quoteData = quoteData.loc[:, columnsToSave]
        quoteData = quoteData[quoteData.loc[:, 'latest'] > 0]
        """ adding part
            1.calculate the mid price for the quote data
        """
        midPlist = []
        for row in zip(quoteData['askPrice'], quoteData['bidPrice']):
            askPrice1 = float(row[0].split(' ')[0])
            bidPrice1 = float(row[1].split(' ')[0])
            midp = (askPrice1 + bidPrice1) / 2
            midPlist.append(midp)
        quoteData['midp'] = midPlist
        return quoteData

    # older version to read the data from csv with level2 data.
    def ReadTradeData(self):
        """

        :return: read tradeData
        """

    def StructIndexData(self):
        indexData = {}
        if isinstance(self.indexmiccode, str):
            # data_key = 'index_' + self.indexmiccode.replace('.','') + '_' + self.tradeDate.replace('-', '')
            # tradeData = self.hdf_buffer.read(data_key)
            # if tradeData.empty:
            indexData[self.indexmiccode] = self.ReadIndexData(self.dataPath, self.indexmiccode, self.tradeDate)
                # self.hdf_buffer.write(data_key, tradeData)
            # else:
            #     indexData[self.indexmiccode] = tradeData
        else:
            for symbol in self.indexmiccode:
                # data_key = 'index_' + symbol.replace('.','') + '_' + self.tradeDate.replace('-', '')
                # tradeData = self.hdf_buffer.read(data_key)
                # if tradeData.empty:
                indexData[symbol] = self.ReadIndexData(self.dataPath, symbol, self.tradeDate)
                    # self.hdf_buffer.write(data_key,tradeData)
                # else:
                #     indexData[symbol] = tradeData

        return indexData

    def StructFutureData(self):
        """
        :param type: index future type: IH IF IC with contract maturity month and year
        :return: index future data
        """
        futureData = {}
        if isinstance(self.futureSymbol,str):
            # data_key = 'future_' + self.futureSymbol.replace('.','') + '_' + self.tradeDate.replace('-', '')
            # tradeData = self.hdf_buffer.read(data_key)
            # if tradeData.empty:
            futureData[self.futureSymbol] = self.ReadSingleQuoteDataLevel2(self.dataPath,self.futureSymbol,self.tradeDate,type = 'Future')
                # self.hdf_buffer.write(data_key, tradeData)
            # else:
            #     futureData[self.futureSymbol] = tradeData
        else:
            for symbol in self.futureSymbol:
                # data_key = 'future_' + symbol.replace('.','') + '_' + self.tradeDate.replace('-', '')
                # tradeData = self.hdf_buffer.read(data_key)
                # if tradeData.empty:
                futureData[symbol] = self.ReadSingleQuoteDataLevel2(self.dataPath,symbol,self.tradeDate,type = 'Future')
                    # self.hdf_buffer.write(data_key,tradeData)
                # else:
                #     futureData[symbol] = tradeData

        return futureData
        # return self.ReadSingleQuoteDataLevel2(self.dataPath,self.futureSymbol,self.tradeDate,type = 'future')

    def ReadIndexData(self,dataPath,symbol,tradeDate):
        """  Read data from file"""
        quoteData = self.ReadFile(dataPath,symbol=symbol,tradeDate = tradeDate,dataType="Index",symbolType="Index")
        if quoteData is None:
            return None
        # quoteData.loc[:,'tradeVolume'] = quoteData.loc[:,'tradeVolume'] * self.volumeMultiplier
        """  convert the data into the type we need"""
        # quoteData.loc[quoteData.loc[:, 'bidPrice1'] == 0, 'bidPrice1'] = list(quoteData.loc[quoteData.loc[:, 'bidPrice1'] == 0, 'askPrice1'])
        # bidPrice1 = quoteData.loc[:, 'bidPrice1']/self.priceMultiplier
        # quoteData.loc[quoteData.loc[:, 'askPrice1'] == 0, 'askPrice1'] = list(quoteData.loc[quoteData.loc[:, 'askPrice1'] == 0, 'bidPrice1'])
        # askPrice1 = quoteData.loc[:, 'askPrice1']/self.priceMultiplier
        # quoteData = quoteData.assign(midp=((bidPrice1 + askPrice1) / 2).values)
        quoteData['latest'] = quoteData['latest']/self.priceMultiplier
        quoteData.loc[quoteData.loc[:,'latest'] == 0,'latest'] = np.nan
        quoteData = quoteData.iloc[2:-2,:]
        # quoteData = pd.concat([quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:15:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 11:30:00'), '%Y%m%d %H:%M:%S'):],
        #                       quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 13:00:00'),
        #                                                                '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
        #                           str(self.tradeDate + ' 15:01:00'), '%Y%m%d %H:%M:%S'):]])
        # data = data[data.loc[:, ' midp'] > 0]
        # quoteData = quoteData[quoteData.loc[:, 'latest'] > 0] # filter the error data
        quoteData = pd.concat([quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:25:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 11:30:00'), '%Y%m%d %H:%M:%S'):],
                              quoteData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 13:00:00'),
                                                                       '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
                                  str(self.tradeDate + ' 15:01:00'), '%Y%m%d %H:%M:%S'):]])
        # ## expand the data to second data.
        # secondTempData = pd.concat([pd.DataFrame(index=pd.date_range(self.tradeDate + " 9:25:00", self.tradeDate + " 11:30:00", freq='1S'), columns=quoteData.columns),
        #                              pd.DataFrame(index=pd.date_range(self.tradeDate + " 13:00:00", self.tradeDate + " 15:01:00", freq='1S'),
        #                                           columns=quoteData.columns)]) ## s means second
        #
        # secondTempData.loc[quoteData.index,:] = quoteData  ## TODO: the future time stamp need to convert into second instead of micro second.
        # secondData = secondTempData.fillna(method = 'ffill')
        # secondData = secondData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 15:00:00'), '%Y%m%d %H:%M:%S'),:]

        # return quoteData
        if self.freq != '1S':
            secondData = quoteData
            secondData = secondData.resample(self.freq,label = 'right').last()
            secondData = secondData.fillna(method='ffill')
            secondData = pd.concat([secondData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(str(self.tradeDate + ' 11:30:00'), '%Y%m%d %H:%M:%S'),:],
                                    secondData.loc[datetime.datetime.strptime(str(self.tradeDate + ' 13:00:01'),
                                                                              '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
                                        str(self.tradeDate + ' 15:00:00'), '%Y%m%d %H:%M:%S'), :]])
            quoteData = secondData
        return quoteData

    def ReadDataFromCpp(self,type = 'Quote'):
        #  需要import给定的模块和对应的接口，假设给定的模块为hello_ext，给定的函数为GetData(dataType)，返回list，再将list处理成data frame，从而操作。
        #  import hello_ext
        # dataParser = hello_ext.DataParser(self.dataPath,self.tradeDate,type)
        # dataList = dataParser.GetData(type)
        # data2save = pd.DataFrame(dataList)
        # data2save.columns = hello_ext.GetColNames()
        # data2save.set_index('szWIndCode')
        # return data2save # 得到的data2save包含了所有股票，包括etf等。需要根据symbol去split成单独的数据。假设数据量太大，需要分批转换（对list取范围）。
        return 0

    def ReadDataFromCsv(self, dataPath, tradeDate, symbol, type = 'Quote'):
        if type == 'Quote':
            exchange = symbol.split('.')[1].lower()
            dataPath = dataPath + '/' + exchange + tradeDate[:6] + 'd' + '/' + exchange + '_' + tradeDate
            fileName = dataPath + '/' + symbol.split('.')[0] + '_' + tradeDate + '.csv'
            # colnames: 市场代码,证券代码,时间(yyyy-mm-dd),最新,成交笔数,成交额,成交量,方向,买一价,买二价,买三价,买四价,买五价,卖一价,卖二价,卖三价,卖四价,卖五价,买一量,买二量,买三量,买四量,买五量,卖一量,卖二量,卖三量,卖四量,卖五量
            columns = ['exchangeCode', 'micCode', 'exchangeTime', 'latest', 'tradeNos',
                       'totalTurnover', 'tradeVolume', 'side',
                       'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
                       'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',
                       'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',
                       'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']

        elif type == 'Trade':
            fileName = './Data/000001.SZ_20170419_trade.csv'
            columns = ['exchangeCode', 'micCode', 'exchangeTime', 'latest', 'tradeNos',
                       'totalTurnover', 'tradeVolume', 'side',
                       'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
                       'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',
                       'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',
                       'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']
        elif type == 'Index':
            exchange = symbol.split('.')[1].lower()
            dataPath = dataPath + '/index_tick_' + exchange + tradeDate[:6] + 'd' + '/' + exchange + '_' + tradeDate
            fileName = dataPath + '/' + symbol.split('.')[0] + '_' + tradeDate + '.csv'
            # colnames: 市场代码,证券代码,时间(yyyy-mm-dd),最新,成交笔数,成交额,成交量,方向,买一价,买二价,买三价,卖一价,卖二价,卖三价,买一量,买二量,买三量,卖一量,卖二量,卖三量
            columns = ['exchangeCode', 'micCode', 'exchangeTime', 'latest', 'tradeNos',
                       'totalTurnover', 'tradeVolume', 'side',
                       'bidPrice1', 'bidPrice2', 'bidPrice3',
                       'askPrice1', 'askPrice2', 'askPrice3',
                       'bidVolume1', 'bidVolume2', 'bidVolume3',
                       'askVolume1', 'askVolume2', 'askVolume3']
        elif type == 'Future':
            dataPath = dataPath + '/sf5_c' + tradeDate[:6] + 'd' + '/' + tradeDate
            files = os.listdir(dataPath)
            futureType = 'IH'
            contractDate = '1711'
            for file in files:
                if futureType + contractDate in file:
                    break
            fileName = dataPath + '/' + file
            # fileName = dataPath + '/' + symbol.split('.')[0] + '_' + tradeDate + '.csv'
            # 市场代码,合约代码,时间,最新,持仓,增仓,成交额,成交量,开仓,平仓,成交类型,方向,买一价,买二价,买三价,买四价,买五价,卖一价,卖二价,卖三价,卖四价,卖五价,买一量,买二量,买三量,买四量,买五量,卖一量,卖二量,卖三量,卖四量,卖五量

            columns = ['exchangeCode', 'micCode', 'exchangeTime', 'latest', 'oi', 'add_oi',
                       'totalTurnover', 'tradeVolume', 'openPosition', 'closePosition', 'transactionType', 'side',
                       'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
                       'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',
                       'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',
                       'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']
        elif type == 'Queue':

            return ''
        elif tupe == 'Order':
            return ''
        else:
            raise('Wrong data type.')

        if os.path.exists(fileName) is False:
            return None
        quoteData = pd.read_csv(fileName, skiprows=1, header=None,
                                encoding='oem')  ## NOTICE: need to check the file encoding before reading the file
        quoteData.columns = columns

        return quoteData



if __name__ == '__main__':
    """
    test the class
    """
    # data = Data('E:/personalfiles/to_zhixiong/to_zhixiong/level2_data_with_factor_added','600030.SH','20170516')
    dataPath = 'E:/data/stock/wind'
    ## /sh201707d/sh_20170703
    tradeDate = '20180330'
    # symbol = ['600519.SH','600887.SH'] # 食品与饮料
    # symbol = ['600048.SH', '600340.SH', '600606.SH'] # 房地产
    # symbols = ['000856.SZ', '300137.SZ', '600340.SH', '600550.SH']  # 雄安新区板块
    # symbols = ['601211.SH', '601688.SH', '600030.SH', '000776.SZ']  # 证券
    # symbols = ['002049.SZ', '600703.SH', '600460.SH', '600584.SH']  # 半导体
    # symbols = ['000001.SZ', '600000.SH', '600036.SH', '601166.SH']  # 银行
    # symbols = ['000156.SZ', '300251.SZ', '600037.SH', '600373.SH']  # 传媒
    # symbols = ['000709.SZ',   '000959.SZ',   '600010.SH',   '600019.SH']  # 钢铁
    # symbols = ['000983.SZ',   '600157.SH',   '600188.SH', '601088.SH', '601225.SH']  # 煤炭
    # symbols = ['000009.SZ', '000413.SZ', '600516.SH', '601877.SH', '603133.SH','000009.SZ', '000413.SZ', '600516.SH', '601877.SH', '603133.SH','000009.SZ', '000413.SZ', '600516.SH', '601877.SH', '603133.SH','000009.SZ', '000413.SZ', '600516.SH', '601877.SH', '603133.SH','000009.SZ', '000413.SZ', '600516.SH', '601877.SH', '603133.SH','000009.SZ', '000413.SZ', '600516.SH', '601877.SH', '603133.SH']  # 石墨烯
    symbols = ['000001.SZ']
    # exchange = symbol.split('.')[1].lower()
    data = Data(dataPath, symbols, tradeDate,dataReadType= 'gzip', RAWDATA = 'True')
    # priceDf = pd.concat(list(map(lambda symbol: pd.DataFrame(list(data.quoteData[symbol].loc[:, 'midp']/data.quoteData[symbol].loc[:, 'midp'][300]), columns=[symbol], index = data.quoteData[symbol].index), symbols)), 1)
    print(data.quoteData)
    # print(data.indexData)
    print("Test reading data done")