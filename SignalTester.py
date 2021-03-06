# -*- coding: utf-8 -*-
"""
Created on 2017-12-26

@author: zhixiong

use: to test the signal by our idea.
"""

import pandas as pd
import Data
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import os
import bisect
import datetime
import numpy as np
import math
import itertools
import SignalLibrary
from dateutil.parser import parse
from Utils import *

class SignalTester(object):

    def __init__(self, Data, dailyData,  tradeDate, symbol,fee, sectorData = pd.DataFrame(), dataSavePath='./Plot', type='Quote',
                 sectorAllData=pd.DataFrame()):
        """
                init data here.

        :param Data: Data that from Data class
        :param dailyData: The daily freq data with all symbols contains columns: open, high, low, close, volume, turnover. To do some calculation
        :param sectorData: dataframe, include the symbol sector infomation
        :param tradeDate: str, date to test
        :param symbol: list, symbols to test
        :param dataSavePath: str, path to save the result
        :param type:
        :param sectorAllData: dataframe, contains the sector columns
        """
        if not os.path.exists(dataSavePath):
            os.makedirs(dataSavePath)
        self.dataSavePath = dataSavePath
        if type is 'Quote':
            self.allQuoteData = Data.quoteData
            # self.Data = Data.quoteData[symbol]
            # self.signals        = Data.quoteqa
            self.priceColumn = 'midp'
            self.allTradeData = Data.tradeData
            # self.Data['tradeVolume'] =
        elif type is 'Trade':
            self.Data = Data.tradeData
            # self.signals        = Data.tradeqa
            self.priceColumn = ' nPrice'
        elif type is 'Future':
            self.allQuoteData = Data.futureData
            self.priceColumn = 'midp'
        self.fee = fee
        self.logReturns = dict()
        self.dailyData = dailyData
        self.tradeDate = parse(tradeDate) # parse the str into datetime format
        self.symbol = symbol
        self.sectorData = sectorData
        self.sectorAllData = sectorAllData
        self.sectorRevisedData = {} # use dict in order to change the component conveniently
        self.sectorCashFlow = {}
        self.futureData = Data.futureData
        self.symbolFilter = {}
        self.data2save = []

    def CalLogReturn(self, Data, targetTime, window=30):
        """
        Calculate look ahead minute log return
        :param window: the future minute window(look ahead)
        :param targetTime: the current time of the data.
        :return: log return
		ssssssssssssssssssssssss
		ssssssssssssssssssssssss
        """
        ## add minute can use :  + datetime.timedelta(minutes = 30)
        i = bisect.bisect_left(Data.index, targetTime + datetime.timedelta(minutes=window)) - 1

        if Data.index[i].hour >= 15 and Data.index[i].minute >= 15:
            futurePrice = np.nan
        else:
            futurePrice = Data.loc[Data.index[i], self.priceColumn]
            if type(futurePrice) is pd.core.series.Series:
                futurePrice = futurePrice[0]
        currentPrice = Data.loc[targetTime, self.priceColumn]
        if type(currentPrice) is pd.core.series.Series:
            currentPrice = currentPrice[0]
        logReturn = math.log(futurePrice / float(currentPrice))
        return logReturn

    def GetSignalSeries(self, signal, symbol, lbWindow,paraset, startTime='', endTime=''):
        """

        :param signal: signal name
        :param symbol: stock symbol
        :param lbWindow: signal look back window
        :param startTime: signal start time.
        :param endTime: signal end time
        :return: signal series
        """
        # if signal not in self.Data.columns:
        #     signalSeries = self.Data.volumeRatio
        # else:
        signalSeries = self.allQuoteData[symbol].loc[:, signal + '_' + str(lbWindow) + '_min']
        outputSeries = signalSeries
        return outputSeries

    def GetLogReturnSeries(self, symbol, window, startTime='', endTime=''):
        """

        :param window: future minute window.,int
        :return: log return series
        """
        # priceSeries = map(lambda targetTime:self.CalLogReturn(Data = self.allQuoteData[symbol],targetTime=targetTime,window = window),self.allQuoteData[symbol].index)
        # outputSeries = pd.Series(priceSeries)
        colName = 'lr_' + str(window) + '_min'
        if colName in self.allQuoteData[symbol].columns:
            outputSeries = self.allQuoteData[symbol].loc[:, colName]
        else:
            outputSeries = self.CalLBReturn(self.allQuoteData[symbol], -window)
            self.allQuoteData[symbol].loc[:, colName] = list(outputSeries)
        return outputSeries

    def PlotSignalNReturn(self, signal, symbol, logReturns, lbWindow=10, laWindow=0, paraset = 0, timeIndex='',
                          diffTime=0):
        """
        plot the signal with the log return under different axis.
        :param signals: signal series with lb window
        :param symbol: signal series with lb window
        :param lbWindow: signal look back window (minute)
        :param logReturns: log return series with la window, should have the same time axis with the signal.
        :param diffTime: lay back window
        :return: plot with signal and return(or price).
        """
        signals = self.GetSignalSeries(signal, symbol, lbWindow,paraset)
        if timeIndex == '':
            timeIndex = self.allQuoteData[symbol].index[300:]
        Font_Size = 9  # the font size on th

        # e plot
        # if symbol == '603133.SH':
        #     print('Test')
        Data1 = signals[300:]
        # Data2 = logReturns[300:] / logReturns.fillna(method='bfill')[300] - 1  # /logReturns[300] may get nan value
        Data2 = logReturns[300:]
        Df = pd.DataFrame(list(Data1), index=timeIndex, columns=['signals'])
        # Df['signals'] = Data1
        Df2 = pd.DataFrame(list(Data2), index=timeIndex, columns=['logreturns'])
        # Df2['logreturns'] = Data2
        # TODO:: skip the noon time
        if (signal is 'sectorAction') or (signal is 'sectorActionLead'):
            if sum(signals) == 0:
                return None
            Df.loc[:, 'rollSum'] = Df.loc[:, 'signals'].rolling(60).sum()
            # posDf       = resultDf.loc[(resultDf.loc[:, 'sig'] > 0) & (resultDf.loc[:, 'sig'].diff(1) > 0) & (Df.loc[:, 'rollSum'] == 1), :]
            # negDf       = resultDf.loc[(resultDf.loc[:, 'sig'] < 0) & (resultDf.loc[:, 'sig'].diff(1) < 0) & (Df.loc[:, 'rollSum'] == -1), :]
            Df3 = Df.loc[(Df.loc[:, 'signals'] > 0) & (Df.loc[:, 'signals'].diff(1) > 0) & (
                    Df.loc[:, 'rollSum'] == 1), 'signals'] * Df2.loc[:, 'logreturns']
            Df4 = abs(Df.loc[(Df.loc[:, 'signals'] < 0) & (Df.loc[:, 'signals'].diff(1) < 0) & (
                    Df.loc[:, 'rollSum'] == -1), 'signals']) * Df2.loc[:, 'logreturns']
            Df3[Df3 == 0] = np.nan
            Df4[Df4 == 0] = np.nan

            plt.figure(figsize=(20, 12))
            plt.plot(Df2)
            plt.gcf().autofmt_xdate()
            uparrow = u'$\u2191$'  # the shape of up arrow
            downarrow = u'$\u2193$'  # the shape of down arrow
            plt.scatter(x=Df.index.tolist(), y=Df3, marker=uparrow,
                        c='g', label='SectorAction', s=90)
            plt.scatter(x=Df.index.tolist(), y=Df4, marker=downarrow,
                        c='r', label='SectorAction', s=90)
            plt.legend(['logreturn', signal + '_Pos', signal + '_Neg'])
            plt.title('Signal_Return_Compare of symbol' + symbol + ' with lbWinow = ' + str(
                lbWindow) + ' and threshold = ' + str(parameter))
            plt.savefig(self.dataSavePath + '/Signal_Price_Compare_' + signal + '_' + symbol + '_' + str(lbWindow) + '_'
                        + str(laWindow) + '_' + str(parameter) + '.jpg')
            plt.clf()
            plt.close('all')  ## need to add 'all' in the parameter
            """
            Df3Value = list(Df3.iloc[:])
            Df4Value = list(Df4.iloc[:])
            Df2Value = np.array(list(Df2.iloc[:]))

            # plot the log return
            fig,ax = plt.subplots(1,  figsize=(20,12), sharex=True)
            
            ax.plot(Df2Value, c='r', label='logreturns')
            # plot the signal
            uparrow = u'$\u2191$'  # the shape of up arrow
            downarrow = u'$\u2193$'  # the shape of down arrow
            ax.scatter(Df3Value, marker=uparrow,
                        c='g', label='SectorAction', s=90)
            ax.scatter(Df4Value, marker=downarrow,
                        c='r', label='SectorAction', s=90)

            # add legend
            ax.legend(['logreturn', signal + '_Pos', signal + '_Neg'])

            # draw the picture in order to get the ticks.
            
            fig.canvas.draw()

            # set x axis name
            labels = [item.get_text() for item in ax.get_xticklabels()]
            # print(labels)
            if len(labels) != 0:
                labels[1:-1] = Df2.index[list(map(int, labels[1:-1]))]
                ax.set_xticklabels(labels, rotation=45)
            # plt.xticks(xvalue,labels,rotation = 'vertical')

            # add title

            ax.set_title('Signal_Return_Compare of symbol' + symbol + ' with lbWinow = ' + str(
                lbWindow) + ' and threshold = ' + str(parameter))
                
            if IFSAVE:
                plt.savefig(self.dataSavePath + '/plt_' + add + '.jpg')
            """


        else:
            print('Plot with price series and signals on the same plot')
            savePath = self.dataSavePath + '/test/' + signal + '_' + str(lbWindow)
            #savePath = self.dataSavePath + '/' + signal + '_' + str(lbWindow) + '/' + str(self.tradeDate.date()).replace('-','')
            if os.path.exists(savePath) is False:
                os.makedirs(savePath)

            fig, ax = plt.subplots(1, figsize=(20, 12), sharex=True)
            subQuoteData = self.allQuoteData.get(symbol)
            if subQuoteData is None:
                return None
            else:
                # bidPrice1 = subQuoteData.loc[:, 'bidPrice1']
                # askPrice1 = subQuoteData.loc[:, 'askPrice1
                midp = subQuoteData.loc[:, 'midp']
                yvalue = list(midp.iloc[:])
                ax.plot(yvalue, label=symbol + '_midquote')
                '''
                midp_2 = subQuoteData.loc[:, 'filter_ewm']
                m_value = list(midp_2.iloc[:])
                ax.plot(m_value, label=symbol + 'filter_midquote')
                '''
                if signal == 'obi_extreme_future':
                    #Jpanag
                    upper_value =list( subQuoteData.loc[:, 'upper_bound'])
                    loerer_value =list( subQuoteData.loc[:, 'lower_bound'])
                    not_values = list( subQuoteData.loc[:, 'not'])
                    ax.plot(upper_value, label=symbol + '_upper_bound')
                    ax.plot(loerer_value, label=symbol + '_lower_bound')
                    ax.plot(not_values, label=symbol + '_not')
                    #subQuoteData.to_csv(savePath + '/' + symbol +str(self.tradeDate.date()).replace('-','')+ '.csv')

                uparrow = u'$\u2191$'  # the shape of up arrow
                downarrow = u'$\u2193$'  # the shape of down arrow
                arrowSize = 180

                signals = subQuoteData.loc[:, signal + '_' + str(lbWindow) + '_min'] * midp
                longSignal = signals.copy()
                longSignal[longSignal <= 0] = np.nan
                shortSignal = signals.copy()
                shortSignal[shortSignal >= 0] = np.nan

                # ax.plot(yvalue, )
                ax.scatter(y=list(longSignal.iloc[:]), x=range(len(yvalue)), marker=uparrow, s=arrowSize, c='red')
                ax.scatter(y=list(abs(shortSignal).iloc[:]), x=range(len(yvalue)), marker=downarrow, s=arrowSize,
                           c='green')

            fig.canvas.draw()

            # set x axis name
            # labels = [item.get_text() for item in ax.get_xticklabels()]
            # # print(labels)
            # if len(labels) != 0:
            #     labels[1:-1] = midp.index[list(map(int, labels[1:-1]))]
            #     ax.set_xticklabels(labels, rotation=45)

            ax.legend([])

            #ax.set_title('Signal ' + signal + ' with midquote change of stock ' + symbol + ' lbwindow = ' + str(lbWindow))
            plt.savefig(savePath + '/' + symbol + '.jpg')
            plt.savefig(savePath + '/' + symbol +str(self.tradeDate.date()).replace('-','')+ '.jpg')
            #print(savePath + '/' + symbol +str(self.tradeDate.date()).replace('-','')+ '.jpg')
            plt.close('all')

    def CalculateCOR(self, signal,symbol,lbWindow, logReturns, type='spearman',paraset = list()):
        """
        Calculate the cor under some condition
        :param signal: signal name with lb window
        :param logReturns: logreturn series with la window.
        :return: the correlation between signals and log returns.
        """
        signals = self.GetSignalSeries(signal,symbol, lbWindow,paraset)
        corDf = pd.DataFrame({'signal': signals, 'logReturns': logReturns})
        return corDf.corr(method=type).iloc[0, 1]

    def CheckSignal(self, symbol, signal, lbWinodw=10, laWindow=10,paraset = list()):
        """
        To check whether this signal is useful under our adjudgement
        :param signal: signal name to check
        :param symbol: symbol code for stock
        :param laWindow: look back window (minute or ticks)
        :param lbWinodw: look ahead window (minute or ticks)
        :return: boolean variable. T: signal useful F:signal useless
        step1. calculate signal. step2. calculate return. step3. plot. step4. calculate sts. step5. return T or F
        """
        # step1 : calculate signal
        if signal not in self.allQuoteData[symbol].columns:
            self.CalSignal(symbol, 0, signal, lbWinodw,paraset)
        # step2 : calculate future log return
        logReturns = self.GetLogReturnSeries(symbol, laWindow)  # use future log reuturn to calculate the WR
        # step3 : plot the signal plots

        #self.PlotSignalNReturn(signal, symbol, logReturns, lbWinodw, 0, paraset)

        # step4 : calculate the statistics

        stsDf = self.CalSts(signal, symbol, lbWinodw, laWindow,paraset = paraset)
        print(stsDf)

        # step5 : run the signal model:
        # todo: finish the model part.

        return stsDf

    def CheckAllSignals(self):
        """
        To check all signals we have
        :return: boolean variable series.
        """

    def CalSignal(self, symbol='', threshold=0, signal='volumeRatio', window=10,paraset = list()):
        """

        :param signal: the signal name, such as 'obi' or 'tfi'
        :param window: minute or ticks, look back
        :return: signal series, stadard output: 1为buy in信号，0为不操作信号， -1为卖出信号
        """
        signalLibrary = SignalLibrary.SignalLibrary(symbol,self.allQuoteData[symbol], signal=signal,tradeData = self.allTradeData[symbol],window = window)

        self.allQuoteData[symbol] = signalLibrary.getSignal()

        #self.allQuoteData[symbol].to_csv(self.dataSavePath+'/'+signal+'_'+str(self.tradeDate.date())+'.csv')
        print(self.dataSavePath+'/'+signal+'_'+str(self.tradeDate.date())+'.csv')
        #print('Calculate obi here for symbol = ', symbol, 'with lbwindow = ', window)


    def CalculateTimeDiff(self, targetTime, compareTime=''):
        if compareTime == '':
            compareTime = datetime.datetime.strptime(str(self.tradeDate.date()).replace('-', '') + str('93000000'),
                                                     '%Y%m%d%H%M%S%f')
        timeDiff = targetTime - compareTime  # suppose the diff is second
        return timeDiff.total_seconds()  # use total seconds means converting the timedelta into seconds

    def GetLast5Days(self, tradeDate=''):
        """

        :return: last 5 trade days
        """
        if tradeDate == '':
            tradeDate = str(self.tradeDate.date())
        tradingDays = self.dailyData.index
        totalNoDays = len(tradingDays)
        tradeDayPosition = np.array(range(totalNoDays))[tradingDays == tradeDate]
        if len(tradeDayPosition) == 0:
            raise ('Can not find the trade volume for this date :', tradeDate)
        else:
            last5DatePosition = tradeDayPosition - 5
            lastDatePosition = tradeDayPosition
            tradeDaysToReturn = tradingDays[last5DatePosition: lastDatePosition]
        return tradeDaysToReturn

    def CalLBReturn(self, data, lbwindow=2):
        """

        :param data: quote data with all columns
        :param lbwindow: minute or ticks to calculate the return
        :return: lb series
        """
        # logmidp = pd.Series(map(math.log,data.loc[:,'midp']))
        logmidp = np.log(data.loc[:, 'midp'])
        if lbwindow > 0:
            lbreturn = logmidp.diff(lbwindow)
        else:
            # if lb window < 0 ,means we want to calculate the la return.
            lbreturn = -logmidp.diff(lbwindow)
            # colName =
        return lbreturn

    def CalSts(self, signal, symbol, lbWindow, laWindow, logReturns=pd.Series(), IFPLOT = False,paraset = list()):
        """

        :param signal: The signal name
        :param symbol: The stock code
        :param lbWindow: Signal look back window (minute)
        :return: times, WR, WR(exclude cost)
        """
        if len(logReturns) == 0:
            logReturns = self.GetLogReturnSeries(symbol, laWindow)  # use future log reuturn to calculate the WR
        data = self.allQuoteData[symbol]
        signals = self.GetSignalSeries(signal, symbol, lbWindow,paraset)  # signal series(continuous or point)
        resultDf = pd.DataFrame({'lr': list(logReturns), 'sig': signals},
                                index=self.allQuoteData[symbol].index)  # combine together.
        resultDf = resultDf[~np.isnan(resultDf['lr'])]
        if (signal is 'sectorAction') or (
                signal is 'sectorActionLead') or (
                signal is 'sectorStockLead') or (
                signal is 'sectorCashFlow'):  # use signal to fiter the data,to find out the same data.
            # timesPos    = sum(resultDf.loc[(resultDf.loc[:,'sig'] != 0) & (resultDf.loc[:,'sig'].diff(1) != 0)& (resultDf.loc[:,'sig'].diff(60) != 0),'sig'])
            # timesNeg    = sum(resultDf.loc[(resultDf.loc[:,'sig'] != 0) & (resultDf.loc[:,'sig'].diff(1) != 0)& (resultDf.loc[:,'sig'].diff(60) != 0),'sig'])
            resultDf.loc[:, 'rollSum'] = resultDf.loc[:, 'sig'].rolling(
                60).sum()  # 60 means the last minute that occure times.
            posDf = resultDf.loc[(resultDf.loc[:, 'sig'] > 0) & (resultDf.loc[:, 'sig'].diff(1) > 0) & (
                    resultDf.loc[:, 'rollSum'] == 1), :]
            negDf = resultDf.loc[(resultDf.loc[:, 'sig'] < 0) & (resultDf.loc[:, 'sig'].diff(1) < 0) & (
                    resultDf.loc[:, 'rollSum'] == -1), :]
            times = len(posDf) + len(negDf)
            if times == 0:
                if signal is 'sectorCashFlow':
                    return pd.concat([pd.DataFrame({'times': 0, 'WR': np.nan, 'WR_excost': np.nan,'signal_return': 0}, index=[symbol]),pd.DataFrame({'cashFlowFlag':0, 'upFlag' : 0},index = [symbol])],1)
                else:
                    return pd.DataFrame({'times': 0, 'WR': np.nan, 'WR_excost': np.nan, 'signal_return': 0}, index=[symbol])
            # posWin      = self.CalWinsBySignal(data, posDf, laWindow, side = 'buy')
            # negWin      = self.CalWinsBySignal(data, negDf, laWindow, side = 'sell')
            # signalretPos= self.CalSignalRet(data,posDf,laWindow,'buy')
            # signalretNeg= self.CalSignalRet(data,negDf,laWindow,'sell')
            signalretPos, posWin = self.CalSignalRetAndWins(data, posDf, laWindow, 'buy')
            signalretNeg, negWin = self.CalSignalRetAndWins(data, negDf, laWindow, 'sell')
            if IFPLOT:
                self.SignalFutureReturn(data, posDf, 'buy', laWindow)
                self.SignalFutureReturn(data, negDf, 'sell',  laWindow)
            WR_excost = (posWin + negWin) / times
            WR = (sum(resultDf.loc[(resultDf.loc[:, 'sig'] > 0) & (resultDf.loc[:, 'sig'].diff(1) > 0) & (
                    resultDf.loc[:, 'rollSum'] == 1), 'lr'] > 0) + sum(resultDf.loc[(resultDf.loc[:, 'sig'] < 0) & (
                    resultDf.loc[:, 'sig'].diff(1) < 0) & (resultDf.loc[:, 'rollSum'] == -1), 'lr'] < 0)) / times
            signalret = signalretNeg + signalretPos
            # WR_excost   = (sum(resultDf.loc[(resultDf.loc[:, 'sig'] > 0) & (resultDf.loc[:,'sig'].diff(1) > 0)& (resultDf.loc[:,'sig'].diff(60) > 0),'lr'] - 0.002 > 0) + sum(resultDf.loc[(resultDf.loc[:,'sig'] < 0) & (resultDf.loc[:,'sig'].diff(1) < 0)& (resultDf.loc[:,'sig'].diff(60) < 0),'lr'] + 0.02 < 0) )/times
            # if IFPLOT:
            #     return pd.DataFrame({'times': times, 'WR': WR, 'WR_excost': WR_excost, 'signal_return': signalret},
            #                         index=[symbol]), posFig, negFig
            # else:
            if signal == 'sectorCashFlow':
                sector = self.sectorData.loc[self.sectorData['secucode'] == symbol,:].index[0]
                return pd.concat([pd.DataFrame({'times': times, 'WR': WR, 'WR_excost': WR_excost, 'signal_return': signalret},
                                    index=[symbol]),pd.DataFrame(self.symbolFilter[sector].loc[symbol,:]).T],1)
            else:
                return pd.DataFrame({'times': times, 'WR': WR, 'WR_excost': WR_excost, 'signal_return': signalret},
                                index=[symbol])
        elif (signal is 'tfi'):  # use signal to fiter the data,to find out the same data.
            upthreshold = 0.7
            downthreshold = -0.7
            posDf = resultDf.loc[(resultDf.loc[:, 'sig'] < downthreshold),:]
            negDf = resultDf.loc[(resultDf.loc[:, 'sig'] > upthreshold),:]
            times = len(posDf) + len(negDf)
            if times == 0:
                if signal is 'sectorCashFlow':
                    return pd.concat([pd.DataFrame(
                        {'times': 0, 'WR': np.nan, 'WR_excost': np.nan, 'signal_return': 0}, index=[symbol]),
                                      pd.DataFrame({'cashFlowFlag': 0, 'upFlag': 0}, index=[symbol])], 1)
                else:
                    return pd.DataFrame({'times': 0, 'WR': np.nan, 'WR_excost': np.nan, 'signal_return': 0},
                                        index=[symbol])
            signalretPos, posWin = self.CalSignalRetAndWins(data, posDf, laWindow, 'buy')
            signalretNeg, negWin = self.CalSignalRetAndWins(data, negDf, laWindow, 'sell')
            if IFPLOT:
                self.SignalFutureReturn(data, posDf, 'buy', laWindow)
                self.SignalFutureReturn(data, negDf, 'sell', laWindow)
            WR_excost = (posWin + negWin) / times
            WR = np.nan
            signalret = signalretNeg + signalretPos
            # WR_excost   = (sum(resultDf.loc[(resultDf.loc[:, 'sig'] > 0) & (resultDf.loc[:,'sig'].diff(1) > 0)& (resultDf.loc[:,'sig'].diff(60) > 0),'lr'] - 0.002 > 0) + sum(resultDf.loc[(resultDf.loc[:,'sig'] < 0) & (resultDf.loc[:,'sig'].diff(1) < 0)& (resultDf.loc[:,'sig'].diff(60) < 0),'lr'] + 0.02 < 0) )/times
            # if IFPLOT:
            #     return pd.DataFrame({'times': times, 'WR': WR, 'WR_excost': WR_excost, 'signal_return': signalret},
            #                         index=[symbol]), posFig, negFig
            # else:
            if signal == 'sectorCashFlow':
                sector = self.sectorData.loc[self.sectorData['secucode'] == symbol, :].index[0]
                return pd.concat([pd.DataFrame(
                    {'times': times, 'WR': WR, 'WR_excost': WR_excost, 'signal_return': signalret},
                    index=[symbol]), pd.DataFrame(self.symbolFilter[sector].loc[symbol, :]).T], 1)
            else:
                return pd.DataFrame(
                    {'times': times, 'WR': WR, 'WR_excost': WR_excost, 'signal_return': signalret},
                    index=[symbol])
        else:
            upthreshold = 1  # or some value
            downthreshold = -1  # or some value
            posDf = resultDf[resultDf.loc[:, 'sig'] >= upthreshold]
            negDf = resultDf[resultDf.loc[:, 'sig'] <= downthreshold]
            if len(posDf) + len(negDf) == 0:
                return None
            posRet, posWin = self.CalSignalRetAndWins(data, posDf, laWindow, side='buy')
            negRet, negWin = self.CalSignalRetAndWins(data, negDf, laWindow, side='sell')
            WR_excost = (posWin + negWin) / (len(posDf) + len(negDf))  # len(dataframe) returns the rows.
            # pnl = (posRet/len(posDf) + negRet/len(negDf))/2
            pnl = (posRet + negRet)/ (len(posDf) + len(negDf))
            if len(posDf) == 0:
                posRet = 0
            else:
                posRet = posRet / len(posDf)
            if len(negDf) == 0:
                negRet = 0
            else:
                negRet = negRet / len(negDf)
            return pd.DataFrame({'times': len(posDf) + len(negDf), 'WR_excost': WR_excost, 'pnl':pnl,'posRet':posRet,'negRet':negRet},
                                index=[symbol])

            # print('Waiting for other signal result to complete the sts calculation')
            # return None

    def CalWinsBySignal(self, data, posDf, laWindow, side='buy'):
        win = 0
        for itime in posDf.index:
            timePos = pd.Index(data.index).get_loc(itime)
            if timePos > (data.shape[0] - laWindow * 60):  # should pass the laWindow to here
                endPos = data.shape[0]
            else:
                endPos = timePos + laWindow * 60
            if side == 'buy':
                openClosePricePos = pd.Index(data.columns).get_loc('askPrice1')
                closePircePos = pd.Index(data.columns).get_loc('bidPrice1')
            else:
                openClosePricePos = pd.Index(data.columns).get_loc('bidPrice1')
                closePircePos = pd.Index(data.columns).get_loc('askPrice1')

            # midPos = pd.Index(data.columns).get_loc('midp')
            midps = data.iloc[endPos, openClosePricePos]
            midp = data.iloc[timePos, closePircePos]
            if side is 'buy':
                if np.log(midps / midp) >   self.fee:
                    win = win + 1
            else:
                if np.log(midps / midp) < -  self.fee:
                    win = win + 1

        return win

    def CalSignalRet(self, data, posDf, laWindow, side='buy'):
        """
        This function aims to calculate the signal return (buy one and sell one)
        :return:
        """
        ret = 0
        win = 0
        for itime in posDf.index:
            timePos = pd.Index(data.index).get_loc(itime)
            if timePos > (data.shape[0] - laWindow * 60):  # should pass the laWindow to here
                endPos = data.shape[0]
            else:
                endPos = timePos + laWindow * 60
            if side == 'buy':
                openClosePricePos = pd.Index(data.columns).get_loc('askPrice1')
                closePircePos = pd.Index(data.columns).get_loc('bidPrice1')
            else:
                openClosePricePos = pd.Index(data.columns).get_loc('bidPrice1')
                closePircePos = pd.Index(data.columns).get_loc('askPrice1')

            # midPos = pd.Index(data.columns).get_loc('midp')
            midps = data.iloc[endPos, openClosePricePos]
            midp = data.iloc[timePos, closePircePos]
            if side is 'buy':
                # if max(list(map(math.log, data.iloc[timePos:endPos, midPos]/midp))) > 0.003:
                #     ret = ret + 0.001
                # else:
                ret = ret + math.log(midps / midp) -   self.fee  # set cost here
                if np.log(midps / midp) >   self.fee:
                    win = win + 1
            else:
                # if min(list(map(math.log, data.iloc[timePos:endPos, midPos]/midp))) < -0.003:
                #     # ret = ret + 0.001
                # else:
                ret = ret + math.log(midp / midps) -   self.fee
                if np.log(midps / midp) < -  self.fee:
                    win = win + 1

        return ret, win

    def CalSignalRetAndWins(self, data, posDf, laWindow, side='buy'):
        """
        This function aims to calculate the signal return (buy one and sell one)
        :return:
        """
        ret = 0
        win = 0
        for itime in posDf.index:
            timePos = pd.Index(data.index).get_loc(itime)
            #if isinstance(timePos,np.int64) is False:

            if isinstance(timePos,np.int) is False:
                print(timePos)
                timePos = timePos.stop

            if timePos > (data.shape[0] - laWindow):  # should pass the laWindow to here
                endPos = data.shape[0]
            else:
                endPos = timePos + laWindow
            if side == 'buy':
                openPricePos = pd.Index(data.columns).get_loc('askPrice1')
                closePircePos = pd.Index(data.columns).get_loc('bidPrice1')
            else:
                openPricePos = pd.Index(data.columns).get_loc('bidPrice1')
                closePircePos = pd.Index(data.columns).get_loc('askPrice1')

            # midPos = pd.Index(data.columns).get_loc('midp')
            midp = data.iloc[timePos, openPricePos]
            #print()
            midps = data.iloc[endPos, closePircePos]
            if side is 'buy':
                if np.log(midps / midp) >   self.fee:
                    win = win + 1
                ret = ret + math.log(midps / midp) -   self.fee  # set cost here
            else:
                if np.log(midps / midp) < -  self.fee:
                    win = win + 1
                ret = ret + math.log(midp / midps) -   self.fee

        return ret, win

    def SignalFutureReturn(self, data, posDf, side = 'buy', laWindow = 15):
        """
        用来观察信号之后的return（或者mid quote）的变化情况
        :param data: DataFrame，股票秒级数据
        :param posDf: DataFrame，信号（秒级，有信号为1或者-1，无信号为0）
        :param laWindow: int，最长区间段
        #:param side: 买信号或者卖信号
        :return: plot，未来价格的均值变化图
        """
        lrDfList = list()
        for itime in posDf.index:
            timePos = pd.Index(data.index).get_loc(itime)
            if timePos > (data.shape[0] - laWindow * 60):  # should pass the laWindow to here
                continue
            else:
                endPos = timePos + laWindow * 60
            midPos = pd.Index(data.columns).get_loc('midp')
            midps = data.iloc[(timePos + 1):endPos, midPos]
            midp = data.iloc[timePos, midPos]
            logreturns = np.log(midps/midp)
            lrDfList.append(pd.DataFrame({str(itime): list(logreturns)}))

        if len(lrDfList) == 0:
            return 0
        lrDf = pd.concat(lrDfList, 1)
        lrDfMean = lrDf.mean(1)

        fig1, ax1 = plt.subplots(1, figsize=(20, 12), sharex=True)


        # plt.figure(figsize=(20,12))
        ax1.plot(lrDfMean, c = 'b')
        ax1.set_title('Mean future log return with signal in la window = ' +  str(laWindow) + ' of symbol = ' + str(data.loc[itime,'exchangeCode']) + ' of side = ' + side)

        fig1.canvas.draw()

        plt.savefig(self.dataSavePath + '/futureLR_' +  str(laWindow) + '_' + str(data.loc[itime,'exchangeCode']) + '_' + side + '.jpg')

        plt.close('all')
        return 0

    def CheckSectorCashFlow(self, symbol='', sector='', lbWindow=5, orderType='netCashFlow',normalize = False,revise = False):
        if symbol is '' and sector is '':
            raise ('Error symbol and sector')
        elif sector is '':
            sector = self.sectorData.loc[self.sectorData.loc[:, 'secucode'] == symbol].index[0]
        sectorCashFlow = self.GetSectorCashFlow(
            sector,normalize = normalize,revise = revise)  # here, the secotrCashFlow should be the dataframe with different types of order cash flow with time as index.
        """
        板块资金是为了确定该板块发生了趋势，即有大量的资金突然流入或者大量的资金突然流出，可以辅助确定同个版块内的信号
        因为使用的是板块内的所有股票（不仅仅是选出来的股票），因此可以直接看资金流向是否与信号方向相匹配？
        即累积资金流 > 0 或者 净资金流 > 0, 才是对应的方向？
        Ideas here:
        1. 资金流速 at some level, like 0.5?：如何计算资金流速？
        2. turns from net negative into positive.
        3. 主力资金所占比例？或者单纯看主力资金的增快和减少
        # TODO: 加入资金流信息，即IC = abs(CF/Amount), CF为资金流净额，Amount为交易金额。即为所计算的net_ratio，其值在10%以上才有足够的信息量。
        4. 资金流信息>0.1的情况下
        返回结果：
        series：满足条件的正向为1，负向为-1，index为时间。
        
        #  ge期货资金流做法：
        1. 过去n分钟的资金净流入净流出（股票）超大单、大单不做
        2. 最近n分钟资金净流入>threshold，资金回转（n分钟不同方向）
        3. 最近n分钟都是净流入：趋势
        4. 净流入：
        
        """
        # sectorCashFlowCheck = sectorCashFlow.loc
        # [:, orderType]
        sectorRollSum = sectorCashFlow.loc[:, orderType].rolling(int(lbWindow * 60)).sum()  # 得到当前板块所有股票资金流在过去lbwindow下的和，。。。？
        # 资金流流速快，即在一定时间内有大量的资金流入，净大单流入
        # 分钟资金流向：（分钟大单资金流入资金量-分钟大单资金流出资金量）/该股流通市值
        # 当前秒钟资金流向：（秒钟主力流入资金量-秒钟主力流出资金量）/该股流通市值（假设恒定）
        # 秒钟资金流向 > 过去至今的资金净流入或者流出的绝对值的n倍（数值如何定，或者超过一定比例）（暂定5倍或者变化的数值）
        # sectorLiquidSpeed = sectorRollSum / (lbWindow * 60)  # the speed of cash. How to evaluate the speed effect?
        sectorCashFlow.loc[:, 'cashFlowCheck'] = 0

        # method 1

        # sectorCashFlow.loc[sectorCashFlow.loc[:, orderType + '_netRatio'] > 0.1, 'cashFlowCheck'] = 1  # check the current cash flow
        # sectorCashFlow.loc[sectorCashFlow.loc[:, orderType + '_netRatio'] < -0.1, 'cashFlowCheck'] = -1  # check the current cash flow

        # method 2

        # sectorCashFlow.loc[sectorRollSum > 0, 'cashFlowCheck'] = 1  # 看当前净现金流是否大于0，或者过去一段时间的累积现金流
        # sectorCashFlow.loc[sectorRollSum < 0, 'cashFlowCheck'] = -1  # check the current cash flow

        # method 3

        # sectorCashFlow.loc[sectorCashFlow.loc[:, orderType] > 0, 'cashFlowCheck'] = 1  # check the current cash flow
        # sectorCashFlow.loc[sectorCashFlow.loc[:, orderType] < 0, 'cashFlowCheck'] = -1  # check the current cash flow

        # method 4
        # 计算每一秒钟的资金净流入（是否需要用不同bar时间去计算现金流，order Type对应列已经是净现金流），以及是过去最大净资金流的多少倍
        # 需要用分钟数据，以及用到相对于流通市值的比例，以达到标准化，消除不同股票之间的不同量值。

        # firstTime = sectorCashFlow.index[0]
        minuteCashFlow = sectorCashFlow.resample('T',label = 'right').sum()
        minuteCashFlow = self.SkipNoonBreak(minuteCashFlow)
        firstTime = minuteCashFlow.index[0]
        # sectorCashFlow.loc[:,'currentMax'] = list(map(lambda time:max(abs(sectorCashFlow.loc[firstTime:time,orderType].max()),abs(sectorCashFlow.loc[firstTime:time,orderType].min())), sectorCashFlow.index))
        # sectorCashFlow.loc[:,'currentMin'] = list(map(lambda time:-max(abs(sectorCashFlow.loc[firstTime:time,orderType].max()),abs(sectorCashFlow.loc[firstTime:time,orderType].min())), sectorCashFlow.index))
        minuteCashFlow.loc[:,'currentMax'] = list(map(lambda time:max(abs(minuteCashFlow.loc[firstTime:time,orderType].max()),abs(minuteCashFlow.loc[firstTime:time,orderType].min())), minuteCashFlow.index))
        minuteCashFlow.loc[:,'currentMin'] = list(map(lambda time:-max(abs(minuteCashFlow.loc[firstTime:time,orderType].max()),abs(minuteCashFlow.loc[firstTime:time,orderType].min())), minuteCashFlow.index))
        minuteCashFlow.loc[:,'currentMax'] = minuteCashFlow.loc[:,'currentMax'].shift(1)   #shift -1 due to the mismatch of the max and min
        minuteCashFlow.loc[:,'currentMin'] = minuteCashFlow.loc[:,'currentMin'].shift(1)   #shift -1 due to the mismatch of the max and min
        # sectorCashFlow.loc[sectorCashFlow.loc[:,orderType] > abs(sectorCashFlow.loc[:,'currentMax']) * lbWindow,'cashFlowCheck'] = 1  # 暂定5倍
        # sectorCashFlow.loc[sectorCashFlow.loc[:,orderType] < -abs(sectorCashFlow.loc[:,'currentMin']) * lbWindow,'cashFlowCheck'] = -1
        minuteCashFlow.loc[minuteCashFlow.loc[:,orderType] > abs(minuteCashFlow.loc[:,'currentMax']) * lbWindow,'cashFlowCheck'] = 1  # 暂定5倍
        minuteCashFlow.loc[minuteCashFlow.loc[:,orderType] < -abs(minuteCashFlow.loc[:,'currentMin']) * lbWindow,'cashFlowCheck'] = -1
        sectorCashFlow.loc[minuteCashFlow.index,'cashFlowCheck'] = list(minuteCashFlow.loc[:,'cashFlowCheck'])
        output = sectorCashFlow.loc[:, 'cashFlowCheck']
        output.iloc[:301] = 0  # 121 因为前n分钟样本数量太少，无法计算

        #  如何判断当有该现金流情况下，剩余股票的现金流的贡献
        if output.sum() != 0:  # 有对应现金流情况
            cashFlowTimes = output[output!=0].index
            symbolFilterList = list()
            symbols = self.sectorData.loc[sector,'secucode']
            for symbol in symbols:
                stockData = self.allTradeData.get(symbol)
                if stockData is not None:
                    stockCashFlow = self.SkipNoonBreak(stockData.resample('T',label = 'right').sum()).loc[:,orderType]  # 个股现金流

                    # stockmidp     = self.allQuoteData[symbol].loc[minuteCashFlow.index,'midp']
                    # cashflowPart = stockCashFlow.loc[cashFlowTime]/minuteCashFlow.loc[cashFlowTime,orderType]
                    cashFlowFlag = 0
                    upFlag = 0
                    for cashFlowTime in cashFlowTimes:
                        cashflowPart = stockCashFlow.loc[cashFlowTime] / minuteCashFlow.loc[cashFlowTime, orderType]
                        if cashflowPart  > 0.01:  # set parameter here
                            cashFlowFlag = cashFlowFlag + 1
                        direction = 'Buy' if output.loc[cashFlowTime]>0 else 'Sell'
                        if direction == 'Buy':
                            maxPrice = self.allQuoteData[symbol].loc[cashFlowTime:,'midp'].iloc[:3001].max()  # 30分钟内的最大涨幅
                            if np.log(maxPrice/self.allQuoteData[symbol].loc[cashFlowTime,'midp']) >= 0.004:
                                upFlag = upFlag + 1
                            # else:
                            #     upFlag = 0
                            # if cashflowPart / minuteCashFlow.loc[cashFlowTime,orderType] > 0.05:  # set parameter here
                            #     cashFlowFlag = 1
                            # else:
                            #     cashFlowFlag = 0
                        else: #means direction is 'sell'
                            minPrice = self.allQuoteData[symbol].loc[cashFlowTime:, 'midp'].iloc[
                                       :301].min()  # 5分钟内的最大跌幅
                            if np.log(minPrice / self.allQuoteData[symbol].loc[cashFlowTime, 'midp']) <= -0.004:
                                upFlag = upFlag + 1
                            # else:
                            #     upFlag = 0

                    symbolFilterList.append(pd.DataFrame({'upFlag':upFlag,'cashFlowFlag':cashFlowFlag},index=[symbol]))

            self.symbolFilter[sector] = pd.concat(symbolFilterList, 0)
        else:
            symbols = self.sectorData.loc[sector, 'secucode']
            self.symbolFilter[sector] = pd.DataFrame({'cashFlowFlag':0,'upFalg':0},index=symbols)

                    #  根据个股的现金流以及个股之后的涨势，推出我们需要的stock lists
                    #  需要的stock list：
                    #  1. 对现金流贡献大的，且股票有上涨趋势
                    #  2. 对现金流无贡献的，股票有上涨趋势
                    #  3. 对现金流有贡献的，股票已经涨到顶
                    #  想法：结合1跟3的现金流做判断，买入1跟2的股票list。
                    #  样本内挑选：现金流为正且达到一定程度的股票，标记为1；在之后5分钟内涨幅达到1%的股票，标记为2。
                    #  统计样本内每只股票出现1和2的次数



        # if output.sum() != 0:
        #     # TODO: 加入累计资金流的曲线
        #     symbols = self.sectorData.loc[sector, 'secucode']
        #     for symbol in symbols:
        #         # fig,ax = plt.figure(figsize=(20,12))
        #         stockData = self.allTradeData.get(symbol)
        #         if stockData is not None:
        #             fig,ax = plt.subplots(3, figsize=(20,12),sharex=True)
        #             stockCashFlow = self.SkipNoonBreak(stockData.resample('T', label='right').sum())
        #             obi = np.log(self.allQuoteData[symbol].loc[:, 'bidVolume1']) - np.log(
        #                 self.allQuoteData[symbol].loc[:, 'askVolume1'])
        #             ax[0].bar(list(range(240)),
        #                     minuteCashFlow.loc[:, orderType].dropna().values.tolist(), 0.4,
        #                     color='green')
        #             ax2 = ax[0].twinx()
        #             ax2.plot(list(range(240)),self.allQuoteData[symbol].loc[minuteCashFlow.index,'midp'].values,linewidth = 2.0)
        #             ax[0].legend(['sector_cashflow', symbol + '_midprice'])
        #             ax2.legend([symbol + '_midprice'])
        #             ax[1].bar(list(range(240)),
        #                       stockCashFlow.loc[:, orderType].dropna().values.tolist(), 0.4,
        #                     color='red')
        #             ax[1].plot(list(range(240)),stockCashFlow.loc[:, orderType].cumsum().dropna().values.tolist(),linewidth = 2.0,color = 'y')
        #             ax[1].legend([symbol + '_main_cashflow', symbol + '_main_cum_cashflow'])
        #             ax[2].plot(list(range(240)),obi.loc[minuteCashFlow.index].values.tolist(),linewidth = 2.0,color = 'green')
        #             ax[2].legend([symbol + '_obi'])
        #             fig.canvas.draw()
        #             # set x axis name
        #             labels = [item.get_text() for item in ax[1].get_xticklabels()]
        #             # print(labels)
        #             if len(labels) != 0:
        #                 labels[1:-1] = minuteCashFlow.index[list(map(int, labels[1:-2]))]
        #                 ax[2].set_xticklabels(labels, rotation=45)
        #             # plt.xticks(xvalue,labels,rotation = 'vertical')
        #
        #             # add title
        #
        #             # plt.bar(list(range(14402)),
        #             #         sectorCashFlow.loc[:, orderType].dropna().values.tolist(), 0.4,
        #             #         color='green')
        #             dateStr = str(self.tradeDate.date()).replace('-','')
        #             outputFolder = './cashflowplot_insample_wind/' + dateStr
        #             if os.path.exists(outputFolder) is False:
        #                 os.makedirs(outputFolder)
        #             plt.savefig(outputFolder + '/' + symbol + '_' + orderType + '.jpg')
        #             plt.close('all')

        "获得一个多月的数据，并将多天的plot合起来，对信号点做标记"
        "生成csv，格式为datafram，行为分钟数，列为板块经资金流、个股经资金流、个股的midp、个股的现金流"
        # if lbWindow == 2:
        symbols = self.sectorData.loc[sector, 'secucode']
        symbolDfList = list()
        for symbol in symbols:
            # fig,ax = plt.figure(figsize=(20,12))
            stockData = self.allTradeData.get(symbol)
            if stockData is not None:
                stockAllCashFlow = self.SkipNoonBreak(stockData.resample('T', label='right').sum())
                stockCashFlow = stockAllCashFlow.loc[:,orderType]
                midp = self.allQuoteData[symbol].loc[minuteCashFlow.index, 'midp']
                allCashFlow = stockAllCashFlow.loc[:,'netCashFlow']
                data2save = pd.DataFrame({symbol + '_cashflow': stockCashFlow.values.tolist(),symbol + '_midp':midp.values.tolist(),symbol + '_allcashflow':allCashFlow.values.tolist()},index = minuteCashFlow.index)
                symbolDfList.append(data2save)
        symbolDf = pd.concat(symbolDfList,1)
        # symbolDf.loc[:, 'sector_cashflow'] = minuteCashFlow.loc[:,orderType].values.tolist()
        # symbolDf.loc[:, 'signal'] = output.loc[minuteCashFlow.index].values.tolist()
        symbolDf.loc[:, sector + '_sector_cashflow'] = minuteCashFlow.loc[:,orderType].values.tolist()
        symbolDf.loc[:, sector + '_signal'] = output.loc[minuteCashFlow.index].values.tolist()

        # symbolDf.to_csv('./cashflowplot_insample_wind/' + str(self.tradeDate.date()).replace('-','') + '_threshold_' + str(lbWindow)+ '_cashflow.csv')

        self.data2save.append(symbolDf)


        # method 5
        # 针对现金流做ema，通过对比当前的值和ema的值来判断信号的生成


        return output

    def SkipNoonBreak(self, data2revise):
        """
        用于针对resample的数据过滤掉中午停盘时间
        :param data2revise:  dataframe，修改的数据
        :return: dataframe，没有中午停盘时间的数据
        """
        date = str(self.tradeDate.date()).replace('-','')
        morningStartTime    = datetime.datetime.strptime(str(date + ' 09:30:00'), '%Y%m%d %H:%M:%S')
        morningEndTime      = datetime.datetime.strptime(str(date + ' 11:30:00'), '%Y%m%d %H:%M:%S')
        noonStartTime       = datetime.datetime.strptime(str(date + ' 13:00:01'),'%Y%m%d %H:%M:%S')
        noonEndTime         = datetime.datetime.strptime(str(date + ' 15:00:00'), '%Y%m%d %H:%M:%S')
        return pd.concat([data2revise.loc[morningStartTime:morningEndTime,:],data2revise.loc[noonStartTime:noonEndTime,:]])

    def CalSectorNetWorth(self, sector, normalize = False,revise = False):
        """

        :param sector: the sector we want to calculate the net cash flow in or out
        :param lbWindow: used to calculate the past cumulative net worth and liquid speed of the cash flow
        :return: net cash flow every second with sector net cash flow as column and timestamp as index
        """
        """
        考虑以下大单流入流出？：
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
        if revise is False:
            symbols = self.sectorData.loc[sector, 'secucode']
        else:
            symbols = self.sectorRevisedData[sector]
        # symbols = self.sectorAllData.loc[sector, 'secucode']
        # if type(symbols) == type(''):  # which means that the symbols contains only one stcok
        if isinstance(symbols, str):  # use is instance is better than the == to compare the types
            symbols = [symbols]
        orderTypes = ['netCashFlow', 'netSuperOrderCashFlow', 'netBigOrderCashFlow', 'netMiddleOrderCashFlow',
                      'netSmallOrderCashFlow', 'netMainOrderCashFlow']
        dfList = list(map(lambda orderType: self.CalSingleOrderTypeCashFlow(symbols, orderType, normalize = normalize),
                          orderTypes))  # calculate the sum and cumsum in one row here.
        cashFlowDf = pd.concat(dfList, 1)
        self.sectorCashFlow[sector] = cashFlowDf  # add the cashflow data frame to the dict
        return cashFlowDf

    def CalSingleOrderTypeCashFlow(self, symbols, orderType, normalize = False):
        """
        This function will calculate the different types of order cash flow for sector: like
        cumulative cash flow in, out, and net cash flow.
        netCashFlow/cumCashFlow
        (in - out)/ (in + out)

        :return: cash flow data frame with time as index and multi cash flow indexes as columns.
        """
        if orderType == 'netCashFlow':
            buyColumn = 'buyTurnover'
            sellColumn = 'sellTurnover'
        else:
            buyColumn = 'buy' + orderType[3].lower() + orderType[4:].replace('CashFlow', '') + 'Turn'
            sellColumn = 'sell' + orderType[3].lower() + orderType[4:].replace('CashFlow', '') + 'Turn'

        columnsToUse = [buyColumn, sellColumn, orderType]
        # dfList = list()
        # for symbol in symbols:
        #     tradeData = self.allTradeData.get(symbol)
        #     if tradeData is not None:
        #         dfList.append(pd.DataFrame(list(self.allTradeData[symbol].loc[:, column]),columns=[symbol],index=self.allTradeData[symbol].index))
        if normalize == True:
            # dfList = pd.concat(list(map(lambda column: pd.concat(
            #                             list(map(lambda symbol:pd.DataFrame(list(self.allTradeData[symbol].loc[:, column]/self.dailyData.loc[symbol,'mkt_freeshares']),columns=[symbol],index=self.allTradeData[symbol].index) if self.allTradeData.get(symbol) is not None else None
            #                                      ,symbols)), 1).sum(1), columnsToUse)), 1)
            dfList = pd.concat(list(map(lambda column: pd.concat(
                                        list(map(lambda symbol:pd.DataFrame(list(self.allTradeData[symbol].loc[:, column]/self.dailyData.loc[symbol,'mkt_freeshares']),columns=[symbol],index=self.allTradeData[symbol].index) if self.allTradeData.get(symbol) is not None else None
                                                 ,symbols)), 1).sum(1), columnsToUse)), 1)
        else:
            dfList = pd.concat(list(map(lambda column: pd.concat(
                                        list(map(lambda symbol:pd.DataFrame(list(self.allTradeData[symbol].loc[:, column]),columns=[symbol],index=self.allTradeData[symbol].index) if self.allTradeData.get(symbol) is not None else None
                                                 ,symbols)), 1).sum(1), columnsToUse)), 1)
        dfList.columns = columnsToUse
        dfCumsum = dfList.cumsum()
        dfCumsum.columns = list(map(lambda column: 'cum' + column, columnsToUse))
        dfToReturn = pd.concat([dfList, dfCumsum], 1)
        dfToReturn.loc[:, orderType + '_netRatio'] = dfToReturn.loc[:, orderType] / (
                dfToReturn.loc[:, buyColumn] + dfToReturn.loc[:, sellColumn])
        dfToReturn.loc[:, orderType + '_ratio'] = dfToReturn.loc[:, orderType] / dfToReturn.loc[:, 'cum' + orderType]
        # Add the indexes here if there are some extensions of the sector cash flow.
        return dfToReturn

    def GetSectorCashFlow(self, sector,normalize = False ,revise = False):
        netCashFlow = self.sectorCashFlow.get(sector, None)
        if netCashFlow is None:
            netCashFlow = self.CalSectorNetWorth(sector,normalize = normalize,revise = revise)
        return netCashFlow

    def CalAllSectorCashFlow(self):
        sectors = np.unique(self.sectorData.index)
        for sector in sectors:
            self.CalSectorNetWorth(sector)

        return None

    def CompareSectorAndStock(self, symbol, orderType='netCashFlow'):
        """
        This function aims to plot the figure that contains three plots:
        1. sector cum cash flow changes
        2. sector net cash flow changes
        3. stock price change
        4. sector price change?
        :param symbol: stock symbol we want to analysis
        :return: plot contains three plots
        """
        # orderType = 'netCashFlow'
        sector = self.sectorAllData.loc[self.sectorAllData.loc[:, 'secucode'] == symbol,  # self.sectorData
                 :].index  # here ,sector is a index series
        sectorCashFlow = self.GetSectorCashFlow(sector[0])
        sectorNetCashFlow = sectorCashFlow.loc[:, orderType].diff(1)
        sectorNetRatio = sectorCashFlow.loc[:, orderType + '_ratio']
        stockPrice = self.allTradeData[symbol].loc[:, 'vwap']
        stockCashFlow = self.allTradeData[symbol].loc[:, orderType]
        stockCumCashFlow = stockCashFlow.cumsum()
        # plot with various axes scales. TODO: change the x axis in order to skip the noon break
        f, axarr = plt.subplots(3,  figsize=(20,12),sharex=True)

        # stock price change
        axarr[0].plot(list(stockPrice.iloc[:]),c = 'r', label = 'stock vwap')
        # plt.yscale('linear')
        axarr[0].set_title('stock ' + symbol + ' vwap')
        axarr[0].grid(True)

        # stock cumulative cash flow changes
        axarr[1].plot(list(stockCumCashFlow.iloc[:]),c = 'g')
        # plt.yscale('log')
        axarr[1].set_title('stock ' + symbol + ' cumulative cash flow changes with order type : ' + orderType)
        axarr[1].grid(True)

        # sector cumulative cash flow
        axarr[2].plot(list(sectorCashFlow.loc[:, 'cum' + orderType]),c = 'b')
        # plt.yscale('symlog', linthreshy=0.01)
        axarr[2].grid()
        axarr[2].set_ylabel('Cumulative cash flow (yuan)')
        ax2 = axarr[2].twinx()
        ax2.plot(list(sectorNetRatio.iloc[:]), label='net_ratio', color='y')
        ax2.set_ylabel('net ratio: (in - out)/(in + out)')
        # Need to set lim here?
        # axarr[2].legend(['cumNetCashFlow', '(in - out)/(in + out)'])
        f.canvas.draw()

        # set x axis name
        labels = [item.get_text() for item in axarr[2].get_xticklabels()]
        # print(labels)
        if len(labels) != 0:
            labels[1:-1] = stockPrice.index[list(map(int, labels[1:-1]))]
            axarr[2].set_xticklabels(labels, rotation=45)
        # plt.xticks(xvalue,labels,rotation = 'vertical')

        # add title

        # ax.set_title('Base diff line of two contracts and ' + add + ' line' + ' in last ' + str(days) + ' days')


        axarr[2].set_title('sector : ' + sector[0].upper() + ' cumulative cash flow of order type : ' + orderType)
        axarr[2].grid(True)

        # adjust hte hspace? test here.
        f.subplots_adjust(hspace=0.3)

        tradeDate = str(self.tradeDate.date()).replace('-','')
        picSavePath = './cashflowplot/' + tradeDate
        if os.path.exists(picSavePath) is False:
            os.makedirs(picSavePath)
        plt.savefig(picSavePath + '/' + symbol + '_' + sector[0] + '.jpg')

        plt.close('all')

        return f

    def DownLevelAndUpLevelConfig(self, nTime, alpha, nChange, etfStmbol='510050.SH', type='figLevel'):
        """
        This function is used to test the idea of down level and up level to see whether the base is down to the another
        level
        加入三个参数的测试：
        nTime：过去多少时间的偏移去确定level变化（n seconds，秒）
        alpha：计算ewm的参数
        nChange：level变化的阈值。思考：nchange是否应该调整为动态
        :param nTime:
        :param alpha:
        :param nChange: (n percent)
        :param type: level calculate type. fixlevel or  or dynamic adjust by mean and sd or dynamic adjust by nChange.
        :return: signal return by up or down level test.
        """
        baseDiff = pd.DataFrame(list(self.allTradeData[etfStmbol].loc[:, 'latest'] - self.futureData.loc[:, 'midp']),
                                columns=['diff'], index=self.allTradeData[etfStmbol].index)
        TRANSACTION_COST = 0.002
        # rollingMean = baseDiff.rolling(window = 60).mean()  # get the rolling mean to help judge the jump level.
        # ewm         = baseDiff.ewm(alpha=alpha)
        # ema         = ewm.mean()
        # emStd       = ewm.std()
        if type == 'dynamic_PctChng':
            lagPct = baseDiff.diff(nTime) / baseDiff  # here is not log return, mathetmatical return instead
            # up level 和 down level的确定：基差是慢慢拉大的。突然拉大的情况很少。假如慢慢拉大，通过什么方法去实现？
            # levelFilg   = 1
            baseDiff.loc[:, 'levelStatusRef'] = 0
            baseDiff.loc[:, 'levelStatus'] = 0
            levelUpSituation = lagPct.loc[:, 'diff'] > nChange
            # levelUpRefSituation = lagPct.loc[:, 'diff'] > (nChange/2)
            #  重复level计算的考虑：当进入到level的变化之后，之后一段时间之内类似的涨幅不应再计算为level的变化。
            #  因为同一段时间之内的涨跌幅类似的话，则视为同一个level，不再记录为一个level。目前使用方法：对nTime里面重复出现的level up 或者 down先暂时不考虑累加level
            levelDownSituation = lagPct.loc[:, 'diff'] < -nChange
            # levelDownRefSituation = lagPct.loc[:, 'diff'] < -(nChange/2)
            baseDiff.loc[levelUpSituation, 'levelStatusRef'] = 1
            baseDiff.loc[(baseDiff.loc[:, 'levelStatusRef'].diff(1) == 1) & (
                    baseDiff.loc[:, 'levelStatusRef'].rolling(nTime).sum() == 1), 'levelStatus'] = 1
            baseDiff.loc[levelDownSituation, 'levelStatusRef'] = -1
            baseDiff.loc[(baseDiff.loc[:, 'levelStatusRef'].diff(1) <= -1) & (
                    baseDiff.loc[:, 'levelStatusRef'].rolling(nTime).sum() == -1), 'levelStatus'] = -1
            baseDiff.loc[:, 'levelStatus'] = baseDiff.loc[:,
                                             'levelStatus'].cumsum()  # here, cumsum can get the level change today. The expected change is 1(n1) 0(n2) -1(n3) 0(n4) 1(n5)
            baseDiff.loc[:, 'periodMean'] = list(
                baseDiff.groupby('levelStatus').apply(lambda x: x.loc[:, 'diff'].ewm(alpha=alpha).mean()).T.sort_index(
                    level=1).reset_index().loc[:,
                'diff'])  # get the rolling mean by the level status, rolling mean or ewm?

            # TODO: consider the alpha change when calculating the different levels of data. And Also: use the different level data to calculate the ema and emstd with different alpha
            baseDiff.loc[:, 'periodStd'] = list(
                baseDiff.groupby('levelStatus').apply(lambda x: x.loc[:, 'diff'].ewm(alpha=alpha).std()).T.sort_index(
                    level=1).reset_index().loc[:, 'diff'])  # get the rolling std by the level status
            baseDiff.loc[:, 'levelBuySignal'] = baseDiff.loc[:, 'diff'] < (
                    baseDiff.loc[:, 'periodMean'] - baseDiff.loc[:,
                                                    'periodStd'])  # get the trading signal by the period change.
            baseDiff.loc[:, 'levelSellSignal'] = baseDiff.loc[:, 'diff'] > (
                    baseDiff.loc[:, 'periodMean'] + baseDiff.loc[:,
                                                    'periodStd'])  # get the trading signal by the period change.
            baseDiff.loc[:, 'levelCloseSignal'] = baseDiff.loc[:, 'diff'] < (
                    baseDiff.loc[:, 'periodMean'] - baseDiff.loc[:,
                                                    'periodStd'])  # get the trading signal by the period change.



        elif type == 'dynamic_sd':
            print('calculate level by dynamic sd.')
            # TODO: 1. Calculate long term mean and short term mean by ewm function
            #       2. Adjust the long term mean and short term mean by noticing the spread.
            #       3. Buy when the diff lower than the std and sell when it reaches the current mean.
            longTermAlpha = 0.999
            shortTermAlpha = 0.99
            baseDiff.loc[:, 'ewmstd'] = baseDiff.ewm(
                alpha=1 - longTermAlpha).std()  # the alpha in ewm means the weight of the new item
            baseDiff.loc[:, 'longTermMean'] = baseDiff.ewm(
                alpha=1 - longTermAlpha).mean()  # the alpha in ewm means the weight of the new item
            baseDiff.loc[:, 'shortTermMean'] = baseDiff.ewm(
                alpha=1 - shortTermAlpha).mean()  # the alpha in ewm means the weight of the new item
            # longTermMeanList = list()
            # shortTermMeanList = list()
            # lastMean = baseDiff.loc[:, 'diff'][0]  # suppose it is not nan here. otherwise we should drop the na before calculating.
            lastMeanList = list()
            # sd = 1.5  # which is the standard deviation line. dynamic adjust by past ewmstd
            levelAdjustFlagList = list()
            openFlagList = list()
            closeFlagList = list()
            position = 0
            positionList = list()
            levelFlag = 0
            # longTermMean = baseDiff.loc[:, 'diff'][0]
            # shortTermMean = baseDiff.loc[:, 'diff'][0]
            for item in zip(baseDiff.loc[:, 'diff'], baseDiff.loc[:, 'ewmstd'], baseDiff.loc[:, 'longTermMean'],
                            baseDiff.loc[:, 'shortTermMean']):
                price = item[0]
                sd = item[1]
                longTermMean = item[2]
                shortTermMean = item[3]
                openFlag = 0
                closeFlag = 0
                # longTermMean    = longTermAlpha * longTermMean + price * (1 - longTermAlpha)  # not last mean here.!
                # shortTermMean   = shortTermAlpha * shortTermMean + price * (1 - shortTermAlpha)
                if abs(longTermMean - shortTermMean) <= sd:
                    lastMean = longTermMean
                    if levelFlag == 0:
                        levelAdjustFlagList.append(0)
                    else:
                        # lastMean = shortTermMean
                        if levelFlag > 0:
                            levelAdjustFlagList.append(-1)
                            levelFlag = 0
                        else:
                            levelAdjustFlagList.append(1)
                            levelFlag = 0
                else:
                    lastMean = shortTermMean
                    if levelFlag == 0:
                        if longTermMean > shortTermMean:
                            levelAdjustFlagList.append(-1)
                            levelFlag = -1
                        else:
                            levelAdjustFlagList.append(1)
                            levelFlag = 1
                    else:
                        levelAdjustFlagList.append(0)
                # longTermMeanList.append(longTermMean)
                # shortTermMeanList.append(shortTermMean)
                lastMeanList.append(lastMean)

                # check the close condition
                if (position >= 1) and (price > lastMean):
                    position = position - 1
                    closeFlag = -1
                elif (position <= -1) and (price < lastMean):
                    position = position + 1
                    closeFlag = 1

                # check the open condition
                if (price < (lastMean - sd)) and (position == 0):
                    position = position + 1
                    openFlag = 1
                elif (price > (lastMean + sd)) and (position == 0):
                    position = position - 1
                    openFlag = -1

                positionList.append(position)
                openFlagList.append(openFlag)
                closeFlagList.append(closeFlag)

            # baseDiff.loc[:, 'longTermMean']     = longTermMeanList
            # baseDiff.loc[:, 'shortTermMean']    = shortTermMeanList
            baseDiff.loc[:, 'periodMean'] = lastMeanList
            baseDiff.loc[:, 'periodStd'] = baseDiff.loc[:, 'ewmstd']
            baseDiff.loc[:, 'levelFlag'] = levelAdjustFlagList
            # baseDiff.loc[baseDiff.loc[:,'levelFlag'].diff(1) == 0, 'levelFlag']        = 0
            baseDiff.loc[:, 'levelStatus'] = baseDiff.loc[:, 'levelFlag'].cumsum()
            baseDiff.loc[:, 'position'] = positionList
            baseDiff.loc[:, 'openFlag'] = openFlagList
            baseDiff.loc[:, 'closeFlag'] = closeFlagList
            baseDiff.loc[:, 'cost'] = 0
            baseDiff.loc[baseDiff.loc[:, 'position'].diff(1) != 0, 'cost'] = -TRANSACTION_COST
            nTimes = np.sum(abs(np.array(levelAdjustFlagList)))
            openTimes = np.sum(abs(baseDiff.loc[:, 'openFlag']))
            closeTimes = np.sum(abs(baseDiff.loc[:, 'closeFlag']))
            # pnl         = np.prod(baseDiff.loc[:, 'diff'].diff(1), baseDiff.loc[:, 'position'])  # sth wrong
            pnl = np.multiply(np.array(-baseDiff.loc[:, 'diff'].diff(-1)), np.array(baseDiff.loc[:, 'position']))
            totalPnL = np.nansum(pnl - abs(np.array(baseDiff.loc[:,
                                                    'cost'])))  # use the nansum instead of sum due to that there is nan existing in the series.

            self.PlotLevelChange(baseDiff, addTradeSignal=True)

            return pd.DataFrame({'nTimes': nTimes, 'pnl': totalPnL, 'tradeTimes': openTimes + closeTimes})

        elif type == 'fixLevel':
            print('Load the fix bin data and use them to down or up level.')

            levels = 3  # should be divided by int with row number of bin data
            pctg = [0, 0.25, 0.5, 0.75]
            binData = pd.read_csv('./index_data/etf_future/' + self.tradeDate.replace('-', '') + '/freq_5.csv')
            levelSplit = binData.shape[0] / levels
            if type(levelSplit) is not type(1):
                raise ('Uncorrect levels')
            binData.loc[:, 'levels'] = list(itertools.chain.from_iterable(
                list(map(lambda x: [x] * int(levelSplit), np.arange(int(-levels / 2), int(levels / 2) + 1)))))
            # bin data is the data frame with history diff distribution
            bins = binData.shape[0]
            middleBin = int(bins / 2)
            middlePrice = binData.iloc[middleBin, 1]
            levelPrices = binData.iloc[middleBin, 1]
            levelPosition = pd.DataFrame()
            # levelPosition.loc[:, 'pctg'] = pctg
            # levelPosition.loc[:, 'levels'] = np.arange(levels)
            levelPosition, index = np.unique(binData.loc[:, 'levels'])
            levelPosition.loc[:, 'position'] = 0

            # suppose FindPriceLevel could find the price locate in which place.
            position = 0
            positionList = list()
            openFlagList = list()
            closeFlagList = list()
            levelStatusList = list()
            periodMean = list()
            # 计算return时，应该保留该level仓位，等回到该level时再平。
            for item in zip(baseDiff.loc[:, 'diff']):
                price = item[0]
                # first step, find the price in which level now.
                # second step, check whether the current level has the position now, if has, check the close condition, if no, turn to third step..
                # third step, check whether the price reaches the buy or sell condition under the level situation.
                currentLevel = self.FindPriceLevelByBin(binData, price)
                curPosition = levelPosition.loc[currentLevel, 'position']
                subBin = binData.loc[binData.loc['levels'] == currentLevel, 'bin']
                meanPrice = np.mean(subBin)
                upperMean = np.mean(subBin.iloc[:int(len(subBin) / 2)])
                lowerMean = np.mean(subBin.iloc[int(len(subBin) / 2):])
                openFlag = 0
                closeFlag = 0
                if curPosition == 0:
                    if price > upperMean:
                        openFlag = -1
                        curPosition = -1
                    elif price < lowerMean:
                        openFlag = 1
                        curPosition = 1
                    else:
                        curPosition = 0
                else:
                    if curPosition > 0:
                        if price >= meanPrice:
                            closeFlag = 1
                            curPosition = 0
                    elif curPosition < 0:
                        if price <= meanPrice:
                            closeFlag = -1
                            curPosition = 0
                position = position + curPosition
                levelPosition.loc[currentLevel, 'position'] = position
                positionList.append(position)
                openFlagList.append(openFlag)
                closeFlagList.append(closeFlag)
                levelStatusList.append(currentLevel)
                periodMean.append(meanPrice)

                print('Loop by price to find out the level change')

            baseDiff.loc[:, 'position'] = positionList
            baseDiff.loc[:, 'openFlag'] = openFlagList
            baseDiff.loc[:, 'closeFlag'] = closeFlagList
            baseDiff.loc[:, 'levelStatus'] = levelStatusList
            baseDiff.loc[:, 'periodMean'] = periodMean
            baseDiff.loc[:, 'cost'] = 0
            baseDiff.loc[baseDiff.loc[:, 'position'].diff(1) != 0, 'cost'] = -TRANSACTION_COST
            openTimes = np.sum(abs(baseDiff.loc[:, 'openFlag']))
            closeTimes = np.sum(abs(baseDiff.loc[:, 'closeFlag']))
            pnl = np.multiply(np.array(-baseDiff.loc[:, 'diff'].diff(-1)), np.array(baseDiff.loc[:, 'position']))
            totalPnL = np.sum(pnl - np.array(baseDiff.loc[:, 'cost'])[1:])

            self.PlotLevelChange(baseDiff, addTradeSignal=True)

            return pd.DataFrame({'nTimes': np.sum(baseDiff.loc[:, 'levelStatus'].diff(1) != 0), 'pnl': totalPnL,
                                 'tradeTimes': openTimes + closeTimes})
        else:
            print('calculate level by fix level by loading the previous data.')
        # TODO: evaluate the effect of different combination of parameters in order to calibrate the parameters we need.
        """
        方法1. 直接测试交易信号，通过信号的returns来衡量。信号：上下穿过1.5倍标准差，回到均值就卖。以及level down和level up的次数，太多也不合适。
        方法2. 衡量这段时间的偏离程度？：标准差
        方法3. 对mean和std的区分是否有效，即区分度，如何衡量？
        """

        return 0

    def PlotLevelChange(self, levelDf, addTradeSignal=False):
        """
        This function is used to plot the level change situation, with the following line:
        price (or base diff) change line with time
        rolling mean( or ewm) change line with time
        level change with line, use the hline to make the virtual diff.
        :type addTradeSignal: True or False
        :param levelDf: column: price levelstatus mean std periodMean periodStd. rows: time ( second interval best)
        :return: plot
        """
        fig, ax = plt.subplots(1, figsize=(20, 12))  # set figsize here in case of the small plot

        xvalue = np.arange(len(levelDf))
        diffValue = list(levelDf.loc[:, 'diff'])
        meanValue = list(levelDf.loc[:, 'periodMean'])
        stdValue = np.array(levelDf.loc[:, 'periodStd'])

        ax.plot(diffValue)
        ax.set_xlim([0, len(diffValue)])
        ax.plot(meanValue, color='r')
        ax.plot(meanValue + stdValue, color='y')
        ax.plot(meanValue - stdValue, color='g')

        fig.canvas.draw()

        changeDf = levelDf.loc[:, 'levelStatus'].diff()
        levelChangePos = xvalue[changeDf != 0]

        for xpos in levelChangePos:
            plt.axvline(x=xpos, color='k')

        ax.legend(['baseDiff', 'meanLine', 'mean + std', 'mean-std'])

        if addTradeSignal:
            openFlag = np.array(levelDf.loc[:, 'openFlag'])
            # openPosition = np.prod(openFlag,diffValue)
            openPosition = np.multiply(openFlag, np.array(diffValue))
            buyPosition = openPosition.copy()
            buyPosition[buyPosition <= 0] = np.nan
            sellPosition = -openPosition.copy()
            sellPosition[sellPosition <= 0] = np.nan
            closeFlag = np.array(levelDf.loc[:, 'closeFlag'])
            closePosition = np.multiply(abs(closeFlag), np.array(diffValue))
            closePosition[closePosition == 0] = np.nan
            uparrow = u'$\u2191$'  # the shape of up arrow
            downarrow = u'$\u2193$'  # the shape of down arrow
            ax.scatter(y=buyPosition, x=xvalue, marker=uparrow,
                       c='r', label='open long', s=90)
            ax.scatter(y=sellPosition, x=xvalue, marker=downarrow,
                       c='g', label='open short', s=90)
            ax.scatter(y=closePosition, x=xvalue, marker="*",
                       c='y', label='close position', s=90)
            ax.legend(['open long', 'open short', 'close position'])

        labels = [item.get_text() for item in ax.get_xticklabels()]
        # print(labels)
        if len(labels) != 0:
            labels[1:-1] = levelDf.index[list(map(int, labels[1:-1]))]
            ax.set_xticklabels(labels, rotation=45)

        ax.set_title('Base diff with different levels and period mean and std.')

        return fig

    def FindPriceLevelByBin(self, binData, price):
        """
        Find out the price in which level
        :param binData:
        :param price:
        :return:
        """
        # should be the first pos value in the diff
        minLocation = (price - binData.loc[:, 'bin']).argmin()
        if price - binData.iloc[minLocation:, 0] > 0:
            minLocation = minLocation - 1
        else:
            minLocation = minLocation

        return binData.loc[minLocation, 'levels']

    def FindSimilarCashflow(self,symbol = '', sector = '', correlationThreshold = 0.35):
        """
        本函数用于寻找板块内在一段时间之内的累积现金流相似的股票（主力资金流流入形状、相关性高？通过相关性高和低的图来做点区分）
        方法：（是否考虑用聚类分析？分为两类，找出每天都在同一个类里的股票pair？）
        1. 得到每只股票的主力净资金流（秒级，可以合成分钟级去判断，相关性可能会提升）
        2. 对资金流的data frame求相关性
        3. 超过threshold的抽出来，并对比图片是否为所需
        :param symbol: str, 股票代码，找出对应板块
        :param sector: str, 板块代码，用于过滤的板块
        :return: list, dataframe，list为相似度高的股票列表，dataframe为correlation的情况
        """
        if symbol is '' and sector is '':
            raise ('Error symbol and sector')
        elif sector is '':
            sector = self.sectorData.loc[self.sectorData.loc[:, 'secucode'] == symbol].index[0]
        symbols = self.sectorData.loc[sector,'secucode']  # 获取当前symbols
        orderType = 'netCashFlow'
        # cashFlow = pd.concat(list(map()))
        cashFlowList = list()
        columnName = list()
        for symbol in symbols:
            tradeData = self.allTradeData.get(symbol,None)
            if tradeData is None:
                continue
            else:
                cashFlowList.append(tradeData.loc[:,orderType])
                columnName.append(symbol)
        cashFlowDf = pd.concat(cashFlowList, 1)
        cashFlowDf.columns = columnName
        cashFlowDf = cashFlowDf.cumsum()
        cashFlowDf = cashFlowDf.resample('T',label='right').last()  # 得到每分钟的cumsum，避免秒级的数据导致相关性都偏低，T才是1分钟，m为月份
        corrDf = cashFlowDf.corr()
        ## 得到相关性矩阵之后，如何将互相相关性都较高的股票提取出来？
        # 相关性高的归为一类

        # for symbol in symbols:
        #     subCorrDf = corrDf.loc[symbol,:]
        #     subCorrDf = subCorrDf[subCorrDf>correlationThreshold]
        #     if subCorrDf.shape[0] == 0:
        #         continue
        #     else:
        #         largeCorSymbols = list(subCorrDf.index)
        #         for subSymbol in largeCorSymbols:
        #             subCorrDf = corrDf.loc[subSymbol,:]

        largeCorDf = list()
        for symbol1 in symbols:

            for symbol2 in symbols:

                if symbol1 != symbol2:
                    cor = corrDf.loc[symbol1,symbol2]
                    if cor >= correlationThreshold:
                        largeCorDf.append(pd.DataFrame({'symbol1':symbol1,'symbol2':symbol2},index = [0]))

        if len(largeCorDf) == 0:
            return None
        else:
            largeCorDf = pd.concat(largeCorDf,0)
            # selectedSymbols = np.union1d(largeCorDf)  # ?.problem here.
            # 根据相关性高的做分类，取个数最多的一类
            countsDf1 = largeCorDf.groupby('symbol1').count()
            countsDf2 = largeCorDf.groupby('symbol2').count()
            countsDf1.columns = ['symbol']
            countsDf2.columns = ['symbol']
            countsDf = countsDf1 + countsDf2
            targetSymbol = countsDf.sort_values('symbol',ascending=False).index[0]
            selectedSymbols = largeCorDf.loc[largeCorDf.iloc[:,0] == targetSymbol, 'symbol2'].values.tolist()
            selectedSymbols.append(targetSymbol)

            return selectedSymbols

        #  选择出所有相关性高的pairs之后，取出有交集的list，整理得到最终list

        # if 用聚类，方法为sklearn.cluster中引入KMeans方法，针对现金流进行kmeans算法分类，分为两类？如何判断分出来的类是我们需求的类
        # 或者分成n类，return数量最多的一类

        return 0

    def PlotSectorData(self,sector,signal,window,savePath = './'):
        sectorData = self.sectorData.loc[sector, :]
        symbols = sectorData.loc[:,'secucode']
        colors = ['green', 'blue', 'grown', 'yellow', 'red','k']
        fig, ax = plt.subplots(1, figsize=(20, 12), sharex=True)
        legendList = list()
        for symbol in symbols:
            subQuoteData = self.allQuoteData.get(symbol)
            if subQuoteData is None:
                continue
            else:
                # bidPrice1 = subQuoteData.loc[:, 'bidPrice1']
                # askPrice1 = subQuoteData.loc[:, 'askPrice1']
                midp = subQuoteData.loc[:,'midp']
                yvalue = list(midp.iloc[:])
                ax.plot(yvalue/ yvalue[0], label=symbol + '_midquote')
                legendList.append(symbol + '_midquote_')
                uparrow = u'$\u2191$'  # the shape of up arrow
                downarrow = u'$\u2193$'  # the shape of down arrow
                arrowSize = 180
                # if leadFlag == 0:
                #     longSignal = subQuoteData.loc[:, 'longShort'] * subQuoteData.loc[:, 'askPrice1']/ yvalue[3] # starts from 3
                #     longCloseFlag = subQuoteData.loc[:, 'closeFlag'] * subQuoteData.loc[:, 'bidPrice1']/ yvalue[3]
                #     shortSignal = subQuoteData.loc[:, 'longShort'] * subQuoteData.loc[:, 'bidPrice1']/ yvalue[3]
                #     shortCloseFlag = subQuoteData.loc[:, 'closeFlag'] * subQuoteData.loc[:, 'askPrice1']/ yvalue[3]
                #
                #     longSignal[longSignal <= 0] = np.nan
                #     longCloseFlag[longCloseFlag >= 0] = np.nan
                #     shortSignal[shortSignal >= 0] = np.nan
                #     shortCloseFlag[shortCloseFlag <= 0] = np.nan
                #
                #     # ax.plot(yvalue, label = symbol + '_midquote_leadFlag_' + str(leadFlag))
                #     ax.scatter(y = list(longSignal.iloc[:]), x = range(len(yvalue)),marker = uparrow,s = arrowSize,c = 'red')
                #     ax.scatter(y = list(abs(longCloseFlag).iloc[:]), x = range(len(yvalue)),marker = 'v',s = arrowSize, c = 'green')
                #     ax.scatter(y = list(abs(shortSignal).iloc[:]), x = range(len(yvalue)),marker = downarrow,s = arrowSize,c= 'green')
                #     ax.scatter(y = list(shortCloseFlag.iloc[:]), x = range(len(yvalue)),marker = '^',s = arrowSize,c='red')

                # else:
                signals = subQuoteData.loc[:, signal + '_' + str(window) + '_min'] * midp / yvalue
                longSignal = signals.copy()
                longSignal[longSignal <= 0] = np.nan
                shortSignal = signals.copy()
                shortSignal[shortSignal >= 0] = np.nan

                # ax.plot(yvalue, )
                ax.scatter(y = list(longSignal.iloc[:]),x = range(len(yvalue)),marker = uparrow,s = arrowSize, c = 'red')
                ax.scatter(y = list(abs(shortSignal).iloc[:]),x = range(len(yvalue)),marker = downarrow,s = arrowSize, c= 'green')

        fig.canvas.draw()

        # set x axis name
        labels = [item.get_text() for item in ax.get_xticklabels()]
        # print(labels)
        if len(labels) != 0:
            labels[1:-1] = midp.index[list(map(int, labels[1:-1]))]
            ax.set_xticklabels(labels, rotation=45)

        ax.legend(legendList)

        ax.set_title('Sector action with ' + sector)
        plt.savefig(savePath + 'sector action with ' + sector)
        plt.close('all')

if __name__ == '__main__':
    """
    test the class
    """
    # data = Data('E:/personalfiles/to_zhixiong/to_zhixiong/level2_data_with_factor_added','600030.SH','20170516')
    dataPath = 'E:/data/stock/wind'
    ## /sh201707d/sh_20170703
    tradeDate = '20190115'
    # symbol = ['600519.SH','600887.SH'] # 食品与饮料
    # symbol = ['600048.SH', '600340.SH', '600606.SH'] # 房地产
    # symbols = ['000856.SZ', '300137.SZ', '600340.SH', '600550.SH']  # 雄安新区板块
    # symbols = ['601211.SH', '601688.SH', '600030.SH', '000776.SZ']  # 证券
    # symbols = ['002049.SZ', '600703.SH', '600460.SH', '600584.SH']  # 半导体
    # symbols = ['000001.SZ', '600000.SH', '600036.SH', '601166.SH']  # 银行
    # symbols = ['000156.SZ', '300251.SZ', '600037.SH', '600373.SH']  # 传媒
    # symbols = ['000709.SZ',   '000959.SZ',   '600010.SH',   '600019.SH']  # 钢铁
    # symbols = ['000983.SZ',   '600157.SH',   '600188.SH', '601088.SH', '601225.SH']  # 煤炭
    # symbols = ['000009.SZ', '000413.SZ', '600516.SH', '601877.SH', '603133.SH']  # 石墨烯
    symbols = ['603993.SH']
    # exchange = symbol.split('.')[1].lower()
    #print(dataPath)
    data = Data.Data(dataPath, symbols, tradeDate,dataReadType= 'gzip', RAWDATA = 'True')
    sectorData = pd.DataFrame(
        {'secucode': symbols[0], 'secuname': 'bank', 'citics_ind_type': 'bank', 'citics_ind_eng': 'bank'}, index=[0])
    sectorData = sectorData.set_index('citics_ind_eng')
    signalTester = SignalTester(data, dailyData=pd.DataFrame, sectorData=sectorData, tradeDate=tradeDate,
                                symbol=symbols[0], dataSavePath='./test')
    # signalTester.CompareSectorAndStock(symbols[0], orderType='netMainOrderCashFlow')
    signalTester.CheckSignal(symbols[0],'obi',20,20)
    print('Test end')
