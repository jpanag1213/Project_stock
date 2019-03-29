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
import time
from Utils import *
import matplotlib.pyplot as plt
import datetime
from numba import jit

class Strategy(object):

    def __init__(self, symbol, qty, quoteData, signal, tradingDay, lbwindow, lawindow, times, closeType, fee=15 / 10000,
                 outputpath='./strategy', stockType='low', asset=''):
        self.symbol = symbol
        self.qty = qty

        self.quoteData = quoteData
        self.signal = signal
        self.lbwindow = lbwindow
        self.lawindow = lawindow
        self.times = times
        self.closeType = closeType
        self.stockType = stockType
        self.asset = asset
        self.tradeDate = tradingDay
        self.sts = pd.DataFrame(columns=['wr', 'pnl', 'times', 'todayup', 'trade_qty', 'total_qty'], index=[symbol])
        self.outputpath = outputpath
        self.quoteData = self.opentime()
        self.fee = fee
        if self.asset == 'Future':
            self.openQty = round(self.qty / self.times / (200 / 12))
            print(openQty)
        else:
            self.openQty = round(self.qty / self.times, -2)

        ### 策略初始化
        self.pnl = 0
        self.openPrice = 0
        self.totalTimes = 0
        self.winTimes = 0
        self.trade_qty = 0
        self.trade_flag = list()
        self.currentQty = 0  # 当前持仓，用来看当前仓位的变化
        self.currentQtyList = list()
        self.holdTime = 0  # 用来记录持仓时长
        self.count = 0  # 记录交易次数


        if os.path.exists(outputpath) is False:
            os.makedirs(outputpath)

    def Plot(self):
        """
        根据交易记录和持仓情况画出交易记录图，包括如下：
        1. 价格变化
        2. 开仓时间点，信号持续点，按时间窗口平仓，按反向信号平仓
        3. 持有仓位变化图
        :return: 图，并保存到指定位置
        """
        print('Plot with price series and signals on the same plot')
        savePath = self.outputpath + '/' + self.signal + '_' + str(self.lbwindow) + '/'
        if os.path.exists(savePath) is False:
            os.mkdir(savePath)

        fig, axs = plt.subplots(2, figsize=(20, 12), sharex=True)
        ax = axs[0]

        midp = self.quoteData.loc[:, 'midp']
        yvalue = list(midp.iloc[:])
        ax.plot(yvalue, label=self.symbol + '_midquote')
        uparrow = u'$\u2191$'  # the shape of up arrow
        upt = '^'
        downt = 'v'
        closet = 's'
        opposite = 'o'
        downarrow = u'$\u2193$'  # the shape of down arrow
        arrowSize = 100

        signals = self.quoteData.loc[:, 'trade_flag']
        longOpen = signals.copy()
        longHold = signals.copy()
        longClose = signals.copy()
        longOpposite = signals.copy()
        shortOpen = signals.copy()
        shortHold = signals.copy()
        shortClose = signals.copy()
        shortOpposite = signals.copy()
        longOpen[longOpen != 1] = np.nan
        longHold[longHold != 2] = np.nan
        longClose[longClose != 3] = np.nan
        longOpposite[longOpposite != 4] = np.nan
        shortOpen[shortOpen != -1] = np.nan
        shortHold[shortHold != -2] = np.nan
        shortClose[shortClose != -3] = np.nan
        shortOpposite[shortOpposite != -4] = np.nan

        # ax.plot(yvalue, )
        ax.scatter(y=list(longOpen.iloc[:] * midp), x=range(len(yvalue)), marker=uparrow, s=arrowSize, c='red')
        ax.scatter(y=list(longHold.iloc[:] * midp / 2), x=range(len(yvalue)), marker=upt, s=arrowSize, c='red')
        ax.scatter(y=list(longClose.iloc[:] * midp / 3), x=range(len(yvalue)), marker=closet, s=arrowSize, c='red')
        ax.scatter(y=list(longOpposite.iloc[:] * midp / 4), x=range(len(yvalue)), marker=opposite, s=arrowSize, c='red')
        ax.scatter(y=list(abs(shortOpen).iloc[:] * midp), x=range(len(yvalue)), marker=downarrow, s=arrowSize,
                   c='green')
        ax.scatter(y=list(abs(shortHold).iloc[:] * midp / 2), x=range(len(yvalue)), marker=downt, s=arrowSize,
                   c='green')
        ax.scatter(y=list(abs(shortClose).iloc[:] * midp / 3), x=range(len(yvalue)), marker=closet, s=arrowSize,
                   c='green')
        ax.scatter(y=list(abs(shortOpposite).iloc[:] * midp / 4), x=range(len(yvalue)), marker=opposite, s=arrowSize,
                   c='green')
        ax.legend(
            ['midp', 'long open', 'long hold', 'long close', 'long opposite', 'short open', 'short hold', 'short close',
             'short opposite'])

        # set x axis name
        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # # print(labels)
        # if len(labels) != 0:
        #     labels[1:-1] = midp.index[list(map(int, labels[1:-1]))]
        #     ax.set_xticklabels(labels, rotation=45)
        ax2 = axs[1]
        pnl = list(self.quoteData['cumpnl'].iloc[:])
        positions = list(self.quoteData['currentQty'].iloc[:])
        ax3 = ax2.twinx()
        ax2.plot(pnl, c='blue', label='pnl')
        ax3.plot(positions, label='current_position', color='y')
        ax3.set_ylabel('pnl vs current_position')
        ax2.legend(['cumpnl'])

        fig.canvas.draw()

        ax.set_title(
            'Strategy by signal = ' + self.signal + ' with midquote change of stock ' + self.symbol + ' lbwindow = ' + str(
                self.lbwindow))
        plt.savefig(savePath + '/' + self.symbol + '.jpg')
        plt.close('all')
        return 0

    def opentime(self):
        quoteData = self.quoteData
        quoteData.loc[:, 'openstatus'] = 0
        quoteData.loc[
        datetime.datetime.strptime(str(self.tradeDate + ' 09:30:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
            str(self.tradeDate + ' 14:50:00'), '%Y%m%d %H:%M:%S'), 'openstatus'] = 1
        quoteData.loc[
        datetime.datetime.strptime(str(self.tradeDate + ' 14:50:00'), '%Y%m%d %H:%M:%S'):datetime.datetime.strptime(
            str(self.tradeDate + ' 14:57:00'), '%Y%m%d %H:%M:%S'), 'openstatus'] = 2

        return quoteData

    def SummaryStrategy(self):
        # if self.sts.empty:
        #     self.run()
        # else:
        #     print(self.sts)
        self.run()
        print(self.sts)


    def run(self):
        """
        designed to test the signal strategy result
        :param symbol: str, the stock we want to test
        :param qty: int, the position we suppose have
        :param quoteData: data frame, the data frame include the signal.
        :param lbwindow: int, the look back window
        :param lawindow: int, the look ahead window
        :param closeType: str, the way we close our position.
        :param stockType: str, the low price stock or high price stock
        :return: pd.DataFrame, strategy summary of the stock.
        """
        fee = self.fee



        cumpnl = 0
        cumpnlList = list()

        pnl_list = list()
        openpricelist = list()
        bidPricelist = list()
        askPricelist = list()

        ''' 
        debug list :record_ 
        '''

        # self.quoteData.to_csv('./'+self.symbol+'_test_.csv')

        for row in zip(self.quoteData.loc[:, self.signal + '_' + str(self.lbwindow) + '_min'],
                       self.quoteData['bidPrice1'], self.quoteData['askPrice1'], self.quoteData['midp'],
                       self.quoteData.openstatus):
            longShort = row[0]  # 1 is long, -1 is short
            self.longShort = longShort


            bidPrice = row[1]
            askPrice = row[2]
            lastPrice = row[3]
            openstatus = row[4]
            if (openstatus != 0):
                if (openstatus == 2):  ##收盘平仓
                    if self.currentQty > 0:
                        self.LongClose(bidPrice,self.openPrice)
                    elif self.currentQty < 0:
                        self.ShortClose(askPrice, self.openPrice)
                    else:
                        self.trade_flag.append(np.nan)
                        self.pnl = 0
                elif (openstatus == 1):
                    if longShort == 1:
                        if self.currentQty > 0:
                            self.holdTime = 0  # 当有持续信号时，暂时不考虑重复开仓，防止记录麻烦。之后需要改进。因此这里需要重新记录持仓时间
                            self.pnl = 0
                            self.trade_flag.append(2)
                        elif self.currentQty < 0:
                            self.ShortClose(askPrice, self.openPrice)
                            #self.LongOpen(askPrice)
                        elif self.currentQty == 0:
                            self.LongOpen(askPrice)



                    elif longShort == -1:

                        if self.currentQty < 0:
                            self.holdTime = 0  # 当有持续信号时，暂时不考虑重复开仓，防止记录麻烦。之后需要改进。因此这里需要重新记录持仓时间
                            self.pnl = 0
                            self.trade_flag.append(-2)
                        elif self.currentQty > 0:
                            self.LongClose(bidPrice, self.openPrice)
                            #self.ShortOpen(bidPrice)
                        elif self.currentQty == 0:
                            self.ShortOpen(bidPrice)



                    else:
                        if self.currentQty == 0:
                            self.pnl = 0
                            self.trade_flag.append(np.nan)
                        elif self.currentQty > 0:
                            if self.holdTime < self.lawindow:
                                self.holdTime = self.holdTime  + 1
                                self.pnl = 0
                                self.trade_flag.append(np.nan)
                            else:
                                self.LongClose(bidPrice,self.openPrice)
                        elif self.currentQty < 0:
                            if self.holdTime < self.lawindow:
                                ###时间止损
                                self.holdTime = self.holdTime + 1
                                self.pnl = 0
                                self.trade_flag.append(np.nan)
                            else:
                                self.ShortClose(askPrice,self.openPrice)
            else:
                self.pnl = 0
                self.trade_flag.append(np.nan)

            openpricelist.append(self.openPrice)
            bidPricelist.append(bidPrice)
            askPricelist.append(askPrice)

            self.currentQtyList.append( self.currentQty)
            # currentQtyList.append(currentQty)
            cumpnlList.append(cumpnl + self.pnl + self.currentQty * (lastPrice - self.openPrice))
            if self.pnl != 0:
                cumpnl = cumpnl + self.pnl

        self.quoteData.loc[:, 'cumpnl'] = cumpnlList
        self.quoteData.loc[:, 'currentQty'] = self.currentQtyList
        #print(len(cumpnlList))
        #print(len(self.trade_flag))
        self.quoteData.loc[:, 'trade_flag'] = self.trade_flag
        if self.totalTimes != 0:
            wr = self.winTimes / self.totalTimes
        else:
            wr = 0
        pnlpct = cumpnlList[-1] / (lastPrice * self.qty)
        todayUpDowns = lastPrice / self.quoteData['midp'].iloc[0] - 1
        self.sts.loc[self.symbol, 'wr'] = wr
        self.sts.loc[self.symbol, 'pnl'] = pnlpct
        self.sts.loc[self.symbol, 'todayup'] = todayUpDowns
        self.sts.loc[self.symbol, 'trade_qty'] = self.trade_qty
        self.sts.loc[self.symbol, 'times'] = self.totalTimes

        self.sts.loc[self.symbol, 'total_qty'] = self.qty
        self.sts.loc[self.symbol, 'actualpnl'] = cumpnlList[-1]

        # df_ = pd.DataFrame([currentQtyList,record_,trade_flag])
        #self.quoteData.to_csv('./pnl.csv')
        return 0

    def LongOpen(self,enterPrice):
        if self.count < self.times:
            self.currentQty = self.currentQty + self.openQty
            self.openPrice = enterPrice
            self.pnl = 0
            self.trade_flag.append(1)
        else:
            self.trade_flag.append(np.nan)
            ##。。。。
            self.pnl = 0
        return 0


    def LongClose(self,exitPrice,enterPrice):
        if self.longShort == -1:
            self.trade_flag.append(4)
        elif self.longShort != -1:
            self.trade_flag.append(3)
        self.pnl = (exitPrice - enterPrice - enterPrice * self.fee) * self.currentQty
        # pnl = (bidPrice - openPrice - openPrice * 0.0015)*currentQty
        self.trade_qty = self.trade_qty + abs(self.currentQty)
        self.count = self.count + 1
        self.holdTime = 0
        self.currentQty = 0
        self.totalTimes = self.totalTimes + 1
        if self.pnl > 0:
            self.winTimes = self.winTimes + 1
        return 0


    def ShortOpen(self,enterPrice):
        if self.count < self.times:
            self.currentQty = self.currentQty - self.openQty
            self.openPrice = enterPrice
            self.pnl = 0
            self.trade_flag.append(-1)
        else:
            self.trade_flag.append(np.nan)
            ##。。。。
            self.pnl = 0
        return 0


    def ShortClose(self,exitPrice,enterPrice):
        if self.longShort == 1:
            self.trade_flag.append(-4)
        elif self.longShort != 1:
            self.trade_flag.append(-3)
        self.pnl = -(enterPrice - exitPrice - enterPrice * self.fee) * self.currentQty
        #pnl = -(openPrice - askPrice - openPrice * 0.0015)*currentQty
        self.trade_qty = self.trade_qty + abs(self.currentQty)
        self.count = self.count + 1
        self.holdTime = 0
        self.currentQty = 0
        self.totalTimes = self.totalTimes + 1
        if self.pnl > 0:
            self.winTimes = self.winTimes + 1


        return 0


