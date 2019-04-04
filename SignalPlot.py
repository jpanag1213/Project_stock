# -*- coding: utf-8 -*-
"""
Created on 2019-04-04
plot!!!plotly!!!
matplotlib!!!
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
import sys

import stats
import plotly_express as px
import plotly
import  plotly.graph_objs as go


class SignalPlot(object):

    def __init__(self, symbol, tradedate, quoteData,tradeData = None,futureData =None, outputpath = 'D:/SignalTest/SignalTest/Plotly'):
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
        print(sys.path[0])
        if os.path.exists(outputpath) is False:
            os.makedirs(outputpath)



    def plotly_Plot(self,*config_dict):
        plot_column,subplot_plt,mode_column = config_dict
        quotedata = self.quoteData[self.symbol]
        t1 = time.time()
        rows_num = np.unique(subplot_plt).shape[0]
        fig = plotly.tools.make_subplots(rows=rows_num, cols=1, print_grid=True, shared_xaxes=True)

        ## 单独plot quotedata的情况下

        quotedata.loc[:,'buypoint'] = quotedata.loc[:,'marker'] == 1
        quotedata.loc[quotedata.loc[:, 'buypoint'] ==0, 'buypoint'] = np.nan
        quotedata.loc[:,'sellpoint'] = quotedata.loc[:,'marker'] == -1
        quotedata.loc[quotedata.loc[:, 'sellpoint'] == 0, 'sellpoint'] = np.nan
        midp = quotedata.loc[:, 'midp']
        for plot_pair in zip(plot_column,subplot_plt,mode_column):
            column_name = plot_pair[0]

            subplot_loc = plot_pair[1]
            mode = plot_pair[2]
            y_data = (quotedata.loc[:, column_name])
            if mode == '':
                mode = 'lines'
                trace = go.Scatter(x = list(quotedata.loc[:,'exchangeTime']), y=list(y_data),name = column_name, mode = mode)
            else:
                try:
                    trace = go.Scatter( x=list(quotedata.loc[:,'exchangeTime']),y =list(y_data.iloc[:] * midp),name = column_name, mode = 'markers',marker= dict( size = 7,symbol =mode))
                except:
                    print('No such marker, circle instead')
                    trace = go.Scatter(x=list(quotedata.loc[:, 'exchangeTime']), y=list(y_data.iloc[:] * midp),name=column_name, mode='markers', marker=dict(size=7))

            fig.append_trace(trace,subplot_loc,1)

        fig.layout
        print( self.outputpath+'/test_plotly.html')
        #plotly.
        plotly.offline.plot(fig,filename= self.outputpath+'/test_plotly.html',image='svg',auto_open=False)
        #plot_html = plotly.offline.offline._plot_html(fig ,False, "", True, '100%', 525,auto_play= False)
        #plotly.offline.offline._plot_html()
        #'config', 'validate', 'default_width', 'default_height', 'global_requirejs', and 'auto_play'
        t2 = time.time()
        #print(plot_html)
        print('printTime: %f'%(t2-t1))
        return 0




if __name__ == '__main__':
    dataPath = '//192.168.0.145/data/stock/wind'
    ## /sh201707d/sh_20170703
    t1 = time.time()
    tradeDate = '20190115'
    symbols = ['600086.SH']
    data = Data.Data(dataPath,symbols, tradeDate,'' ,dataReadType= 'gzip', RAWDATA = 'True')
    stats   = stats.Stats(symbols,tradeDate,data.quoteData,data.tradeData)
    quotedata = stats.price_volume_fun(symbols[0])
    plt = SignalPlot(symbols[0], tradeDate, quotedata, data.tradeData)
    #print(data.tradeData[symbols[0]])
    t2 = time.time()

    t3 = time.time()
    print('total:' + str(t3 - t2))
    ## subplot_plot 画在那个subplot
    ## plot_column  plot Dataframe 的指定属性
    ## mode_column  图标 '' ：直线 非"" 指定
    plot_column = ['consistence_mean', 'midp', 'buypoint', 'sellpoint']

    subplot_plt = [2, 1, 1, 1]


    mode_column = ['', '', 'triangle-down', 'abc']
    plt.plotly_Plot(plot_column,subplot_plt,mode_column)
    print('readData_time:' + str(t2 - t1))
    #

    print('Test end')
