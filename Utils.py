# -*- coding: utf-8 -*-
"""
Created on 2017-09-08

@author: zhixiong

use: some functions may be used to do sth
"""
import datetime
import pandas as pd
import numpy as np
import itertools
import os


def GetLastNdates(n=5, tradeDate = '',tradingDays = ''):
    if tradeDate == '':
        tradeDate = str(datetime.datetime.today().date())
    tradingDays = pd.Index(tradingDays.loc[:, 'date'])
    tradeDayPosition = tradingDays.get_loc(str(tradeDate))
    if tradeDayPosition is None:
        raise ('Can not find the trade date for this date :', tradeDate)
    else:
        lastNDatePosition = tradeDayPosition - n + 1
        lastDatePosition = tradeDayPosition + 1  # plus 1 due to the position incorrect
        tradeDaysToReturn = tradingDays[lastNDatePosition: lastDatePosition]
    return tradeDaysToReturn.values.tolist()

def search_sorted_subsets_field(df, field, keywords):
    subsets = []
    for keyword in keywords:
        left_bound = df[field].searchsorted(keyword,'left')[0]
        right_bound = df[field].searchsorted(keyword,'right')[0]
        if (right_bound-left_bound) > 0:
            subsets.append(df[left_bound:right_bound])
    return subsets

def search_sorted_subsets(df, searches):
    subsets = [df]
    for field, values in searches:
        subsets = list(itertools.chain(*[search_sorted_subsets_field(subset, field, values) for subset in subsets]))
    return pd.DataFrame(np.vstack(subsets) if len(subsets)>0 else None,columns=df.columns).convert_objects()

def CheckStockSuspend(symbol, dataPath, rundt,Asset = ''):
    exchange = symbol.split('.')[1].upper()
    miccode = symbol.split('.')[0]
    if Asset == 'Future':
        dataPath = dataPath + '/FutTick/' + exchange + '/' + rundt[:6] + '/' + rundt + '/'
        fileName = dataPath + '/' + miccode + '_' + rundt + '.csv.gz'
        print(fileName + '   '+'Utils_CheckStock_suspend')
    else:
        dataPath = dataPath + '/Transaction/' + exchange + '/' + rundt[:6] + '/' + rundt + '/'
        fileName = dataPath + '/' + miccode + '_' + rundt + '.csv.gz'
    return os.path.exists(fileName)
