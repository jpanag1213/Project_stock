# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:20:23 2019

@author: pan
"""

import pandas as pd
import numpy as np
from Utils import *

tradingDayFile ='D:/SignalTest/SignalTest/ref_data/TradingDay.csv'
tradingDays = pd.read_csv(tradingDayFile)
tradingDays = pd.Index(tradingDays.loc[:, 'date'])
tradeDate ='2019-01-02'
tradeDayPosition = tradingDays.get_loc(str(tradeDate))