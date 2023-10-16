# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:37:31 2023

@author: docs9
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../Libraries/trade"))


import polars as pl
import numpy as np
from numpy import array
import pandas as pd

from datetime import datetime, timedelta
#from timeframe import Timeframe
from const import Const as C
from candle_chart import CandleChart, BandPlot, makeFig, gridFig, Colors
from time_utils import TimeUtils


POSITIVE = 1 
NEGATIVE = -1 


def load_data():
    file = './data/US30Cash_M1_202310020100_202310132359.csv'
    df = pd.read_csv(file, delimiter='\t')
    date = df['<DATE>'].tolist()
    time = df['<TIME>'].tolist()
    op = df['<OPEN>'].tolist()
    hi = df['<HIGH>'].tolist()
    lo = df['<LOW>'].tolist()
    cl = df['<CLOSE>'].tolist()
    
    dt = []
    for d, t in zip(date, time):
        s = d + ' ' + t
        dt.append(datetime.strptime(s, '%Y.%m.%d %H:%M:%S'))
        
    dic = {C.TIME: dt, C.OPEN: op, C.HIGH: hi, C.LOW: lo, C.CLOSE: cl}
    return dic


def mid_price(dic):
    op = dic[C.HIGH]
    cl = dic[C.CLOSE]
    mid = []
    pol = []
    for o, c in zip(op, cl):
        mid.append((o + c ) / 2)
        if c > o:
            pol.append(POSITIVE)
        else:
            pol.append(NEGATIVE)
            
    dic[C.MID] = mid
    dic[C.POLARITY] = pol
    return mid, pol
    
    
def moving_average(dic, window):
    return

    
def main():
    dic = load_data()
    mid, pol = mid_price(dic)
    
    print(len(mid))
    
    for month in range(10, 11):
        for day in range(2, 31):
            try:
                tbegin = datetime(2023, month, day)
            except:
                continue
            tend = tbegin + timedelta(days=1)
            n, data = TimeUtils.slice(dic, dic[C.TIME], tbegin, tend)
            if n < 5:
                continue
            
            fig, ax = makeFig(1, 1, (15, 10))
            chart = CandleChart(fig, ax)
            chart.drawCandle(data[C.TIME], data[C.OPEN], data[C.HIGH], data[C.LOW], data[C.CLOSE])
            
            pass
    
    
    
    
    
    
if __name__ == '__main__':
    main()