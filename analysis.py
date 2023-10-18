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
        t_xm = datetime.strptime(s, '%Y.%m.%d %H:%M:%S')
        # XM -> JST
        # jst =  xm + 7hours (winter)
        # jst = xm + 6hours (summer)
        t_xm += timedelta(hours=7)
        dt.append(t_xm)
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
    
def moving_average(signal, window):
    n = len(signal)
    half = int(window / 2)
    ma = np.full(n, np.nan)
    for i in range(half, n):
        right = i + half
        if i == n - 1:
            left = i - 1 
            right = i 
        elif right >= n:
            right = n - 1 
            left = i - (n - i) + 1 
        else:
            left = i - half + 1 
        d = signal[left: right + 1]
        ma[i] = np.mean(d)
    return ma

def upper_and_lower(signal, shift_value, is_value_percent=False):
    n = len(signal)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(n):
        if is_value_percent:
            upper[i] = signal[i] + signal[i] * shift_value / 100 
            lower[i] = signal[i] - signal[i] * shift_value / 100 
        else:
            upper[i] = signal[i] + shift_value 
            lower[i] = signal[i] - shift_value   
    return upper, lower

def difference(dic, signal):
    n = len(signal)
    plus = np.full(n, 0.0)
    minus = np.full(n, 0.0)
    for i, value in enumerate(signal):    
        if np.isnan(value):
            continue 
        plus[i] = dic[C.HIGH] - value
        minus[i] = value - dic[C.LOW] 
    return plus, minus

def slope(time, signal, window=5, by_percent=True):
    n = len(signal)
    delta = np.full(n, 0.0)
    for i in range(window - 1, n):
        y = signal[i - window + 1: i + 1]
        m, offset = np.polyfit(range(window), np.array(y), 1)
        dt = time[i] - time[i - window + 1]
        m /= (dt.total_seconds() / 60)
        if by_percent:
            m = m / np.mean(y) * 100.0 
        delta[i] = m
    return delta
        
def strength(dic, mid, upper, lower, delta):
    low = dic[C.LOW]
    high = dic[C.HIGH]
    streng = []
    for h, lo, m,  u, l, d in zip(high, low, mid, upper, lower, delta):
        if d > 0:
            s = (l - lo) / m 
        else:
            s = (u - h) / m 
        streng.append(s)
    return streng

def position(level, value):
    for i, l in enumerate(level):
        if value > l:
            break
    return i

def band_domain(dic, center, upper, lower):
    n = len(center)
    high = dic[C.HIGH]
    low = dic[C.LOW]
    mapping = []
    for h, l, c, lo, up in zip(high, low, center, lower, upper):
        if np.isnan(c):
            rates = [0.0, 0.0, 0.0, 0.0]
            mapping.append(rates)
            continue
        levels = [up, c, lo]
        ih = position(levels, h)
        il = position(levels, l)
        height = h - l
        rates = [0.0, 0.0, 0.0, 0.0]
        if ih == il:
            rates[ih] = 1.0
        else:
            r = (h - levels[ih]) / height
            rates[ih] = r
            r = (levels[il - 1] - l) / height
            rates[il] = r
            for i in range(ih + 1, il):
                r = (levels[i-1] - levels[i]) / height
                rates[i] = r
        mapping.append(rates)
        out = [np.full(n, 0.0) for _ in range(4)]
        for i, [d1, d2, d3, d4] in enumerate(mapping):
            out[0][i] = d1
            out[1][i] = d2
            out[2][i] = d3
            out[3][i] = d4
    return out

def main():
    dic = load_data()
    mid_price(dic)
    
    for month in range(10, 11):
        for day in range(9, 16):
            try:
                tbegin = datetime(2023, month, day, 22)
            except:
                continue
            tend = tbegin + timedelta(hours=4)
            n, data = TimeUtils.slice(dic, dic[C.TIME], tbegin, tend)
            if n < 5:
                continue
            
            mid = data[C.MID]
            polarity = data[C.POLARITY]
            ma = moving_average(mid, 9)
            upper, lower = upper_and_lower(ma, 20, is_value_percent=False)
            delta = slope(data[C.TIME], ma)
            strg = strength(dic, mid, upper, lower, delta)            
            domain = band_domain(dic, mid, upper, lower)

            fig, axes = gridFig([5, 2, 2, 1, 1, 1, 1], (25, 15))
            
            chart1 = CandleChart(fig, axes[0], write_time_range=True)
            chart1.drawCandle(data[C.TIME], data[C.OPEN], data[C.HIGH], data[C.LOW], data[C.CLOSE])
            chart1.drawLine(data[C.TIME], ma, color='blue')
            chart1.drawLine(data[C.TIME], upper, color='green')
            chart1.drawLine(data[C.TIME], lower, color='red')
            
            chart2 = CandleChart(fig, axes[1], comment='delta')
            chart2.drawLine(data[C.TIME], delta, color='blue')
            
            chart3 = CandleChart(fig, axes[2], comment='strength')
            chart3.drawLine(data[C.TIME], strg, color='blue')
            
            chart4 = CandleChart(fig, axes[3])
            chart4.drawLine(data[C.TIME], domain[0])
            chart5 = CandleChart(fig, axes[4])
            chart5.drawLine(data[C.TIME], domain[1])
            chart6 = CandleChart(fig, axes[5])
            chart6.drawLine(data[C.TIME], domain[2])
            chart7 = CandleChart(fig, axes[6])
            chart7.drawLine(data[C.TIME], domain[3])
            pass
    
if __name__ == '__main__':
    main()