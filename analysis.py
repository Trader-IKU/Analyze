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

MA = 'ma'
MA_MULTI = 'ma_multi'
UPPER = 'upper'
LOWER = 'lower'
DELTA = 'delta'
STRENGTH = 'strength'
DOMAIN = 'domain'
U_OVER = 'u_over'
L_OVER = 'l_over'
MA_CHANGE = 'ma_change'

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

def moving_average_multi(signal, window):
    n = len(signal)
    half = int(window / 2)
    mas = [np.full(n, np.nan) for _ in range(half + 1)]
    change = np.full(n, np.nan)
    for i in range(half, n - half):
        for j in range(half + 1):
            if j == 0:
                begin = i - 1 
                end = i
            else:
                begin = i - j 
                end = i + j 
            d = signal[begin: end + 1]
            mas[j][i] = np.mean(d)
        change[i] = mas[half][i] - mas[0][i]
    return mas, change

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

def upper_over(dic: dict, upper):
    n = len(upper)
    high = dic[C.HIGH]
    out = np.full(n, np.nan)
    for i in range(n):
        if high[i] > upper[i]:
            out[i] = high[i]
    return out

def lower_over(dic: dict, lower):
    n = len(lower)
    low = dic[C.LOW]
    out = np.full(n, np.nan)
    for i in range(n):
        if low[i] < lower[i]:
            out[i] = low[i]
    return out

def indicators(dic):
    mid_price(dic)
    mid = dic[C.MID]
    ma = moving_average(mid, 9)
    dic[MA] = ma
    upper, lower = upper_and_lower(ma, 20, is_value_percent=False)
    dic[UPPER] = upper
    dic[LOWER] = lower
    dic[U_OVER] = upper_over(dic, upper)
    dic[L_OVER] = lower_over(dic, lower)
    delta = slope(dic[C.TIME], ma)
    dic[DELTA] = delta
    dic[STRENGTH] = strength(dic, mid, upper, lower, delta)            
    dic[DOMAIN] = band_domain(dic, mid, upper, lower)
    mas, change = moving_average_multi(mid, 9)
    dic[MA_MULTI] = mas
    dic[MA_CHANGE] = change    

def main():
    dic = load_data()
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
            
            indicators(data)
            mid = data[C.MID]
            polarity = data[C.POLARITY]
    
            fig, axes = gridFig([2, 2, 1], (25, 15))
            time = data[C.TIME]
            chart1 = CandleChart(fig, axes[0], write_time_range=True)
            chart1.drawCandle(time, data[C.OPEN], data[C.HIGH], data[C.LOW], data[C.CLOSE])
            chart1.drawLine(time, data[MA], color='blue')
            chart1.drawLine(time, data[U_OVER], color='green')
            chart1.drawLine(time, data[L_OVER], color='red')
            
            chart2 = CandleChart(fig, axes[1], comment='MA_MULTI')
            ma_multi = data[MA_MULTI]
            for ma in ma_multi:
                chart2.drawLine(time, ma, color='yellow')
            
            chart3 = CandleChart(fig, axes[2], comment='strength')
            chart3.drawLine(time, data[MA_CHANGE], color='blue')
            
            pass
    
if __name__ == '__main__':
    main()