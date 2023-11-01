# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:28:50 2023

@author: docs9
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, initial, risk_by_reward, lot, target_pips, leverage, win_rate):
        self.initial = initial 
        self.risk_by_reward = risk_by_reward 
        self.lot = lot
        self.target_pips = target_pips
        self.leverage = leverage
        self.win_rate = win_rate
        
    def run(self, rep_max=1000):
        profit = 0
        profit_acc = []
        profits = []
        for i in range(rep_max):
            p = self.target_pips * self.lot * self.leverage
            r = random.random()
            if r > (1.0 - self.win_rate):
                profit += p
                profit_acc.append(profit)
                profits.append(p)
            else:
                loss = -1 * p * self.risk_by_reward
                profit += loss
                profit_acc.append(profit)
                profits.append(loss)
            if profit >= self.initial:
                print('Complet', i)
                return i,profits, profit_acc
            if profit < -1 * (self.initial * 0.8):
                print('Finish', i)
                return -1, profits, profit_acc
        return 0, profits, profit_acc
    
    
def search():
    initial = 100000
    leverage = 1000
    lot = 1
    cond = []
    for target_pips in np.arange(0.05, 0.4, 0.05):
        for win_rate in np.arange(0.1, 0.8, 0.1):
            for risk_by_reward in np.arange(0.1, 0.9, 0.1):
                sim = Simulation(initial, risk_by_reward, lot, target_pips, leverage, win_rate)
                n, profits, profit_cc = sim.run()
                cond.append([n, target_pips, win_rate, risk_by_reward])
    df = pd.DataFrame(data=cond, columns = ['n', 'target_pips', 'win_rate', 'risk_by_reward'])
    df.to_excel('/record.xlsx', index=False)
    
    
def main():
    initial = 100000
    leverage = 1000
    usdjpy = 150
    price = 100000 / leverage * usdjpy
    
    lot = 1
    target_pips = 0.3
    win_rate = 0.7
    risk_by_reward = 0.6
    sim = Simulation(initial, risk_by_reward, lot, target_pips, leverage, win_rate)
    n, profits, profit_acc = sim.run()
    fig, axes = plt.subplots(2)
    axes[0].plot(range(len(profit_acc)), profit_acc)
    axes[1].plot(range(len(profits)), profits)
    fig.show()
    
if __name__ == '__main__':
    main()