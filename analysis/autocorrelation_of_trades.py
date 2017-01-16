# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:22:39 2016

@author: ilkhem
"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import datetime as dt

from statsmodels.graphics.api import qqplot

if __name__ == '__main__':
    # dat = pd.DataFrame()
    # for dirpath, dirnames, files in os.walk('/Volumes/Khmkhm/data/trades/btcusd/2016/05/coinbase'):
    #     for file in files:
    #         dat = dat.append(pd.read_csv(dirpath+'/'+file,usecols=['date','sell']),ignore_index=True)

    dat = pd.read_csv('/Volumes/Khmkhm/data/trades/btcusd/2016/05/coinbase/Coinbase_BTCUSD_trades_2016_05_01.csv',
                      usecols=['date','sell'])

    dat.sort_values('date',ascending=True,inplace=True)
    dat['date'] = dat['date'].map(lambda x : dt.datetime.utcfromtimestamp(x/1000))
    dat['sell'] = 2*dat['sell'] - 1
    dat['sell'] = dat['sell'].map(float)
    dat.set_index('date',inplace=True)


    #sm.tsa.acf(dat,nlags=50,alpha=0.05,qstat=True)

    #dat.plot(figsize=(12,8))

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(dat.values.squeeze(), lags=40, ax=ax1, alpha=0.1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(dat, lags=40, ax=ax2, alpha=0.1)
    plt.show()
    fig.savefig('plots/analysis/autocorrelation_of_trades-cb-01052016.png',format='png')

    print('arma')

    #%%

    arma150 = sm.tsa.ARMA(dat, (15,0)).fit()
    print(arma150.params)

    sm.stats.durbin_watson(arma150.resid.values)

    print('resid')

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax = arma150.resid.plot(ax=ax)
    plt.show()

    resid = arma150.resid
    stats.normaltest(resid)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)
    plt.show()

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
    plt.show()


    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax = dat.ix['2016-05-25':].plot(ax=ax)
    # fig = arma150.plot_predict('2016-05-27', '2016-06-03', dynamic=True, ax=ax, plot_insample=False)