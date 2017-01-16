# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:47:35 2016

@author: ilkhem
"""

__author__ = 'ilkhem'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt


def get_closest_minute(t):
    """
    returns a datetime.datetime object giving the closest minute to a given timestamp
    t : timestamp in MILLISECONDS
    """
    ts = dt.datetime.utcfromtimestamp(t/1000)
    s = ts.second
    if s < 30:
        return dt.datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute)
    else:
        return dt.datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute) + dt.timedelta(minutes=1)


def get_mid_from_file(file):
    '''file : path to ob1 file (.csv)'''
    with open(file) as f:
        f.readline()  # skip header line
        ts = '0'  # current timestamp, initiatie at 0
        b = 0  # limit bid at current timestamp
        a = 0  # limit ask at current timestamp
        mid = []  # list of [timestamp, mid]
        ask = False  # Defines what mean price we are calculating
        for line in f:   
            line_split = line.split(',')
            
            # for a given timestamp, this condition is only true when we first encounter that timestamp in BID side
            # we re-initialize all the variables but the outputs at each new timestamp
            if ts != line_split[0]:
                ts = line_split[0]  # update the timestamp
                b = float(line_split[2])  # the first line for each new timestamp is the limit bid
                ask = False  # we work on the bid side
            
            # for a given timestamp, this condition is only true when we first encounter that timestamp in ASK side
            # we re-initialize all the variables but the outputs each time we move from bids to asks
            if not ask and line_split[1] == 'a':
                ask = True  # we are now working on the ask side
                a = float(line_split[2])
                mid += [[int(ts),(a+b)/2]]
                
    df = pd.DataFrame(mid,columns=['date','mid'])
    df['date'] = df['date'].map(get_closest_minute)
    return df.groupby('date').mean().reset_index()


def get_liquidity_from_file(file):
    '''file : path to ob1 file (.csv)'''
    with open(file) as f:
        f.readline()  # skip header line
        ts = '0'  # current timestamp, initiatie at 0
        v = 0  # top 5 limit ask amount at current timestamp
        bid_liquidity = []  # list of [timestamp, bid_liquidity]
        ask_liquidity = []  # list of [timestamp, ask_liquidity]
        c = 0
        ask = False  # Defines what mean price we are calculating
        for line in f:   
            line_split = line.split(',')
            
            # for a given timestamp, this condition is only true when we first encounter that timestamp in BID side
            # we re-initialize all the variables but the outputs at each new timestamp
            if ts != line_split[0]:
                if ask:
                    ask_liquidity += [[int(ts),v]]
                ts = line_split[0]  # update the timestamp
                ask = False  # we work on the bid side
                # reinitialize
                v = 0
                c = 0
            
            # for a given timestamp, this condition is only true when we first encounter that timestamp in ASK side
            # we re-initialize all the variables but the outputs each time we move from bids to asks
            if not ask and line_split[1] == 'a':
                bid_liquidity += [[int(ts),v]]
                ask = True  # we are now working on the ask side
                # reinitialize
                v = 0
                c = 0
            
            if c < 5:            
                f3 = float(line_split[3])
                v += f3
                c += 1
        ask_liquidity += [[int(ts),v]]
            
    df = pd.DataFrame(bid_liquidity,columns=['date','bid liquidity'])
    df = df.merge(pd.DataFrame(ask_liquidity,columns=['date','ask liquidity']),on='date',how='outer')
    df['date'] = df['date'].map(get_closest_minute)
    return df.groupby('date').mean().reset_index()


def get_trades_from_file(file):
    '''file : path to trades file (.csv)'''
    with open(file) as f:
        f.readline()  # skip header line
        ts = 0  # current timestamp, initiatie at 0
        v = 0  # traded volume at current timestamp
        trades = []  # list of [timestamp, mid]
        flag = False  
        for line in f:   
            line_split = line.split(',')
            x = 60000*(int(line_split[3])//60000)
            
            # for a given timestamp, this condition is only true when we enter a new minute
            if ts != x:
                if flag:
                    trades += [[ts,v]]
                ts = x # update the timestamp
                v = 0
                flag = True
                
            v += float(line_split[5])
            
        trades += [[ts,v]]
        
    df = pd.DataFrame(trades,columns=['date','amount'])
    df['date'] = df['date'].map(get_closest_minute)
    return df.groupby('date').mean().reset_index()


if __name__ == '__main__':
    directory = "/Volumes/Khmkhm/data/"
    tr = pd.DataFrame()
    ob = pd.DataFrame() 
    lb = pd.DataFrame()
    for dirpath, dirnames, files in os.walk(directory+'trades/btcusd/2016/05/bitfinex/'):
        for file in files:
            print(file)
            tr = tr.append(get_trades_from_file(dirpath+'/'+file),
                           ignore_index=True)  
                           
    for dirpath, dirnames, files in os.walk(directory+'ob_1/btcusd/2016/05/bitfinex/'):
        for file in files:
            print(file)
            ob = ob.append(get_mid_from_file(dirpath+'/'+file),
                           ignore_index=True)
    ob.set_index('date',inplace=True) 

    for dirpath, dirnames, files in os.walk(directory+'ob_1/btcusd/2016/05/bitfinex/'):
        for file in files:
            print(file)
            lb = lb.append(get_liquidity_from_file(dirpath+'/'+file),
                           ignore_index=True)
    lb.set_index('date',inplace=True)      
    
    limit = 10 # threshold                 
    df = tr[tr['amount']>=limit].copy().reset_index(drop=True)
    
    l1 = []
    flag = True
    for i in range(len(df)):
        ts = df.loc[i,'date']
        ts_ob = ts
        try:
            x = ob.loc[ts_ob,'mid']
        except:
            try:
                ts_ob = ts + dt.timedelta(minutes=1)
                x = ob.loc[ts_ob,'mid']
            except:
                try:
                    ts_ob = ts - dt.timedelta(minutes=1)
                    x = ob.loc[ts_ob,'mid']
                except:
                    flag = False
        if flag:
            st = ts - dt.timedelta(minutes=5)
            ed = ts + dt.timedelta(minutes=5)
            temp = ob[(ob.index>=st) & (ob.index<=ed)].copy()
            temp['mid'] = (temp['mid'] - x)
            missing_before = 5 - len(temp[temp.index<ts_ob])
            missing_after = 5 - len(temp[temp.index>ts_ob])
            l1 += [[ts,missing_before*[np.nan]+list(temp['mid'])+missing_after*[np.nan]]]
        flag = True
    
    lmean = np.nanmean([x[1] for x in l1],axis=0)
    fig, ax = plt.subplots()
    ax.plot(lmean)
    plt.savefig('plots/analysis/bitfinex-average-variation-around-large-10trades-05.png',dpi=200)
    plt.show()
    
    l2 = []
    flag = True
    for i in range(len(df)):
        ts = df.loc[i,'date']
        ts_lb = ts
        if not ts_lb in lb.index:
            ts_lb = ts + dt.timedelta(minutes=1)
            if not ts_lb in lb.index:
                ts_lb = ts - dt.timedelta(minutes=1)
                if not ts_lb in lb.index:
                    flag = False
        if flag:
            st = ts - dt.timedelta(minutes=5)
            ed = ts + dt.timedelta(minutes=5)
            temp = lb[(lb.index>=st) & (lb.index<=ed)].copy()
            temp['r'] = temp['bid liquidity'] / temp['ask liquidity']
            missing_before = 5 - len(temp[temp.index<ts_lb])
            missing_after = 5 - len(temp[temp.index>ts_lb])
            l2 += [[ts,missing_before*[np.nan]+list(temp['r'])+missing_after*[np.nan]]]
        flag = True
    
    lmean2 = np.nanmean([x[1] for x in l2],axis=0)
    fig, ax = plt.subplots()
    ax.plot(lmean2)
    plt.savefig('plots/analysis/bitfinex-average-liq-variation-around-large-10trades-05.png',dpi=200)
    plt.show()