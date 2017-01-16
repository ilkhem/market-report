# -*- coding: utf-8 -*-
"""
Created on Mon May  2 17:12:26 2016

@author: ilyes
"""

__author__ = 'ilkhem'

import datetime as dt

import numpy as np
import pandas as pd

from utils import get_closest_minute, get_monday


def get_vb_from_csv(file, xch, thresholds=[1, 5, 10, 50, 100, 500, 1000]):
    """
    returns the volume_book from csv orderbook

    file : path to csv (STRING)
    
    xch : exchange name (STRING)

    thresholds: this argument is not used, its purpose is to have the same syntax for get_vb_ and get_sb_
    """

    def get_volume_from_file(file):
        '''
        file : path to ob1 file (.csv)   
        
        works with ob10
        '''
        with open(file) as f:
            f.readline()  # skip header line
            ts = 0  # current timestamp, initiatie at 0
            vb = []  # list of [timestamp, vol bid]
            va = []  # list of [timestamp, vol ask]
            ask = False  # Defines what side we are in at a given timestamp
            v = 0
            for line in f:
                line_split = line.split(',')

                # for a given timestamp, this condition is only true when we first encounter that timestamp in BID side
                # we re-initialize all the variables but the outputs at each new timestamp
                if ts != line_split[0]:
                    if v != 0:
                        if ask:
                            va += [[int(ts), v]]
                        else:
                            vb += [[int(ts), v]]
                    ts = line_split[0]  # update the timestamp
                    ask = False  # we work on the bid side
                    # reinitialize
                    v = 0

                # for a given timestamp, this condition is only true when we first encounter that timestamp in ASK side
                # we re-initialize all the variables but the outputs each time we move from bids to asks
                if not ask and line_split[1] == 'a':
                    if v != 0:
                        vb += [[int(ts), v]]
                    ask = True  # we are now working on the ask side
                    # reinitialize
                    v = 0

                f3 = float(line_split[3])
                v += f3
            va += [[int(ts), v]]
        return vb, va

    volb, vola = get_volume_from_file(file)
    volb = pd.DataFrame(volb, columns=['date', 'bid_volume'])
    vola = pd.DataFrame(vola, columns=['date', 'ask_volume'])
    vb = volb.merge(vola, on='date', how='outer')
    vb['date'] = vb['date'].map(get_closest_minute)
    vb = vb.groupby('date').mean().reset_index()
    vb['exchange'] = xch
    return vb


def get_vb_from_json(df, xch):
    """
    returns volumes in the order book aggregated by minute from dump-like order books
    
    df : raw ob
    
    xch : exchange name
    """

    def get_volume(t):
        try:
            s = 0
            for i in range(len(t)):
                s += t[i][1]
            return s
        except:
            return np.nan

    vb = pd.DataFrame(df.ts)
    vb.ts = vb.ts.map(get_closest_minute)
    vb['bid_amount'] = df['bids'].map(get_volume)
    vb['ask_amount'] = df['asks'].map(get_volume)
    vb = vb.groupby('ts').mean().reset_index()
    vb['exchange'] = xch

    return vb


def get_sb_from_csv(file, xch, thresholds=[1, 5, 10, 50, 100, 500, 1000]):
    """
    returns the spread_book from csv order book

    file : path to csv (STRING)
    
    xch : exchange name (STRING)    
    
    thresholds : list of thresholds for calculating the spread. Default : [1,5,10,50,100,500,1000]
    """

    def get_mid_spread_from_file(file):
        '''
        file : path to ob10 file (.csv)    
        '''
        with open(file) as f:
            f.readline()  # skip header line
            ts = '0'  # current timestamp, initiatie at 0
            s = 0
            mt = 0
            b = 0  # limit bid at current timestamp
            a = 0  # limit ask at current timestamp
            to_add = []
            mid = []  # list of [timestamp, mid]
            pa = []  # list of [timestamp, mean price to sell 1btc, mean price to sell 5btc, etc...]
            pb = []  # list of [timestamp, mean price to buy 1btc, mean price to buy 5btc, etc...]
            flag = True  # is the cumamount lower than the threshold x ?
            ask = False  # Defines what mean price we are calculating
            for line in f:
                line_split = line.split(',')

                # for a given timestamp, this condition is only true when we first encounter that timestamp in BID side
                # we re-initialize all the variables but the outputs at each new timestamp
                if ts != line_split[0]:
                    if to_add != []:
                        if ask:  # if there was an ask side then to_add should be added to pa
                            pa += [[int(ts)] + to_add + (len(thresholds) - len(to_add)) * [np.nan]]
                        else:  # else it should be added to pb
                            pb += [[int(ts)] + to_add + (len(thresholds) - len(to_add)) * [np.nan]]
                    ts = line_split[0]  # update the timestamp
                    b = float(line_split[2])  # the first line for each new timestamp is the limit bid
                    ask = False  # we work on the bid side
                    # reinitialize
                    it = iter(thresholds)  # initialize the iterator
                    x = next(it)  # x gets the first value of the iterator
                    flag = True  # cumamount=0 lower than any threshold
                    s = 0
                    mt = 0
                    to_add = []

                # for a given timestamp, this condition is only true when we first encounter that timestamp in ASK side
                # we re-initialize all the variables but the outputs each time we move from bids to asks
                if not ask and line_split[1] == 'a':
                    if to_add != []:
                        pb += [[int(ts)] + to_add + (len(thresholds) - len(to_add)) * [np.nan]]
                    ask = True  # we are now working on the ask side
                    a = float(line_split[2])
                    mid += [[int(ts), (a + b) / 2]]
                    # reinitialize
                    it = iter(thresholds)  # initialize the iterator
                    x = next(it)  # x gets the first value of the iterator
                    flag = True  # cumamount=0 lower than any threshold
                    s = 0
                    mt = 0
                    to_add = []

                f3 = float(line_split[3])
                f2 = float(line_split[2])
                s += f3
                while s >= x and flag:
                    to_add += [(mt + (x - s + f3) * f2) / x]
                    try:
                        x = next(it)
                    except:
                        flag = False
                mt += f3 * f2
            pa += [[int(ts)] + to_add + (len(thresholds) - len(to_add)) * [np.nan]]
        return mid, pb, pa

    mid, pb, pa = get_mid_spread_from_file(file)
    mid = pd.DataFrame(mid, columns=['date', 'mid'])
    columnnames = ['date'] + ['sp' + str(th) for th in thresholds]
    pb = pd.DataFrame(pb, columns=columnnames)
    columnnames_a = ['date'] + ['sp' + str(th) + 'a' for th in thresholds]
    pa = pd.DataFrame(pa, columns=columnnames_a)
    sb = mid.merge(pb.merge(pa, on='date', how='outer'), on='date', how='outer')
    for x in thresholds:
        sb['sp' + str(x)] = sb.mid - sb['sp' + str(x)]
        sb['sp' + str(x) + 'a'] = sb['sp' + str(x) + 'a'] - sb.mid
    sb.drop('mid', axis=1, inplace=True)
    sb['date'] = sb['date'].map(get_closest_minute)
    sb = sb.groupby('date').mean().reset_index()
    sb['exchange'] = xch
    return sb


def get_sb_from_json(df, xch, weighted=True, thresholds=[1, 5, 10, 50, 100, 500, 1000]):
    """ 
    returns spreads in the orderbook aggregated by minute from dump-like orderbooks
    
    df : raw ob
    
    xch : exchange name
    
    weighted : if True, use weighted spread (default = True) 
    
    thresholds : list of numeric thresholds (default = [1,5,10,50,100,500,1000])
    """

    def get_spread(t, x):
        try:
            s = 0
            for i in range(len(t)):
                s += t[i][1]
                if s >= x:
                    r = t[i][0]
                    return r
            return np.nan
        except:
            return np.nan

    def get_spread_weighted(t, x):
        try:
            s = 0
            mt = 0
            for i in range(len(t)):
                s += t[i][1]
                if s >= x:
                    mt += (x - s + t[i][1]) * t[i][0]
                    r = mt / x
                    return r
                mt += t[i][1] * t[i][0]
            return np.nan
        except:
            return np.nan

    def get_mid(df):
        mid = []
        for i in range(len(df)):
            try:
                mid += [(df.loc[i, 'bids'][0][0] + df.loc[i, 'asks'][0][0]) / 2]
            except:
                mid += [np.nan]
        return mid

    mid = get_mid(df)
    sb = pd.DataFrame(df.ts)
    sb.ts = sb.ts.map(get_closest_minute)
    for x in thresholds:
        if weighted:
            sb['sp' + str(x)] = df['bids'].map(lambda t: get_spread_weighted(t, x))
            sb['sp' + str(x)] = mid - sb['sp' + str(x)]
            sb['sp' + str(x) + 'a'] = df['asks'].map(lambda t: get_spread_weighted(t, x))
            sb['sp' + str(x) + 'a'] = sb['sp' + str(x) + 'a'] - mid
        else:
            sb['sp' + str(x)] = df['bids'].map(lambda t: get_spread(t, x))
            sb['sp' + str(x)] = mid - sb['sp' + str(x)]
            sb['sp' + str(x) + 'a'] = df['asks'].map(lambda t: get_spread(t, x))
            sb['sp' + str(x) + 'a'] = sb['sp' + str(x) + 'a'] - mid
    sb = sb.groupby('ts').mean().reset_index()
    sb['exchange'] = xch

    return sb


# ==============================================================================
# functions to aggregate minute data from spread and volume book into hours,
# days and weeks
# ==============================================================================

def getDay(t):
    return dt.datetime(t.year, t.month, t.day)


def getHour(t):
    return dt.datetime(t.year, t.month, t.day, t.hour)


def getMonth(t):
    return dt.datetime(t.year, t.month, 1)


def aggregate_book(ob, resolution):
    resolutions = {'day': getDay, 'hour': getHour, 'month': getMonth, 'week': get_monday}
    df = ob.copy()
    df['date'] = df['date'].apply(resolutions[resolution])
    return df.groupby(['exchange', 'date']).mean().reset_index()


# ==============================================================================
# functions to check the validity of the spread books
# ==============================================================================

def check_sb_validity(df):
    sb = df.copy()
    sb.fillna(500, inplace=True)
    thresholds = ['1', '5', '10', '50', '100', '500', '1000']
    for i in range(len(thresholds) - 1):
        if not (sb['sp' + thresholds[i + 1]].values >= sb['sp' + thresholds[i]].values).all():
            return False
        if not (sb['sp' + thresholds[i + 1] + 'a'].values >= sb['sp' + thresholds[i] + 'a'].values).all():
            return False
    return True


def correct_sb(sb):
    corrected = []
    thresholds = [str(y) for y in
                  sorted([int(x.split('sp')[1]) for x in list(sb.columns) if 'sp' in x and 'a' not in x])]

    for i in range(len(sb)):
        # bid side
        for j in range(len(thresholds) - 1):
            if np.isnan(sb.loc[i, 'sp' + thresholds[j]]):
                corrected += [i]
                for k in range(j + 1, len(thresholds)):
                    print('correcting ' + str(i) + ', ' + thresholds[k])
                    sb.loc[i, 'sp' + thresholds[k]] = np.nan
                break
            if sb.loc[i, 'sp' + thresholds[j]] > sb.loc[i, 'sp' + thresholds[j + 1]]:
                corrected += [i]
                for k in range(j + 1, len(thresholds)):
                    print('correcting ' + str(i) + ', ' + thresholds[k])
                    sb.loc[i, 'sp' + thresholds[k]] = np.nan
                break

        # ask side
        for j in range(len(thresholds) - 1):
            if np.isnan(sb.loc[i, 'sp' + thresholds[j] + 'a']):
                corrected += [i]
                for k in range(j + 1, len(thresholds)):
                    print('correcting ' + str(i) + ', ' + thresholds[k] + 'a')
                    sb.loc[i, 'sp' + thresholds[k] + 'a'] = np.nan
                break
            if sb.loc[i, 'sp' + thresholds[j] + 'a'] > sb.loc[i, 'sp' + thresholds[j + 1] + 'a']:
                corrected += [i]
                for k in range(j + 1, len(thresholds)):
                    print('correcting ' + str(i) + ', ' + thresholds[k] + 'a')
                    sb.loc[i, 'sp' + thresholds[k] + 'a'] = np.nan
                break
    return corrected