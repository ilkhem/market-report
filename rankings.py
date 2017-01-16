# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:02:54 2016

@author: ilyes
"""

__author__ = 'ilkhem'

import datetime as dt
import json
import urllib.request

import numpy as np
import pandas as pd

from utils import get_dates, nanget, parse_month


# utility function
def get_v(df, col, t, xch):
    try:
        return df[(df['date'] == t) & (df['exchange'] == xch)][col].iloc[0]
    except:
        return np.nan


def weight_ranking(index, from_ts, to_ts, xchanges, symbols):
    """
    returns the exchanges ranking by weights in kaiko's index, and the data used to calculate the ranking

    JAPANESE INDEX INSTANTLY RETURNS EMPTY DATAFRAMES

    :param index: kaiko's price index
    :param from_ts: starting timestamp in seconds
    :param to_ts: finish tiemstamop in seconds
    :param xchanges: config file with the desired exchanges
    :param symbols: config file with the exchanges' symbols (slugs)
    :return: weight_reanking, weight_data
    """
    if to_ts == 'today':
        to_ts = int(dt.datetime.timestamp(dt.datetime.today()))
    if index == 'jp':
        return pd.DataFrame(), pd.DataFrame()
    url = "https://api.kaiko.com/v1/history/indices/weights?indices=" + index + "&from=" + str(from_ts) + "&to=" + str(
        to_ts) + "&resolution=day"
    js = json.loads(urllib.request.urlopen(url).read().decode("utf-8"))[index]
    weights = pd.DataFrame()
    for x in js:
        temp = pd.DataFrame(x['weights'])
        for xch in temp.exchange.unique():
            try:
                xch = symbols[xch]
            except:
                temp.drop(temp[temp['exchange'] == xch].index, axis=0, inplace=True)
        temp['exchange'] = temp['exchange'].map(lambda y: symbols[y])
        temp['timestamp'] = x['timestamp']
        weights = weights.append(temp, ignore_index=True)
    weights['date'] = weights['timestamp'].map(dt.datetime.utcfromtimestamp)
    weights['date'] = weights['date'].map(lambda x: dt.datetime(x.year, x.month, 1))
    weights['weight'] = weights['weight'].map(float)
    weights.drop('timestamp', axis=1, inplace=True)
    weights = weights.groupby(['exchange', 'date']).mean().reset_index()
    ranking = pd.DataFrame({'date': [], 'ranking': []})
    data = pd.DataFrame({'date': [], 'v': []})
    for t in weights['date'].unique():
        rk = list(weights[weights['date'] == t].sort_values('weight', ascending=False).exchange)
        rk = {xch: rk.index(xch) + 1 for xch in rk}
        v = {xch: get_v(weights, 'weight', t, xch) for xch in xchanges}
        ranking.loc[len(ranking)] = {'date': t, 'ranking': rk}
        data.loc[len(data)] = {'date': t, 'v': v}
    for xch in xchanges:
        ranking[xch] = ranking['ranking'].map(lambda x: nanget(x, xch))
        data[xch] = data['v'].map(lambda x: nanget(x, xch))
    ranking.drop('ranking', axis=1, inplace=True)
    data.drop('v', axis=1, inplace=True)
    return ranking.sort_values('date', ascending=True), data.sort_values('date', ascending=True)


def volume_ranking(pair, from_ts, to_ts, xchanges, exchanges_api, symbols):
    """
    returns the exchanges ranking by monthly traded volume
    :param pair: pair to be processed
    :param from_ts: starting timestamp
    :param to_ts: ending timestamp
    :param xchanges: list of exchanges' slugs to be processed
    :param exchanges_api: string of exchanges to be used in the api call
    :param symbols: dict converting exchange names to slugs
    :return: volume ranking, volume data
    """
    if to_ts == 'today':
        to_ts = int(dt.datetime.timestamp(dt.datetime.today()))
    url = "https://api.kaiko.com/v1/history/exchanges?exchanges=" + exchanges_api + "&pairs=" + pair + "&from=" + str(
        from_ts) + "&to=" + str(to_ts) + "&resolution=day&fields=v"
    volumes = pd.DataFrame({'timestamp': [], 'v': [], 'exchange': []})
    js = json.loads(urllib.request.urlopen(url).read().decode("utf-8"))
    for x in js:
        temp = pd.DataFrame(js[x][pair])
        temp['exchange'] = symbols[x]
        volumes = volumes.append(temp, ignore_index=True)
    volumes['date'] = volumes['timestamp'].map(dt.datetime.utcfromtimestamp)
    volumes['date'] = volumes['date'].map(lambda x: dt.datetime(x.year, x.month, 1))
    volumes['v'] = volumes['v'].map(float)
    volumes.drop('timestamp', axis=1, inplace=True)
    volumes = volumes.groupby(['exchange', 'date']).sum().reset_index()
    ranking = pd.DataFrame({'date': [], 'ranking': []})
    data = pd.DataFrame({'date': [], 'v': []})
    for t in volumes.date.unique():
        rk = list(volumes[volumes['date'] == t].sort_values('v', ascending=False).exchange)
        rk = {xch: rk.index(xch) + 1 for xch in rk}
        v = {xch: get_v(volumes, 'v', t, xch) for xch in xchanges}
        ranking.loc[len(ranking)] = {'date': t, 'ranking': rk}
        data.loc[len(data)] = {'date': t, 'v': v}
    for xch in xchanges:
        ranking[xch] = ranking['ranking'].map(lambda x: nanget(x, xch))
        data[xch] = data['v'].map(lambda x: nanget(x, xch))
    ranking.drop('ranking', axis=1, inplace=True)
    data.drop('v', axis=1, inplace=True)
    return ranking.sort_values('date', ascending=True), data.sort_values('date', ascending=True)


def liquidity_ranking(directory, pair, from_ts, to_ts, xchanges):
    """
    returns the exchanges ranking by monthly mean liquidity in ob_1
    :param directory: directory for volume books
    :param pair: pair to be processed
    :param from_ts: starting timestamp
    :param to_ts: ending timestamp
    :param xchanges: slug of exchanges to be processed
    :return: liquidity rankingm liquidity data
    """
    if directory != '' and directory[-1] != '/':
        directory += '/'
    dates = get_dates(from_ts, to_ts)

    liquidities = pd.concat([pd.read_csv(
        directory + 'vb_' + pair + '_' + str(d.year) + '_' + parse_month(d.month) + '.csv', parse_dates=['date']) for d
                             in dates], ignore_index=True)
    liquidities['liquidity'] = liquidities['ask_volume'] + liquidities['bid_volume']
    liquidities.drop(['ask_volume', 'bid_volume'], axis=1, inplace=True)

    liquidities = liquidities[liquidities['date'] >= dt.datetime.utcfromtimestamp(from_ts)]
    liquidities = liquidities[liquidities['date'] <= dt.datetime.utcfromtimestamp(to_ts)]

    liquidities['date'] = liquidities['date'].map(lambda x: dt.datetime(x.year, x.month, 1))
    liquidities = liquidities.groupby(['exchange', 'date']).mean().reset_index()
    ranking = pd.DataFrame({'date': [], 'ranking': []})
    data = pd.DataFrame({'date': [], 'v': []})
    for t in liquidities['date'].unique():
        rk = list(liquidities[liquidities['date'] == t].sort_values('liquidity', ascending=False).exchange)
        rk = {xch: rk.index(xch) + 1 for xch in rk}
        v = {xch: get_v(liquidities, 'liquidity', t, xch) for xch in xchanges}
        ranking.loc[len(ranking)] = {'date': t, 'ranking': rk}
        data.loc[len(data)] = {'date': t, 'v': v}
    for xch in xchanges:
        ranking[xch] = ranking['ranking'].map(lambda x: nanget(x, xch))
        data[xch] = data['v'].map(lambda x: nanget(x, xch))
    ranking.drop('ranking', axis=1, inplace=True)
    data.drop('v', axis=1, inplace=True)
    return ranking.sort_values('date', ascending=True), data.sort_values('date', ascending=True)
