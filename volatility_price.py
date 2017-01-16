# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:02:25 2016

@author: ilyes
"""
__author__ = 'ilkhem'

import datetime as dt
import json
import math
import os
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
from dateutil import rrule

from cassy import create_session
from config import config
from utils import to_float, get_monday, str_month, get_dates, save_plot


def get_mtgox_data(directory, currency):
    """
    reads mtgox data from dumps
    :param currency: the currency te be read (usd, cny, eur)
    :param directory: data directory of mtgox files
    :return: mtgox data in pandas dataframe corresponding to the current pair
    """
    if directory != '' and directory[-1] != '/':
        directory += '/'
    print('getting %s mtgox data ...' % currency)
    filename = '.mtgox' + currency.upper() + '.csv'
    try:
        df = pd.read_csv(directory + filename, header=None, names=['timestamp', 'price', 'amount'])
    except:
        print('MtGox data not found')
        return None
    df.timestamp = df.timestamp.map(dt.datetime.utcfromtimestamp)
    df = df[df['timestamp'] < dt.datetime(2014, 1, 1)]
    df['day'] = df['timestamp'].map(lambda x: dt.datetime(x.year, x.month, x.day))
    df = df[['day', 'price']].groupby('day').mean().reset_index().rename(columns={'day': 'timestamp', 'price': 'c'})
    return df


def get_price_index(index, from_ts, to_ts, resolution, currency=None, convert_currency=True, mtgox=None):
    """
    returns the price index from kaiko's api
    :param index: index
    :param from_ts: starting timestamp
    :param to_ts: finish timestamp
    :param resolution: resolution. {'day','week','hour'}
    :param convert_currency: Default True. converts currency to match the index
    :param mtgox: mtgox dataframe. Default None
    :return: price index in a dataframe. columns are : timestamp, c
    """
    print('getting %s price index ...' % index)
    if to_ts == 'today':
        to_ts = int(dt.datetime.timestamp(dt.datetime.today()))
    if currency == 'btc' or not convert_currency or currency is None:
        url = "https://api.kaiko.com/v1/history/indices?indices=" + index + "&from=" + str(from_ts) + "&to=" + str(
            to_ts) + "&resolution=" + resolution
    else:
        url = "https://api.kaiko.com/v1/history/indices?indices=" + index + "&from=" + str(from_ts) + "&to=" + str(
            to_ts) + "&resolution=" + resolution + "&currency=" + currency
    kpi = pd.DataFrame(json.loads(urllib.request.urlopen(url).read().decode("utf-8"))[index])
    kpi['timestamp'] = kpi['timestamp'].map(lambda x: dt.datetime.utcfromtimestamp(x))
    kpi.drop(['v', 'o', 'l', 'h'], inplace=True, axis=1)
    kpi['c'] = kpi['c'].map(to_float)
    try:
        kpi = kpi.append(mtgox[mtgox['timestamp'] >= dt.datetime.utcfromtimestamp(from_ts)]).groupby(
            'timestamp').mean().reset_index().sort_values('timestamp', ascending=False)
    except TypeError:
        print('No/Erroneous MtGox data')
    # kpi.sort_values('date',inplace=True)
    return kpi


def get_volatility(index, from_ts, to_ts, resolution, currency, convert_currency=True, mtgox=None):
    """
    returns the volatility of the kaiko price index
    :param index: index
    :param from_ts: starting timestamp
    :param to_ts: finish timestamp
    :param resolution: resolution. {'day','week','hour'}
    :param convert_currency: Default True. converts currency to match the index
    :param mtgox: mtgox dataframe. Default None
    :return: volatility in a dataframe. columns are : timestamp, v
    """
    print('getting %s volatility ...' % index)
    if to_ts == 'today':
        to_ts = int(dt.datetime.timestamp(dt.datetime.today()))

    from_ts = int(dt.datetime.timestamp(dt.datetime.utcfromtimestamp(from_ts) - dt.timedelta(days=31)))
    kpi = get_price_index(index, from_ts, to_ts, resolution, currency, convert_currency, mtgox)
    kpi.reset_index(drop=True, inplace=True)
    kpi['c1'] = kpi['c'].shift(-1)
    kpi.drop(kpi.index[-1], axis=0, inplace=True)

    #    kpi['r'] = (kpi['c'] / kpi['c1']) -1
    kpi['r'] = kpi['c'].map(math.log) - kpi['c1'].map(math.log)

    for i in range(len(kpi) - 30):
        kpi.loc[i, 'v'] = np.std(kpi.loc[i:i + 30, 'r']) * 100
    kpi = kpi.loc[:len(kpi) - 31]
    #    kpi.sort_values('date',inplace=True)
    return kpi[['timestamp', 'v']]


def get_historical_prices_from_db(pair, from_ts, to_ts, xchanges, currency):
    """
    :param pair: pair
    :param from_ts: starting date timestamp (int)
    :param to_ts: finish date timestamp (int)
    :return: dataframe with the price of each exchange and the price index
    """
    print('getting historical prices from database for %s ...' % pair)
    session = create_session('read')  # create cassandra session
    #    query_start = dt.datetime(2015,12,6,23) #start time for the query, sunday at 23
    query_start = get_monday(dt.datetime.utcfromtimestamp(from_ts), 23) + dt.timedelta(days=6)
    #    query_end = dt.datetime(2016,6,5,23)
    query_end = get_monday(dt.datetime.utcfromtimestamp(to_ts), 23) + dt.timedelta(days=6)
    date_hours = list(rrule.rrule(rrule.WEEKLY, dtstart=query_start, until=query_end))

    prices = pd.DataFrame({'date': []})
    for dh in date_hours:
        # print(dh - dt.timedelta(days=6))
        query = session.execute(
            "select weights, date, latest_best_bid ,latest_worst_ask, latest_price, price from price_indices_2 where slug = 'regional_price_" + currency + "' and day_hour = '" + str(
                dh) + "';", timeout=60)
        df = pd.DataFrame(query.current_rows, columns=query.column_names)
        df = df[df.index == 0]
        temp = df[['date', 'price']].copy()
        for xch in xchanges:
            try:
                if (df['latest_best_bid'].map(lambda x: x[xch + '_' + pair]).loc[0] != 0) and (
                            df['latest_worst_ask'].map(lambda x: x[xch + '_' + pair]).loc[0] != 0):
                    p = (df['latest_best_bid'].map(lambda x: x[xch + '_' + pair]) + df['latest_worst_ask'].map(
                        lambda x: x[xch + '_' + pair])) / 2
                else:
                    p = df['latest_price'].map(lambda x: x[xch + '_' + pair])
                temp[xch] = p
            except:
                temp[xch] = np.nan
        prices = prices.append(temp)
    prices['date'] = prices['date'].map(get_monday)
    return prices


def get_historical_prices(pair, from_ts, to_ts, exchanges_api, symbols, resolution):
    """deprecated"""
    print('getting historical prices ...')
    if to_ts == 'today':
        to_ts = int(dt.datetime.timestamp(dt.datetime.today()))

    url2 = "https://api.kaiko.com/v1/history/exchanges?exchanges=" + exchanges_api + "&pairs=" + pair + "&from=" + str(
        from_ts) + "&to=" + str(to_ts) + "&resolution=" + resolution
    historical_price = pd.DataFrame({'c': [], 'h': [], 'l': [], 'o': [], 'timestamp': [], 'v': []})
    js = json.loads(urllib.request.urlopen(url2).read().decode("utf-8"))
    for x in js:
        temp = pd.DataFrame(js[x][pair])
        temp['exchange'] = symbols[x]
        historical_price = historical_price.append(temp, ignore_index=True)
    historical_price['timestamp'] = historical_price['timestamp'].map(lambda x: dt.datetime.utcfromtimestamp(x))
    historical_price['c'] = pd.to_numeric(historical_price['c'])
    return historical_price[['timestamp', 'exchange', 'c']]


def get_price_std_deviation(pair, from_ts, to_ts, xchanges, exchanges_api, symbols, resolution):
    """
    Computes the standard deviation between the different exchanges for a given timestamp
    :param pair: pair
    :param from_ts: starting timestamp
    :param to_ts: finish timestamp
    :param resolution: resolution. {'week','day','hour'}
    :return: the standard deviation data frame. columns : timestamp, std_dev
    """
    print('getting price standard deviation for %s ...' % pair)
    if to_ts == 'today':
        to_ts = int(dt.datetime.timestamp(dt.datetime.today()))

    url2 = "https://api.kaiko.com/v1/history/exchanges?exchanges=" + exchanges_api + "&pairs=" + pair + "&from=" + str(
        from_ts) + "&to=" + str(to_ts) + "&resolution=" + resolution
    historical_price = pd.DataFrame({'c': [], 'h': [], 'l': [], 'o': [], 'timestamp': [], 'v': []})
    js = json.loads(urllib.request.urlopen(url2).read().decode("utf-8"))

    for x in js:
        temp = pd.DataFrame(js[x][pair])
        temp['exchange'] = symbols[x]
        historical_price = historical_price.append(temp, ignore_index=True)
    historical_price['timestamp'] = historical_price['timestamp'].map(lambda x: dt.datetime.utcfromtimestamp(x))
    historical_price['c'] = pd.to_numeric(historical_price['c'])

    try:
        a = (historical_price['exchange'] == 'cb') & (historical_price['timestamp'] < dt.datetime(2015, 5, 1))
        historical_price.loc[a, 'c'] = np.nan
    except:
        pass

    X = pd.DataFrame(list(set(historical_price['timestamp'])), columns=['timestamp']).sort_values('timestamp')
    for xch in xchanges:
        X[xch] = [np.nan] * (
            len(X) - len(
                historical_price[historical_price['exchange'] == xch].sort_values('timestamp').c.values)) + list(
            historical_price[historical_price['exchange'] == xch].sort_values('timestamp').c.values)
    X['std_dev'] = np.nanstd(X.drop('timestamp', axis=1).values, axis=1)
    return X[['timestamp', 'std_dev']].sort_values('timestamp', ascending=False)


def plot_price_deviation(prices, index, pair, xchanges, names, colors, from_ts, to_ts, plotly_directory='',
                         auto_open_charts=True, P_DIR='plots/'):
    _, start, end = get_dates(from_ts, to_ts, parsed=True)

    trace = [go.Scatter(
        x=prices.date,
        y=100 * (prices[xch] - prices.price) / prices.price,
        mode='lines',
        opacity=0.9,
        name=names[xch],
        marker=dict(color=colors[xch])) for xch in xchanges]

    title = 'Price deviation from Kaiko ' + index.upper() + ' Price Index from ' + str_month(start[1]) + ' ' + start[
        0] + ' to ' + str_month(end[1]) + ' ' + end[0] + ' (' + pair.upper() + ')'
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Deviation (%)',
        ),

    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, filename=plotly_directory + pair + '-price-deviation', auto_open=auto_open_charts, sharing='private')
    save_plot(fig, P_DIR, title.lower())


def plot_std_dev(std_dev, kpi, index, pair, currency, plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    trace = [
        go.Scatter(x=std_dev.timestamp, y=std_dev.std_dev, mode='lines', opacity=0.8, name='Price standard deviation'),
        go.Scatter(x=kpi.timestamp, y=kpi.c, mode='lines', opacity=0.8, name='Kaiko ' + index.upper() + ' Price Index',
                   line=dict(dash='dash'), yaxis='y2')
    ]

    title = 'Price standard deviation (' + pair.upper() + ')'
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Standard deviation (%)',
            #            showlegend=False,
            rangemode='tozero'
        ),
        yaxis2=dict(
            title='Price (' + currency.upper() + ')',
            overlaying='y',
            side='right',
            showgrid=False,
            rangemode='tozero'
        )
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, filename=plotly_directory + pair + '-price-standard-deviation', auto_open=auto_open_charts,
            sharing='private')
    save_plot(fig, P_DIR, title.lower())


def plot_volatility(vol, kpi, index, pair, currency, plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    trace = [
        go.Scatter(x=vol.timestamp, y=vol.v, mode='lines', opacity=0.8, name='30-day ' + pair.upper() + ' Volatility'),
        go.Scatter(x=kpi.timestamp, y=kpi.c, mode='lines', opacity=0.8, name='Kaiko ' + index.upper() + ' Price Index',
                   line=dict(dash='dash'), yaxis='y2')
    ]

    title = 'Bitcoin volatility (' + pair.upper() + ')'
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Volatility (%)',
            rangemode='tozero'
        ),
        yaxis2=dict(
            title='Price (' + currency.upper() + ')',
            overlaying='y',
            side='right',
            showgrid=False,
            rangemode='tozero'
        ),
        #            showlegend=False,
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, filename=plotly_directory + pair + '-volatility', auto_open=auto_open_charts, sharing='private')
    save_plot(fig, P_DIR, title.lower())


def plot_std_dev_volatility(std_dev, vol, pair, plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    trace = [
        go.Scatter(x=std_dev.timestamp, y=std_dev.std_dev, mode='lines', opacity=0.77, name='Price standard deviation')]

    trace += [
        go.Scatter(x=vol.timestamp, y=vol.v, mode='lines', opacity=0.77, name='30-day ' + pair.upper() + ' Volatility',
                   yaxis='y2')]

    title = 'Bitcoin volatility vs Standard deviation (' + pair.upper() + ')'
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Standard deviation (%)',
            rangemode='tozero'
        ),
        yaxis2=dict(
            title='Volatility (%)',
            overlaying='y',
            side='right',
            showgrid=False,
            rangemode='tozero'
        )
        #            showlegend=False,
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, filename=plotly_directory + pair + '-volatility-vs-std-deviation', auto_open=auto_open_charts,
            sharing='private')
    save_plot(fig, P_DIR, title.lower())


# testing
if __name__ == '__main__':

    P_DIR = 'plots/'
    if not os.path.exists(P_DIR):
        os.makedirs(P_DIR)

    # define parameters

    pair = 'btcusd'
    from_ts = 1293836400  # 1/1/2011
    # from_ts = 1356994800 #1/1/2013
    from_ts1 = 1451606400  # 01/01/2016
    #    from_ts1 = 1462060800   # 01/05/2016 for eth***
    #    from_ts1 = 1464739200   # 01/06/2016 for btcjpy
    to_ts = 1467331199

    pair_to_index = {'btcusd': 'us', 'btceur': 'eu', 'btccny': 'cn', 'ethbtc': 'ethbtc', 'ethusd': 'ethusd',
                     'etheur': 'etheur'}
    index = pair_to_index[pair]

    plotly_directory = 'july-report/'
    auto_open_charts = True

    # Global config files
    exchanges_api = config["exchanges_string_by_pair"][pair]
    xchanges = config["exchanges_list_by_pair"][pair]
    symbols = config["symbols"]
    names = {xch: config["exchanges"][xch]["name"] for xch in xchanges}
    colors = {xch: config["exchanges"][xch]["color"] for xch in xchanges}
    currency = config["currency"][index]
    cryptocurrency = pair[:3]

    # additional variables for formatting

    #    mtgox = get_mtgox_data('E:\\data/mtgox/',currency)
    mtgox = None
    vol = get_volatility(index, from_ts, to_ts, 'day', currency, mtgox=mtgox)
    kpi = get_price_index(index, from_ts, to_ts, 'day', currency, mtgox=mtgox)
    # kpi1 = get_price_index(index,from_ts1,to_ts,'week')
    # historical_price = get_historical_prices(pair,from_ts,to_ts,'day')
    prices = get_historical_prices_from_db(pair, from_ts1, to_ts, xchanges, currency)
    std_dev = get_price_std_deviation(pair, from_ts, to_ts, xchanges, exchanges_api, symbols, 'day')

    plot_price_deviation(prices, index, pair, xchanges, names, colors, from_ts1, to_ts, plotly_directory,
                         auto_open_charts)
    plot_std_dev(std_dev, kpi, index, pair, currency, plotly_directory, auto_open_charts)
    plot_volatility(vol, kpi, index, pair, currency, plotly_directory, auto_open_charts)
    plot_std_dev_volatility(std_dev, vol, pair, plotly_directory, auto_open_charts)
