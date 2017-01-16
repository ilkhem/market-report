# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:06:09 2016

@author: ilyes
"""
__author__ = 'ilkhem'

import datetime as dt
import json
import math
import os
import urllib.request

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py

from utils import get_dates, parse_month, fmt, str_month, save_plot


def get_mean_volume_count(directory, exchange, pair, from_ts, to_ts, api_names):
    """
    get the mean, the count and volume of trades for a given exchange from from_ts to to_ts
    :param directory: directory to trade dumps
    :param exchange: exchange slug (or symbol) e.g.: bf for bitfinex
    :param pair: pair
    :param from_ts: start ts in seconds
    :param to_ts: end timestamp in seconds
    :param api_names: exchange names as in the api (for seeking local files by name)
    :return: the mean (v/n), the volume (v), the count (n)
    """
    print('getting mean, volume and count of trades for %s %s ...' % (exchange, pair))
    dates = get_dates(from_ts, to_ts)
    filepaths = [directory + '/' + pair + '/' + str(d.year) + '/' + parse_month(d.month) + '/' + api_names[
        exchange].lower() + '/' for d in dates]

    n = 0
    v = 0
    for filepath in filepaths:
        # print(filepath)
        for dirpath, dirnames, files in os.walk(filepath):
            for file in files:
                with open(filepath + file) as f:
                    f.readline()
                    for line in f:
                        n += 1
                        v += float(line.split(',')[5])
    if n == 0:
        return 0, 0, 0
    return v / n, v, n


def get_volumes(pair, from_ts, to_ts, resolution, exchanges_api, symbols):
    """
    Get traded volume per exchange using kaiko's api
    :param pair: pair to be processed
    :param from_ts: start timestamp
    :param to_ts: finish timestamp
    :param resolution: resolution for the api call
    :param exchanges_api: list of exchanges for the api exchanges parameters; defined in .cfg file
    :param symbols: dict of symbols (slugs) for the exchanges; defined in .cfg
    :return: volumes dataframe. columns: date, exchange, v
    """
    print('getting total traded volumes for %s ...' % pair)
    if to_ts == 'today':
        to_ts = int(dt.datetime.timestamp(dt.datetime.today()))

    url2 = "https://api.kaiko.com/v1/history/exchanges?exchanges=" + exchanges_api + "&pairs=" + pair + "&from=" + str(
        from_ts) + "&to=" + str(to_ts) + "&resolution=" + resolution
    historical_price = pd.DataFrame({'c': [], 'h': [], 'l': [], 'o': [], 'timestamp': [], 'v': []})
    js = json.loads(urllib.request.urlopen(url2).read().decode("utf-8"))
    for x in js:
        temp = pd.DataFrame(js[x][pair])
        temp['exchange'] = symbols[x]
        temp.sort_values('timestamp', inplace=True, ascending=False)
        historical_price = historical_price.append(temp, ignore_index=True)
    historical_price['date'] = historical_price['timestamp'].map(lambda x: dt.datetime.utcfromtimestamp(x))
    historical_price['c'] = pd.to_numeric(historical_price['c'])
    historical_price['v'] = pd.to_numeric(historical_price['v'])
    historical_price = historical_price[['date', 'exchange', 'v']]
    volumes = historical_price.groupby(['exchange', 'date']).sum().reset_index()

    return volumes


def get_volumes_per_bucket(directory, exchange, pair, from_ts, to_ts, api_names, regroup_small_values=False):
    """
    return the trades volume per bucket for a given pair/exchange.
    buckets are hardcoded in bucket (for btc) and bucket_regroup(for eth)
    buckets are [10e-06, ..., 10e03] for btc
    buckets are [10e-03, ..., 10e04] for eth
    :param directory: directory to trade dumps
    :param exchange: exchange
    :param pair: pair
    :param from_ts: start ts in seconds
    :param to_ts: end ts in seconds
    :param api_names: exchange names as in the api (for seeking local files by name)
    :param regroup_small_values: True if need to use regrouped buckets (eth buckets)
    :return:
    """
    if directory != '' and directory[-1] != '/':
        directory += '/'

    def bucket(x):
        if (x != 0):
            b = math.floor(math.log10(x))
        else:
            b = -6
        if b > 3:
            b = 3
        if b < -6:
            b = -6
        return b + 6

    def bucket_regroup(x):
        if (x != 0):
            b = math.floor(math.log10(x))
        else:
            b = -3
        if b > 4:
            b = 4
        if b < -3:
            b = -3
        return b + 3

    print('getting traded volume per bucket for %s %s ...' % (exchange, pair))

    if regroup_small_values:
        xch6 = [0, 0, 0, 0, 0, 0, 0, 0]
        xchvolumes6 = [0, 0, 0, 0, 0, 0, 0, 0]
        bckt = bucket_regroup
    else:
        xch6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        xchvolumes6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bckt = bucket

    dates = get_dates(from_ts, to_ts)
    filepaths = [
        directory + pair + '/' + str(d.year) + '/' + parse_month(d.month) + '/' + api_names[exchange].lower() + '/' for
        d in dates]

    for filepath in filepaths:
        print(filepath)
        for dirpath, dirnames, files in os.walk(filepath):
            for file in files:
                with open(filepath + file) as f:
                    f.readline()
                    for line in f:
                        try:
                            xch6[bckt(float(line.split(',')[5]))] += 1
                            xchvolumes6[bckt(float(line.split(',')[5]))] += float(line.split(',')[5])
                        except ValueError:
                            pass

    return xch6, xchvolumes6


def read_trades(directory, exchange, pair, from_ts, to_ts, api_names, threshold=500, get_median=False):
    """
    get big_trades, list of trade amounts, and median_prices per exchange by reading the dumps of trades
    :param directory: trades dump directory
    :param exchange: slug of the exchange to be processed
    :param pair: pair to be processed
    :param from_ts: starting timestamp
    :param to_ts: ending timestamp
    :param api_names: dict converting slugs to full names
    :param threshold: threshold for big trades
    :param get_median: if True, returns the median, else return only big trades and list of trades' values
    :return: big_trades, list of trades' values for 6m, list of trades' values for 1m, [if get_median==True: median6m, median1m]
    """
    if directory != '' and directory[-1] != '/':
        directory += '/'
    print('reading trades for %s %s ...' % (exchange, pair))
    if to_ts == 'today':
        to_ts = int(dt.datetime.timestamp(dt.datetime.today()))

    big_trades = pd.DataFrame({'exchange': [], 'date': [], 'amount': []})
    df = pd.DataFrame({'amount': []})
    df6 = pd.DataFrame({'amount': []})

    dates, _, end = get_dates(from_ts, to_ts, parsed=True)
    filepaths = [
        directory + pair + '/' + str(d.year) + '/' + parse_month(d.month) + '/' + api_names[exchange].lower() + '/' for
        d in dates]

    for filepath in filepaths:
        print(filepath)
        for dirpath, dirnames, files in os.walk(filepath):
            for file in files:
                temp = pd.read_csv(filepath + file, usecols=['exchange', 'date', 'amount'], na_values='null')
                temp2 = temp[temp['amount'] >= threshold].copy()
                big_trades = big_trades.append(temp2, ignore_index=True)
                temp2 = None
                temp = pd.read_csv(filepath + file, usecols=['amount'], na_values='null')
                if filepath.split('/')[-3] == end[1]:
                    df = df.append(temp, ignore_index=True)
                df6 = df6.append(temp, ignore_index=True)
    big_trades['date'] = big_trades['date'].map(lambda x: dt.datetime.utcfromtimestamp(x / 1000))
    big_trades.sort_values('date', inplace=True, ascending=False)
    big_trades = big_trades[big_trades['date'] >= dt.datetime.utcfromtimestamp(from_ts)]
    big_trades = big_trades[big_trades['date'] <= dt.datetime.utcfromtimestamp(to_ts)]

    if get_median:
        return big_trades, list(df6.amount), list(df.amount), df6.amount.median(), df.amount.median()
    else:
        return big_trades, list(df6.amount), list(df.amount)


def get_big_trades(directory, exchange, pair, from_ts, to_ts, api_names, threshold=500):
    """
    returns only big_trades from reading the dumps
    :param directory: driectory for trades' dumps
    :param exchange: slug of exchange to be processed
    :param pair: pair to be processed
    :param from_ts: starting timestamp
    :param to_ts: ending timestaamp
    :param api_names: dict cinverting slugs to full names
    :param threshold: threshold for big trades
    :return: datafrane of big trades. columns: exchange, date, amount
    """
    if directory != '' and directory[-1] != '/':
        directory += '/'
    print('reading trades  for %s %s ...' % (exchange, pair))
    if to_ts == 'today':
        to_ts = int(dt.datetime.timestamp(dt.datetime.today()))

    big_trades = pd.DataFrame({'exchange': [], 'date': [], 'amount': []})

    dates, _, end = get_dates(from_ts, to_ts, parsed=True)
    filepaths = [
        directory + pair + '/' + str(d.year) + '/' + parse_month(d.month) + '/' + api_names[exchange].lower() + '/' for
        d in dates]

    for filepath in filepaths:
        print(filepath)
        for dirpath, dirnames, files in os.walk(filepath):
            for file in files:
                temp = pd.read_csv(filepath + file, usecols=['exchange', 'date', 'amount'], na_values='null')
                temp2 = temp[temp['amount'] >= threshold].copy()
                big_trades = big_trades.append(temp2, ignore_index=True)
                temp2 = None
    big_trades['date'] = big_trades['date'].map(lambda x: dt.datetime.utcfromtimestamp(x / 1000))
    big_trades.sort_values('date', inplace=True, ascending=False)
    big_trades = big_trades[big_trades['date'] >= dt.datetime.utcfromtimestamp(from_ts)]
    big_trades = big_trades[big_trades['date'] <= dt.datetime.utcfromtimestamp(to_ts)]

    return big_trades


def plot_volumes_price(volumes, kpi, index, pair, xchanges, names, colors, currency, cryptocurrency, plot_kpi=True,
                       plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    trace = [
        go.Bar(x=volumes[volumes['exchange'] == xch]['date'], y=volumes[volumes['exchange'] == xch]['v'], opacity=0.8,
               name=names[xch], marker=dict(color=colors[xch])) for xch in xchanges]

    title = 'Volumes per exchange (' + pair.upper() + ')'
    layout = go.Layout(
        barmode='stack',  # stack, overlay, group
        title=title,
        xaxis=dict(
            title='',
        ),
        yaxis=dict(
            title='Volume (' + cryptocurrency.upper() + ')',
            rangemode='tozero'
        ),
        legend=dict(traceorder='normal'),
        bargap=0.25,
        bargroupgap=0.3,
    )
    if plot_kpi:
        trace += [go.Scatter(x=kpi['timestamp'], y=kpi['c'], opacity=0.8, yaxis='y2',
                             name='Kaiko ' + index.upper() + ' Price Index', marker=dict(color='#17becf'))]
        layout['yaxis2'] = dict(
            title='Price (' + currency.upper() + ')',
            overlaying='y',
            side='right',
            showgrid=False,
            rangemode='tozero'
        )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private',
            filename=plotly_directory + pair + '-total-xch-value-6m')
    save_plot(fig, P_DIR, title.lower())


def plot_market_share(volumes, pair, xchanges, names, colors, from_ts, to_ts, plotly_directory='',
                      auto_open_charts=True, P_DIR='plots/'):
    _, start, end = get_dates(from_ts, to_ts, parsed=True)

    x = pd.DataFrame(list(set(volumes['date'])), columns=['date']).sort_values('date')

    def fill_empty(x, df):
        temp = x.merge(df, on='date', how='left')
        temp['v'].fillna(0, inplace=True)
        return temp

    trace = []
    yold = [0] * len(x)
    ytotal = [np.nansum(y) for y in zip(*[fill_empty(x, volumes[volumes['exchange'] == xch])['v'] for xch in xchanges])]

    for xch in xchanges:
        y = np.divide([np.nansum(x) for x in zip(fill_empty(x, volumes[volumes['exchange'] == xch])['v'], yold)],
                      ytotal) * 100

        text = (np.divide(fill_empty(x, volumes[volumes['exchange'] == xch])['v'], ytotal) * 100).map(
            lambda x: "%0.3f" % x + '%')
        trace += [
            go.Scatter(
                x=x.date, y=y, text=text, fill='tonexty', hoverinfo='x+text+name',
                mode='lines', line=dict(width=0.5), opacity=0.8,
                name=names[xch], marker=dict(color=colors[xch])
            )
        ]
        yold = [np.nansum(x) for x in zip(fill_empty(x, volumes[volumes['exchange'] == xch])['v'], yold)]
    title = 'Exchange market share from ' + str_month(start[1]) + ' ' + start[0] + ' to ' + str_month(end[1]) + ' ' + \
            end[0] + ' (' + pair.upper() + ')'
    layout = go.Layout(
        title='Exchange market share from ' + str_month(start[1]) + ' ' + start[0] + ' to ' + str_month(end[1]) + ' ' +
              end[0] + ' (' + pair.upper() + ')',
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='',
            nticks=11,
            showgrid=True,
            ticksuffix='%',
            rangemode='tozero'
        ),
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private',
            filename=plotly_directory + pair + 'market-share-' + end[1])
    save_plot(fig, P_DIR, title.lower())


def plot_big_trades(big_trades, kpi, index, pair, threshold, xchanges, names, colors, currency, cryptocurrency,
                    plot_kpi=True, plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    trace = [go.Scatter(x=big_trades[big_trades['exchange'] == xch]['date'],
                        y=big_trades[big_trades['exchange'] == xch]['amount'], name=names[xch],
                        showlegend=True, opacity=0.8, mode='markers', marker=dict(color=colors[xch], size=11)) for xch
             in xchanges]

    title = str(threshold) + '' + cryptocurrency.upper() + '+ trades (' + pair.upper() + ')'
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title='',
        ),
        yaxis=dict(
            title='Size (' + cryptocurrency.upper() + ')',
            rangemode='tozero'
        ),
    )
    if plot_kpi:
        trace += [go.Scatter(x=kpi['timestamp'], y=kpi['c'], opacity=0.8, yaxis='y2',
                             name='Kaiko ' + index.upper() + ' Price Index', marker=dict(color='#17becf'))]
        layout['yaxis2'] = dict(
            title='Price (' + currency.upper() + ')',
            overlaying='y',
            side='right',
            showgrid=False,
            rangemode='tozero'
        )
    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private',
            filename=plotly_directory + pair + '-total-xch-value-6m-bt')
    save_plot(fig, P_DIR, title.lower())


def plot_count_histogram(trades_per_bucket6, pair, xchanges, names, colors, cryptocurrency, from_ts, to_ts,
                         plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    _, start, end = get_dates(from_ts, to_ts, parsed=True)

    if len(trades_per_bucket6[list(trades_per_bucket6.keys())[0]]) == 10:
        x = ['<0.00001', '>0.00001', '>0.0001', '>0.001', '>0.01', '>0.1', '>1', '>10', '>100', '>1000']
    else:
        x = ['<0.001', '>0.01', '>0.1', '>1', '>10', '>100', '>1000', '>10000']

    trace = [go.Bar(x=x, y=trades_per_bucket6[xch], name=names[xch], opacity=0.77, marker=dict(color=colors[xch])) for
             xch in xchanges]

    if len(_) == 1:
        title = 'Number of trades per size per exchange in ' + pair.upper() + ' for ' + str_month(end[1]) + ' ' + end[0]
        filename = plotly_directory + pair + '-' + str_month(end[1]) + end[0]
    else:
        title = 'Number of trades per size per exchange in ' + pair.upper() + ' from ' + str_month(start[1]) + ' ' + \
                start[0] + ' to ' + str_month(end[1]) + ' ' + end[0]
        filename = plotly_directory + pair + '-' + str_month(end[1]) + end[0] + '-6m'

    layout = go.Layout(
        barmode='group',  # stack, overlay, group
        title=title,
        xaxis=dict(
            title='Size (' + cryptocurrency.upper() + ')'
        ),
        yaxis=dict(
            title='Number of trades'
        ),
        bargap=0.25,
        bargroupgap=0.3,
    )
    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private', filename=filename)
    save_plot(fig, P_DIR, title.lower())


def plot_volume_histogram(volumes_per_bucket6, pair, xchanges, names, colors, cryptocurrency, from_ts, to_ts,
                          plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    _, start, end = get_dates(from_ts, to_ts, parsed=True)

    if len(volumes_per_bucket6[list(volumes_per_bucket6.keys())[0]]) == 10:
        x = ['<0.00001', '>0.00001', '>0.0001', '>0.001', '>0.01', '>0.1', '>1', '>10', '>100', '>1000']
    else:
        x = ['<0.001', '>0.01', '>0.1', '>1', '>10', '>100', '>1000', '>10000']

    trace = [go.Bar(x=x, y=volumes_per_bucket6[xch], name=names[xch], opacity=0.77, marker=dict(color=colors[xch])) for
             xch in xchanges]

    if len(_) == 1:
        title = 'Volume per size per exchange in ' + pair.upper() + ' for ' + str_month(end[1]) + ' ' + end[0]
        filename = plotly_directory + pair + '-' + str_month(end[1]) + end[0] + '-volume'
    else:
        title = 'Volume per size per exchange in ' + pair.upper() + ' from ' + str_month(start[1]) + ' ' + start[
            0] + ' to ' + str_month(end[1]) + ' ' + end[0]
        filename = plotly_directory + pair + '-' + str_month(end[1]) + end[0] + '-volume-6m'

    layout = go.Layout(
        barmode='group',  # stack, overlay, group
        title=title,
        xaxis=dict(
            title='Size (' + cryptocurrency.upper() + ')'
        ),
        yaxis=dict(
            title='Volume (' + cryptocurrency.upper() + ')'
        ),
        bargap=0.25,
        bargroupgap=0.3,
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private', filename=filename)
    save_plot(fig, P_DIR, title.lower())


# def plot_mean_median_trade(trades_per_bucket6, volumes_per_bucket6, median_price6,pair, xchanges, names, colors, cryptocurrency,one_month=False):
#    x = [names[xch] for xch in xchanges]
#    y = [np.sum(volumes_per_bucket6[xch])/np.sum(trades_per_bucket6[xch]) for xch in xchanges]
#    y2 = [median_price6[xch] for xch in xchanges]
#    
#    color = [colors[xch] for xch in xchanges]
#    temp = pd.DataFrame({'x':x,'y':y,'y2':y2,'c':color})
#    temp.sort_values('y',ascending=True,inplace=True)
#    x = list(temp['x'].values)
#    y = list(temp['y'].values)
#    y2 = list(temp['y2'].values)
##    c = list(temp['c'].values)
#    temp = None
#    
#    trace = [
#                go.Bar(x=x,y=y,opacity=0.77, name = 'Mean trade size'),
#                go.Bar(x=x,y=y2,opacity=0.77, name = 'Median trade size')
#            ]
#            
#    if one_month:
#        title='Mean and median trade size per exchange in '+pair.upper()+' for '+monthStr+' '+year
#        filename=plotly_directory+pair+'-'+monthStr+year+'-avg-trade-vol'
#    else:
#        title='Mean and median trade size per exchange in '+pair.upper()+' from '+monthStr_end+' '+year_end+' to '+monthStr+' '+year
#        filename=plotly_directory+pair+'-'+monthStr+year+'-avg-trade-vol-6m'
#    
#    layout = go.Layout(
#            barmode='group', #stack, overlay, group
#            title=title,
#            xaxis=dict(
#                title='Exchange'
#            ),
#            yaxis=dict(
#                title='Size ('+cryptocurrency.upper()+')'
#            ),
#            bargap=0.25,
#            bargroupgap=0.3,
#        )
#        
#        
#    fig = go.Figure(data=trace, layout=layout)
#    py.plot(fig,auto_open = auto_open_charts,sharing='private', filename=filename)


# plot price distribution and volume per exchange for 6 months
def plot_price_distribution(trades_per_bucket6, volumes_per_bucket6, pair, xchanges, names, colors, cryptocurrency,
                            from_ts, to_ts, regroup_small_values=False, plotly_directory='', auto_open_charts=True,
                            P_DIR='plots/'):
    _, start, end = get_dates(from_ts, to_ts, parsed=True)

    if regroup_small_values:
        th6 = -3
        th6_text = '100'
    else:
        th6 = -4
        th6_text = '1'

    yi = [[np.sum(volumes_per_bucket6[xch][th6:]) / np.sum(volumes_per_bucket6[xch]) * 100 for xch in xchanges],
          [np.sum(volumes_per_bucket6[xch][:th6]) / np.sum(volumes_per_bucket6[xch]) * 100 for xch in xchanges]]
    ynames = ['≥ ' + th6_text + ' ' + cryptocurrency + '', '< ' + th6_text + ' ' + cryptocurrency + '']
    trace = []
    barcolors = ['#E6E6E6', '#20A0F9']
    for i in range(2):
        x = [names[xch] for xch in xchanges]
        y2 = [np.sum(volumes_per_bucket6[xch]) for xch in xchanges]
        y3 = [np.sum(trades_per_bucket6[xch]) for xch in xchanges]
        y = yi[i]

        temp = pd.DataFrame({'x': x, 'y': y, 'y2': y2, 'y3': y3})
        temp.sort_values('y2', ascending=True, inplace=True)
        x = list(temp['x'].values)
        y = list(temp['y'].values)
        y2 = list(temp['y2'].values)
        y3 = list(temp['y3'].values)
        temp = None

        trace += [go.Bar(x=x, y=y, name=ynames[i], opacity=0.8, marker=dict(color=barcolors[i]))]

    trace += [go.Scatter(x=x, y=y2,
                         yaxis='y2',
                         mode='lines+markers+text', text=list(map(fmt, y2)), textposition='bottomright',
                         hoverinfo='x+text+name',
                         name='Total volume of trades (' + cryptocurrency.upper() + ')', marker=dict(color='#FFA556'))]

    trace += [go.Scatter(x=x, y=y3,
                         yaxis='y3',
                         mode='lines+markers+text', text=list(map(fmt, y3)), textposition='bottomright',
                         hoverinfo='x+text+name',
                         name='Number of trades', marker=dict(color='#144584'))]
    if len(_) == 1:
        title = 'Distribution of trade sizes in ' + pair.upper() + ' for ' + str_month(end[1]) + ' ' + end[0]
        filename = plotly_directory + pair + '-' + str_month(end[1]) + end[0] + '-xchdist'
    else:
        title = 'Distribution of trade sizes in ' + pair.upper() + ' from ' + str_month(start[1]) + ' ' + start[
            0] + ' to ' + str_month(end[1]) + ' ' + end[0]
        filename = plotly_directory + pair + '-' + str_month(end[1]) + end[0] + '-xchdist-m6'

    layout = go.Layout(
        barmode='stack',  # stack, overlay, group
        title=title,
        xaxis=dict(
            title='Exchange'
        ),
        yaxis=dict(
            title='Percentage of traded volume (' + cryptocurrency.upper() + ')',
            ticksuffix='%'
        ),
        yaxis2=dict(
            title='',
            overlaying='y',
            side='right',
            showgrid=False,
            rangemode='tozero',
            showaxeslabels=False,
            showticklabels=False
        ),
        yaxis3=dict(
            title='',
            showgrid=False,
            rangemode='tozero',
            side='right',
            overlaying='y',
            anchor='free',
            position=1,
            showaxeslabels=False,
            showticklabels=False

        ),
        showlegend=True,
        bargap=0.25,
        bargroupgap=0.3,
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private', filename=filename)
    save_plot(fig, P_DIR, title.lower())


def plot_boxplot(trades, pair, xchanges, names, cryptocurrency, from_ts, to_ts,
                 colors=['#20a0f9', '#ffa556', '#c0c0c0'], P_DIR='plots/'):
    _, start, end = get_dates(from_ts, to_ts, parsed=True)

    matplotlib.style.use('ggplot')

    def setBoxColors(bp, colors=colors):
        plt.setp(bp['boxes'][0], color=colors[0])
        plt.setp(bp['caps'][0], color=colors[0])
        plt.setp(bp['caps'][1], color=colors[0])
        plt.setp(bp['whiskers'][0], color=colors[0])
        plt.setp(bp['whiskers'][1], color=colors[0])
        #    plt.setp(bp['fliers'][0], color=colors[0])
        #    plt.setp(bp['fliers'][1], color=colors[0])
        plt.setp(bp['medians'][0], color=colors[0])
        plt.setp(bp['means'][0], color=colors[2])

        plt.setp(bp['boxes'][1], color=colors[1])
        plt.setp(bp['caps'][2], color=colors[1])
        plt.setp(bp['caps'][3], color=colors[1])
        plt.setp(bp['whiskers'][2], color=colors[1])
        plt.setp(bp['whiskers'][3], color=colors[1])
        #    plt.setp(bp['fliers'][2], color=colors[1])
        #    plt.setp(bp['fliers'][3], color=colors[1])
        plt.setp(bp['medians'][1], color=colors[1])
        plt.setp(bp['means'][1], color=colors[2])

    def clean_plot(ax):
        ax.grid(True, color='0.12', axis='y', linestyle='--', linewidth=0.2)
        ax.patch.set_facecolor('1')
        ax.set_axisbelow(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    def get_ymax(bp):
        return np.nanmax([np.nanmax(bp['whiskers'][1].get_ydata()), np.nanmax(bp['whiskers'][3].get_ydata())]) + 1

    # create box plots
    plt.figure(figsize=(18, 12))
    ax = plt.axes()
    plt.hold(True)
    i = 1
    ymax = 0
    for xch in xchanges:
        bp = plt.boxplot(x=trades[xch], whis=[10, 90], showfliers=False, widths=0.7, showmeans=True, meanline=True,
                         positions=[i, i + 1])
        setBoxColors(bp, colors)  # set colors to the boxplot
        ymax = np.max([ymax, get_ymax(bp)])
        i += 3

    # set axes limits and labels
    ax.set_xticklabels([names[xch] for xch in xchanges])
    ax.set_xticks([1.5 + i * 3 for i in range(len(xchanges))])
    plt.ylim(-0.2, ymax)
    plt.xlim(0, 3 * len(xchanges))
    plt.ylabel('Size (' + cryptocurrency.upper() + ')')
    plt.title(
        "Trades' size for " + pair.upper() + " (median, mean, upper and lower quantile, 10th and 90th percentile)")

    # draw temporary red and blue lines and use them to create a legend
    hB, = plt.plot([1, 1], colors[0])
    hR, = plt.plot([1, 1], colors[1])
    hG, = plt.plot([1, 1], colors[2])
    plt.legend((hB, hR, hG), (
        str_month(start[1]) + ' ' + start[0] + ' - ' + str_month(end[1]) + ' ' + end[0],
        str_month(end[1]) + ' ' + end[0],
        'Mean'), frameon=False)
    hB.set_visible(False)
    hR.set_visible(False)
    hG.set_visible(False)

    # clean plot
    clean_plot(ax)

    # show plot
    plt.savefig(P_DIR + pair + '-boxplot' + end[1] + end[0] + '.png', dpi=300)
    # plt.show()
    print(' saved to ' + pair + '-boxplot' + end[1] + end[0] + '.png')
