# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:22:38 2016

@author: ilyes
"""

__author__ = 'ilkhem'

import datetime as dt
import math

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py

import aggregators
from config import config
from utils import parse_month, get_dates, str_month, save_plot


def get_spread_book(directory, from_ts, to_ts, pair, resolution='day'):
    """
    returns a spread book from the aggregated sb, after correcting it, deleting all negative values, and making sure
    spreads are non decreasing the higher the threshold
    :param directory: directory of sb_ files created by aggregate_orderbooks.py
    :param from_ts: starting timestamp
    :param to_ts: ending timestamo
    :param pair: pair to be processed
    :param resolution: resolution of the returned spread book: can be 'hour', 'day', 'week' or 'total'
    :return: spread book
    """
    if directory != '' and directory[-1] != '/':
        directory += '/'
    print('getting spread books for %s ...' % pair)
    dates = get_dates(from_ts, to_ts)
    mys = [[parse_month(d.month), str(d.year)] for d in dates]

    next_month = dt.datetime(int(mys[-1][1]), int(mys[-1][0]), 1) + dt.timedelta(days=32)
    next_month = dt.datetime(next_month.year, next_month.month, 1)

    spread_book = pd.concat(
        [pd.read_csv(directory + 'sb_' + pair + '_' + my[1] + '_' + my[0] + '.csv', parse_dates=['date']) for my in
         mys])
    if resolution == 'hour':
        spread_book = aggregators.aggregate_book(spread_book, 'hour')
    if resolution == 'day':
        spread_book = aggregators.aggregate_book(aggregators.aggregate_book(spread_book, 'hour'), 'day')
    if resolution == 'week':
        try:
            spread_book = spread_book.append(pd.read_csv(
                directory + 'sb_' + pair + '_' + str(next_month.year) + '_' + parse_month(next_month.month) + '.csv',
                parse_dates=['date']))
        except OSError:
            pass
        spread_book = aggregators.aggregate_book(
            aggregators.aggregate_book(aggregators.aggregate_book(spread_book, 'hour'), 'day'), 'week')
    if resolution == 'month' or resolution == 'total':
        spread_book = aggregators.aggregate_book(
            aggregators.aggregate_book(aggregators.aggregate_book(spread_book, 'hour'), 'day'), 'month')

    print('checking sb validity')
    # Check for negative values
    # negative values are due sometimes to orders executed by the exchange but still saved to the orderbook
    for i in range(len(spread_book)):
        b = False
        for c in spread_book.columns:
            if c != 'date' and c != 'exchange':
                if spread_book.loc[i, c] < 0:
                    b = True
                    break
        if b:
            for c in spread_book.columns:
                if c != 'date' and c != 'exchange':
                    try:
                        spread_book.loc[i, c] = spread_book.loc[i - 1, c]
                    except:
                        spread_book.loc[i, c] = np.nan

    print('correcting sb')
    # call to correct_sb from the aggregators module
    # makes sure thresholds values are ascending
    aggregators.correct_sb(spread_book)

    spread_book = spread_book[spread_book['date'] >= dt.datetime.utcfromtimestamp(from_ts)]
    spread_book = spread_book[spread_book['date'] <= dt.datetime.utcfromtimestamp(to_ts)]

    if resolution == 'total':
        spread_book = spread_book.drop('date').groupby('exchange').mean().reset_index()

    return spread_book


def get_volume_book(directory, from_ts, to_ts, pair, resolution='day'):
    """
    return volume book from aggregated vb
    :param directory: dir to vb_ files created by aggregate_ordeebooks.py
    :param from_ts: starting ts
    :param to_ts: ending ts
    :param pair: pair to be processed
    :param resolution: resolution of the returned spread book: can be 'hour', 'day', 'week' or 'total'
    :return:
    """
    if directory != '' and directory[-1] != '/':
        directory += '/'
    print('getting liquidity books for %s ...' % pair)
    dates = get_dates(from_ts, to_ts)
    mys = [[parse_month(d.month), str(d.year)] for d in dates]

    next_month = dt.datetime(int(mys[-1][1]), int(mys[-1][0]), 1) + dt.timedelta(days=32)
    next_month = dt.datetime(next_month.year, next_month.month, 1)

    volume_book = pd.concat(
        [pd.read_csv(directory + 'vb_' + pair + '_' + my[1] + '_' + my[0] + '.csv', parse_dates=['date']) for my in
         mys])
    if resolution == 'hour':
        volume_book = aggregators.aggregate_book(volume_book, 'hour')
    if resolution == 'day':
        volume_book = aggregators.aggregate_book(aggregators.aggregate_book(volume_book, 'hour'), 'day')
    if resolution == 'week':
        try:
            volume_book = volume_book.append(pd.read_csv(
                directory + 'vb_' + pair + '_' + str(next_month.year) + '_' + parse_month(next_month.month) + '.csv',
                parse_dates=['date']))
        except OSError:
            pass
        volume_book = aggregators.aggregate_book(
            aggregators.aggregate_book(aggregators.aggregate_book(volume_book, 'hour'), 'day'), 'week')
    if resolution == 'month' or resolution == 'total':
        volume_book = aggregators.aggregate_book(
            aggregators.aggregate_book(aggregators.aggregate_book(volume_book, 'hour'), 'day'), 'month')

    volume_book = volume_book[volume_book['date'] >= dt.datetime.utcfromtimestamp(from_ts)]
    volume_book = volume_book[volume_book['date'] <= dt.datetime.utcfromtimestamp(to_ts)]

    if resolution == 'total':
        volume_book = volume_book[['exchange', 'ask_volume', 'bid_volume']].groupby('exchange').mean().reset_index()

    return volume_book


def plot_liquidity(vb, pair, xchanges, names, colors, cryptocurrency, from_ts, to_ts, plotly_directory='',
                   auto_open_charts=True, P_DIR='plots/'):
    _, start, end = get_dates(from_ts, to_ts, parsed=True)

    x = pd.DataFrame(list(set(vb['date'])), columns=['date']).sort_values('date')
    trace = []

    # bid liquidity

    def fill_empty(x, df):
        temp = x.merge(df, on='date', how='left')
        temp['bid_volume'].fillna(0, inplace=True)
        temp['ask_volume'].fillna(0, inplace=True)
        return temp

    yold2 = [0] * len(x)
    ytotal2 = [np.nansum(y) for y in zip(*[fill_empty(x, vb[vb['exchange'] == xch])['bid_volume'] for xch in xchanges])]

    for xch in xchanges:
        y2 = np.divide([-np.nansum(z) for z in zip(fill_empty(x, vb[vb['exchange'] == xch])['bid_volume'], yold2)],
                       ytotal2) * 100
        text2 = (np.divide(fill_empty(x, vb[vb['exchange'] == xch])['bid_volume'], ytotal2) * 100).map(
            lambda x: "%0.3f" % x + '%')
        trace += [
            go.Scatter(
                x=x.date, y=y2, text=text2, fill='tonexty', hoverinfo='x+text+name',
                mode='lines', line=dict(width=0.5), opacity=0.8,
                name=names[xch] + ', bid',
                showlegend=False,
                marker=dict(color=colors[xch])
            )
        ]
        yold2 = [np.nansum(z) for z in zip(fill_empty(x, vb[vb['exchange'] == xch])['bid_volume'], yold2)]

    trace += [
        go.Scatter(x=x.date, y=np.negative(ytotal2), mode='lines', line=dict(width=3), opacity=0.8,
                   name='Total bid liquidity', yaxis='y2', marker=dict(color='#073763'))
    ]

    # ask liquidity
    yold1 = [0] * len(x)
    ytotal = [np.nansum(y) for y in zip(*[fill_empty(x, vb[vb['exchange'] == xch])['ask_volume'] for xch in xchanges])]

    i = 0
    for xch in xchanges:
        y1 = np.divide([np.nansum(z) for z in zip(fill_empty(x, vb[vb['exchange'] == xch])['ask_volume'], yold1)],
                       ytotal) * 100
        text1 = (np.divide(fill_empty(x, vb[vb['exchange'] == xch])['ask_volume'], ytotal) * 100).map(
            lambda x: "%0.3f" % x + '%')
        if i == 0:
            i += 1
            trace += [
                go.Scatter(
                    x=x.date, y=y1, text=text1, fill='tozeroy', hoverinfo='x+text+name',
                    mode='lines', line=dict(width=0.5), opacity=0.8,
                    #                                    name = names[xch]+', ask',
                    name=names[xch],
                    marker=dict(color=colors[xch])
                )
            ]
        else:
            trace += [
                go.Scatter(
                    x=x.date, y=y1, text=text1, fill='tonexty', hoverinfo='x+text+name',
                    mode='lines', line=dict(width=0.5), opacity=0.8,
                    #                                    name = names[xch]+', ask',
                    name=names[xch],
                    marker=dict(color=colors[xch])
                )
            ]
        yold1 = [np.nansum(z) for z in zip(fill_empty(x, vb[vb['exchange'] == xch])['ask_volume'], yold1)]

    trace += [
        go.Scatter(x=x.date, y=ytotal, mode='lines', line=dict(width=3), opacity=0.8, name='Total ask liquidity',
                   yaxis='y2', marker=dict(color='#073763'))
    ]

    title = 'Liquidity at +/-1% from mid price from ' + str_month(start[1]) + ' ' + start[0] + ' to ' + str_month(
        end[1]) + ' ' + end[0] + ' (' + pair.upper() + ')'

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Volume (%)',
            #                showgrid=False,
            ticksuffix='%'
            #            rangemode='tozero'
        ),
        yaxis2=dict(
            title='Total liquidity (' + cryptocurrency.upper() + ')',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[-math.floor(np.nanmax(np.concatenate([ytotal, ytotal2])) * 1.2),
                   math.floor(np.nanmax(np.concatenate([ytotal, ytotal2])) * 1.2)]
        ),
        legend=dict(traceorder='normal'),

        #            showlegend=False,
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private', filename=plotly_directory + pair + '-ob-1p-liquidity')
    save_plot(fig, P_DIR, title.lower())


def plot_exchange_liquidity(vb, xch, pair, names, colors, cryptocurrency, from_ts, to_ts, plotly_directory='',
                            auto_open_charts=True, P_DIR='plots/'):
    _, start, end = get_dates(from_ts, to_ts, parsed=True)

    trace = [
        go.Scatter(x=vb[vb['exchange'] == xch].date, y=vb[vb['exchange'] == xch]['bid_volume'], mode='lines',
                   name='1% Bid liquidity'),
        go.Scatter(x=vb[vb['exchange'] == xch].date, y=vb[vb['exchange'] == xch]['ask_volume'], mode='lines',
                   name='1% Ask liquidity')
    ]

    title = 'Liquidity evolution for ' + names[xch] + ' from ' + str_month(start[1]) + ' ' + start[
        0] + ' to ' + str_month(end[1]) + ' ' + end[0] + ' (' + pair.upper() + ')'

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Liquidity (' + cryptocurrency.upper() + ')',
            #                showgrid=False,
            #            rangemode='tozero'
        ),
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private',
            filename=plotly_directory + pair + '-' + xch + '-1p-liquidity')
    save_plot(fig, P_DIR, title.lower())


def plot_liquidity_stacked(vb1, pair, xchanges, names, colors, cryptocurrency, from_ts, to_ts, plotly_directory='',
                           auto_open_charts=True, P_DIR='plots/'):
    _, start, end = get_dates(from_ts, to_ts, parsed=True)

    trace = []

    for xch in xchanges:
        y = vb1[vb1['exchange'] == xch]['ask_volume']
        trace += [
            go.Bar(x=vb1[vb1['exchange'] == xch]['date'], y=y, opacity=0.8, name=names[xch],
                   marker=dict(color=colors[xch]))
        ]
    title = 'Stacked liquidity of 1% orderbook from ' + str_month(start[1]) + ' ' + start[0] + ' to ' + str_month(
        end[1]) + ' ' + end[0] + ' (' + pair.upper() + ')'

    layout = go.Layout(
        barmode='stack',  # stack, overlay, group
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Volume (' + cryptocurrency.upper() + ')',
            #                showgrid=False,
            #            rangemode='tozero'
        ),

        bargap=0.25,
        bargroupgap=0.3,
    )

    text = []
    y_data = []

    for i in range(len(xchanges)):
        t = []
        l = []
        for j in range(len(vb1[vb1['exchange'] == xchanges[i]])):
            try:
                t += [vb1[vb1['exchange'] == xchanges[i]]['ask_volume'].iloc[j] + y_data[i - 1][j]]
            except:
                t += [vb1[vb1['exchange'] == xchanges[i]]['ask_volume'].iloc[j]]
            l += [str(math.floor(vb1[vb1['exchange'] == xchanges[i]]['ask_volume'].iloc[j]))]
        y_data += [t]
        text += [l]

    textcolors = [config["exchanges"][xch]["textcolor"] for xch in xchanges]

    annotations = []

    for i in range(len(xchanges)):
        for j in range(len(vb1[vb1['exchange'] == xchanges[i]])):
            annotations.append(
                dict(x=vb1[vb1['exchange'] == xchanges[i]]['date'].iloc[j], y=y_data[i][j], text=text[i][j],
                     showarrow=False, xanchor='center', yanchor='top', font=dict(color=textcolors[i])))

            #    layout['annotations'] = annotations

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private',
            filename=plotly_directory + pair + '-ob-1p-repartition-stacked-bars')
    save_plot(fig, P_DIR, title.lower())


def plot_spread_per_threshold(sb, xch, thresholds, pair, names, color_by_thresh, currency, cryptocurrency,
                              plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    trace = []

    for th in thresholds:
        #    y1 = sb[sb['exchange']==xch]['sp'+str(th)]
        #    trace += [go.Scatter(x = sb[sb['exchange']==xch]['date'], y = np.negative(y1), mode ='lines', opacity=0.8, name = str(th)+' '+cryptocurrency+', bid', marker = dict(color=color_by_thresh[th]))]

        y2 = sb[sb['exchange'] == xch]['sp' + str(th) + 'a']
        trace += [go.Scatter(x=sb[sb['exchange'] == xch]['date'], y=y2, mode='lines', opacity=0.8,
                             name=str(th) + ' ' + cryptocurrency + ', ask',
                             marker=dict(color=color_by_thresh[str(th)]))]

    title = 'Spread evolution at different thresholds for ' + names[xch] + ' (' + pair.upper() + ')'
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Spread (' + currency.upper() + ')',
            #                showgrid=False,
            type='log'
            #            rangemode='tozero'
        ),

        #            showlegend=False,
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private',
            filename=plotly_directory + pair + '-' + xch + '-ob-1p-spreads')
    save_plot(fig, P_DIR, title.lower())


def plot_spread_per_exchange(sb, th, pair, xchanges, currency, names, colors, cryptocurrency, plotly_directory='',
                             auto_open_charts=True, P_DIR='plots/'):
    # x = sb.date
    trace = []

    for xch in xchanges:
        y1 = list(sb[sb['exchange'] == xch].sort_values('date')['sp' + str(th)])
        y2 = list(sb[sb['exchange'] == xch].sort_values('date')['sp' + str(th) + 'a'])
        trace += [
            go.Scatter(x=sb[sb['exchange'] == xch].date, y=[np.sum(y) for y in zip(y1, y2)], mode='lines', opacity=0.8,
                       name=names[xch], marker=dict(color=colors[xch]))]

    title = 'Bid side spreads per exchange for a threshold of ' + str(
        th) + ' ' + cryptocurrency + ' (' + pair.upper() + ')'
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=''
        ),
        yaxis=dict(
            title='Spread (' + currency.upper() + ')',
            #                showgrid=False,
            type='log',
            #            rangemode='tozero'
        ),
        #            showlegend=False,
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private',
            filename=plotly_directory + pair + '-sp' + str(th) + '-ob-1p-spreads')
    save_plot(fig, P_DIR, title.lower())


def plot_month_summary(sb1, thresholds, pair, xchanges, names, color_by_thresh, currency, from_ts, to_ts,
                       plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    def get_spread(df, xch, th):
        try:
            return df[df['exchange'] == xch]['sp' + str(th)].iloc[0] + \
                   df[df['exchange'] == xch]['sp' + str(th) + 'a'].iloc[0]
        except IndexError:
            pass

    _, _, end = get_dates(from_ts, to_ts, parsed=True)

    thresholds.reverse()

    x = [names[xch] for xch in xchanges if get_spread(sb1, xch, '1') != None]

    trace = [
        go.Bar(x=x,
               y=[get_spread(sb1, xch, th) for xch in xchanges if get_spread(sb1, xch, th) != None],
               name=str(th),
               opacity=0.8,
               marker=dict(color=color_by_thresh[str(th)])
               )
        for th in thresholds
        ]
    title = 'Average threshold spread per exchange for ' + str_month(end[1]) + ' ' + end[0] + ' (' + pair.upper() + ')'
    layout = go.Layout(
        barmode='overlay',  # stack, overlay, group
        title=title,
        xaxis=dict(
            title='Exchange'
        ),
        yaxis=dict(
            title='Spread (' + currency.upper() + ')',
            #                type='log',
            nticks=20
        ),
        bargap=0.25,
        bargroupgap=0.3,
    )
    thresholds.reverse()
    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, auto_open=auto_open_charts, sharing='private',
            filename=plotly_directory + pair + '-monthly-avg-spreads' + end[1] + end[0])
    save_plot(fig, P_DIR, title.lower())
