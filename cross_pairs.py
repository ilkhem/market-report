# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 09:56:09 2016

@author: ilkhem

Script to be run separately from main.py for plotting cross pair charts (and pie charts)
"""
__author__ = 'ilkhem'

import argparse
import datetime as dt
import json
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py

from config import config
from utils import parse_date, eod_to_ts


def get_cross_pair_liquidity(directory, from_ts, to_ts, bpairs):
    """
    get total liquidity over a period of time (from from_ts to to_ts) by pai
    :param directory: directory for volune books
    :param from_ts: starting timestamp
    :param to_ts: ending timestamp
    :param bpairs: list of the pairs to include
    :return: dict of pair: total liquidity
    """
    print('getting cross pair liquidity ... ')
    liq = {pair: 0 for pair in bpairs}
    for p in bpairs:
        df = get_volume_book(directory, from_ts, to_ts, p, 'month')
        df['total'] = df.bid_volume + df.ask_volume
        df = df[['date', 'total']]
        df = df.groupby('date').sum().reset_index()
        liq[p] = int(df.total.mean())
    return liq


def get_cross_pair_weights(bpairs, cryptocurrency, volumes=None, liquidities=None):
    """
    returns weights by pair in the global price index
    :param bpairs: list of pairs
    :param cryptocurrency: btc or eth
    :param volumes: if cryptocurrency == eth, dict of volumes by pair
    :param liquidities: if cryptocurrency == eth, dict of liquidities by pair
    :return: dict of pair: total weight in the global price index
    """
    print('getting cross pair weights ... ')

    def sum_weights(r):
        s = {pair: 0 for pair in bpairs}
        for x in r:
            s[x.split('_')[1]] += r[x]
        return s

    w = {pair: 0 for pair in bpairs}
    if cryptocurrency == 'btc':
        session = create_session('read')
        query = session.execute(
            "select period,weights from index_weights_aggregated_by_week_3 where slug = 'global_price';")
        weights = pd.DataFrame(query.current_rows, columns=query.column_names)
        weights['s'] = weights.weights.map(sum_weights)
        for p in bpairs:
            weights[p] = weights.s.map(lambda x: x[p])
            w[p] = weights[p].mean()

    if cryptocurrency == 'eth':
        for x in bpairs:
            vol = volumes[x] / np.sum(list(volumes.values()))
            liq = liquidities[x] / np.sum(list(liquidities.values()))
            w[x] = 0.3 * vol + 0.7 * liq
    return w


def get_cross_pair_indices(from_ts, to_ts, indices):
    print('getting indices ... ')
    url = "https://api.kaiko.com/v1/history/indices?from=" + str(from_ts) + "&to=" + str(to_ts) + "&fields=c"
    js = json.loads(urllib.request.urlopen(url).read().decode("utf-8"))

    df = pd.DataFrame({'timestamp': []})
    for x in indices:
        temp = pd.DataFrame(js[x])
        temp['indice'] = x
        df = df.append(temp)
    df.timestamp = df.timestamp.map(dt.datetime.utcfromtimestamp)
    return df


def get_local_bitcoins(from_ts, to_ts, resolution='week'):
    print('getting local bitcoins ... ')
    if to_ts == 'today':
        to_ts = int(dt.datetime.timestamp(dt.datetime.today()))
    url = "https://api.kaiko.com/v1/history/exchanges?exchanges=localbitcoins&fields=v&resolution=" + resolution + "&from=" + str(
        from_ts) + "&to=" + str(to_ts) + "&pairs=" + config["localbitcoins"]["pairs_api"]
    js = json.loads(urllib.request.urlopen(url).read().decode("utf-8"))["localbitcoins"]
    lbtc = pd.DataFrame({'timestamp': []})
    for x in js:
        temp = pd.DataFrame(js[x])
        temp['pair'] = x
        lbtc = lbtc.append(temp)
    lbtc.timestamp = lbtc.timestamp.map(dt.datetime.utcfromtimestamp)
    return lbtc


def plot_cross_pair_mean_trade(means, xp, bpairs, names, cryptocurrency, plotly_directory='', auto_open_charts=True,
                               P_DIR='plots/'):
    trace = [go.Bar(x=[names[xch] for xch in xp[p]], y=[means[x][p] for x in xp[p]], name=p.upper()) for p in bpairs]
    # trace = [go.Bar(x = pairs[xch], y = [means[xch][p] for p in pairs[xch]], name = xch) for xch in xchanges]
    layout = go.Layout(
        barmode='group',
        title="Average trade size per exchange per pair",
        yaxis=dict(
            title='Size (' + cryptocurrency.upper() + ')'
        ),
        bargap=0.25,
        bargroupgap=0.3,
    )
    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, filename=plotly_directory + 'cross Pair mean trade size', auto_open=auto_open_charts,
            sharing='private')
    save_plot(fig, P_DIR, '')


def plot_cross_pair_volume(volumes, cryptocurrency, plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    full_name = {'btc': 'Bitcoin', 'eth': 'Ethereum'}

    labels = sorted([x.upper() for x in volumes])
    values = [volumes[l.lower()] for l in labels]
    trace = [
        go.Pie(
            values=values,
            labels=labels,
            marker=dict(colors=['#ffa556', '#20a0f9', '#b7e8ad']),
            #      domain= {"x": [0, .48]},
            name=cryptocurrency.upper() + " volume distribution",
            #          text = dict(info = 'val+%')
        )
    ]

    layout = go.Layout(
        title=full_name[cryptocurrency] + " volume distribution between the main currencies (1 month)",
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, filename=plotly_directory + cryptocurrency.upper() + ' Pie chart -1m', auto_open=auto_open_charts,
            sharing='private')
    save_plot(fig, P_DIR, '')


def plot_cross_pair_liquidity(liq, cryptocurrency, plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    full_name = {'btc': 'Bitcoin', 'eth': 'Ethereum'}

    labels = sorted([x.upper() for x in liq])
    values = [liq[l.lower()] for l in labels]
    trace = [
        go.Pie(
            values=values,
            labels=labels,
            #      domain= {"x": [0, .48]},
            marker=dict(colors=['#ffa556', '#20a0f9', '#b7e8ad']),
            name=cryptocurrency.upper() + " Liquidity distribution",
        )
    ]

    layout = go.Layout(
        title=full_name[cryptocurrency] + " liquidity distribution between the main currencies (1 month)",
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, filename=plotly_directory + cryptocurrency.upper() + ' Pie chart liquidity -1m', sharing='private',
            auto_open=auto_open_charts)
    save_plot(fig, P_DIR, '')


def plot_cross_pair_weights(w, cryptocurrency, plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    full_name = {'btc': 'Bitcoin', 'eth': 'Ethereum'}

    labels = sorted([x.upper() for x in w])
    values = [w[l.lower()] for l in labels]
    trace = [
        go.Pie(
            values=values,
            labels=labels,
            marker=dict(colors=['#ffa556', '#20a0f9', '#b7e8ad']),
            #      domain= {"x": [0, .48]},
            name=cryptocurrency.upper() + " weights distribution",
        )
    ]

    layout = go.Layout(
        title=full_name[cryptocurrency] + " weights distribution between the main currencies",
    )

    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig, filename=plotly_directory + cryptocurrency.upper() + ' Pie chart weights', auto_open=auto_open_charts,
            sharing='private')
    save_plot(fig, P_DIR, '')


def plot_cross_pair_indices(df, indices, cryptocurrency, from_ts, to_ts, plotly_directory='', auto_open_charts=True,
                            P_DIR='plots/'):
    _, _, end = get_dates(from_ts, to_ts, parsed=True)

    full_name = {'btc': 'Bitcoin', 'eth': 'Ethereum'}

    trace = [go.Scatter(x=df[df['indice'] == i].timestamp, mode='line', y=df[df['indice'] == i].c,
                        name='Kaiko ' + i + ' price index') for i in indices]
    layout = go.Layout(
        title="Kaiko Price Indices",
        xaxis=dict(),
        yaxis=dict(title='Price (USD)')
    )
    fig = go.Figure(data=trace, layout=layout)
    py.plot(fig,
            filename=plotly_directory + 'Kaiko ' + full_name[cryptocurrency] + ' Price Indices-' + end[1] + '-' + end[
                0],
            auto_open=auto_open_charts, sharing='private')
    save_plot(fig, P_DIR, '')


def plot_local_bitcoin(lbtc, from_ts, to_ts, plotly_directory='', auto_open_charts=True, P_DIR='plots/'):
    _, _, end = get_dates(from_ts, to_ts, parsed=True)

    for lbp in config["localbitcoins"]["pairs"]:
        df = lbtc[lbtc['pair'] == lbp]
        trace = [go.Scatter(x=df.timestamp, y=df.v,
                            name=config["localbitcoins"]["pair_names"][lbp],
                            fill='tonexty', mode='lines')]
        layout = go.Layout(
            title=config["localbitcoins"]["pair_names"][lbp],
            yaxis=dict(
                title='Volume (BTC)'
            ),
            font=dict(family='helvetica, light')
        )
        fig = go.Figure(data=trace, layout=layout)
        py.plot(fig, filename=plotly_directory + 'lbc-' + lbp + '-' + end[1] + '-' + end[0],
                auto_open=auto_open_charts, sharing='private')
        save_plot(fig, P_DIR, '')


# testing
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot cross pair charts')
    parser.add_argument('cryptocurrency', help='processed cryptocurrency. Currently support btc and eth')
    parser.add_argument('directory', help='directory for dumps')
    parser.add_argument('from_ts', help='start timestamp in seconds')
    parser.add_argument('to_ts', help='end timestamp in seconds')

    parser.add_argument('-lpd', '--local-plot-directory', default='plots/', action='store',
                        help="local directory to store charts, default ='plots/' ")
    parser.add_argument('-pd', '--plotly-directory', default='test/', action='store',
                        help="plotly directory to store charts online, default ='test/' ")
    parser.add_argument('-aoc', '--auto-open-charts', action='store_true', default=False,
                        help='open all charts in browser')
    parser.add_argument('-n', '--no-cross-pair-mean-size', action='store_false', default=True, dest='cross_mean',
                        help='do not plot cross pair mean trade size by exchange')

    args_dict = vars(parser.parse_args())

    cryptocurrency = args_dict['cryptocurrency']
    directory = args_dict['directory']
    if directory[-1] != '/':
        directory += '/'
    from_ts = parse_date(args_dict['from_ts'])
    to_ts = eod_to_ts(parse_date(args_dict['to_ts']))
    P_DIR = args_dict['local_plot_directory']
    if P_DIR[-1] != '/':
        P_DIR += '/'
    plotly_directory = args_dict['plotly_directory']
    if plotly_directory[-1] != '/':
        plotly_directory += '/'
    auto_open_charts = args_dict['auto_open_charts']
    cross_mean = args_dict['cross_mean']

    bpairs = config['pairs'][cryptocurrency]
    xchanges = [config['symbols'][x] for x in sorted([config['exchanges'][xch]['name'].lower()
                                                      for xch in config['exchanges'].keys()])]
    pairs = {xch: config['exchanges'][xch]['pairs'] for xch in xchanges}
    names = {xch: config["exchanges"][xch]["name"] for xch in xchanges}
    api_names = {xch: config["exchanges"][xch]["api_name"] for xch in xchanges}
    xp = config['exchanges_list_by_pair']

    from cassy import create_session
    from orderbooks import get_volume_book
    from transactions import get_mean_volume_count
    from utils import get_dates, save_plot

    if cross_mean:
        means = {xch: {p: get_mean_volume_count(directory + 'trades/', xch, p, from_ts, to_ts, api_names)[0]
                       for p in pairs[xch]}
                 for xch in xchanges}

        plot_cross_pair_mean_trade(means, xp, bpairs, names, cryptocurrency, plotly_directory, auto_open_charts, P_DIR)

    volumes = {p: np.sum([get_mean_volume_count(directory + 'trades/', xch, p, from_ts, to_ts, api_names)[1]
                          for xch in config['exchanges_list_by_pair'][p]])
               for p in bpairs}
    liq = get_cross_pair_liquidity(directory + 'vb/', from_ts, to_ts, bpairs)
    w = get_cross_pair_weights(bpairs, cryptocurrency, volumes, liq)

    plot_cross_pair_volume(volumes, cryptocurrency, plotly_directory, auto_open_charts, P_DIR)
    plot_cross_pair_liquidity(liq, cryptocurrency, plotly_directory, auto_open_charts, P_DIR)
    plot_cross_pair_weights(w, cryptocurrency, plotly_directory, auto_open_charts, P_DIR)

    indices = ['global', 'us', 'eu', 'cn']
    df = get_cross_pair_indices(from_ts, to_ts, indices)
    plot_cross_pair_indices(df, indices, cryptocurrency, from_ts, to_ts, plotly_directory, auto_open_charts, P_DIR)

    lbtc = get_local_bitcoins(from_ts, to_ts)
    plot_local_bitcoin(lbtc, from_ts, to_ts, plotly_directory, auto_open_charts, P_DIR)
