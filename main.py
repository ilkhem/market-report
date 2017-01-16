# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:21:27 2016

@author: ilyes

Main script to be run in order to plot trade, order book, volatility and bubble charts, as well as produce the rankings
for a given pair and time period.
"""
__author__ = 'ilkhem'

import argparse
import datetime as dt
import os

import pandas as pd

import bubbles
import orderbooks
import rankings
import transactions
import volatility_price
from config import config
from utils import str_bool, parse_date, eod_to_ts


def parse_args():
    parser = argparse.ArgumentParser(description='Creates plots and rankings for market report')
    parser.add_argument('pair',
                        help='the pair to be processed')
    parser.add_argument('directory',
                        help='directory for dumps')
    parser.add_argument('from_ts',
                        help='start timestamp in seconds')
    parser.add_argument('to_ts',
                        help='end timestamp in seconds')
    parser.add_argument('-f1', '--volatility-from', default='1293836400', dest='from_ts1',
                        help='start timestamp for volatility charts, default = 1/1/2011')
    parser.add_argument('-e', '--exchange', nargs='+',
                        help='run script for the specified exchanges, override exchange list from .cfg')
    parser.add_argument('-lpd', '--local-plot-directory', default='plots/', action='store',
                        help="local directory to store charts, default ='plots/' ")
    parser.add_argument('-lrd', '--local-ranking-directory', default='rankings/', action='store',
                        help="local directory to store rankings, default ='rankings/' ")
    parser.add_argument('-pd', '--plotly-directory', default='test/', action='store',
                        help="plotly directory to store charts online, default ='test/' ")
    parser.add_argument('-aoc', '--auto-open-charts', action='store_true', default=False,
                        help='open all charts in browser')
    parser.add_argument('-nt', '--no-trades', action='store_false', default=True, dest='trades',
                        help="don't plot trade charts")
    parser.add_argument('-no', '--no-orderbooks', action='store_false', default=True, dest='orderbooks',
                        help="don't plot order book charts")
    parser.add_argument('-nv', '--no-volatility', action='store_false', default=True, dest='volatility',
                        help="don't plot volatility charts")
    parser.add_argument('-nb', '--no-bubbles', action='store_false', default=True, dest='bubbles',
                        help="don't plot bubble charts")
    parser.add_argument('-nr', '--no-rankings', action='store_false', default=True, dest='rankings',
                        help="don't create rankings")
    parser.add_argument('-ot', '--only-trades', action='store_true', default=False, dest='o_trades',
                        help="plot trade charts only. Only use one -o* flag, non predictible behavior if you use 2 or + ")
    parser.add_argument('-oo', '--only-orderbooks', action='store_true', default=False, dest='o_orderbooks',
                        help="plot order book charts only. Only use one -o* flag, non predictible behavior if you use 2 or + ")
    parser.add_argument('-ov', '--only-volatility', action='store_true', default=False, dest='o_volatility',
                        help="plot volatility charts only. Only use one -o* flag, non predictible behavior if you use 2 or + ")
    parser.add_argument('-ob', '--only-bubbles', action='store_true', default=False, dest='o_bubbles',
                        help="plot bubble charts only. Only use one -o* flag, non predictible behavior if you use 2 or + ")
    parser.add_argument('-or', '--only-rankings', action='store_true', default=False, dest='o_rankings',
                        help="create rankings only. Only use one -o* flag, non predictible behavior if you use 2 or + ")
    parser.add_argument('--index',
                        help='index')
    parser.add_argument('--big-trades-threshold', type=int,
                        help='threshold for big trades')
    parser.add_argument('--regroup-small-values',
                        help='bool for regrouping small values. Can be True, true, t, T, False, false, f, F')
    parser.add_argument('--plot-kpi',
                        help='bool for plotting price index. Can be True, true, t, T, False, false, f, F')
    parser.add_argument('--use-mtgox-data',
                        help='bool for getting mtgox data. Can be True, true, t, False, false, f')

    args = parser.parse_args()
    return vars(args)


def deactivate_others(e):
    elem = ['trades', 'orderbooks', 'volatility', 'bubbles', 'rankings']
    ix = elem.index(e)
    res = [False] * 5
    res[ix] = True
    return res


if __name__ == '__main__':

    args_dict = parse_args()

    pair = args_dict['pair']
    directory = args_dict['directory']
    if directory[-1] != '/':
        directory += '/'
    from_ts = parse_date(args_dict['from_ts'])
    to_ts = eod_to_ts(parse_date(args_dict['to_ts']))
    P_DIR = args_dict['local_plot_directory']
    if P_DIR[-1] != '/':
        P_DIR += '/'
    plotly_directory = args_dict['plotly_directory']
    if plotly_directory != '' and plotly_directory[-1] != '/':
        plotly_directory += '/'
    ranking_directory = args_dict['local_ranking_directory']
    if ranking_directory[-1] != '/':
        ranking_directory += '/'
    auto_open_charts = args_dict['auto_open_charts']
    from_ts1 = parse_date(args_dict['from_ts1'])

    process_trades = args_dict['trades']
    process_orderbooks = args_dict['orderbooks']
    process_volatility = args_dict['volatility']
    process_bubbles = args_dict['bubbles']
    process_rankings = args_dict['rankings']

    if args_dict['o_trades']:
        process_trades, process_orderbooks, process_volatility, process_bubbles, process_rankings = deactivate_others(
            'trades')
    if args_dict['o_orderbooks']:
        process_trades, process_orderbooks, process_volatility, process_bubbles, process_rankings = deactivate_others(
            'orderbooks')
    if args_dict['o_volatility']:
        process_trades, process_orderbooks, process_volatility, process_bubbles, process_rankings = deactivate_others(
            'volatility')
    if args_dict['o_bubbles']:
        process_trades, process_orderbooks, process_volatility, process_bubbles, process_rankings = deactivate_others(
            'bubbles')
    if args_dict['o_rankings']:
        process_trades, process_orderbooks, process_volatility, process_bubbles, process_rankings = deactivate_others(
            'rankings')
    # pair to (index, big_trades_threshold, regroup_small_values, plot_kpi, use_mtgox_data)
    # Those parameters can be overwritten
    pair_to_index = {'btcusd': ('us', 500, False, True, True),
                     'btceur': ('eu', 200, False, True, True),
                     'btccny': ('cn', 500, False, True, True),
                     'ethbtc': ('ethbtc', 10000, True, True, False),
                     'ethusd': ('ethusd', 10000, True, True, False),
                     'etheur': ('etheur', 10000, True, True, False),
                     'btcjpy': ('jp', 200, False, False, False)}
    index, big_trades_threshold, regroup_small_values, plot_kpi, use_mtgox_data = pair_to_index[pair]

    if not os.path.exists(P_DIR):
        os.makedirs(P_DIR)
    if not os.path.exists(ranking_directory):
        os.makedirs(ranking_directory)

    if to_ts == 'today':
        to_ts = int(dt.datetime.utcnow().timestamp())

    if args_dict['index'] is not None:
        index = args_dict['index']
    if args_dict['big_trades_threshold'] is not None:
        big_trades_threshold = args_dict['big_trades_threshold']
    if args_dict['regroup_small_values'] is not None:
        regroup_small_values = str_bool(args_dict['regroup_small_values'])
    if args_dict['plot_kpi'] is not None:
        plot_kpi = str_bool(args_dict['plot_kpi'])
    if args_dict['use_mtgox_data'] is not None:
        use_mtgox_data = str_bool(args_dict['use_mtgox_data'])

    # Once all variables are ready, import modules
    print('\n> Kaiko Market Report charting... \n')

    # Global config files
    exchanges_api = config["exchanges_string_by_pair"][pair]
    xchanges = config["exchanges_list_by_pair"][pair]
    if args_dict['exchange'] is not None:
        xchanges = args_dict['exchange']
        exchanges_api = ','.join(config["exchanges"][xch]["api_name"] for xch in xchanges)
    symbols = config["symbols"]
    names = {xch: config["exchanges"][xch]["name"] for xch in xchanges}
    api_names = {xch: config["exchanges"][xch]["api_name"] for xch in xchanges}
    colors = {xch: config["exchanges"][xch]["color"] for xch in xchanges}
    currency = config["currency"][index]
    cryptocurrency = pair[:3]
    # order books specific config
    thresholds = sorted([int(th) for th in config["thresholds"].keys()])
    color_by_thresh = config["thresholds"]

    # VOLATILITY AND PRICE STD DEVIATION
    if process_volatility:
        print('\n> processing volatility:')

        if use_mtgox_data:
            mtgox = volatility_price.get_mtgox_data(directory + 'mtgox/', currency)
        else:
            mtgox = None
        vol = volatility_price.get_volatility(index, from_ts1, to_ts, 'day', currency, mtgox=mtgox)
        kpi = volatility_price.get_price_index(index, from_ts1, to_ts, 'day', currency, mtgox=mtgox)
        prices = volatility_price.get_historical_prices_from_db(pair, from_ts, to_ts, xchanges, currency)
        std_dev = volatility_price.get_price_std_deviation(pair, from_ts1, to_ts, xchanges, exchanges_api, symbols,
                                                           'day')

        volatility_price.plot_price_deviation(prices, index, pair, xchanges, names, colors, from_ts, to_ts,
                                              plotly_directory, auto_open_charts, P_DIR=P_DIR)
        volatility_price.plot_std_dev(std_dev, kpi, index, pair, currency, plotly_directory, auto_open_charts,
                                      P_DIR=P_DIR)
        volatility_price.plot_volatility(vol, kpi, index, pair, currency, plotly_directory, auto_open_charts,
                                         P_DIR=P_DIR)
        volatility_price.plot_std_dev_volatility(std_dev, vol, pair, plotly_directory, auto_open_charts, P_DIR=P_DIR)

    # TRANSACTIONS
    if process_trades:
        print('\n> processing trades:')
        trades_per_bucket = {}
        volumes_per_bucket = {}
        trades_per_bucket6 = {}
        volumes_per_bucket6 = {}
        median_price = {}
        median_price6 = {}
        trades = {}
        big_trades = pd.DataFrame({'exchange': [], 'date': [], 'amount': []})

        kpi = volatility_price.get_price_index(index, from_ts, to_ts, 'day', currency)
        volumes = transactions.get_volumes(pair, from_ts, to_ts, 'week', exchanges_api, symbols)
        for xch in xchanges:
            trades_per_bucket6[xch], volumes_per_bucket6[xch] = transactions.get_volumes_per_bucket(
                directory + 'trades/', xch, pair, from_ts, to_ts,
                api_names, regroup_small_values)
            trades[xch] = [None, None]
            temp, trades[xch][0], trades[xch][1] = transactions.read_trades(directory + 'trades/', xch, pair,
                                                                            from_ts, to_ts, api_names,
                                                                            big_trades_threshold)
            #        temp = get_big_trades(directory+'trades/',xch,pair,from_ts,to_ts,api_names,big_trades_threshold)
            big_trades = big_trades.append(temp, ignore_index=False)
            temp = None

        transactions.plot_volumes_price(volumes, kpi, index, pair, xchanges, names, colors, currency, cryptocurrency,
                                        plot_kpi,
                                        plotly_directory, auto_open_charts, P_DIR=P_DIR)
        transactions.plot_market_share(volumes, pair, xchanges, names, colors, from_ts, to_ts, plotly_directory,
                                       auto_open_charts, P_DIR=P_DIR)
        transactions.plot_big_trades(big_trades, kpi, index, pair, big_trades_threshold, xchanges, names, colors,
                                     currency, cryptocurrency,
                                     plot_kpi, plotly_directory, auto_open_charts, P_DIR=P_DIR)
        transactions.plot_count_histogram(trades_per_bucket6, pair, xchanges, names, colors, cryptocurrency, from_ts,
                                          to_ts,
                                          plotly_directory, auto_open_charts, P_DIR=P_DIR)
        transactions.plot_volume_histogram(volumes_per_bucket6, pair, xchanges, names, colors, cryptocurrency, from_ts,
                                           to_ts,
                                           plotly_directory, auto_open_charts, P_DIR=P_DIR)
        transactions.plot_price_distribution(trades_per_bucket6, volumes_per_bucket6, pair, xchanges, names, colors,
                                             cryptocurrency,
                                             from_ts, to_ts, regroup_small_values, plotly_directory, auto_open_charts,
                                             P_DIR=P_DIR)
        transactions.plot_boxplot(trades, pair, xchanges, names, cryptocurrency, from_ts, to_ts, P_DIR=P_DIR)

    # ORDERBOOKS
    if process_orderbooks:
        print('\n> processing order books:')

        vb = orderbooks.get_volume_book(directory + 'vb/', from_ts, to_ts, pair, 'day')
        orderbooks.plot_liquidity(vb, pair, xchanges, names, colors, cryptocurrency, from_ts, to_ts, plotly_directory,
                                  auto_open_charts, P_DIR=P_DIR)
        for xch in xchanges:
            orderbooks.plot_exchange_liquidity(vb, xch, pair, names, colors, cryptocurrency, from_ts, to_ts,
                                               plotly_directory,
                                               auto_open_charts, P_DIR=P_DIR)

        vb1 = orderbooks.get_volume_book(directory + 'vb/', from_ts, to_ts, pair, 'week')
        orderbooks.plot_liquidity_stacked(vb1, pair, xchanges, names, colors, cryptocurrency, from_ts, to_ts,
                                          plotly_directory,
                                          auto_open_charts, P_DIR=P_DIR)

        sb = orderbooks.get_spread_book(directory + 'sb/', from_ts, to_ts, pair, 'day')
        for xch in xchanges:
            orderbooks.plot_spread_per_threshold(sb, xch, thresholds, pair, names, color_by_thresh, currency,
                                                 cryptocurrency,
                                                 plotly_directory, auto_open_charts, P_DIR=P_DIR)

        sb2 = orderbooks.get_spread_book(directory + 'sb/', from_ts, to_ts, pair, 'week')
        for th in thresholds:
            orderbooks.plot_spread_per_exchange(sb2, th, pair, xchanges, currency, names, colors, cryptocurrency,
                                                plotly_directory,
                                                auto_open_charts, P_DIR=P_DIR)

        from_ts2 = (dt.datetime.utcfromtimestamp(to_ts) - dt.timedelta(days=2))
        from_ts2 = dt.datetime.timestamp(dt.datetime(from_ts2.year, from_ts2.month, 1))
        sb1 = orderbooks.get_spread_book(directory + 'sb/', from_ts2, to_ts, pair, 'month')
        orderbooks.plot_month_summary(sb1, thresholds, pair, xchanges, names, color_by_thresh, currency, from_ts2,
                                      to_ts,
                                      plotly_directory, auto_open_charts, P_DIR=P_DIR)

    # BUBBLE CHARTS
    if process_bubbles:
        print('\n> processing bubbles:')

        means, volumes, counts, liquidities = {}, {}, {}, {}
        vb = orderbooks.get_volume_book(directory + 'vb/', from_ts, to_ts, pair, resolution='total')
        for xch in xchanges:
            temp = transactions.get_mean_volume_count(directory + 'trades/', xch, pair, from_ts, to_ts,
                                                      api_names)
            if temp[2] != 0:
                means[xch], volumes[xch], counts[xch] = temp
                liquidities[xch] = vb[vb['exchange'] == xch]['ask_volume'].iloc[0] + \
                                   vb[vb['exchange'] == xch]['bid_volume'].iloc[0]

        bubbles.plot_bubble_chart(volumes, liquidities, counts, pair, xchanges, names, colors, plotly_directory,
                                  auto_open_charts, P_DIR=P_DIR)
        print(
            '\n Warning, for an unknown reason, bubble charts are not correctly saved locally. Do it manually from plot.ly')

    # RANKINGS
    if process_rankings:
        print('\n> processing rankings:')

        vrk, vdata = rankings.volume_ranking(pair, from_ts, to_ts, xchanges, exchanges_api, symbols)
        lrk, ldata = rankings.liquidity_ranking(directory + 'vb/', pair, from_ts, to_ts, xchanges)
        wrk, wdata = rankings.weight_ranking(index, from_ts, to_ts, xchanges, symbols)

        wrk.to_excel(ranking_directory + 'weight_rk_' + pair + '.xlsx', index=False)
        lrk.to_excel(ranking_directory + 'liquidity_rk_' + pair + '.xlsx', index=False)
        vrk.to_excel(ranking_directory + 'volume_rk_' + pair + '.xlsx', index=False)
        wdata.to_excel(ranking_directory + 'weight_data_' + pair + '.xlsx', index=False)
        ldata.to_excel(ranking_directory + 'liquidity_data_' + pair + '.xlsx', index=False)
        vdata.to_excel(ranking_directory + 'volume_data_' + pair + '.xlsx', index=False)

    print('\nLocal ranking directory %s:' % ranking_directory)
    print('Local chart directory: %s' % P_DIR)
    print('Online (plotly) chart directory: %s' % plotly_directory)

    print('\n> done!\n')
