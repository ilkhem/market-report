"""
Script for calculating a 'volume-index' equal to the volume normalized by the spread i.e. the volume for a
constant spread of 1USD.


Data to extract from dumps:
from trade dumps: id,exchange,symbol,date,price,amount,sell (14698813,bf,btcusd,1451606422000,429.17,1.6817,false)
    - price per minute: price of last trade of this minute
    - number of trades per minute
    - volume of trades per minute

from ob1 dumps: date,type,price,amount (1451606456000,b,428.98,5.7347)
    - spread per minute


Clipping functions: (alpha equals to 2, but can be changed to change the width of the interval)
f1: clip to [0, mean + alpha*std]
f2: clip to [0, mean + alpha*rolling_std]
f3: clip to [0, rolling_mean + alpha*rolling_std]
f4: clip to [0, clean_mean + alpha*rolling_std] (clean_* is for * calculated by excluding the outliers)
f5: clip to [0, clean_mean + alpha*clean_std] (clean_* is for * calculated by excluding the outliers)
-> f5 showed the best performance overall (especially in crisis periods, and when exchanges go dark), it is
the one we will use.


RATES define CURRENCY to USD conversion rate.
@author: ilkhem
"""
import argparse
import glob
import json
import os
import urllib.request

import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py

from utils import parse_date, parse_month, get_dates, eod_to_ts

__author__ = 'ilkhem'

MISSING_DUMPS = ''


def rates_by_minute(from_ts, to_ts):
    """
    returns CNY to USD, USD to USD and EUR to USD rates from the European Central Bank using the api of fixer.io
    :param from_ts: starting timestamp
    :param to_ts:  ending timestamp
    :return: df; index: dates with freq=1Min, columns: ['btcusd','btccny','btceur'] (pairs used instead of currencies)
    """
    url = 'http://api.fixer.io/%s?%s'
    # date range for rates to be used with the api
    days = pd.date_range(pd.to_datetime(from_ts, unit='s').floor('D'), pd.to_datetime(to_ts, unit='s').floor('D'),
                         freq='D')
    params = 'base=USD&symbols=CNY,EUR'
    # a minute frequency date range to construct a rate by minute df based on the returns of the api,
    # and by forward filling
    rng = pd.date_range(pd.to_datetime(from_ts, unit='s'), pd.to_datetime(to_ts, unit='s'), freq='1Min')
    rates = pd.DataFrame(index=rng)

    for day in days:
        js = json.loads(urllib.request.urlopen(url % (day.date(), params)).read().decode("utf-8"))
        rates.loc[day, 'btcusd'] = 1
        rates.loc[day, 'btceur'] = 1 / js["rates"]["EUR"]
        rates.loc[day, 'btccny'] = 1 / js["rates"]["CNY"]
    # since days may include daytime values not in rng (the day is considered at midnight, if start is not at midinght,
    # this will be the case. So drop the value that is prior to rng[0]
    return rates.sort_index().ffill().drop(rates[rates.index < rng[0]].index)


def ts_to_min(ts):
    """
    transforms a timestamp in MILLISECONDS into the timestamp of the minute it belongs to in SECONDS
    :param ts: timestamp in MILLISECONDS, can be either a string or an int
    :return: timestamp of the minute in SECONDS
    """
    t = int(ts)
    return t - t % 60000


def read_trades(file):
    """
    read trade dumps to extract the price, trade count, and volume per minute
    :param file: path to a trade dump file (.csv like file)
    :return: a list of lists, each element is a list of [timestamp, volume, count, price]
    """
    with open(file) as f:
        f.readline()  # skip header line
        ts = 0  # current timestamp, initialize at 0
        vcp = []  # list of [ts,v,c,p]
        v = 0  # volumes
        c = 0  # count
        p = 0  # price
        for line in f:
            line_split = line.split(',')

            # for a given timestamp, this condition is only true when we first enter a minute
            # we re-initialize all the variables but the outputs at each new timestamp
            if ts != ts_to_min(line_split[3]):
                if c != 0:
                    vcp += [[ts, v, c, p]]
                ts = ts_to_min(line_split[3])
                # reinitialize
                c = 0
                v = 0
                p = 0

            c += 1
            v += float(line_split[5])
            p = float(line_split[4])
        vcp += [[ts, v, c, p]]
    return vcp


def read_ob(file):
    """
    read ob_1/ob_10 dumps to extract the spread per minute. works faster with ob_1
    :param file: path to a ob_1(0) dump file (.csv like file)
    :return: a list of lists, each element is a list of [timestamp, spread]
    """
    with open(file) as f:
        f.readline()  # skip header line
        ts = 0  # current timestamp, initialize at 0
        sl = []  # list of [ts,v,c,p]
        b = 0  # limit bid at current timestamp
        a = 0  # limit ask at current timestamp
        ask = False
        for line in f:
            line_split = line.split(',')

            # for a given timestamp, this condition is only true when we first encounter that timestamp in BID side
            # we re-initialize all the variables but the outputs at each new timestamp
            if ts != ts_to_min(line_split[0]):
                ts = ts_to_min(line_split[0])  # update the timestamp
                b = float(line_split[2])  # the first line for each new timestamp is the limit bid
                ask = False  # we work on the bid side

            # for a given timestamp, this condition is only true when we first encounter that timestamp in ASK side
            # we re-initialize all the variables but the outputs each time we move from bids to asks
            if not ask and line_split[1] == 'a':
                ask = True
                a = float(line_split[2])
                sl += [[ts, a - b]]
    return sl


# clipping functions
def f1(ts, alpha):
    return ts.clip(0,
                   ts.mean() + alpha * ts.std())


def f2(ts, alpha, w):
    return ts.clip(0,
                   ts.mean() + alpha * ts.rolling(w).std().fillna(ts.std()))


def f3(ts, alpha, w1, w2):
    return ts.clip(0,
                   ts.rolling(w1).mean().fillna(ts.mean()) + alpha * ts.rolling(w2).std().fillna(ts.std()))


def f4(ts, alpha, w):
    clean_ts = ts.drop(ts[ts > (ts.mean() + alpha * ts.rolling(w).std().fillna(ts.std()))].index)
    return ts.clip(0,
                   clean_ts.mean() + alpha * ts.rolling(w).std().fillna(ts.std()))


def f5(ts, alpha):
    clean_ts = ts.drop(ts[ts > (ts.mean() + alpha * ts.std())].index)
    return ts.clip(0,
                   clean_ts.mean() + alpha * clean_ts.std())


def process_exchange(directory, xch, pair, from_ts, to_ts, orig=False):
    """
    reads the dumps for a given exchange, extracts the price, volume, trade count and spread per minute,
    filling empty values with 0 for volume and count, and a forward fill for price and spread
    :param directory: general directory of the dumps, as downloaded using aws.py
    :param xch: FULL exchange name (not the slug)
    :param pair: pair
    :param from_ts: timestamp of the starting month, can be a timestamp or a date (e.g.: '2016', '2016-03-31').
    all the month is loaded even if the timestamp is not of the month's start
    :param to_ts: timestamp of the ending month, can be a timestamp or a date (e.g.: '2016', '2016-03-31').
    all the month is loaded even if the timestamp is not of the month's end
    :param orig: return original values before applying a clipping function
    :return: df, or (df,df_orig) if orig. index : date, columns: 's', 'v', 'v', 'p'
    """
    global MISSING_DUMPS  # for modifying global variable

    if directory != '' and directory[-1] != '/':
        directory += '/'
    dates = get_dates(parse_date(from_ts), parse_date(to_ts))
    filepaths = [
        directory + '%s/' + pair + '/' + str(d.year) + '/' + parse_month(d.month) + '/' + xch + '/' for
        d in dates]
    df = pd.DataFrame()
    df_orig = pd.DataFrame()

    for fp in filepaths:
        tr = []
        ob = []
        fpt = fp % 'trades'
        if not os.path.exists(fpt):
            MISSING_DUMPS += fpt + ' not found\n'
        for f in glob.glob(fpt + '*.csv'):
            print(f)
            tr += read_trades(f)
        fpo = fp % 'ob_1'
        if not os.path.exists(fpo):
            MISSING_DUMPS += fpo + ' not found\n'
        for f in glob.glob(fpo + '*.csv'):
            print(f)
            ob += read_ob(f)

        if tr != [] and ob != []:
            tr_df = pd.DataFrame(tr, columns=['date', 'v', 'c', 'p'])
            tr_df = tr_df.set_index(pd.to_datetime(tr_df.date.values / 1000, unit='s')).drop('date', axis=1).resample(
                '1Min').mean().sort_index().fillna({'c': 0, 'v': 0}).ffill()

            ob_df = pd.DataFrame(ob, columns=['date', 's'])
            ob_df = ob_df.set_index(pd.to_datetime(ob_df.date.values / 1000, unit='s')).drop('date', axis=1).resample(
                '1Min').mean().sort_index().ffill()

            ob_df_orig = ob_df.copy()  # keep a copy of the spreads before smoothing

            # smooth spreads
            # first, clip outliers using one of the clipping functions (f1, f2, ..., f5)
            # best performance overall is achieved by using the f5 clip function
            ob_df.s = f5(ob_df.s, 2)
            # second, merge the spreads with volumes and prices
            df = df.append(pd.merge(tr_df, ob_df,
                                    how='inner', left_index=True, right_index=True))
            df_orig = df_orig.append(pd.merge(tr_df, ob_df_orig,
                                              how='inner', left_index=True,
                                              right_index=True))  # a copy of df before smoothing
            # third, remove lines where the spread is higher than 1% of the price
            # since we used the f5 clipping function, this step is just an additional layer of protection against outliers
            df = df.drop(df[df.s >= 0.01 * df.p].index, axis=0)

    # Since reading files is done by months (due to how get_dates is coded), from_ts and to_ts are applied here
    if not df.empty:
        df = df.loc[pd.to_datetime(from_ts, unit='s'):pd.to_datetime(to_ts, unit='s')]
        df_orig = df_orig.loc[pd.to_datetime(from_ts, unit='s'):pd.to_datetime(to_ts, unit='s')]
    if orig:
        return df, df_orig
    return df


def process(directory, xch_list, from_ts, to_ts):
    """
    processes a list of (exchange,pair), and returns the new normalized volume per exchange_pair
    :param directory: general directory of the dumps, as downloaded using aws.py
    :param xch_list: a list of (exchange,pair) pairs to be processed
    :param from_ts: timestamp of the starting month, can be a timestamp or a date (e.g.: '2016', '2016-03-31').
    all the month is loaded even if the timestamp is not of the month's start
    :param to_ts: timestamp of the ending month, can be a timestamp or a date (e.g.: '2016', '2016-03-31').
    all the month is loaded even if the timestamp is not of the month's end
    :return: df. index: date, columns: 'exchange_pair' for exchange, pair in xch_list
    """

    dfs = {}
    df = pd.DataFrame()

    for xch, pair in xch_list:
        dfs[xch + '_' + pair] = process_exchange(directory, xch, pair, from_ts, to_ts)

    if not all([dfs[x].empty for x in dfs]):
        rates = rates_by_minute(from_ts, to_ts)

        df = pd.DataFrame(index=dfs[
            sorted({x: len(dfs[x]) for x in dfs}, key={x: len(dfs[x]) for x in dfs}.get, reverse=True)[0]].index)

        for xch, pair in xch_list:
            try:
                df[xch + '_' + pair] = dfs[xch + '_' + pair]['v'] * ((dfs[xch + '_' + pair]['s'] * rates[pair]) ** 2)
            except KeyError:
                pass

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the volume-index for selected (exchange,pair)s')
    parser.add_argument('directory',
                        help='directory to dumps as downloaded with aws.py')
    parser.add_argument('from_ts',
                        help='start timestamp in seconds')
    parser.add_argument('to_ts',
                        help='end timestamp in seconds')
    parser.add_argument('-e', '--exchange', nargs='+', required=True,
                        help='<Required> List of exchange_pair to be processed. '
                             'Entry should be in this particular format: exchange_pair.'
                             'Exemple use: -e bitfinex_btcusd huobi_btccny')
    parser.add_argument('-vd', '--volume-directory', default='', action='store',
                        help="local directory to store normalized-volume files, default is current folder ")
    args = parser.parse_args()
    args_dict = vars(args)

    directory = args_dict['directory']
    if directory[-1] != '/':
        directory += '/'
    from_ts = parse_date(args_dict['from_ts'])
    to_ts = eod_to_ts(parse_date(args_dict['to_ts']))
    xch_list = [(xp.split('_')[0].lower(), xp.split('_')[1].lower()) for xp in args_dict['exchange']]

    volume_directory = args_dict['volume_directory']
    if volume_directory != '':
        if volume_directory[-1] != '/':
            volume_directory += '/'
        if not os.path.exists(volume_directory):
            os.makedirs(volume_directory)

    print('\nCalculating normalized volume ... \n')

    df = process(directory, xch_list, from_ts, to_ts)

    print('\nprocessing done!\n')

    filename = 'normalized-volume-' + str(from_ts) + '-' + str(to_ts) + '-' + '-'.join(
        xp for xp in args_dict['exchange'])
    if not df.empty:
        df.to_csv('%s%s.csv' % (volume_directory, filename))
        print(' saved to %s%s.csv' % (volume_directory, filename))

        grouped = df.resample('1D').sum()
        grouped.to_csv('%s%s-day.csv' % (volume_directory, filename))
        print(' saved to %s%s-day.csv\n' % (volume_directory, filename))

        # Plotting grouped

        trace = [go.Scatter(x=grouped.index, y=grouped[column], mode='lines+markers', name=column) for column in
                 grouped]
        layout = go.Layout(
            title='Standardized volume (' + ', '.join(x + '_' + p for x, p in xch_list) + ')',
            xaxis=dict(
                title=''
            ),
            yaxis=dict(
                title='Standardized volume',
            ),
        )
        fig = go.Figure(data=trace, layout=layout)
        plot_url = py.plot(fig, sharing='private',
                           filename='standardized-volume/' + '-'.join(x + '_' + p for x, p in xch_list) + '-' + str(
                               from_ts) + '-' + str(to_ts), auto_open=False)

        print(' Plot url:', plot_url)
        print(MISSING_DUMPS)
        print('\n> done!\n')



    else:
        print(MISSING_DUMPS)
        print('\nno data was found, please check your dumps\n')
