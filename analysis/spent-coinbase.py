import argparse
import json
import os
import time
import urllib.request

import bs4
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py

soup = bs4.BeautifulSoup(urllib.request.urlopen('https://www.walletexplorer.com/').read(), 'lxml')
EXCHANGES = [x.get_text() for x in soup.find_all('td')[0].find_all('a') if 'old' not in x.get_text()]
POOLS = [x.get_text() for x in soup.find_all('td')[1].find_all('a') if 'old' not in x.get_text()]
SERVICES = [x.get_text() for x in soup.find_all('td')[2].find_all('a') if 'old' not in x.get_text()]
GAMBLING = [x.get_text() for x in soup.find_all('td')[3].find_all('a') if 'old' not in x.get_text()]


def load_wallet(url, limit):
    print('loading wallet at %s' % url)
    df = pd.DataFrame()
    for i in range(1, limit + 1):
        df = df.append(
            pd.read_csv(url + '?format=csv&page={0}'.format(i), header=1, index_col='date', parse_dates=True))
    return df


def get_cb_transactions(df):
    """transactions sent from coinbase have '-' in 'received from' column"""
    return df.loc[dat['received from'] == '-', 'transaction']


def get_next_tx(tx):
    tx_url = 'https://www.walletexplorer.com/txid/%s' % tx
    try:
        soup = bs4.BeautifulSoup(urllib.request.urlopen(tx_url).read(), 'lxml')
        time.sleep(0.5)
    except Exception as e:
        print(e)
        print(tx)
        return None
    for x in soup.find_all('a'):
        if x.get_text() == 'next tx':
            return x['href'].split('/')[-1]
    return None


def get_next_tx_2(tx):
    print('getting next tx for %s' % tx)
    tx_url = 'https://api.kaiko.com/v1/transactions/%s' % tx
    js = json.loads(urllib.request.urlopen(tx_url).read().decode('utf-8'))
    return js['outputs'][0]['next_hash']


def get_spent_cb_transaction_hash(df):
    print('getting next transactions')
    res = df.to_frame(name='cb_tx')
    res['next_tx'] = res['cb_tx'].map(get_next_tx_2)
    return res


def assign_type(s):
    if s in EXCHANGES or s.split('-old')[0] in EXCHANGES:
        return 'exchange'
    elif s in POOLS or s.split('-old')[0] in POOLS:
        return 'pool'
    elif s in SERVICES or s.split('-old')[0] in SERVICES:
        return 'service'
    elif s in GAMBLING or s.split('-old')[0] in GAMBLING:
        return 'gambling'
    else:
        return 'unknown'


def get_output_addr(tx):
    print('getting outputs for %s' % tx)
    if tx is not None:
        tx_url = 'https://www.walletexplorer.com/txid/%s' % tx
        try:
            df = pd.read_html(tx_url, attrs={'class': 'empty'})[1]
        except Exception as e:
            print(e)
            print(tx)
            return pd.DataFrame()
        else:
            df.columns = ['addr', 'unique_addr', 'amount', 'spent']
            df['spent'] = df['spent'] == 'next tx'
            df['category'] = df['unique_addr'].map(assign_type)
            return df
    else:
        return pd.DataFrame()


def first_layer(df):
    print('exploring first layer')
    res = pd.DataFrame()
    for tx in df['next_tx'].unique():
        res = res.append(get_output_addr(tx).assign(next_tx=tx), ignore_index=True)
    print('first layer generated')
    return res.set_index(['next_tx', 'addr'])[['unique_addr', 'amount', 'spent', 'category']]


def pie_plots(fl, miner):
    # category count:
    cat_count = {x: list(fl.category).count(x) for x in fl.category.unique()}

    # important unique address count
    addr_count = {}
    for x in fl.unique_addr.unique():
        if assign_type(x) == 'unknown':
            if list(fl.unique_addr).count(x) >= 0.005 * len(fl):
                addr_count[x] = list(fl.unique_addr).count(x)
        else:
            addr_count[x] = list(fl.unique_addr).count(x)

    # plot categories
    labels = sorted([x for x in cat_count])
    values = [cat_count[l] for l in labels]
    trace = [
        go.Pie(
            values=values,
            labels=labels,
            name='category',
        )
    ]
    layout = go.Layout(
        title='%s: Output categories' % miner,
    )
    fig = go.Figure(data=trace, layout=layout)
    url1 = py.plot(fig, filename='test/%s-category-count' % miner, sharing='private')

    # plot addresses
    labels = sorted([x for x in addr_count])
    values = [addr_count[l] for l in labels]
    trace = [
        go.Pie(
            values=values,
            labels=labels,
            name='address',
        )
    ]
    layout = go.Layout(
        title='%s: Output addresses' % miner,
    )
    fig = go.Figure(data=trace, layout=layout)
    url2 = py.plot(fig, filename='test/%s-addresses-count' % miner, sharing='private')
    return url1, url2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore first layer of spend coinbase bitcoins')
    parser.add_argument('miner',
                        help='miner page as it figures in the url of www.wallet-explorer.com')
    parser.add_argument('-l', '--limit', default=10, action='store', type=int,
                        help='limit of pages to load, default=10')
    parser.add_argument('-d', '--directory', default='', action='store',
                        help="directory to store the output csv, default = ''")
    args = parser.parse_args()
    args_dict = vars(args)

    miner = args_dict['miner']
    limit = args_dict['limit']

    directory = args_dict['directory']
    if directory != '':
        if directory[-1] != '/':
            directory += '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

    url = 'https://www.walletexplorer.com/wallet/%s' % miner
    dat = load_wallet(url, limit)
    df = get_cb_transactions(dat)
    dta = get_spent_cb_transaction_hash(df)
    fl = first_layer(dta)
    fl.to_csv('%s%s_first_layer_limit%s.csv' % (directory, miner, limit))
    print(' saved to %s%s_first_layer_limit%s.csv\n' % (directory, miner, limit))
    plot_urls = pie_plots(fl, miner)
    print(' Plots url:', plot_urls)
