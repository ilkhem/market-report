# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:19:23 2016

@author: ilkhem
"""
__author__ = 'ilkhem'

import argparse
import calendar
import os
import time

import pandas as pd

import aggregators
from config import config
from utils import parse_month


def process_orderbooks(main_directory, sub_folder, output, pair, year, month, thresholds=[1, 5, 10, 50, 100, 500, 1000],
                       start_day='01', finish_day=None):
    '''
    Creates spread books and volume (liquidity) books from ob_1 and ob_10 dumps
    for spread books, columns are: date, exchange, sp1, sp5, ..., sp1a, sp5a, ... (spreads depend on thresholds, 1,5 are exemples)
    for volume books, columns are: date, exchange, ask_volume, bid_volume
    :param main_directory: data directory storing ob dumps as downloaded by aws.py
    :param sub_folder: ob_1 or ob_10
    :param output: sb or vb
    :param pair: pair to be processed
    :param year: year to be processed
    :param month: month to be processed
    :param thresholds: thresholds for spread books, ignored when processing volume books, default=[1,5,10,50,100,500,1000]
    :param start_day: optional, day from which to start processing
    :param finish_day: optional, included, day until which process data.
    :return: spread book or volume book for the given pair including a date column and an exchange column
    '''
    print('Creating %s from %s for %s, %s %s' % (output, sub_folder, pair, month, year))
    symbols = config["symbols"]
    df = pd.DataFrame()

    if output == 'sb':
        getter = aggregators.get_sb_from_csv
    elif output == 'vb':
        getter = aggregators.get_vb_from_csv
    else:
        print('output must be either "sb" or "vb"')
        return
    start_time = time.time()
    if finish_day is None:
        finish_day = str(calendar.monthrange(int(year), int(month))[1])
    directory = main_directory + sub_folder + '/' + pair + '/' + year + '/' + month + '/'
    for dirpath, dirnames, files in os.walk(directory):
        for file in files:
            s = file.split('.csv')[0].split('_')[-1]
            if int(start_day) <= int(s) <= int(finish_day):
                try:
                    print('reading', dirpath + '/' + file)
                    xch = symbols[file.split('_')[0].lower()]
                    df = df.append(getter(dirpath + '/' + file, xch, thresholds))
                except:
                    pass

    if not os.path.exists(main_directory + output + '/'):
        os.makedirs(main_directory + output + '/')

    print('saving to ' + main_directory + output + '/' + output + '_' + pair + '_' + year + '_' + month + '.csv')
    df.to_csv(main_directory + output + '/' + output + '_' + pair + '_' + year + '_' + month + '.csv', index=False)
    print('done !')
    # print("--- %s seconds ---" % (time.time() - start_time))
    return df


# testing
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='transforms orderbooks into spread books and volume (liquidity) books')
    parser.add_argument('data_dir', help='directory that contains data')
    parser.add_argument('sub_folder', help='sub-folder to be treated: ob_1 for liquidity and ob_10 for spreads')
    parser.add_argument('output', help='desired output: vb (for volume books) or sb (for spread books)')
    parser.add_argument('pair', help='pair')
    parser.add_argument('year', help='year to be aggregated')
    parser.add_argument('month', help='month to be aggregated')
    parser.add_argument('--from', default='01', help='starting day, optional')
    parser.add_argument('--to', help='finish day, optional')
    parser.add_argument('-t', action='append', dest='thresholds', type=int,
                        help='Use to override thresholds defined in .cfg file. NOT RECOMMANDED. '
                             'This will only modify the thresholds in the aggregated files, and '
                             'will lead to KeyError when launching main IF .cfg file is not updated. '
                             'Only use if you want to have spread books with different spreads not to be used '
                             'with the main file. \n'
                             'E.g.: -t 1 -t 2 -t 3 will add 1,2,3 to []. Thresholds would be [1,2,3]. -t 1 2 3 raises an Error')

    args_dict = vars(parser.parse_args())

    main_directory = args_dict['data_dir']
    if main_directory[-1] != '/':
        main_directory += '/'
    sub_folder = args_dict['sub_folder']
    output = args_dict['output']
    pair = args_dict['pair']
    year = args_dict['year']
    month = parse_month(args_dict['month'])
    start_day = args_dict['from']
    finish_day = args_dict['to']
    if args_dict['thresholds'] is not None:
        thresholds = args_dict['thresholds']
    else:
        thresholds = sorted([int(th) for th in config["thresholds"].keys()])

    process_orderbooks(main_directory, sub_folder, output, pair, year, month, thresholds, start_day, finish_day)
