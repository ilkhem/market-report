# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:59:19 2016

@author: ilkhem
"""
__author__ = 'ilkhem'

import datetime as dt

import numpy as np
import pandas as pd
from dateutil import rrule


def parse_month(month):
    """
    transforms an int representing a month into a string
    e.g.: 1 -> '01', 11 -> '11' 
    """
    if len(str(month)) < 2:
        return '0' + str(month)
    else:
        return str(month)


def parse_date(st):
    """
    Tries to parse the string date st, and transforms it to a timestamp.
    st can be a timestamp, or a date.
    :param st: string giving a date. Either a timestamp or a date
    :return: an integer timestamp representing date st
    """
    s = str(st)
    try:
        ts = int(
            (pd.to_datetime(s, infer_datetime_format=True).to_datetime() - dt.datetime(1970, 1, 1)).total_seconds())
    except:
        if len(s) == 10:
            ts = int(s)
        else:
            print('Wrong date format')
            raise TypeError
    return ts


def eod_to_ts(ts):
    """
    given a ts, either return it, or the timestamp of the end of the day if the input timestamp is of the start of a day
    i.e: timestamp for 2016-01-01 will return timestamop for 2016-01-01 23:59:59, timestamp for 2016-01-01 12:23:43 will
    remain the same.
    This function helps simplify defining to_ts if the whole end day is to be considered in the processing.
    For exemple, if charts are needed for 3 entire days, from_ts should be midnight for the first day, to_ts can be
    midnight for the third day, and it will be autmatically transformed into a minute before midnight of the fourth day.
    :param ts: timestamp in SECONDS
    :return: a timestamp in SECONDS
    """
    t = pd.to_datetime(ts, unit='s')
    if not t.floor('D') == t:  # ts is not a timestamp for the start of a day
        return ts  # hence return it
    else:  # ts is the timestamp of the start of a day
        return ts + 60 * 60 * 24 - 1  # hence return the timestamp of the start of the next day minus 1 second


def get_dates(from_ts, to_ts, parsed=False):
    """
    returns a list of months that ranges from from_ts to to_ts
    if parsed, also returns the starting month's year and month 
    and the finishing month's year and month
    -> returns
    or 
    -> returns dates, [starting year, starting month], [ending year, ending month]
    """
    start_month = dt.datetime.utcfromtimestamp(from_ts).month
    start_year = dt.datetime.utcfromtimestamp(from_ts).year
    finish_month = dt.datetime.utcfromtimestamp(to_ts).month
    finish_year = dt.datetime.utcfromtimestamp(to_ts).year
    dates = list(rrule.rrule(rrule.MONTHLY, dtstart=dt.datetime(start_year, start_month, 1),
                             until=dt.datetime(finish_year, finish_month, 1)))
    if not parsed:
        return dates
    else:
        return dates, [str(start_year), parse_month(start_month)], [str(finish_year), parse_month(finish_month)]


def get_closest_minute(t):
    """
    returns a datetime.datetime object giving the closest minute to a given timestamp
    t : timestamp in MILLISECONDS 
    """
    ts = dt.datetime.utcfromtimestamp(t / 1000)
    s = ts.second
    if s < 30:
        return dt.datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute)
    else:
        return dt.datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute) + dt.timedelta(minutes=1)


def fmt(z):
    """
    format big integers to short strings
    e.g. : 15000 -> 15k / 1200000 -> 1.2m
    """
    x = int(z)
    if x < 1000:
        return str(x)
    elif x < 1000000 and x >= 1000:
        return str(x // 1000) + '.' + str(x % 1000)[:1] + 'k'
    else:
        return str(x // 1000000) + '.' + str(x % 1000000)[:2] + 'm'


def nanget(x, xch):
    """a wrapper to get that returns nan when get raises a KeyError exception"""
    try:
        return x[xch]
    except KeyError:
        return np.nan


def to_float(s):
    """
    a float() wrapper to transform non string objects into np.nan
    """
    try:
        return (float(s))
    except:
        return np.nan


def get_monday(date, offset=0):
    """
    :param date: datetime object
    :param offset: hours after midnight
    :return: monday of the week
    """
    return dt.datetime(date.year, date.month, date.day, offset) - dt.timedelta(days=date.weekday())


def str_month(m):
    """takes in a month (string or int) and returns its name"""
    month_to_monthStr = {'01': 'January', '02': 'February', '03': 'March', '04': 'April', '05': 'May', '06': 'June',
                         '07': 'July', '08': 'August', '09': 'September', '10': 'October', '11': 'November',
                         '12': 'December'}
    return month_to_monthStr[parse_month(m)]


def save_plot(fig, directory, name, width='1000', height='750'):
    """
    save plotly files locally
    :param fig: a plotly figure (plotly.graph_objs.Figure)
    :param directory: dir to which save the plots locally
    :param name: name to save under, removed in next version
    :param width: image width
    :param height: image height
    :return: nothing
    """
    import plotly.plotly as py
    def get_title(x):
        try:
            return x.split(') ')[1]
        except:
            return x

    def length_legend(fig):
        i = 0
        for x in fig.data:
            if x['showlegend'] != False:
                i += 1
        return i

    if length_legend(fig) > 8:
        fig.layout.legend = dict(orientation='h', font=dict(size=10))
    else:
        fig.layout.legend = dict(orientation='h')
    fig.layout.font = dict(family='helvetica')
    title = fig.layout.title
    fig.layout.title = ''
    sname = ''.join(e for e in title if e.isalnum())
    py.image.save_as(fig, directory + sname + '.png', width=width, height=height)
    print(' saved', title)


def str_bool(s):
    bool_string_to_string = {'False': False, 'True': True, 'false': False,
                             'true': True, 't': True, 'f': False, 'T': True, 'F': False}
    return bool_string_to_string[s]
