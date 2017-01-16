# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:42:00 2016

@author: ilkhem
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pprint
import statsmodels.tsa.stattools as ts
import statsmodels.formula.api as sm
from numpy import log, polyfit, sqrt, std, subtract
from volatility_price import get_price_index


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)
    
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    
    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


def plot_price_series(df, ts1, ts2):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.show()
    
    
def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()
    

def plot_residuals(df):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()
    plt.show()
    

def estimate_parameters(df):
    """estimate the parameters theta and mu for dZ = Theta*(Mu - Zt) + dWt"""
    # df must have a "res" column    
    dat = pd.DataFrame(index = df.index)
    dz = []
    for i in range(len(df)-1):
        dz += [df['res'][i+1] - df['res'][i]]
    dz += [np.nan]
    dat['Z'] = df['res']
    dat['dZ'] = dz
    dat.drop(dat.index[-1],inplace=True)
    res = sm.ols('dZ ~ Z',dat).fit()
    Theta = -log(1+res.params['Z'])
    Mu = res.params['Z']/res.params['Intercept']
    return Theta, Mu
    
if __name__ == "__main__":
    
    ts1 = "EU"
    ts2 = "CN"
    
    us = get_price_index(ts1.lower(),1451602800,1466774444,'day').sort_values('timestamp').set_index('timestamp')
    cn = get_price_index(ts2.lower(),1451602800,1466774444,'day').sort_values('timestamp').set_index('timestamp') 

    df = pd.DataFrame(index=us.index)
    df[ts1] = us["c"]
    df[ts2] = cn["c"]

    # Plot the two time series
    plot_price_series(df, ts1, ts2)

    # Display a scatter plot of the two time series
    plot_scatter_series(df, ts1, ts2)

    # Calculate optimal hedge ratio "beta"
    res = sm.ols(ts2+" ~ "+ts1,df).fit()
    beta_hr = res.params[ts1]

    # Calculate the residuals of the linear combination
    df["res"] = df[ts2] - beta_hr*df[ts1]

    # Plot the residuals
    plot_residuals(df)

    # Calculate and output the CADF test on the residuals
    # cadf = ts.adfuller(df["res"],1,autolag=None)
    cadf = ts.adfuller(df["res"])
    pprint.pprint(cadf)
    print('Theta = %s, process half life: %s' % (estimate_parameters(df)[0],log(2)/estimate_parameters(df)[0]))
    # Calculate and output the Hurst exponent
    print("Hurst Exponent: %s" % hurst(df['res']))