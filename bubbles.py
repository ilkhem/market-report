# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:43:02 2016

@author: ilkhem
"""
__author__ = 'ilkhem'

import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py

from utils import save_plot


# %%

def plot_bubble_chart(x, y, s, pair, xchanges, names, colors, plotly_directory='', auto_open_charts=True,
                      P_DIR='plots/'):
    """
    plot bubble chart with given x, y and size s
    :param x: dict of x values by exchange
    :param y: dict of y values by exchange
    :param s: dict of sizes by exchange. Sizes are then lineary scaled using get_coeff and scale
    :param pair: pair
    :param xchanges: list of exchanges to be plotted
    :param names: names
    :param colors: colors
    :param plotly_directory: plotly directory to where save the plot
    :param auto_open_charts: open charts in browser if True, default True
    :param P_DIR: local dir for plots
    :return: nothing
    """

    def get_coeff(x, l=6, h=20):
        m = np.min([x[y] for y in x])
        M = np.max([x[y] for y in x])
        a = (h - l) / (M - m)
        b = h - a * M
        return a, b

    def scale(x, coeff):
        return coeff[0] * x + coeff[1]

    coeff = get_coeff(s, 10, 25)
    trace = []

    for xch in xchanges:
        try:
            trace += [go.Scatter(x=x[xch], y=y[xch], textposition='right', hoverinfo='none',
                                 mode='text+markers', name=names[xch], text=names[xch],
                                 marker=dict(color=colors[xch], size=scale(s[xch], coeff)))
                      ]
        except:
            pass
    title = 'Bubble Chart (number of trades) (' + pair.upper() + ')'
    layout = go.Layout(
        title=title,
        showlegend=False,
        xaxis=dict(
            title='Volume'
        ),
        yaxis=dict(
            title='Liquidity'
        ),
    )
    fig = go.Figure(data=trace, layout=layout)
    py.image.save_as(fig, 'test.png', width=1000, height=750)
    py.plot(fig, filename=plotly_directory + pair + '-bubbles', auto_open=auto_open_charts, sharing='private')
    save_plot(fig, P_DIR, title.lower())
