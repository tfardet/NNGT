#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Tools to plot graph properties """

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

from ..analysis import degree_distrib, betweenness_distrib


def degree_distribution(network, deg_type="total", use_weights=True,
                        logx=False, logy=False):
    '''
    Plotting the degree distribution of a graph.
    
    Parameters
    ----------
    graph : :class:`Graph` or subclass
        the graph to analyze.
    deg_type : string or tuple, optional (default: "total")
        type of degree to consider ("in", "out", or "total")
    use_weights : bool, optional (default: True)
        use weighted degrees (do not take the sign into account : all weights
        are positive).
    logx : bool, optional (default: False)
        use log-spaced bins.
    logy : bool, optional (default: False)
        use logscale for the degree count.
    '''
    fig, ax1 = plt.subplots(111)
    if isinstance(deg_type, str):
        counts,bins = degree_distrib(network.graph, deg_type, use_weights, logx)
        line = ax1.scatter(bins, counts)
        s_legend = deg_type[0].upper() + deg_type[1:] + " degree"
        fig.legend(line, s_legend, 'upper left')
    else:
        for s_type in deg_type:
            counts,bins = degree_distrib(network.graph, deg_type, use_weights,
                                        logx)
            line = ax1.scatter(bins, counts)
            s_legend = deg_type[0].upper() + deg_type[1:] + " degree"
            fig.legend(line, s_legend, 'upper left')
    if logx:
        ax1.set_xscale("log")
    if logy:
        ax1.set_yscale("log")
    ax1.set_title("Degree distribution for {}".format(network.name))
    plt.show()
            
def betweenness_distrib(network, use_weights=True, logx=False, logy=False):
    pass
