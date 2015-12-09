#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Tools to plot graph properties """

import matplotlib.pyplot as plt
import numpy as np

from .custom_plt import palette, format_exponent
from ..analysis import degree_distrib, betweenness_distrib


__all__ = [
            'degree_distribution',
            'betweenness_distribution',
            '_set_new_plot'
          ]



#-----------------------------------------------------------------------------#
# Figure management
#------------------------
#

def _set_new_plot(fignum, num_new_plots=1):
    # get the figure and compute the new number of rows and cols
    fig = plt.figure(num=fignum)
    num_axes = len(fig.axes)+num_new_plots
    num_rows = max(int(np.floor(np.sqrt(num_axes))),1)
    ratio = num_axes/float(num_rows)
    num_cols = int(ratio)+1 if int(ratio)!=int(np.ceil(ratio)) else int(ratio)
    # change the geometry
    for i in range(num_axes-num_new_plots):
        fig.axes[i].change_geometry(num_rows, num_cols, i+1)
    lst_new_axes = []
    n_old = num_axes-num_new_plots+1
    for i in range(num_new_plots):
        lst_new_axes.append(fig.add_subplot(num_rows, num_cols, n_old+i))
    return fig, lst_new_axes


#-----------------------------------------------------------------------------#
# Plotting distributions
#------------------------
#

def degree_distribution(network, deg_type="total", node_list=None, num_bins=50,
                        use_weights=True, logx=False, logy=False, fignum=None,
                        show=True):
    '''
    Plotting the degree distribution of a graph.
    
    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        the graph to analyze.
    deg_type : string or tuple, optional (default: "total")
        type of degree to consider ("in", "out", or "total")
    node_list : list or numpy.array of ints, optional (default: None)
        Restrict the distribution to a set of nodes (default: all nodes).
    use_weights : bool, optional (default: True)
        use weighted degrees (do not take the sign into account : all weights
        are positive).
    logx : bool, optional (default: False)
        use log-spaced bins.
    logy : bool, optional (default: False)
        use logscale for the degree count.
    show : bool, optional (default: True)
        Show the Figure right away if True, else keep it warm for later use.
    '''
    fig, lst_axes = _set_new_plot(fignum)
    ax1 = lst_axes[0]
    ax1.axis('tight')
    maxcounts,maxbins,minbins = 0,0,np.inf
    if isinstance(deg_type, str):
        counts,bins = degree_distrib(network, deg_type, node_list,
                                     use_weights, logx, num_bins)
        maxcounts,maxbins,minbins = counts.max(),bins.max(),bins.min()
        line = ax1.scatter(bins, counts)
        s_legend = deg_type[0].upper() + deg_type[1:] + " degree"
        ax1.legend((s_legend,))
    else:
        colors = palette(np.linspace(0.,0.5,len(deg_type)))
        m = ["o","s","D"]
        lines, legends = [], []
        for i,s_type in enumerate(deg_type):
            counts,bins = degree_distrib(network, s_type, node_list,
                                         use_weights, logx, num_bins)
            maxcounts_tmp,mincounts_tmp = counts.max(),counts.min()
            maxbins_tmp,minbins_tmp = bins.max(),bins.min()
            maxcounts = max(maxcounts,maxcounts_tmp)
            maxbins = max(maxbins,maxbins_tmp)
            minbins = min(minbins,minbins_tmp)
            lines.append(ax1.scatter(bins, counts, c=colors[i], marker=m[i]))
            legends.append(s_type[0].upper() + s_type[1:] + " degree")
        ax1.legend(lines, legends)
    ax1.set_xlim([0.9*minbins, 1.1*maxbins])
    ax1.set_ylim([0, 1.1*maxcounts])
    if logx:
        ax1.set_xscale("log")
        ax1.set_xlim([max(0.8,0.8*minbins), 1.5*maxbins])
    if logy:
        ax1.set_yscale("log")
        ax1.set_ylim([0.8, 1.5*maxcounts])
    ax1.set_title("Degree distribution for {}".format(network.name))
    if show:
        plt.show()
            
def betweenness_distribution(network, btype="both", use_weights=True,
                             logx=False, logy=False, fignum=None, show=True):
    '''
    Plotting the betweenness distribution of a graph.
    
    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        the graph to analyze.
    btype : string, optional (default: "both")
        type of betweenness to display ("node", "edge" or "both")
    use_weights : bool, optional (default: True)
        use weighted degrees (do not take the sign into account : all weights
        are positive).
    logx : bool, optional (default: False)
        use log-spaced bins.
    logy : bool, optional (default: False)
        use logscale for the degree count.
    fignum : int, optional (default: None)
        Number of the Figure on which the plot should appear
    show : bool, optional (default: True)
        Show the Figure right away if True, else keep it warm for later use.
    '''
    num_axes = 2 if btype == "both" else 1
    fig, lst_axes  = _set_new_plot(fignum, num_axes)
    ax1 = lst_axes[0]
    ax1.axis('tight')
    ax2 = lst_axes[-1]
    ax2.axis('tight')
    ncounts,nbins,ecounts,ebins = betweenness_distrib(network,
                                                      use_weights, logx)
    colors = palette(np.linspace(0.,0.5,2))
    if btype in ("node","both"):
        line = ax1.plot(nbins, ncounts, c=colors[0], linestyle="--", marker="o")
        ax1.legend(["Node betweenness"],bbox_to_anchor=[1,1],loc='upper right')
        ax1.set_xlim([nbins.min(), nbins.max()])
        ax1.set_ylim([0, 1.1*ncounts.max()])
        ax1.set_xlabel("Node betweenness")
        ax1.set_ylabel("Node count")
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3,2))
    if btype in ("edge","both"):
        line = ax2.scatter(ebins, ecounts, c=colors[1])
        ax2.legend(["Edge betweenness"],bbox_to_anchor=[1,1],loc='upper right')
        ax2.set_xlim([ebins.min(), ebins.max()])
        ax2.set_ylim([0, 1.1*ecounts.max()])
        ax2.set_xlabel("Edge betweenness")
        ax2.set_ylabel("Edge count")
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3,2))
    if btype == "both":
        ax2.legend(["Edge betweenness"],bbox_to_anchor=[1.,0.88],loc='upper right')
        ax1.legend(["Node betweenness"],bbox_to_anchor=[1.,0.88],loc='lower right')
        ax1.spines['top'].set_color('none')
        ax1.spines['right'].set_color('none')
        plt.subplots_adjust(top=0.85)
        ax2.patch.set_visible(False)
        ax2.xaxis.set_label_position("top")
        ax2.grid(False)
        ax2.xaxis.tick_top()
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        plt.title("Betweenness distribution for {}".format(network.name), y=1.12)
        ax2 = format_exponent(ax2, 'x', (1.,1.1))
        ax1 = format_exponent(ax1, 'x', (1.,-0.05))
    else:
        plt.title("Betweenness distribution for {}".format(network.name))
    if logx:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    if logy:
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    if show:
        plt.show()
