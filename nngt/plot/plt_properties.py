#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Tools to plot graph properties """

import matplotlib.pyplot as plt
import numpy as np

from .custom_plt import palette, format_exponent
from nngt.lib import nonstring_container
from nngt.analysis import degree_distrib, betweenness_distrib, node_attributes


__all__ = [
            'degree_distribution',
            'betweenness_distribution',
            'node_attributes_distribution'
          ]


# ---------------------- #
# Plotting distributions #
# ---------------------- #

def degree_distribution(network, deg_type="total", nodes=None, num_bins=50,
                        use_weights=False, logx=False, logy=False, fignum=None,
                        show=True):
    '''
    Plotting the degree distribution of a graph.
    
    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        the graph to analyze.
    deg_type : string or tuple, optional (default: "total")
        type of degree to consider ("in", "out", or "total")
    nodes : list or numpy.array of ints, optional (default: all nodes)
        Restrict the distribution to a set of nodes.
    num_bins : int, optional (default: 50):
        Number of bins used to sample the distribution.
    use_weights : bool, optional (default: False)
        use weighted degrees (do not take the sign into account : only the
        magnitude of the weights is considered).
    logx : bool, optional (default: False)
        use log-spaced bins.
    logy : bool, optional (default: False)
        use logscale for the degree count.
    fignum : int, optional (default: ``None``)
        Index of the figure on which the plot should be drawn (default creates
        a new figure).
    show : bool, optional (default: True)
        Show the Figure right away if True, else keep it warm for later use.
    '''
    fig, lst_axes = _set_new_plot(fignum)
    ax1 = lst_axes[0]
    ax1.axis('tight')
    maxcounts,maxbins,minbins = 0,0,np.inf
    if isinstance(deg_type, str):
        counts,bins = degree_distrib(network, deg_type, nodes,
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
            counts,bins = degree_distrib(network, s_type, nodes,
                                         use_weights, logx, num_bins)
            maxcounts_tmp,mincounts_tmp = counts.max(),counts.min()
            maxbins_tmp,minbins_tmp = bins.max(),bins.min()
            maxcounts = max(maxcounts,maxcounts_tmp)
            maxbins = max(maxbins,maxbins_tmp)
            minbins = min(minbins,minbins_tmp)
            lines.append(ax1.scatter(bins, counts, c=colors[i], marker=m[i]))
            legends.append(s_type[0].upper() + s_type[1:] + " degree")
        ax1.legend(lines, legends)
    ax1.set_title("Degree distribution for {}".format(network.name))
    _set_scale(ax1, maxbins, minbins, maxcounts, logx, logy)
    if show:
        plt.show()


def attribute_distribution(network, attribute, num_bins=50, logx=False,
                           logy=False, fignum=None, show=True):
    '''
    Plotting the distribution of a graph attribute (e.g. "weight", or
    "distance" is the graph is spatial).
    
    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph to analyze.
    attribute : string or tuple of strings
        Name of a graph attribute.
    num_bins : int, optional (default: 50):
        Number of bins used to sample the distribution.
    logx : bool, optional (default: False)
        use log-spaced bins.
    logy : bool, optional (default: False)
        use logscale for the degree count.
    fignum : int, optional (default: ``None``)
        Index of the figure on which the plot should be drawn (default creates
        a new figure).
    show : bool, optional (default: True)
        Show the Figure right away if True, else keep it warm for later use.
    '''
    fig, lst_axes = _set_new_plot(fignum)
    ax1 = lst_axes[0]
    ax1.axis('tight')
    maxcounts, maxbins, minbins = 0, 0, np.inf
    if isinstance(attribute, str):
        values = network.attributes(name=attribute)
        bins = np.logspace(np.log(values.min()), np.log(values.max()),
                           num_bins) if logx else num_bins
        counts, bins = np.histogram(values, bins=bins)
        maxcounts, maxbins, minbins = counts.max(), bins.max(), bins.min()
        line = ax1.scatter(bins[:-1], counts)
        s_legend = attribute
        ax1.legend((s_legend,))
    else:
        raise NotImplementedError("Multiple attribute plotting not ready yet")
        #~ colors = palette(np.linspace(0.,0.5,len(deg_type)))
        #~ m = ["o", "s", "D", "x"]
        #~ lines, legends = [], []
        #~ for i,s_type in enumerate(deg_type):
            #~ counts,bins = degree_distrib(network, s_type, nodes,
                                         #~ use_weights, logx, num_bins)
            #~ maxcounts_tmp,mincounts_tmp = counts.max(),counts.min()
            #~ maxbins_tmp,minbins_tmp = bins.max(),bins.min()
            #~ maxcounts = max(maxcounts,maxcounts_tmp)
            #~ maxbins = max(maxbins,maxbins_tmp)
            #~ minbins = min(minbins,minbins_tmp)
            #~ lines.append(ax1.scatter(bins, counts, c=colors[i], marker=m[i]))
            #~ legends.append(attribute)
        #~ ax1.legend(lines, legends)
    _set_scale(ax1, maxbins, min_bins, maxcounts, logx, logy)
    ax1.set_title("Attribute distribution for {}".format(network.name))
    if show:
        plt.show()


def betweenness_distribution(network, btype="both", use_weights=True,
                             nodes=None, logx=False, logy=False,
                             num_nbins=None, num_ebins=None, fignum=None,
                             show=True):
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
    nodes : list or numpy.array of ints, optional (default: all nodes)
        Restrict the distribution to a set of nodes (taken into account only
        for the node attribute).
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
    ncounts, nbins, ecounts, ebins = betweenness_distrib(
        network, use_weights, nodes=nodes, num_nbins=num_nbins,
        num_ebins=num_ebins, log=logx)
    print(nbins, ncounts)
    colors = palette(np.linspace(0.,0.5,2))
    if btype in ("node", "both"):
        line = ax1.plot(
            nbins, ncounts, c=colors[0], linestyle="--", marker="o")
        ax1.legend(["Node betweenness"],bbox_to_anchor=[1,1],loc='upper right')
        ax1.set_xlabel("Node betweenness")
        ax1.set_ylabel("Node count")
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3,2))
        _set_scale(ax1, nbins.max(), nbins.min(), ncounts.max(), logx, logy)
    if btype in ("edge", "both"):
        line = ax2.scatter(ebins, ecounts, c=colors[1])
        ax2.legend(["Edge betweenness"],bbox_to_anchor=[1,1],loc='upper right')
        ax2.set_xlim([ebins.min(), ebins.max()])
        ax2.set_ylim([0, 1.1*ecounts.max()])
        ax2.set_xlabel("Edge betweenness")
        ax2.set_ylabel("Edge count")
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3,2))
        _set_scale(ax2, ebins.max(), ebins.min(), ecounts.max(), logx, logy)
    if btype == "both":
        ax2.legend(
            ["Edge betweenness"], bbox_to_anchor=[1.,0.88], loc='upper right')
        ax1.legend(
            ["Node betweenness"], bbox_to_anchor=[1.,0.88], loc='lower right')
        ax1.spines['top'].set_color('none')
        ax1.spines['right'].set_color('none')
        plt.subplots_adjust(top=0.85)
        ax2.patch.set_visible(False)
        ax2.xaxis.set_label_position("top")
        ax2.grid(False)
        ax2.xaxis.tick_top()
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        plt.title(
         "Betweenness distribution for {}".format(network.name), y=1.12)
        ax2 = format_exponent(ax2, 'x', (1.,1.1))
        ax1 = format_exponent(ax1, 'x', (1.,-0.05))
    else:
        plt.title("Betweenness distribution for {}".format(network.name))
    if show:
        plt.show()


# ------------------------ #
# Plotting node attributes #
# ------------------------ #

def node_attributes_distribution(network, attributes, nodes=None, num_bins=50):
    '''
    Return node `attributes` for a set of `nodes`.
    
    Parameters
    ----------
    network : :class:`~nngt.Graph`
        The graph where the `nodes` belong.
    attributes : str or list
        Attributes which should be returned, among:
        * "betweenness"
        * "clustering"
        * "in-degree", "out-degree", "total-degree"
        * "subgraph_centrality"
    nodes : list, optional (default: all nodes)
        Nodes for which the attributes should be returned.
    num_bins : int or list, optional (default: 50)
        Number of bins to plot the distributions. If only one int is provided,
        it is used for all attributes, otherwize a list containing one int per
        attribute in `attributes` is required.
    '''
    if not nonstring_container(attributes):
        attributes = [attributes]
    if nonstring_container(num_bins):
        assert len(num_bins) == len(attributes), "One entry per attribute " +\
            "required for `num_bins`."
    else:
        num_bins = [num_bins for _ in range(len(attributes))]
    fig = plt.figure()
    # plot degrees if required
    degrees = []
    for name in attributes:
        if "degree" in name.lower():
            degrees.append(name[:name.find("-")])
    if degrees:
        indices = []
        for i, name in enumerate(attributes):
            if "degree" in name:
                indices.append(i)
        indices.sort()
        deg_bin = int(np.average(np.array(num_bins)[indices]))
        for idx in indices[::-1]:
            del num_bins[idx]
            del attributes[idx]
        degree_distribution(
            network, deg_type=degrees, nodes=nodes, num_bins=deg_bin,
            fignum=fig.number, show=False)
    # plot betweenness if needed
    if "betweenness" in attributes:
        idx = attributes.index("betweenness")
        betweenness_distribution(
            network, btype="nodes", nodes=nodes, fignum=fig.number, show=False)
        del attributes[idx]
        del num_bins[idx]
    # plot the rest
    print(attributes)
    values = node_attributes(network, attributes, nodes=None)
    fig, axes = _set_new_plot(fignum=fig.number, names=attributes)
    for i, (attr, val) in enumerate(values.items()):
        print(attr, axes)
        counts, bins = np.histogram(val, num_bins[i])
        bins = bins[:-1] + 0.5*np.diff(bins)
        axes[i].scatter(bins, counts)
    plt.tight_layout()
    plt.show()


# ----------------- #
# Figure management #
# ----------------- #

def _set_new_plot(fignum=None, num_new_plots=1, names=None, sharex=None):
    # get the figure and compute the new number of rows and cols
    fig = plt.figure(num=fignum)
    num_axes = len(fig.axes) + num_new_plots
    if names is not None:
        num_axes = len(fig.axes) + len(names)
        num_new_plots = len(names)
    num_cols = max(int(np.floor(np.sqrt(num_axes))),1)
    ratio = num_axes/float(num_cols)
    num_rows = int(ratio)+1 if int(ratio)!=int(np.ceil(ratio)) else int(ratio)
    # change the geometry
    for i in range(num_axes - num_new_plots):
        fig.axes[i].change_geometry(num_rows, num_cols, i+1)
    lst_new_axes = []
    n_old = num_axes-num_new_plots+1
    for i in range(num_new_plots):
        if fig.axes:
            lst_new_axes.append( fig.add_subplot(num_rows, num_cols, n_old+i,
                                                 sharex=sharex) )
        else:
            lst_new_axes.append(fig.add_subplot(num_rows, num_cols, n_old+i))
        if names is not None:
            lst_new_axes[-1].name = names[i]
    return fig, lst_new_axes


def _set_scale(ax1, maxbins, minbins, maxcounts, logx, logy):
    ax1.set_xlim([0.9*minbins, 1.1*maxbins])
    ax1.set_ylim([0, 1.1*maxcounts])
    if logx:
        ax1.set_xscale("log")
        ax1.set_xlim([max(0.8,0.8*minbins), 1.5*maxbins])
    if logy:
        ax1.set_yscale("log")
        ax1.set_ylim([0.8, 1.5*maxcounts])
