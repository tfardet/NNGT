#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Tools to plot graph properties """

import matplotlib.pyplot as plt
import numpy as np

from .custom_plt import palette, format_exponent
from nngt.lib import InvalidArgument, nonstring_container
from nngt.analysis import degree_distrib, betweenness_distrib, node_attributes


__all__ = [
    'degree_distribution',
    'betweenness_distribution',
    'node_attributes_distribution',
    'compare_population_attributes',
    "correlation_to_attribute",
]


# ---------------------- #
# Plotting distributions #
# ---------------------- #

def degree_distribution(network, deg_type="total", nodes=None, num_bins=50,
                        use_weights=False, logx=False, logy=False, fignum=None,
                        axis_num=None, colors=None, norm=False, show=True):
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
    fig, lst_axes = plt.figure(fignum), None
    # create new axes or get them from existing ones
    if axis_num is None:
        fig, lst_axes = _set_new_plot(fig.number)
        axis_num = 0
    else:
        lst_axes = fig.get_axes()
    ax1 = lst_axes[axis_num]
    ax1.axis('tight')
    # get degrees
    maxcounts, maxbins, minbins = 0, 0, np.inf
    if isinstance(deg_type, str):
        counts, bins = degree_distrib(network, deg_type, nodes,
                                      use_weights, logx, num_bins)
        if norm:
            counts = counts / float(np.sum(counts))
        maxcounts, maxbins, minbins = counts.max(), bins.max(), bins.min()
        s_legend = deg_type[0].upper() + deg_type[1:] + " degree"
        line = ax1.scatter(bins, counts, label=s_legend)
    else:
        if colors is None:
            colors = palette(np.linspace(0.,0.5, len(deg_type)))
        m = ["o", "s", "D"]
        lines, legends = [], []
        for i,s_type in enumerate(deg_type):
            counts, bins = degree_distrib(network, s_type, nodes,
                                          use_weights, logx, num_bins)
            if norm:
                counts = counts / float(np.sum(counts))
            maxcounts_tmp, mincounts_tmp = counts.max(), counts.min()
            maxbins_tmp, minbins_tmp = bins.max(), bins.min()
            maxcounts = max(maxcounts, maxcounts_tmp)
            maxbins = max(maxbins, maxbins_tmp)
            minbins = min(minbins, minbins_tmp)
            legend = s_type[0].upper() + s_type[1:] + " degree"
            lines.append(ax1.plot(
                bins, counts, ls="--", c=colors[i], marker=m[i], label=legend))
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Node count")
    ax1.set_title(
        "Degree distribution for {}".format(network.name), x=0., y=1.05,
        loc='left')
    _set_scale(ax1, maxbins, minbins, maxcounts, logx, logy)
    plt.legend()
    if show:
        plt.show()


def attribute_distribution(network, attribute, num_bins=50, logx=False,
                           logy=False, fignum=None, axis_num=None, norm=False,
                           show=True):
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
    fig, lst_axes = plt.figure(fignum), None
    # create new axes or get them from existing ones
    if axis_num is None:
        fig, lst_axes = _set_new_plot(fignum)
        axis_num = 0
    else:
        lst_axes = fig.get_axes()
    ax1 = lst_axes[axis_num]
    ax1.axis('tight')
    # get attribute
    maxcounts, maxbins, minbins = 0, 0, np.inf
    if isinstance(attribute, str):
        values = network.attributes(name=attribute)
        bins = np.logspace(np.log(values.min()), np.log(values.max()),
                           num_bins) if logx else num_bins
        counts, bins = np.histogram(values, bins=bins)
        if norm:
            counts /= np.sum(counts)
        bins = bins[:-1] + 0.5*np.diff(bins)
        maxcounts, maxbins, minbins = counts.max(), bins.max(), bins.min()
        line = ax1.plot(bins, counts, linestyle="--", marker="o")
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
    ax1.set_xlabel(attribute.replace("_", "\\_"))
    ax1.set_ylabel("Node count")
    _set_scale(ax1, maxbins, min_bins, maxcounts, logx, logy)
    ax1.set_title(
        "Attribute distribution for {}".format(network.name), x=0., y=1.05,
        loc='left')
    if show:
        plt.show()


def betweenness_distribution(network, btype="both", use_weights=True,
                             nodes=None, logx=False, logy=False,
                             num_nbins=None, num_ebins=None, fignum=None,
                             axis_num=None, colors=None, norm=False,
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
    if btype not in ("node", "edge", "both"):
        raise InvalidArgument('`btype` must be one of the following: '
                              '"node", "edge", "both".')
    num_axes = 2 if btype == "both" else 1
    # create new axes or get them from existing ones
    fig, lst_axes = plt.figure(fignum), None
    if axis_num is None:
        fig, lst_axes  = _set_new_plot(fignum, num_axes)
        axis_num = (0, -1)
    else:
        lst_axes = fig.get_axes()
    ax1 = lst_axes[axis_num[0]]
    ax1.axis('tight')
    ax2 = lst_axes[axis_num[-1]]
    ax2.axis('tight')
    # get betweenness
    ncounts, nbins, ecounts, ebins = betweenness_distrib(
        network, use_weights, nodes=nodes, num_nbins=num_nbins,
        num_ebins=num_ebins, log=logx)
    if norm:
        ncounts = ncounts / float(np.sum(ncounts))
        ecounts = ecounts / np.sum(ecounts)
    # plot
    if colors is None:
        colors = palette(np.linspace(0., 0.5, 2))
    if btype in ("node", "both"):
        line = ax1.plot(
            nbins, ncounts, c=colors[0], linestyle="--", marker="o")
        ax1.legend(
            ["Node betweenness"], bbox_to_anchor=[1, 1], loc='upper right')
        ax1.set_xlabel("Node betweenness")
        ax1.set_ylabel("Node count")
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3, 2))
        _set_scale(ax1, nbins.max(), nbins.min(), ncounts.max(), logx, logy)
    if btype in ("edge", "both"):
        line = ax2.scatter(ebins, ecounts, c=colors[1])
        ax2.legend(
            ["Edge betweenness"], bbox_to_anchor=[1, 1], loc='upper right')
        ax2.set_xlim([ebins.min(), ebins.max()])
        ax2.set_ylim([0, 1.1*ecounts.max()])
        ax2.set_xlabel("Edge betweenness")
        ax2.set_ylabel("Edge count")
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3, 2))
        _set_scale(ax2, ebins.max(), ebins.min(), ecounts.max(), logx, logy)
    if btype == "both":
        ax2.legend(
            ["Edge betweenness"], bbox_to_anchor=[1., 0.88], loc='upper right')
        ax1.legend(
            ["Node betweenness"], bbox_to_anchor=[1., 0.88], loc='lower right')
        ax1.spines['top'].set_color('none')
        ax1.spines['right'].set_color('none')
        plt.subplots_adjust(top=0.85)
        ax2.patch.set_visible(False)
        ax2.xaxis.set_label_position("top")
        ax2.grid(False)
        ax2.xaxis.tick_top()
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2 = format_exponent(ax2, 'x', (1., 1.1))
        ax1 = format_exponent(ax1, 'x', (1., -0.05))
    ax1.set_title(
        "Betweenness distribution for {}".format(network.name), x=0., y=1.05,
        loc='left')
    if show:
        plt.show()


# ------------------------ #
# Plotting node attributes #
# ------------------------ #

def node_attributes_distribution(network, attributes, nodes=None, num_bins=50,
                                 show=True):
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
        * "b2" (requires NEST)
        * "firing_rate" (requires NEST)
    nodes : list, optional (default: all nodes)
        Nodes for which the attributes should be returned.
    num_bins : int or list, optional (default: 50)
        Number of bins to plot the distributions. If only one int is provided,
        it is used for all attributes, otherwize a list containing one int per
        attribute in `attributes` is required.
    '''
    if not nonstring_container(attributes):
        attributes = [attributes]
    else:
        attributes = [name for name in attributes]
    if nonstring_container(num_bins):
        assert len(num_bins) == len(attributes), "One entry per attribute " +\
            "required for `num_bins`."
    else:
        num_bins = [num_bins for _ in range(len(attributes))]
    fig = plt.figure()
    num_plot = 0
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
        num_plot += 1
    # plot betweenness if needed
    if "betweenness" in attributes:
        idx = attributes.index("betweenness")
        betweenness_distribution(
            network, btype="node", nodes=nodes, fignum=fig.number, show=False)
        del attributes[idx]
        del num_bins[idx]
        num_plot += 1
    # plot the remaining attributes
    values = node_attributes(network, attributes, nodes=None)
    fig, axes = _set_new_plot(fignum=fig.number, names=attributes)
    for i, (attr, val) in enumerate(values.items()):
        counts, bins = np.histogram(val, num_bins[i])
        bins = bins[:-1] + 0.5*np.diff(bins)
        axes[i].plot(bins, counts, ls="--", marker="o")
        axes[i].set_title("{}{} distribution for {}".format(
            attr[0].upper(), attr[1:].replace("_", "\\_"), network.name),
            x=0., y=1.05)
    # adjust space, set title, and show
    _format_and_show(fig, num_plot, values, title, show)


def correlation_to_attribute(network, reference_attribute, other_attributes,
                             nodes=None, title=None, show=True):
    '''
    For each node plot the value of `reference_attributes` against each of the
    `other_attributes` to check for correlations.
    
    Parameters
    ----------
    network : :class:`~nngt.Graph`
        The graph where the `nodes` belong.
    reference_attribute : str or array-like
        Attribute which should serve as reference, among:

        * "betweenness"
        * "clustering"
        * "in-degree", "out-degree", "total-degree"
        * "subgraph_centrality"
        * "b2" (requires NEST)
        * "firing_rate" (requires NEST)
        * a custom array of values, in which case one entry per node in `nodes`
          is required.
    other_attributes : str or list
    nodes : list, optional (default: all nodes)
        Nodes for which the attributes should be returned.
    '''
    if not nonstring_container(other_attributes):
        other_attributes = [other_attributes]
    fig = plt.figure()
    # get reference data
    ref_data = reference_attribute
    if isinstance(reference_attribute, str):
        ref_data = node_attributes(network, reference_attribute, nodes=nodes)
    # plot the remaining attributes
    values = node_attributes(network, other_attributes, nodes=nodes)
    fig, axes = _set_new_plot(fignum=fig.number, names=other_attributes)
    for i, (attr, val) in enumerate(values.items()):
        # reference nodes
        axes[i].plot(val, ref_data, ls="", marker="o")
        axes[i].set_xlabel(attr[0].upper() + attr[1:].replace("_", "\\_"))
        axes[i].set_ylabel(
            reference_attribute[0].upper() +
            reference_attribute[1:].replace("_", "\\_"))
        axes[i].set_title("{}{} vs {} for each ".format(
            attr[0].upper(), attr[1:].replace("_", "\\_"),
            reference_attribute.replace("_", "\\_"), network.name) +\
            "node in {}".format(network.name), loc='left', x=0., y=1.05)
    # adjust space, set title, and show
    _format_and_show(fig, 0, values, title, show)


def compare_population_attributes(network, attributes, nodes=None,
                                  reference_nodes=None, num_bins=50,
                                  reference_color="gray", title=None,
                                  show=True):
    '''
    Compare node `attributes` between two sets of nodes. Since number of nodes
    can vary, normalized distributions are used.
    
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
        * "b2" (requires NEST)
        * "firing_rate" (requires NEST)
    nodes : list, optional (default: all nodes)
        Nodes for which the attributes should be returned.
    reference_nodes : list, optional (default: all nodes)
        Reference nodes for which the attributes should be returned in order
        to compare with `nodes`.
    num_bins : int or list, optional (default: 50)
        Number of bins to plot the distributions. If only one int is provided,
        it is used for all attributes, otherwize a list containing one int per
        attribute in `attributes` is required.
    '''
    if not nonstring_container(attributes):
        attributes = [attributes]
    else:
        attributes = [name for name in attributes]
    if isinstance(reference_color, str):
        reference_color = [reference_color]
    else:
        raise InvalidArgument("`reference_color` must be a valid matplotlib "
                              "color string.")
    if nonstring_container(num_bins):
        assert len(num_bins) == len(attributes), "One entry per attribute " +\
            "required for `num_bins`."
    else:
        num_bins = [num_bins for _ in range(len(attributes))]
    fig = plt.figure()
    num_plot = 0
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
        # reference nodes
        degree_distribution(
            network, deg_type=degrees, nodes=reference_nodes, num_bins=deg_bin,
            fignum=fig.number, colors=reference_color*len(degrees), norm=True,
            show=False)
        # nodes
        degree_distribution(
            network, deg_type=degrees, nodes=nodes, num_bins=deg_bin,
            fignum=fig.number, axis_num=0, norm=True, show=False)
        # set legend
        lines = fig.get_axes()[num_plot].get_lines()
        for i in range(int(len(lines) / 2)):
            lines[i].set_label("{}-degree reference".format(degrees[i]))
            lines[i+len(degrees)].set_label(
                "{}-degree nodes".format(degrees[i]))
        plt.legend(
            loc='center', bbox_to_anchor=[0.9, 1.], ncol=1, frameon=True)
        num_plot += 1
    # plot betweenness if needed
    if "betweenness" in attributes:
        idx = attributes.index("betweenness")
        # reference nodes
        betweenness_distribution(
            network, btype="node", nodes=reference_nodes, fignum=fig.number,
            colors=2*reference_color, norm=True, show=False)
        # nodes
        betweenness_distribution(
            network, btype="node", nodes=nodes, fignum=fig.number,
            axis_num=(1,), norm=True, show=False)
        # set legend
        lines = fig.get_axes()[num_plot].get_lines()
        lines[0].set_label("reference")
        lines[1].set_label("nodes")
        plt.legend(
            loc='center', bbox_to_anchor=[0.9, 1.], ncol=1, frameon=True)
        num_plot += 1
        del attributes[idx]
        del num_bins[idx]
    # plot the remaining attributes
    values = node_attributes(network, attributes, nodes=nodes)
    values_ref = node_attributes(network, attributes, nodes=reference_nodes)
    fig, axes = _set_new_plot(fignum=fig.number, names=attributes)
    for i, (attr, val) in enumerate(values.items()):
        counts, bins = np.histogram(val, num_bins[i])
        bins = bins[:-1] + 0.5*np.diff(bins)
        counts_ref, bins_ref = np.histogram(values_ref[attr], num_bins[i])
        bins_ref = bins_ref[:-1] + 0.5*np.diff(bins_ref)
        # normalize
        counts = counts / float(np.sum(counts))
        counts_ref = counts_ref / float(np.sum(counts_ref))
        # reference nodes
        axes[i].plot(
            bins_ref, counts_ref, ls="--", c=reference_color[0], marker="o",
            label="reference")
        # nodes
        axes[i].plot(bins, counts, ls="--", marker="o", label="nodes")
        axes[i].set_xlabel(attr[0].upper() + attr[1:])
        axes[i].set_ylabel("Node count")
        axes[i].set_title("{}{} distribution for {}".format(
            attr[0].upper(), attr[1:], network.name), loc='left', x=0., y=1.05)
        axes[i].legend(loc='center', bbox_to_anchor=[0.9, 1.], ncol=1,
                   frameon=True)
    # adjust space, set title, and show
    _format_and_show(fig, num_plot, values, title, show)


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
    num_cols = max(int(np.ceil(np.sqrt(num_axes))), 1)
    ratio = num_axes/float(num_cols)
    num_rows = int(ratio)
    if int(ratio) != int(np.ceil(ratio)):
        num_rows += 1
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
    if 0.9*minbins < ax1.get_xlim()[0]:
        ax1.set_xlim(left=0.9*minbins)
    if 1.1*maxbins > ax1.get_xlim()[1]:
        ax1.set_xlim(right=1.1*maxbins)
    if 1.1*maxcounts > ax1.get_ylim()[1]:
        ax1.set_ylim([0, 1.1*maxcounts])
    if logx:
        ax1.set_xscale("log")
        ax1.set_xlim([max(0.8, 0.8*minbins), 1.5*maxbins])
    if logy:
        ax1.set_yscale("log")
        ax1.set_ylim([0.8, 1.5*maxcounts])


def _format_and_show(fig, num_plot, values, title, show):
    num_cols = max(int(np.ceil(np.sqrt(num_plot + len(values)))), 1)
    ratio = (num_plot + len(values)) / float(num_cols)
    num_rows = int(ratio)
    if int(ratio) != int(np.ceil(ratio)):
        num_rows += 1
    plt.subplots_adjust(hspace=num_rows*0.2, wspace=num_cols*0.1, left=0.075,
                        right=0.95, top=0.9, bottom=0.075)
    if title is not None:
        fig.suptitle(title)
    if show:
        plt.show()
