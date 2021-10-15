#-*- coding:utf-8 -*-
#
# plot/plt_properties.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Tools to plot graph properties """

import numpy as np

import nngt
from nngt.lib import InvalidArgument, nonstring_container
from nngt.analysis import (degree_distrib, betweenness_distrib,
                           node_attributes, binning)
from .custom_plt import palette_continuous, palette_discrete, format_exponent

from matplotlib.gridspec import SubplotSpec


__all__ = [
    'degree_distribution',
    'betweenness_distribution',
    'edge_attributes_distribution',
    'node_attributes_distribution',
    'compare_population_attributes',
    'correlation_to_attribute',
]


# ---------------------- #
# Plotting distributions #
# ---------------------- #

def degree_distribution(network, deg_type="total", nodes=None,
                        num_bins='doane', weights=False, logx=False,
                        logy=False, axis=None, colors=None,
                        norm=False, show=True, title=None, **kwargs):
    '''
    Plotting the degree distribution of a graph.

    .. versionchanged :: 2.5.0
        Removed unused `axis_num` argument.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        The graph to analyze.
    deg_type : string or N-tuple, optional (default: "total")
        Type of degree to consider ("in", "out", or "total")
    nodes : list or numpy.array of ints, optional (default: all nodes)
        Restrict the distribution to a set of nodes.
    num_bins : str, int or N-tuple, optional (default: 'doane'):
        Number of bins used to sample the distribution. Defaults to 'doane'.
        Use to 'auto' for numpy automatic selection or 'bayes' for unsupervised
        Bayesian blocks method.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    logx : bool, optional (default: False)
        Use log-spaced bins.
    logy : bool, optional (default: False)
        Use logscale for the degree count.
    axis : :class:`matplotlib.axes.Axes` instance, optional (default: new one)
        Axis which should be used to plot the histogram, if None, a new one is
        created.
    colors : (list of) matplotlib colors, optional (default: from palette)
        Colors associated to each degree type.
    title : str, optional (default: no title)
        Title of the axis.
    show : bool, optional (default: True)
        Show the Figure right away if True, else keep it warm for later use.
    **kwargs : keyword arguments for :func:`matplotlib.axes.Axes.bar`.
    '''
    import matplotlib.pyplot as plt

    if axis is None:
        fig, axis = plt.subplots()

    empty_axis = axis.has_data()

    axis.axis('tight')

    single_deg = isinstance(deg_type, str) or len(deg_type) == 1

    if "alpha" not in kwargs:
        kwargs["alpha"] = 1 if single_deg else 0.5

    labels = kwargs.get('label', None)

    if not nonstring_container(labels) and labels is not None:
        labels = [labels]

    # get degrees
    mincounts, maxcounts, allbins = np.inf, 0, []

    deg_string = None

    if isinstance(deg_type, str):
        counts, bins = degree_distrib(network, deg_type, nodes,
                                      weights, logx, num_bins)

        if norm:
            counts = counts / float(np.sum(counts))

        max_nnz = np.where(counts > 0)[0][-1]
        maxcounts, mincounts = counts.max(), np.min(counts[counts > 0])
        allbins.extend(bins)

        deg_string = deg_type[0].upper() + deg_type[1:] + " degree"

        if "label" not in kwargs:
            kwargs["label"] = deg_string
    
        axis.bar(bins[:-1], counts, np.diff(bins), **kwargs)
    else:
        if colors is None:
            colors = palette_continuous(np.linspace(0.,0.5, len(deg_type)))

        if not nonstring_container(num_bins):
            num_bins = [num_bins for _ in range(len(deg_type))]

        for i, s_type in enumerate(deg_type):
            counts, bins = degree_distrib(network, s_type, nodes,
                                          weights, logx, num_bins[i])

            if norm:
                counts = counts / float(np.sum(counts))

            maxcounts_tmp = counts.max()
            mincounts_tmp = np.min(counts[counts>0])
            
            mincounts = min(mincounts, mincounts_tmp)
            maxcounts = max(maxcounts, maxcounts_tmp)

            allbins.extend(bins)

            deg_string = s_type[0].upper() + s_type[1:] + "-degree"

            if labels is None:
                kwargs['label'] = deg_string
            else:
                kwargs['label'] = labels[i]

            axis.bar(
                bins[:-1], counts, np.diff(bins), color=colors[i],
                align='edge', **kwargs)

    if single_deg:
        axis.set_xlabel(deg_string)
    else:
        axis.legend()
        axis.set_xlabel("Degree")

    axis.set_ylabel("Node count")

    title_start = (deg_type[0].upper() + deg_type[1:] + '-d'
                   if isinstance(deg_type, str) else 'D')

    if title != "":
        str_title = title
        if title is None:
            str_title = "{}egree distribution for {}".format(
                title_start, network.name)
        axis.set_title(str_title, x=0., y=1.05, loc='left')

    # restore ylims and xlims and adapt if necessary
    _set_scale(axis, np.array(allbins), mincounts, maxcounts, logx, logy)

    if show:
        plt.show()


def attribute_distribution(network, attribute, num_bins='auto', logx=False,
                           logy=False, axis=None, norm=False, show=True):
    '''
    Plotting the distribution of a graph attribute (e.g. "weight", or
    "distance" is the graph is spatial).

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph to analyze.
    attribute : string or tuple of strings
        Name of a graph attribute.
    num_bins : int or 'auto', optional (default: 'auto'):
        Number of bins used to sample the distribution. Defaults to
        unsupervised Bayesian blocks method.
    logx : bool, optional (default: False)
        use log-spaced bins.
    logy : bool, optional (default: False)
        use logscale for the degree count.
    axis : :class:`matplotlib.axis.Axis` instance, optional (default: new one)
        Axis which should be used to plot the histogram, if None, a new one is
        created.
    show : bool, optional (default: True)
        Show the Figure right away if True, else keep it warm for later use.
    '''
    import matplotlib.pyplot as plt
    if axis is None:
        axis = plt.gca()

    mincounts, maxcounts, bins = 0, 0, None

    if isinstance(attribute, str):
        values = network.get_edge_attributes(name=attribute)
        counts, bins = _hist(
            values, num_bins, norm, logx, attribute, axis, **kwargs)
        maxcounts, mincounts = counts.max(), np.min(counts[counts>0])
        maxbins, minbins     = bins.max(), bins.min()
    else:
        raise NotImplementedError("Multiple attribute plotting not ready yet")
        #~ colors = palette(np.linspace(0.,0.5,len(deg_type)))
        #~ m = ["o", "s", "D", "x"]
        #~ lines, legends = [], []
        #~ for i,s_type in enumerate(deg_type):
            #~ counts,bins = degree_distrib(network, s_type, nodes,
                                         #~ weights, logx, num_bins)
            #~ maxcounts_tmp,mincounts_tmp = counts.max(),counts.min()
            #~ maxbins_tmp,minbins_tmp = bins.max(),bins.min()
            #~ maxcounts = max(maxcounts,maxcounts_tmp)
            #~ maxbins = max(maxbins,maxbins_tmp)
            #~ minbins = min(minbins,minbins_tmp)
            #~ lines.append(ax1.scatter(bins, counts, c=colors[i], marker=m[i]))
            #~ legends.append(attribute)
        #~ ax1.legend(lines, legends)
    if nngt._config['use_tex']:
        axis.set_xlabel(attribute.replace("_", "\\_"))

    axis.set_ylabel("Node count")

    _set_scale(ax1, bins, mincounts, maxcounts, logx, logy)

    axis.set_title(
        "Attribute distribution for {}".format(network.name), x=0., y=1.05,
        loc='left')

    plt.legend()

    axis.axis('tight')

    if show:
        plt.show()


def betweenness_distribution(
    network, btype="both", weights=False, nodes=None, logx=False, logy=False,
    num_nbins=None, num_ebins=None, axes=None, colors=None, norm=False,
    legend_location='right', title=None, show=True, **kwargs):
    '''
    Plotting the betweenness distribution of a graph.

    .. versionchanged :: 2.5.0
        Added `title` argument.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        the graph to analyze.
    btype : string, optional (default: "both")
        type of betweenness to display ("node", "edge" or "both")
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    nodes : list or numpy.array of ints, optional (default: all nodes)
        Restrict the distribution to a set of nodes (taken into account only
        for the node attribute).
    logx : bool, optional (default: False)
        use log-spaced bins.
    logy : bool, optional (default: False)
        use logscale for the degree count.
    num_nbins : int or 'auto', optional (default: None):
        Number of bins used to sample the node distribution. Defaults to
        `max(num_nodes / 50., 10)`.
    num_ebins : int or 'auto', optional (default: None):
        Number of bins used to sample the edge distribution. Defaults to
        `max(num_edges / 500., 10)` ('auto' method will be slow).
    axes : list of :class:`matplotlib.axis.Axis`, optional (default: new ones)
        Axes which should be used to plot the histogram, if None, new ones are
        created.
    legend_location : str, optional (default; 'right')
        Location of the legend.
    title : str, optional (default: auto-generated)
        Title of the axis.
    show : bool, optional (default: True)
        Show the Figure right away if True, else keep it warm for later use.
    '''
    import matplotlib.pyplot as plt

    if btype not in ("node", "edge", "both"):
        raise InvalidArgument('`btype` must be one of the following: '
                              '"node", "edge", "both".')

    num_axes = 2 if btype == "both" else 1

    # create new axes or get them from existing ones
    ax1, ax2 = None, None

    if axes is None:
        fig, ax1 = plt.subplots()
        fig.patch.set_visible(False)

        axes = [ax1]

        if num_axes == 2:
            ax2 = ax1.twiny()
            axes.append(ax2)
            ax2.grid(False)
            ax2.yaxis.set_visible(True)
            ax2.yaxis.set_ticks_position("right")
            ax2.yaxis.set_label_position("right")
        else:
            ax2 = ax1
    else:
        ax1 = axes[0]

        ax1.yaxis.tick_left()
        ax1.yaxis.set_label_position("left")
        ax2 = axes[-1]

    # get betweenness
    if num_ebins is None:
        num_ebins = int(max(network.edge_nb() / 500., 10))

    if num_nbins is None:
        num_nbins = int(max(network.node_nb() / 50., 10))

    ncounts, nbins, ecounts, ebins = betweenness_distrib(
        network, weights, nodes=nodes, num_nbins=num_nbins,
        num_ebins=num_ebins, log=logx)

    if norm:
        ncounts = ncounts / float(np.sum(ncounts))
        ecounts = ecounts / np.sum(ecounts)

    if colors is None:
        colors = palette_continuous(np.linspace(0.05, 0.5, 2))

    alpha = kwargs.get("alpha", None)

    if alpha is None:
        kwargs["alpha"] = 0.5 if btype == "both" else 1

    
    x = 0 if legend_location == 'left' else 1

    # plot
    if btype in ("node", "both"):
        ax1.bar(
            nbins[:-1], ncounts, np.diff(nbins), color=colors[0], align='edge',
            **kwargs)

        ax1.set_xlabel("Node betweenness")
        ax1.set_ylabel("Node count")
        ax1.ticklabel_format(axis='x', style='sci', scilimits=(-3, 2))

        _set_scale(ax1, nbins, np.min(ncounts[ncounts>0]), ncounts.max(),
                   logx, logy)

    if btype in ("edge", "both"):
        ax2.bar(
            ebins[:-1], ecounts, np.diff(ebins), color=colors[-1],
            align='edge', **kwargs)

        ax2.set_xlim([ebins.min(), ebins.max()])
        ax2.set_ylim([0, 1.1*ecounts.max()])
        ax2.set_xlabel("Edge betweenness")
        ax2.set_ylabel("Edge count")
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3, 2))

        _set_scale(ax2, ebins, np.min(ecounts[ecounts>0]), ecounts.max(),
                   logx, logy)

    if btype == "both":
        ax2.legend(
            ["Edge betweenness"], bbox_to_anchor=[x, 0.88],
            loc='upper ' + legend_location)
        ax1.legend(
            ["Node betweenness"], bbox_to_anchor=[x, 0.88],
            loc='lower ' + legend_location)

        plt.subplots_adjust(top=0.85)

        ax1.grid(False)
        ax2.grid(False)

        if not logx:
            ax2 = format_exponent(ax2, 'x', (1., 1.1))
            ax1 = format_exponent(ax1, 'x', (1., -0.05))

    y = 1.15 if btype == "both" else 1.05 

    title = "Betweenness distribution for {}".format(network.name) \
            if title is None else title

    ax1.set_title(title, x=0., y=y, loc='left')

    plt.tight_layout()

    if show:
        plt.show()


# ------------------------ #
# Plotting node attributes #
# ------------------------ #

def node_attributes_distribution(network, attributes, nodes=None,
                                 num_bins='auto', logx=False, logy=False,
                                 norm=False, title=None, axtitles=None,
                                 colors=None, axes=None, show=True, **kwargs):
    '''
    Return node `attributes` for a set of `nodes`.

    .. versionchanged :: 2.5.0
        Added `axtitles` and `axes` arguments.

    Parameters
    ----------
    network : :class:`~nngt.Graph`
        The graph where the `nodes` belong.
    attributes : str or list
        Attributes which should be returned, among:
        * any user-defined node attribute
        * "betweenness"
        * "clustering"
        * "closeness"
        * "in-degree", "out-degree", "total-degree"
        * "subgraph_centrality"
        * "b2" (requires NEST)
        * "firing_rate" (requires NEST)
    nodes : list, optional (default: all nodes)
        Nodes for which the attributes should be returned.
    num_bins : int or list, optional (default: 'auto')
        Number of bins to plot the distributions. If only one int is provided,
        it is used for all attributes, otherwise a list containing one int per
        attribute in `attributes` is required. Defaults to unsupervised
        Bayesian blocks method.
    logx : bool or list, optional (default: False)
        Use log-spaced bins.
    logy : bool or list, optional (default: False)
        use logscale for the node count.
    norm : bool, optional (default: False)
        Whether the histogram should be normed such that the sum of the counts
        is 1.
    title : str, optional (default: no title)
        Title of the figure.
    axtitles : list of str, optional (default: auto-generated)
        Titles of the axes. Use "" or False to turn them of.
    colors : (list of) matplotlib colors, optional (default: from palette)
        Colors associated to each degree type.
    axes : list of :class:`matplotlib.axis.Axis`, optional (default: new ones)
        Axess which should be used to plot the histograms, if None, a new axis
        is created for each attribute.
    show : bool, optional (default: True)
        Show the Figure right away if True, else keep it warm for later use.
    **kwargs : keyword arguments for :func:`matplotlib.axes.Axes.bar`.
    '''
    import matplotlib.pyplot as plt

    if not nonstring_container(attributes):
        attributes = [attributes]
    else:
        attributes = [name for name in attributes]

    if axtitles in ("", False):
        axtitles = [""]*len(attributes)
    elif nonstring_container(axtitles):
        assert len(axtitles) == len(attributes), \
            "One entry per attribute is required for `axtitles`."

    num_attr = len(attributes)
    num_bins = _format_arg(num_bins, num_attr, 'num_bins')
    colors = _format_arg(colors, num_attr, 'num_bins')
    logx = _format_arg(logx, num_attr, 'logx')
    logy = _format_arg(logy, num_attr, 'logy')
    num_plot = 0

    # kwargs that will not be passed:
    ignore = ["degree", "betweenness"] + attributes
    new_kwargs = {k: v for k, v in kwargs.items() if k not in ignore}

    fig = None

    if axes is None:
        if new_kwargs == kwargs:
            fig = plt.figure()
            fig.patch.set_visible(False)
        else:
            fig = plt.figure(plt.get_fignums()[-1])
    else:
        if not nonstring_container(axes):
            axes = [axes]

        fig = axes[0].get_figure()

        assert len(axes) == len(attributes), \
            "One entry per attribute is required."

    # plot degrees if required
    degrees = []

    for name in attributes:
        if "degree" in name.lower():
            degrees.append(name[:name.find("-")])

    if degrees:
        # get the indices where a degree-related attribute is required
        indices, colors_deg, logx_deg, logy_deg = [], [], 0, 0

        for i, name in enumerate(attributes):
            if "degree" in name:
                indices.append(i)
                if colors is not None:
                    colors_deg.append(colors[i])
                logx_deg += logx[i]
                logy_deg += logy[i]

        colors_deg = None if colors is None else colors_deg

        indices.sort()

        deg_bin = [num_bins[i] for i in indices]

        for idx in indices[::-1]:
            del num_bins[idx]
            del attributes[idx]
            del logx[idx]
            del logy[idx]

            if colors is not None:
                del colors[idx]

        if "degree" in kwargs:
            degree_distribution(
                network, deg_type=degrees, nodes=nodes, num_bins=deg_bin,
                logx=logx_deg, logy=logy_deg, norm=norm, axis=kwargs["degree"],
                colors=colors_deg, show=False, **new_kwargs)
        else:
            ax = None

            if axes is None:
                fig, ax = _set_new_plot(
                    fignum=fig.number, num_new_plots=1,
                    names=['Degree distribution'])
            else:
                ax = [axes[indices[0]]]

                for idx in indices:
                    del axes[idx]

            degree_distribution(
                network, deg_type=degrees, nodes=nodes, num_bins=deg_bin,
                logx=logx_deg, logy=logy_deg, axis=ax[0], colors=colors_deg,
                norm=norm, show=False)

        num_plot += 1

    # plot betweenness if needed
    if "betweenness" in attributes:
        idx = attributes.index("betweenness")

        if "betweenness" in kwargs:
            betweenness_distribution(
                network, btype="node", nodes=nodes, logx=logx[idx],
                logy=logy[idx], axes=kwargs["betweenness"],
                colors=[colors[idx]], norm=norm, show=False, **new_kwargs)
        else:
            ax = None

            if axes is None:
                fig, ax = _set_new_plot(
                    fignum=fig.number, num_new_plots=1,
                    names=['Betweenness distribution'])
            else:
                ax = [axes[idx]]

                del axes[idx]

            betweenness_distribution(
                network, btype="node", nodes=nodes, logx=logx[idx], axes=ax,
                logy=logy[idx], colors=[colors[idx]], norm=norm, show=False)

        del attributes[idx]
        del num_bins[idx]
        del logx[idx]
        del logy[idx]
        del colors[idx]

        num_plot += 1

    # plot the remaining attributes
    values = node_attributes(network, attributes, nodes=nodes)

    for i, (attr, val) in enumerate(values.items()):
        if attr in kwargs:
            new_kwargs['color'] = colors[i]
            counts, bins = _hist(
                val, num_bins[i], norm, logx[i], attr, kwargs[attr],
                **new_kwargs)
        else:
            ax = None

            if axes is None:
                fig, ax = _set_new_plot(fignum=fig.number, names=[attr])
                ax = ax[0]
            else:
                ax = axes[i]

            counts, bins = _hist(
                val, num_bins[i], norm, logx[i], attr, ax, color=colors[i],
                **kwargs)

            end_attr = attr[1:]

            if nngt._config["use_tex"]:
                end_attr = end_attr.replace("_", "\\_")

            ax.set_title("{}{} distribution for {}".format(
                attr[0].upper(), end_attr, network.name), y=1.05)
            ax.set_ylabel("Node count")
            ax.set_xlabel(attr[0].upper() + end_attr)
            _set_scale(ax, bins, np.min(counts[counts>0]),
                       counts.max(), logx[i], logy[i])

        num_plot += 1

    # adjust space, set title, and show
    _format_and_show(fig, num_plot, values, title, show)


def edge_attributes_distribution(network, attributes, edges=None,
                                 num_bins='auto', logx=False, logy=False,
                                 norm=False, title=None, axtitles=None,
                                 colors=None, axes=None, show=True, **kwargs):
    '''
    Return node `attributes` for a set of `nodes`.

    .. versionchanged :: 2.5.0
        Added `axtitles` and `axes` arguments.

    Parameters
    ----------
    network : :class:`~nngt.Graph`
        The graph where the `nodes` belong.
    attributes : str or list
        Attributes which should be returned (e.g. "betweenness", "delay",
        "weight").
    edges : list, optional (default: all edges)
        Edges for which the attributes should be returned.
    num_bins : int or list, optional (default: 'auto')
        Number of bins to plot the distributions. If only one int is provided,
        it is used for all attributes, otherwise a list containing one int per
        attribute in `attributes` is required. Defaults to unsupervised
        Bayesian blocks method.
    logx : bool or list, optional (default: False)
        Use log-spaced bins.
    logy : bool or list, optional (default: False)
        use logscale for the node count.
    norm : bool, optional (default: False)
        Whether the histogram should be normed such that the sum of the counts
        is 1.
    title : str, optional (default: no title)
        Title of the figure.
    axtitles : list of str, optional (default: auto-generated)
        Titles of the axes. Use "" or False to turn them of.
    colors : (list of) matplotlib colors, optional (default: from palette)
        Colors associated to each degree type.
    axes : list of :class:`matplotlib.axis.Axis`, optional (default: new ones)
        Axess which should be used to plot the histograms, if None, a new axis
        is created for each attribute.
    show : bool, optional (default: True)
        Show the Figure right away if True, else keep it warm for later use.
    **kwargs : keyword arguments for :func:`matplotlib.axes.Axes.bar`.
    '''
    import matplotlib.pyplot as plt

    if not nonstring_container(attributes):
        attributes = [attributes]
    else:
        attributes = [name for name in attributes]

    if axtitles in ("", False):
        axtitles = [""]*len(attributes)
    elif nonstring_container(axtitles):
        assert len(axtitles) == len(attributes), \
            "One entry per attribute is required for `axtitles`."

    num_attr = len(attributes)
    num_bins = _format_arg(num_bins, num_attr, 'num_bins')
    colors   = _format_arg(colors, num_attr, 'num_bins')
    logx     = _format_arg(logx, num_attr, 'logx')
    logy     = _format_arg(logy, num_attr, 'logy')
    num_plot = 0

    # kwargs that will not be passed:
    ignore = ["weight", "delay", "betweenness"] + attributes
    new_kwargs = {k: v for k, v in kwargs.items() if k not in ignore}

    fig = None

    if axes is None:
        fig = plt.figure()
        fig.patch.set_visible(False)
    else:
        if not nonstring_container(axes):
            axes = [axes]

        fig = axes[0].get_figure()

        assert len(axes) == len(attributes), \
            "One entry per attribute is required."

    # plot betweenness if needed
    if "betweenness" in attributes:
        idx = attributes.index("betweenness")

        ax = None

        if axes is None:
            fig, ax = _set_new_plot(
                fignum=fig.number, num_new_plots=1,
                names=['Betweenness distribution'])
        else:
            ax = [axes[idx]]

            del axes[idx]

        axtitle = axtitles[idx] if axtitles is not None else None

        betweenness_distribution(
            network, btype="edge", logx=logx[idx], logy=logy[idx], norm=norm,
            title=axtitle, axes=ax, colors=[colors[idx]], show=False)

        if axtitles is not None:
            del axtitles[idx]

        del attributes[idx]
        del num_bins[idx]
        del logx[idx]
        del logy[idx]
        del colors[idx]

        num_plot += 1

    # plot the remaining attributes
    for i, attr in enumerate(attributes):
        val = network.get_edge_attributes(edges=edges, name=attr)

        if attr in kwargs:
            new_kwargs['color'] = colors[i]
            counts, bins = _hist(
                val, num_bins[i], norm, logx[i], attr, kwargs[attr],
                **new_kwargs)
        else:
            ax = None

            if axes is None:
                fig, ax = _set_new_plot(fignum=fig.number, names=[attr])
                ax = ax[0]
            else:
                ax = axes[i]

            counts, bins = _hist(
                val, num_bins[i], norm, logx[i], attr, ax, color=colors[i],
                **kwargs)

            end_attr = attr[1:]

            if nngt._config["use_tex"]:
                end_attr = end_attr.replace("_", "\\_")

            axtitle = None if axtitles is None else axtitles[i]

            if axtitle != "":
                if axtitle is None:
                    axtitle = "{}{} distribution for {}".format(
                        attr[0].upper(), end_attr, network.name)
                ax.set_title(axtitle, y=1.05)

            ax.set_ylabel("Edge count")
            ax.set_xlabel(attr[0].upper() + end_attr)

            _set_scale(ax, bins, np.min(counts[counts>0]), counts.max(),
                       logx[i], logy[i])

        num_plot += 1

    # adjust space, set title, and show
    _format_and_show(fig, num_plot, attributes, title, show)


def correlation_to_attribute(network, reference_attribute, other_attributes,
                             attribute_type="node", nodes=None, edges=None,
                             fig=None, title=None, show=True):
    '''
    For each node plot the value of `reference_attributes` against each of the
    `other_attributes` to check for correlations.

    .. versionchanged :: 2.0
        Added `fig` argument.

    Parameters
    ----------
    network : :class:`~nngt.Graph`
        The graph where the `nodes` belong.
    reference_attribute : str or array-like
        Attribute which should serve as reference, among:

        * "betweenness"
        * "clustering"
        * "in-degree", "out-degree", "total-degree"
        * "in-strength", "out-strength", "total-strength"
        * "subgraph_centrality"
        * "b2" (requires NEST)
        * "firing_rate" (requires NEST)
        * a custom array of values, in which case one entry per node in `nodes`
          is required.
    other_attributes : str or list
        Attributes that will be compared to the reference.
    attribute_type : str, optional (default: 'node')
        Whether we are dealing with 'node' or 'edge' attributes
    nodes : list, optional (default: all nodes)
        Nodes for which the attributes should be returned.
    edges : list, optional (default: all edges)
        Edges for which the attributes should be returned.
    fig : :class:`matplotlib.figure.Figure`, optional (default: new Figure)
        Figure to which the plot should be added.
    title : str, optional (default: automatic).
        Custom title, use "" to remove the automatic title.
    show : bool, optional (default: True)
        Whether the plot should be displayed immediately.
    '''
    import matplotlib.pyplot as plt

    if not nonstring_container(other_attributes):
        other_attributes = [other_attributes]

    fig = plt.figure() if fig is None else fig
    fig.patch.set_visible(False)

    # get reference data
    ref_data = reference_attribute

    if isinstance(reference_attribute, str):
        if attribute_type == "node":
            ref_data = node_attributes(network, reference_attribute,
                                       nodes=nodes)
        else:
            ref_data = network.get_edge_attributes(edges=edges,
                                                   name=reference_attribute)
    else:
        reference_attribute = "user defined attribute"

    # plot the remaining attributes
    assert isinstance(other_attributes, (str, list)), \
        "Only attribute names are allowed for `other_attributes`"

    values = {}

    if attribute_type == "node":
        values = node_attributes(network, other_attributes, nodes=nodes)
    else:
        for name in other_attributes:
            values[name] = network.get_edge_attributes(edges=edges, name=name)

    fig, axes = _set_new_plot(fignum=fig.number, names=other_attributes)

    for i, (attr, val) in enumerate(values.items()):
        end_attr = attr[1:]
        end_ref_attr = reference_attribute[1:]

        if nngt._config["use_tex"]:
            end_attr = end_attr.replace("_", "\\_")
            end_ref_attr = end_ref_attr.replace("_", "\\_")

        # reference nodes
        axes[i].plot(val, ref_data, ls="", marker="o")
        axes[i].set_xlabel(attr[0].upper() + end_attr)
        axes[i].set_ylabel(reference_attribute[0].upper() + end_ref_attr)
        axes[i].set_title(
            "{}{} vs {}".format(
                reference_attribute[0].upper(), end_ref_attr, attr[0] + \
                end_attr),
            loc='left', x=0., y=1.05)

    fig.suptitle(network.name)

    plt.tight_layout()

    # adjust space, set title, and show
    _format_and_show(fig, 0, values, title, show)


def compare_population_attributes(network, attributes, nodes=None,
                                  reference_nodes=None, num_bins='auto',
                                  reference_color="gray", title=None,
                                  logx=False, logy=False, show=True, **kwargs):
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
    num_bins : int or list, optional (default: 'auto')
        Number of bins to plot the distributions. If only one int is provided,
        it is used for all attributes, otherwize a list containing one int per
        attribute in `attributes` is required. Defaults to unsupervised
        Bayesian blocks method.
    logx : bool or list, optional (default: False)
        Use log-spaced bins.
    logy : bool or list, optional (default: False)
        use logscale for the node count.
    '''
    import matplotlib.pyplot as plt
    if not isinstance(reference_color, str):
        raise InvalidArgument("`reference_color` must be a valid matplotlib "
                              "color string.")
    # plot the non reference nodes
    node_attributes_distribution(network, attributes, nodes=nodes,
                                 num_bins=num_bins, logx=logx, logy=logx,
                                 norm=True, title=title, show=False, **kwargs)
    # get the last figure and put the axes to a dict
    # (order is degree, betweenness, attributes)
    fig = plt.figure(plt.get_fignums()[-1])
    fig.patch.set_visible(False)
    axes = fig.get_axes()
    ref_kwargs = kwargs.copy()
    ref_kwargs.update({'alpha': 0.5})
    for ax in axes:
        if ax.name == 'Degree distribution':
            ref_kwargs['degree'] = ax
        elif ax.name == 'Betweenness distribution':
            ref_kwargs['betweenness'] = [ax]  # expect list
        else:
            ref_kwargs[ax.name] = ax
    node_attributes_distribution(
        network, attributes, nodes=reference_nodes, num_bins=num_bins,
        logx=logx, logy=logx, colors=reference_color, norm=True, title=title,
        show=show, **ref_kwargs)


# --------- #
# Histogram #
# --------- #

def _hist(values, num_bins, norm, logx, label, axis, **kwargs):
    '''
    Compute and draw the histogram.

    Returns
    -------
    counts, bins
    '''
    bins = binning(values, bins=num_bins, log=logx)

    counts, bins = np.histogram(values, bins=bins)

    if norm:
        counts = np.divide(counts, float(np.sum(counts)))

    axis.bar(
        bins[:-1], counts, np.diff(bins), label=label, **kwargs)

    if logx:
        axis.set_xscale("log")

    return counts, bins


# ----------------- #
# Figure management #
# ----------------- #

def _set_new_plot(fignum=None, num_new_plots=1, names=None, sharex=None):
    import matplotlib.pyplot as plt
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
    gs = fig.add_gridspec(num_rows, num_cols)

    for i in range(num_axes - num_new_plots):
        y = i // num_cols
        x = i - num_cols*y
        fig.axes[i].set_subplotspec(gs[y, x])

    lst_new_axes = []
    n_old = num_axes-num_new_plots+1

    for i in range(num_new_plots):
        if fig.axes:
            lst_new_axes.append(
                fig.add_subplot(num_rows, num_cols, n_old+i, sharex=sharex))
        else:
            lst_new_axes.append(fig.add_subplot(num_rows, num_cols, n_old+i))

        if names is not None:
            lst_new_axes[-1].name = names[i]

    return fig, lst_new_axes


def _set_scale(ax1, xbins, mincounts, maxcounts, logx, logy):
    if logx:
        minposbin = xbins[xbins > 0][0]
        ax1.set_xscale("symlog", linthresh=0.2*minposbin)
        next_power = np.ceil(np.log10(xbins.max()))
        ax1.set_xlim([0.8*xbins.min(), 10**next_power])
    else:
        maxbins, minbins = xbins.max(), xbins.min()
        bin_margin = 0.05*(maxbins - minbins)
        if minbins - bin_margin < ax1.get_xlim()[0]:
            ax1.set_xlim(left=(minbins - bin_margin))
        if maxbins + bin_margin > ax1.get_xlim()[1]:
            ax1.set_xlim(right=(maxbins + bin_margin))

    if logy:
        ax1.set_ylim(0, 2*maxcounts)
        ax1.set_yscale("symlog", linthresh=1)
    else:
        if 1.05*maxcounts > ax1.get_ylim()[1]:
            ax1.set_ylim([0, 1.05*maxcounts])


def _set_ax_lims(ax, maxx, minx, maxy, miny, logx=False, logy=False):
    if ax.has_data():
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        if not logx:
            Dx = maxx - minx
            minx = minx - 0.01*Dx
            maxx = maxx + 0.01*Dx
            Dy = maxy - miny
            miny = miny - 0.01*Dy
            maxy = maxy + 0.01*Dy
        else:
            minx /= 1.5
            maxx *= 1.5
        if minx > xlims[0]:
            minx = xlims[0]
        if maxx < xlims[1]:
            maxx = xlims[1]
        if miny > ylims[0]:
            miny = ylims[0]
        if maxy < ylims[1]:
            maxy = ylims[1]
    _set_xlim(ax, maxx, minx, logx)
    _set_ylim(ax, maxy, miny, logy)


def _set_xlim(ax, maxx, minx, log):
    if log:
        ax.set_xscale("log")
        ax.set_xlim([max(minx, 1e-10)/1.5, 1.5*maxx])
    else:
        Dx = maxx - minx
        ax.set_xlim([minx - 0.01*Dx, maxx + 0.01*Dx])


def _set_ylim(ax, maxy, miny, log):
    if log:
        ax.set_yscale("log")
        ax.set_ylim([max(miny, 1e-10)/1.5, 1.5*maxy])
    else:
        Dy = maxy - miny
        ax.set_ylim([miny - 0.01*Dy, maxy + 0.01*Dy])


def _format_and_show(fig, num_plot, values, title, show):
    import matplotlib.pyplot as plt

    plt.tight_layout()
    
    if title is not None:
        fig.suptitle(title)
    if show:
        plt.show()


def _format_arg(arg, num_expected, arg_name):
    if nonstring_container(arg):
        assert len(arg) == num_expected, "One entry per attribute " +\
            "required for `" + arg_name + "`."
    elif arg is not None:
        arg = [arg for _ in range(num_expected)]
    elif arg is None:
        arg = [None]*num_expected

    return arg
