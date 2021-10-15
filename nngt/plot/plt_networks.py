#-*- coding:utf-8 -*-
#
# plot/plt_networks.py
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

from itertools import cycle
from collections import defaultdict

import numpy as np

import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch, ArrowStyle, FancyArrow, Circle
from matplotlib.patches import Arc, RegularPolygon, PathPatch, Patch
from matplotlib.cm import get_cmap
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.colors import ListedColormap, Normalize, ColorConverter
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import nngt
from nngt.lib import POS, nonstring_container, is_integer
from .custom_plt import palette_continuous, palette_discrete, format_exponent
from .chord_diag import chord_diagram as _chord_diag
from .hive_helpers import *


'''
Network plotting
================

Implemented
-----------

Simple representation for spatial graphs, random distribution if non-spatial.
Support for edge-size (according to betweenness or synaptic weight).


Objectives
----------

Implement the spring-block minimization.

If edges have varying size, plot only those that are visible (size > min)

'''

__all__ = ["chord_diagram", "draw_network", "hive_plot", "library_draw"]


# ------- #
# Drawing #
# ------- #

def draw_network(network, nsize="total-degree", ncolor=None, nshape="o",
                 esize=None, ecolor="k", curved_edges=False, threshold=0.5,
                 decimate_connections=None, spatial=True,
                 restrict_sources=None, restrict_targets=None,
                 restrict_nodes=None, restrict_edges=None,
                 show_environment=True, fast=False, size=(600, 600),
                 xlims=None, ylims=None, dpi=75, axis=None, colorbar=False,
                 cb_label=None, layout=None, show=False, **kwargs):
    '''
    Draw a given graph/network.

    Parameters
    ----------
    network : :class:`~nngt.Graph` or subclass
        The graph/network to plot.
    nsize : float, array of float or string, optional (default: "total-degree")
        Size of the nodes as a percentage of the canvas length. Otherwise, it
        can be a string that correlates the size to a node attribute among
        "in/out/total-degree", "in/out/total-strength", or "betweenness".
    ncolor : float, array of floats or string, optional
        Color of the nodes; if a float in [0, 1], position of the color in the
        current palette, otherwise a string that correlates the color to a node
        attribute or "in/out/total-degree", "betweenness" and "group".
        Default to red or one color per group in the graph if not specified.
    nshape : char, array of chars, or groups, optional (default: "o")
        Shape of the nodes (see `Matplotlib markers <http://matplotlib.org/api/
        markers_api.html?highlight=marker#module-matplotlib.markers>`_).
        When using groups, they must be pairwise disjoint; markers will be
        selected iteratively from the matplotlib default markers.
    nborder_color : char, float or array, optional (default: "k")
        Color of the node's border using predefined `Matplotlib colors
        <http://matplotlib.org/api/colors_api.html?highlight=color
        #module-matplotlib.colors>`_).
        or floats in [0, 1] defining the position in the palette.
    nborder_width : float or array of floats, optional (default: 0.5)
        Width of the border in percent of canvas size.
    esize : float, str, or array of floats, optional (default: 0.5)
        Width of the edges in percent of canvas length. Available string values
        are "betweenness" and "weight".
    ecolor : str, char, float or array, optional (default: "k")
        Edge color. If ecolor="groups", edges color will depend on the source
        and target groups, i.e. only edges from and toward same groups will
        have the same color.
    curved_edges : bool, optional (default: False)
        Whether the edges should be curved or straight.
    threshold : float, optional (default: 0.5)
        Size under which edges are not plotted.
    decimate_connections : int, optional (default: keep all connections)
        Plot only one connection every `decimate_connections`.
        Use -1 to hide all edges.
    spatial : bool, optional (default: True)
        If True, use the neurons' positions to draw them.
    restrict_sources : str, group, or list, optional (default: all)
        Only draw edges starting from a restricted set of source nodes.
    restrict_targets : str, group, or list, optional (default: all)
        Only draw edges ending on a restricted set of target nodes.
    restrict_nodes : str, group, or list, optional (default: plot all nodes)
        Only draw a subset of nodes.
    restrict_edges : list of edges, optional (default: all)
        Only draw a subset of edges.
    show_environment : bool, optional (default: True)
        Plot the environment if the graph is spatial.
    fast : bool, optional (default: False)
        Use a faster algorithm to plot the edges. Zooming on the drawing made
        using this method leaves the size of the nodes and edges unchanged, it
        is therefore not recommended when size consistency matters, e.g. for
        some spatial representations.
    size : tuple of ints, optional (default: (600,600))
        (width, height) tuple for the canvas size (in px).
    dpi : int, optional (default: 75)
        Resolution (dot per inch).
    axis : matplotlib axis, optional (default: create new axis)
        Axis on which the network will be plotted.
    colorbar : bool, optional (default: False)
        Whether to display a colorbar for the node colors or not.
    cb_label : str, optional (default: None)
        A label for the colorbar.
    layout : str, optional (default: random or spatial positions)
        Name of a standard layout to structure the network. Available layouts
        are: "circular" or "random". If no layout is provided and the network
        is spatial, then node positions will be used by default.
    show : bool, optional (default: True)
        Display the plot immediately.
    **kwargs : dict
        Optional keyword arguments.

        ================  ==================  =================================
              Name               Type            Purpose and possible values
        ================  ==================  =================================
                                              Desired node colormap (default is
        node_cmap         str                 "magma" for continuous variables
                                              and "Set1" for groups)
        ----------------  ------------------  ---------------------------------
        title             str                 Title of the plot
        ----------------  ------------------  ---------------------------------
        max_*             float               Maximum value for `nsize` or
                                              `esize`
        ----------------  ------------------  ---------------------------------
        min_*             float               Minimum value for `nsize` or
                                              `esize`
        ----------------  ------------------  ---------------------------------
        nalpha            float               Node opacity in [0, 1]`, default 1
        ----------------  ------------------  ---------------------------------
        ealpha            float               Edge opacity, default 0.5
        ----------------  ------------------  ---------------------------------
                                              Color of the border for nodes (n)
        *border_color     color               or edges (e).
                                              Default to black.
        ----------------  ------------------  ---------------------------------
                                              Border size for nodes (n) or edges
        *border_width     float               (e). Default to .5 for nodes and
                                              .3 for edges (if `fast` is False).
        ----------------  ------------------  ---------------------------------
                                              Whether to use simple nodes (that
        simple_nodes      bool                are always the same size) or
                                              patches (change size with zoom).
        ================  ==================  =================================
    '''
    import matplotlib.pyplot as plt

    # figure and axes
    size_inches = (size[0]/float(dpi), size[1]/float(dpi))

    fig = None

    if axis is None:
        fig = plt.figure(facecolor='white', figsize=size_inches,
                         dpi=dpi)
        axis = fig.add_subplot(111, frameon=0, aspect=1)
    else:
        fig = axis.get_figure()

    fig.patch.set_visible(False)

    # projections for geographic plots

    proj = kwargs.get("proj", None)

    kw = {} if proj is None else {"transform": proj}

    if proj is None:
        axis.set_axis_off()

    pos = None

    # arrow style
    arrowstyle = "-|>" if network.is_directed() else "-"

    # restrict sources and targets
    restrict_sources = _convert_to_nodes(restrict_sources,
                                         "restrict_sources", network)

    restrict_targets = _convert_to_nodes(restrict_targets,
                                         "restrict_targets", network)

    restrict_nodes = _convert_to_nodes(restrict_nodes,
                                       "restrict_nodes", network)

    if restrict_nodes is not None and restrict_sources is not None:
        restrict_sources = \
            set(restrict_nodes).intersection(restrict_sources)
    elif restrict_nodes is not None:
        restrict_sources = set(restrict_nodes)

    if restrict_nodes is not None and restrict_targets is not None:
        restrict_targets = \
            set(restrict_nodes).intersection(restrict_targets)
    elif restrict_nodes is not None:
        restrict_targets = set(restrict_nodes)

    # get nodes and edges
    n = network.node_nb() if restrict_nodes is None \
                          else len(restrict_nodes)

    adj_mat = network.adjacency_matrix(weights=None)

    if restrict_sources is not None:
        remove = np.array(
            [1 if node not in restrict_sources else 0
             for node in range(network.node_nb())],
            dtype=bool)
        adj_mat[remove] = 0

    if restrict_targets is not None:
        remove = np.array(
            [1 if node not in restrict_targets else 0
             for node in range(network.node_nb())],
            dtype=bool)
        adj_mat[:, remove] = 0

    edges = (np.array(adj_mat.nonzero()).T if restrict_edges is None else
             np.asarray(restrict_edges))

    e = len(edges)

    decimate_connections = 1 if decimate_connections is None\
                           else decimate_connections

    # get positions (all cases except circular layout which is done below the
    # node sizes
    pos = None

    spatial *= network.is_spatial()

    if nonstring_container(layout):
        assert np.shape(layout) == (n, 2), "One position per node is required."
        pos = np.asarray(layout).astype(float)
        spatial = False
    elif spatial:
        if show_environment:
            nngt.geometry.plot.plot_shape(network.shape, axis=axis, show=False)

        nodes = None if restrict_nodes is None else list(restrict_nodes)

        pos = network.get_positions(nodes=nodes).astype(float)
    elif layout in (None, "random"):
        pos = np.random.uniform(size=(n, 2)) - 0.5

        pos[:, 0] *= size[0]
        pos[:, 1] *= size[1]
    elif layout not in ("circular", "random", None):
        raise ValueError("Unknown `layout`: {}".format(layout))

    # get node and edge size extrema and drawing properties
    simple_nodes = kwargs.get("simple_nodes", fast)

    dist = min(size)

    if pos is not None:
        dist = min(pos[:, 0].max() - pos[:, 0].min(),
                   pos[:, 1].max() - pos[:, 1].min())

    max_nsize = kwargs.get("max_nsize", 100 if simple_nodes else 0.05*dist)
    min_nsize = kwargs.get("min_nsize", 0.2*max_nsize)

    max_esize = kwargs.get("max_esize", 5 if fast else 0.05*dist)
    min_esize = kwargs.get("min_esize", 0)

    if fast:
        simple_nodes = True
        max_nsize *= 0.01*min(size)
        min_nsize *= 0.01*min(size)
        max_esize *= 0.005*min(size)
        min_esize *= 0.005*min(size)
        threshold *= 0.005*min(size)

    if esize is None:
        esize = 0.5*max_esize

    # circular layout
    if isinstance(layout, str) and layout == "circular":
        pos = _circular_layout(network, max_nsize)

    # check axis extent
    xmax = pos[:, 0].max()
    xmin = pos[:, 0].min()
    ymax = pos[:, 1].max()
    ymin = pos[:, 1].min()

    height = ymax - ymin
    width = xmax - xmin

    if not show_environment or not spatial or proj is not None:
        # axis.get_data()
        _set_ax_lim(axis, xmax, xmin, ymax, ymin, height, width, xlims, ylims,
                    max_nsize, fast)

    # get node and edge shape/size properties
    markers, nsize, esize = _node_edge_shape_size(
        network, nshape, nsize, max_nsize, min_nsize, esize, max_esize,
        min_esize, restrict_nodes, edges, size, threshold,
        simple_nodes=simple_nodes)

    # node color information
    if ncolor is None:
        if network.structure is not None:
            ncolor = "group"
        else:
            ncolor = "r"

    nborder_color = kwargs.get("nborder_color", "k")
    nborder_width = kwargs.get("nborder_width", 0.5)

    eborder_color = kwargs.get("eborder_color", "k")
    eborder_width = kwargs.get("eborder_width", 0.3)

    discrete_colors, default_ncmap = _get_ncmap(network, ncolor)

    nalpha = kwargs.get("nalpha", 1)
    ealpha = kwargs.get("ealpha", 0.5)

    ncmap = get_cmap(kwargs.get("node_cmap", default_ncmap))

    node_color, nticks, ntickslabels, nlabel = _node_color(
        network, restrict_nodes, ncolor, discrete_colors=discrete_colors)

    if nonstring_container(ncolor) and not len(ncolor) in (3, 4):
        assert len(ncolor) == n, "For color arrays, one " +\
            "color per node is required."
        ncolor = "custom"

    c = node_color

    if not nonstring_container(nborder_color):
        nborder_color = np.repeat(nborder_color, n)

    # prepare node colors
    if nonstring_container(c) and not isinstance(c[0], (str, np.ndarray)):
        # make the colorbar for the nodes
        cmap = ncmap
        cnorm = None

        if discrete_colors:
            cmap = _discrete_cmap(len(nticks), ncmap, discrete_colors)
            cnorm = Normalize(nticks[0]-0.5, nticks[-1] + 0.5)
        else:
            cnorm = Normalize(np.min(c), np.max(c))
            c = cnorm(c)

        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)

            if discrete_colors:
                sm.set_array(nticks)
            else:
                sm.set_array(c)

            plt.subplots_adjust(right=0.95)
            divider = make_axes_locatable(axis)
            cax     = divider.append_axes("right", size="5%", pad=0.05)

            if discrete_colors:
                cb = plt.colorbar(sm, ticks=nticks, cax=cax, shrink=0.8)
                cb.set_ticklabels(ntickslabels)
                if nlabel:
                    cb.set_label(nlabel)
            else:
                cb = plt.colorbar(sm, cax=cax, shrink=0.8)

                if cb_label is not None:
                    cb.ax.set_ylabel(cb_label)
        else:
            cmin, cmax = np.min(c), np.max(c)

            if cmin != cmax:
                c = (c - cmin) / (cmax - cmin)

        c = cmap(c)
    else:
        if not nonstring_container(c) and not isinstance(c, str):
            minc = np.min(node_color)

            c = np.array(
                [ncmap((node_color - minc)/(np.max(node_color) - minc))]*n)

    # check edge color
    group_based = False

    default_ecmap = (palette_discrete() if not nonstring_container(ncolor) and
                     ecolor == "group" else palette_continuous())

    if ecolor == "groups" or ecolor == "group":
        if network.structure is None:
            raise TypeError(
                "The graph must have a Structure/NeuralPop to use "
                "`ecolor='groups'`.")

        group_based = True
        ecolor = {}

        for i, src in enumerate(network.structure):
            if network.structure[src].ids:
                idx1 = network.structure[src].ids[0]
                for j, tgt in enumerate(network.structure):
                    if network.structure[tgt].ids:
                        idx2 = network.structure[tgt].ids[0]
                        if src == tgt:
                            ecolor[(src, tgt)] = c[idx1]
                        else:
                            ecolor[(src, tgt)] = 0.7*c[idx1] + 0.3*c[idx2]
    elif not nonstring_container(ecolor):
        ecolor = np.repeat(ecolor, e)

    # plot nodes
    scatter = []

    if simple_nodes:
        if nonstring_container(nshape):
            # matplotlib scatter does not support marker arrays
            if isinstance(nshape[0], nngt.Group):
                for g in nshape:
                    ids = g.ids if restrict_nodes is None \
                          else list(set(g.ids).intersection(restrict_nodes))

                    scatter.append(
                        axis.scatter(pos[ids, 0], pos[ids, 1], color=c[ids],
                                     s=0.5*np.array(nsize)[ids],
                                     marker=markers[ids[0]], zorder=2,
                                     edgecolors=nborder_color,
                                     linewidths=nborder_width, alpha=nalpha))
            else:
                ids = range(network.node_nb()) if restrict_nodes is None \
                      else restrict_nodes

                for i in ids:
                    scatter.append(axis.scatter(
                        pos[i, 0], pos[i, 1], color=c[i], s=0.5*nsize[i],
                        marker=nshape[i], zorder=2, edgecolors=nborder_color[i],
                        linewidths=nborder_width, alpha=nalpha))
        else:
            scatter.append(axis.scatter(
                pos[:, 0], pos[:, 1], color=c, s=0.5*np.array(nsize),
                marker=nshape, zorder=2, edgecolor=nborder_color,
                linewidths=nborder_width, alpha=nalpha))
    else:
        nodes = []
        axis.set_aspect(1.)

        if network.structure is not None:
            converter = None

            if restrict_nodes is not None:
                converter = {n: i for i, n in enumerate(restrict_nodes)}

            for group in network.structure.values():
                idx = group.ids

                if restrict_nodes is not None:
                    idx = [converter[n]
                           for n in set(restrict_nodes).intersection(idx)]

                for i, fc in zip(idx, c[idx]):
                    m = MarkerStyle(markers[i]).get_path()
                    center = np.average(m.vertices, axis=0)
                    m = Path(m.vertices - center, m.codes)
                    transform = Affine2D().scale(
                        0.5*nsize[i]).translate(pos[i][0], pos[i][1])
                    patch = PathPatch(
                        m.transformed(transform), facecolor=fc,
                        lw=nborder_width, edgecolor=nborder_color[i],
                        alpha=nalpha)
                    nodes.append(patch)
        else:
            for i, ci in enumerate(c):
                m = MarkerStyle(markers[i]).get_path()
                center = np.average(m.vertices, axis=0)
                m = Path(m.vertices - center, m.codes)
                transform = Affine2D().scale(0.5*nsize[i]).translate(
                    pos[i, 0], pos[i, 1])
                patch = PathPatch(
                    m.transformed(transform), facecolor=ci,
                    lw=nborder_width, edgecolor=nborder_color[i], alpha=nalpha)
                nodes.append(patch)

        scatter = PatchCollection(nodes, match_original=True, alpha=nalpha)
        scatter.set_zorder(2)
        axis.add_collection(scatter)

    # draw the edges
    arrows = []

    if e and decimate_connections != -1:
        avg_size = np.average(nsize)

        if group_based:
            for src_name, src_group in network.structure.items():
                for tgt_name, tgt_group in network.structure.items():
                    s_ids = src_group.ids

                    if restrict_sources is not None:
                        s_ids = list(set(restrict_sources).intersection(s_ids))

                    t_ids = tgt_group.ids

                    if restrict_targets is not None:
                        t_ids = list(set(restrict_targets).intersection(t_ids))

                    if t_ids and s_ids:
                        s_min, s_max = np.min(s_ids), np.max(s_ids) + 1
                        t_min, t_max = np.min(t_ids), np.max(t_ids) + 1

                        edges = np.array(
                            adj_mat[s_min:s_max, t_min:t_max].nonzero(),
                            dtype=int).T

                        edges[:, 0] += s_min
                        edges[:, 1] += t_min

                        strght_edges, self_loops, strght_sizes, loop_sizes = \
                            _split_edges_sizes(edges, esize,
                                               decimate_connections)

                        # plot
                        ec = ecolor[(src_name, tgt_name)]

                        if len(strght_edges) and fast:
                            dl       = 0 if simple_nodes else 0.5*np.max(nsize)
                            arrow_x  = pos[strght_edges[:, 1], 0] - \
                                       pos[strght_edges[:, 0], 0]
                            arrow_x -= np.sign(arrow_x) * dl
                            arrow_y  = pos[strght_edges[:, 1], 1] - \
                                       pos[strght_edges[:, 0], 1]
                            arrow_x -= np.sign(arrow_y) * dl

                            axis.quiver(
                                pos[strght_edges[:, 0], 0],
                                pos[strght_edges[:, 0], 1], arrow_x,
                                arrow_y, scale_units='xy', angles='xy',
                                scale=1, alpha=ealpha,
                                width=3e-3, linewidths=0.5*strght_sizes,
                                edgecolors=ec, color=ec, zorder=1, **kw)
                        elif len(strght_edges):
                            for i, (s, t) in enumerate(strght_edges):
                                xs, ys = pos[s, 0], pos[s, 1]
                                xt, yt = pos[t, 0], pos[t, 1]

                                sA = 0 if simple_nodes else 0.5*nsize[s]
                                sB = 0 if simple_nodes else 0.5*nsize[t]

                                cs = 'arc3,rad=0.2' if curved_edges else None

                                astyle = ArrowStyle.Simple(
                                    head_length=0.7*strght_sizes[i],
                                    head_width=0.7*strght_sizes[i],
                                    tail_width=0.3*strght_sizes[i])

                                arrows.append(FancyArrowPatch(
                                    posA=(xs, ys), posB=(xt, yt),
                                    arrowstyle=astyle, connectionstyle=cs,
                                    alpha=ealpha, fc=ec, zorder=1,
                                    shrinkA=0.5*nsize[s], shrinkB=0.5*nsize[t],
                                    lw=eborder_width, ec=eborder_color))

                        for i, s in enumerate(self_loops):
                            loop = _plot_loop(
                                i, s, pos, loop_sizes, nsize, max_nsize, xmax,
                                xmin, ymax, ymin, height, width, ec, ealpha,
                                eborder_width, eborder_color, fast, network,
                                restrict_nodes)

                            axis.add_artist(loop)
        else:
            strght_colors, loop_colors = [], []

            strght_edges, self_loops, strght_sizes, loop_sizes = \
                _split_edges_sizes(edges, esize, decimate_connections,
                ecolor, strght_colors, loop_colors)

            # keep only desired edges
            if None not in (restrict_sources, restrict_targets):
                new_edges = []
                new_colors = []

                for edge, ec in zip(strght_edges, strght_colors):
                    s, t = edge

                    if s in restrict_sources and t in restrict_targets:
                        new_edges.append(edge)
                        new_colors.append(ec)

                strght_edges = np.array(new_edges, dtype=int)
                strght_colors = new_colors

                if restrict_nodes is not None:
                    nodes = list(self_loops)
                    nodes.sort()

                    new_loops = set()
                    new_colors = []

                    for i, node in enumerate(restrict_nodes):
                        strght_edges[strght_edges == node] = i

                        if node in self_loops:
                            idx = nodes.index(node)
                            new_loops.add(i)
                            new_colors.append(loop_colors[idx])

                    self_loops = new_loops
                    loop_colors = new_colors
            elif restrict_sources is not None:
                new_edges = []
                new_colors = []

                for edge, ec in zip(strght_edges, strght_colors):
                    s, _ = edge

                    if s in restrict_sources:
                        new_edges.append(edge)
                        new_colors.append(ec)

                strght_edges = np.array(new_edges, dtype=int)

                loop_colors = [ec for ec, n in zip(loop_colors, self_loops)
                                if n in restrict_sources]
                self_loops  = self_loops.intersection(restrict_sources)
            elif restrict_targets is not None:
                new_edges = []
                new_colors = []

                for edge, ec in zip(strght_edges, strght_colors):
                    _, t = edge

                    if t in restrict_targets:
                        new_edges.append(edge)
                        new_colors.append(ec)

                strght_edges = np.array(new_edges, dtype=int)

                loop_colors = [ec for ec, n in zip(loop_colors, self_loops)
                                if n in restrict_targets]
                self_loops  = self_loops.intersection(restrict_targets)

            if fast:
                if len(strght_edges):
                    dl = 0.5*np.max(nsize) if not simple_nodes else 0.

                    arrow_x  = pos[strght_edges[:, 1], 0] - \
                                pos[strght_edges[:, 0], 0]
                    arrow_x -= np.sign(arrow_x) * dl
                    arrow_y  = pos[strght_edges[:, 1], 1] - \
                                pos[strght_edges[:, 0], 1]
                    arrow_x -= np.sign(arrow_y) * dl

                    axis.quiver(
                        pos[strght_edges[:, 0], 0], pos[strght_edges[:, 0], 1],
                        arrow_x, arrow_y, scale_units='xy', angles='xy',
                        scale=1, alpha=ealpha, width=3e-3,
                        linewidths=0.5*strght_sizes, ec=ecolor, fc=ecolor,
                        zorder=1)
            else:
                if len(strght_edges):
                    for i, (s, t) in enumerate(strght_edges):
                        xs, ys = pos[s, 0], pos[s, 1]
                        xt, yt = pos[t, 0], pos[t, 1]

                        astyle = ArrowStyle.Simple(
                            head_length=0.7*strght_sizes[i],
                            head_width=0.7*strght_sizes[i],
                            tail_width=0.3*strght_sizes[i])

                        sA = 0 if simple_nodes else 0.5*nsize[s]
                        sB = 0 if simple_nodes else 0.5*nsize[t]

                        cs = 'arc3,rad=0.2' if curved_edges else None

                        arrows.append(FancyArrowPatch(
                            posA=(xs, ys), posB=(xt, yt), arrowstyle=astyle,
                            connectionstyle=cs, alpha=ealpha, fc=ecolor[i],
                            zorder=1, shrinkA=sA, shrinkB=sB, lw=eborder_width,
                            ec=eborder_color))

            for i, s in enumerate(self_loops):
                ec = loop_colors[i]
                loop = _plot_loop(
                    i, s, pos, loop_sizes, nsize, max_nsize, xmax, xmin,
                    ymax, ymin, height, width, ec, ealpha, eborder_width,
                    eborder_color, fast, network, restrict_nodes)

                axis.add_artist(loop)

    # add patch arrows
    arrows = PatchCollection(arrows, match_original=True, alpha=ealpha)
    arrows.set_zorder(1)
    axis.add_collection(arrows)

    if kwargs.get('tight', True):
        plt.tight_layout()
        plt.subplots_adjust(
            hspace=0., wspace=0., left=0., right=0.95 if colorbar else 1.,
            top=1., bottom=0.)

    # annotations
    annotations = kwargs.get("annotations",
        [str(i) for i in range(network.node_nb())] if restrict_nodes is None
        else [str(i) for i in restrict_nodes])

    if isinstance(annotations, str):
        assert annotations in network.node_attributes, \
            "String values for `annotations` must be a node attribute."

        if restrict_nodes is None:
            annotations = network.node_attributes[annotations]
        else:
            annotations = network.get_node_attributes(
                nodes=list(restrict_nodes), name=annotations)
    elif len(annotations) == network.node_nb() and restrict_nodes is not None:
        annotations = [annotations[i] for i in restrict_nodes]
    else:
        assert len(annotations) == n, "One annotation per node is required."

    annotate = kwargs.get("annotate", True)

    if annotate:
        annot = axis.annotate(
            "", xy=(0,0), xytext=(10,10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"))

        annot.set_visible(False)

        def update_annot(ind):
            annot.xy = pos[ind["ind"][0]]
            text = "\n".join([annotations[n] for n in ind["ind"]])
            annot.set_text(text)
            annot.get_bbox_patch().set_facecolor("w")

        def hover(event):
            if hover.bg is None:
                # first run, save the current plot
                hover.bg = fig.canvas.copy_from_bbox(fig.bbox)

            vis = annot.get_visible()
            if event.inaxes == axis:
                if fast or simple_nodes:
                    for sc in scatter:
                        cont, ind = sc.contains(event)
                        if cont:
                            update_annot(ind)
                            fig.canvas.restore_region(hover.bg)
                            annot.set_visible(True)
                            axis.draw_artist(annot)
                            fig.canvas.blit(fig.bbox)
                        else:
                            if vis:
                                annot.set_visible(False)
                                fig.canvas.restore_region(hover.bg)
                                fig.canvas.blit(fig.bbox)
                else:
                    cont, ind = scatter.contains(event)
                    if cont:
                        update_annot(ind)
                        fig.canvas.restore_region(hover.bg)
                        annot.set_visible(True)
                        axis.draw_artist(annot)
                        fig.canvas.blit(fig.bbox)
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.restore_region(hover.bg)
                            fig.canvas.blit(fig.bbox)

                fig.canvas.flush_events()

        hover.bg = None

        fig.canvas.mpl_connect("motion_notify_event", hover)

    if show:
        plt.show()


def hive_plot(network, radial, axes=None, axes_bins=None, axes_range=None,
              axes_angles=None, axes_labels=None, axes_units=None,
              intra_connections=True, highlight_nodes=None,
              highlight_edges=None, nsize=None, esize=None, max_nsize=10,
              max_esize=1, axes_colors=None, edge_colors=None, edge_alpha=0.05,
              nborder_color="k", nborder_width=0.2, show_names=True,
              show_circles=False, axis=None, tight=True, show=False):
    '''
    Draw a hive plot of the graph.

    Note
    ----
    For directed networks, the direction of intra-axis connections is
    counter-clockwise.
    For inter-axes connections, the default edge color is closest to the color
    of the source group (i.e. from a red group to a blue group, edge color will
    be a reddish violet , while from blue to red, it will be a blueish violet).

    Parameters
    ----------
    network : :class:`~nngt.Graph`
        Graph to plot.
    radial : str, list of str or array-like
        Values that will be used to place the nodes on the axes. Either one
        identical property is used for all axes (traditional hive plot) or
        one radial coordinate per axis is used (custom hive plot).
        If radial is a string or a list of strings, then these must correspond
        to the names of node attributes stored in the graph.
    axes : str, or list of str, optional (default: one per radial coordinate)
        Name of the attribute(s) that will be used to make each of the axes
        (i.e. each group of nodes).
        This can be either "groups" if the graph has a structure or is a
        :class:`~nngt.Network`, a list of (Meta)Group names, or any (list of)
        node attribute(s).
        If a single node attribute is used, `axes_bins` must be provided to
        make one axis for each range of values.
        If there are multiple radial coordinates, then leaving `axes` blanck
        will plot all nodes on each of the axes (one per radial coordinate).
    axes_bins : int or array-like, optional (default: all nodes on each axis)
        Required if there is a single radial coordinate and a single axis
        entry: provides the bins that will be used to separate the nodes
        into groups (one per axis). For N axes, there must therefore be N + 1
        entries in `axes_bins`, or `axis_bins` must be equal to N, in which
        case the nodes are separated into N evenly sized bins.
    axes_units : str, optional
        Units used to scale the axes. Either "native" to have them scaled
        between the minimal and maximal radial coordinates among all axes,
        "rank", to use the min and max ranks of the nodes on all axes, or
        "normed", to have each axis go from zero (minimal local radial
        coordinate) to one (maximal local radial coordinate).
        "native" is the default if there is a single radial coordinate,
        "normed" is the default for multiple coordinates.
    axes_angles : list of angles, optional (default: automatic)
        Angles for each of the axes, by increasing degree. If
        `intra_connections` is True, then angles of duplicate axes must be
        adjacent, e.g. ``[a1, a1bis, a2, a2bis, a3, a3bis]``.
    axes_labels : str or list of str, optional
        Label of each axis. For binned axes, it can be automatically formatted
        via the three entries ``{name}``, ``{start}``, ``{stop}``.
        E.g. "{name} in [{start}, {stop}]" would give "CC in [0, 0.2]" for
        a first axis and "CC in [0.2, 0.4]" for a second axis.
    intra_connections : bool, optional (default: True)
        Show connections between nodes belonging to the same axis. If true,
        then each axis is duplicated to display intra-axis connections.
    highlight_nodes : list of nodes, optional (default: all nodes)
        Highlight a subset of nodes and their connections, all other nodes
        and connections will be gray.
    highlight_edges : list of edges, optional (default: all edges)
        Highlight a subset of edges; all other connections will be gray.
    nsize : float, str, or array-like, optional (default: automatic)
        Size of the nodes on the axes. Either a fixed size, the name of a
        node attribute, or a list of user-defined values.
    esize : float or str, optional (default: 1)
        Size of the edges. Either a fixed size or the name of an edge
        attribute.
    max_nsize : float, optional (default: 10)
        Maximum node size if `nsize` is an attribute or a list of
        user-defined values.
    max_esize : float, optional (default: 1)
        Maximum edge size if `esize` is an attribute.
    axes_colors : valid matplotlib color/colormap, optional (default: Set1)
        Color associated to each axis.
    nborder_color : matplotlib color, optional (default: "k")
        Color of the node's border.
        or floats in [0, 1] defining the position in the palette.
    nborder_width : float, optional (default: 0.2)
        Width of the border.
    edge_colors : valid matplotlib color/colormap, optional (default: auto)
        Color of the edges. By default it is the intermediate color between
        two axes colors. To provide custom colors, they must be provided as
        a dictionnary of axes edges ``{(0, 0): "r", (0, 1): "g", (1, 0): "b"}``
        with default color being black.
    edge_alpha : float, optional (default: 0.05)
        Edge opacity.
    show_names : bool, optional (default: True)
        Show axes names and properties.
    show_circles : bool, optional (default: False)
        Show the circles associated to the maximum value of each axis.
    axis : matplotlib axis, optional (default: create new axis)
        Axis on which the network will be plotted.
    tight : bool, optional (default: True)
        Set figure layout to tight (set to False if plotting multiple axes on
        a single figure).
    show : bool, optional (default: True)
        Display the plot immediately.
    '''
    import matplotlib.pyplot as plt

    # get numer of axes and radial coordinates
    num_axes, num_radial = _get_axes_radial_coord(
        radial, axes, axes_bins, network)

    # get axes names, associated nodes, and radial values
    ax_names, ax_nodes, ax_radco = _get_axes_nodes(
        network, radial, axes, axes_bins, num_axes, num_radial)

    # get highlighted nodes and edges
    if highlight_nodes:
        highlight_nodes = set(highlight_nodes)
    else:
        highlight_nodes= set()

    if highlight_edges is not None:
        highlight_edges = {tuple(e) for e in highlight_edges}

    # get units, maximum values for the axes, renormalize radial values
    if axes_units is None:
        axes_units = "normed" if num_radial > 1 else "native"

    radial_values = _get_radial_values(ax_radco, axes_units, network)

    # compute the angles
    angles = None

    if axes_angles is None:
        dtheta = 2 * np.pi / num_axes

        if intra_connections:
            angles = []

            for i in range(num_axes):
                angles.extend(((i - 0.125)*dtheta, (i + 0.125)*dtheta))
        else:
            angles = [i*dtheta for i in range(num_axes)]
    else:
        angles = [a*np.pi/180 for a in ax_angles]

    # renormalize the sizes
    nsize = _get_size(nsize, max_nsize, ax_nodes, network)

    nedges = network.edge_nb()

    esize = np.ones(nedges) if esize is None else network.edge_attributes[esize]
    esize *= max_esize / esize.max()

    esize = {tuple(e): s for e, s in zip(network.edges_array, esize)}

    # get the colors
    ncolors, ecolors = _get_colors(axes_colors, edge_colors, angles, num_axes,
                                   intra_connections, network)

    # make the figure
    if axis is None:
        _, axis = plt.subplots()

    # plot the nodes and axes
    node_pos  = []
    max_radii = []

    for i, (nn, rr) in enumerate(zip(ax_nodes, radial_values)):
        if len(nn):
            # max radii
            rax = np.array([RMIN, rr[nn].max()])

            max_radii.extend([rax[-1]]*(1 + intra_connections))

            # plot max radii
            if show_circles:
                aa = np.arange(0, 2*np.pi, 0.02)
                xx = rax[-1]*np.cos(aa)
                yy = rax[-1]*np.sin(aa)
                axis.plot(xx, yy, color="grey", alpha=0.2, zorder=1)

            # comppute angles
            aa = [angles[2*i] if intra_connections else angles[i]]

            if intra_connections:
                aa += [angles[2*i+1]]

            for j, a in enumerate(aa):
                # plot axes lines
                lw = 1 if j % 2 else 2

                axis.plot(rax*np.cos(a), rax*np.sin(a), color="grey", lw=lw,
                          zorder=1)

                # compute node positions
                xx = rr*np.cos(a)
                yy = rr*np.sin(a)

                node_pos.append(np.array([xx, yy]).T)

                if highlight_nodes:
                    greys = list(set(nn).difference(highlight_nodes))

                    _plot_nodes(greys, nsize, xx, yy, "grey",
                                nborder_width, nborder_color, axis, zorder=3)

                hlght = (nn if not highlight_nodes
                         else list(highlight_nodes.intersection(nn)))

                _plot_nodes(hlght, nsize, xx, yy, ncolors[i],
                            nborder_width, nborder_color, axis, zorder=4)
        else:
            node_pos.extend([[]]*(1 + intra_connections))
            max_radii.extend([RMIN]*(1 + intra_connections))

    # plot the edges
    xs, ys = [], []

    for i, n1 in enumerate(ax_nodes):
        targets = ax_nodes if network.is_directed() else ax_nodes[i:]

        for j, n2 in enumerate(ax_nodes):
            # ignore i = j if intra_connections is True
            if i == j and not intra_connections:
                continue

            # find which axes should be used
            idx_s, idx_t = _get_ax_angles(
                angles, i, j, intra_connections)

            # get the edges
            edges = network.get_edges(source_node=n1, target_node=n2)

            if len(edges):
                color = ecolors[(i, j)]

                paths_greys = []
                paths_hghlt = []

                lw = []

                for (ns, nt) in edges:
                    pstart = node_pos[idx_s][ns]
                    pstop  = node_pos[idx_t][nt]

                    contains = True

                    if highlight_edges is not None:
                        contains = (ns, nt) in highlight_edges
                    elif highlight_nodes is not None:
                        contains = \
                            ns in highlight_nodes or nt in highlight_nodes

                    if highlight_edges is None or contains:
                        paths_hghlt.append(_plot_bezier(
                            pstart, pstop, angles[idx_s], angles[idx_t],
                            radial_values[i][ns], radial_values[j][nt], i, j,
                            num_axes, xs, ys))

                        lw.append(esize[(ns, nt)])
                    else:
                        paths_greys.append(_plot_bezier(
                            pstart, pstop, angles[idx_s], angles[idx_t],
                            radial_values[i][ns], radial_values[j][nt], i, j,
                            num_axes, xs, ys))

                if paths_greys:
                    pcol = PathCollection(
                        paths_greys, facecolors="none", edgecolors="grey",
                        alpha=0.1*edge_alpha, zorder=1)

                    axis.add_collection(pcol)

                alpha = 0.7 if highlight_nodes else edge_alpha

                pcol = PathCollection(paths_hghlt, facecolors="none", lw=lw,
                                      edgecolors=color, alpha=alpha, zorder=2)

                axis.add_collection(pcol)

    _set_names_lims(ax_names, angles, max_radii, xs, ys, intra_connections,
                    show_names, axis, show_circles)


    axis.set_aspect(1)
    axis.axis('off')

    if tight:
        plt.tight_layout()

    if show:
        plt.show()


def library_draw(network, nsize="total-degree", ncolor=None, nshape="o",
                 nborder_color="k", nborder_width=0.5, esize=1., ecolor="k",
                 ealpha=0.5, curved_edges=False, threshold=0.5,
                 decimate_connections=None, spatial=True,
                 restrict_sources=None, restrict_targets=None,
                 restrict_nodes=None, restrict_edges=None,
                 show_environment=True, size=(600, 600), xlims=None,
                 ylims=None, dpi=75, axis=None, colorbar=False,
                 show_labels=False, layout=None, show=False, **kwargs):
    '''
    Draw a given :class:`~nngt.Graph` using the underlying library's drawing
    functions.

    .. versionadded:: 2.0

    .. warning::
        When using igraph or graph-tool, if you want to use the `axis`
        argument, then you must first switch the matplotlib backend to its
        cairo version using e.g. ``plt.switch_backend("Qt5Cairo")`` if your
        normal backend is Qt5 ("Qt5Agg").

    Parameters
    ----------
    network : :class:`~nngt.Graph` or subclass
        The graph/network to plot.
    nsize : float, array of float or string, optional (default: "total-degree")
        Size of the nodes as a percentage of the canvas length. Otherwise, it
        can be a string that correlates the size to a node attribute among
        "in/out/total-degree", or "betweenness".
    ncolor : float, array of floats or string, optional (default: 0.5)
        Color of the nodes; if a float in [0, 1], position of the color in the
        current palette, otherwise a string that correlates the color to a node
        attribute or "in/out/total-degree", "betweenness" and "group".
        Default to red or one color per group in the graph if not specified.
    nshape : char, array of chars, or groups, optional (default: "o")
        Shape of the nodes (see `Matplotlib markers <http://matplotlib.org/api/
        markers_api.html?highlight=marker#module-matplotlib.markers>`_).
        When using groups, they must be pairwise disjoint; markers will be
        selected iteratively from the matplotlib default markers.
    nborder_color : char, float or array, optional (default: "k")
        Color of the node's border using predefined `Matplotlib colors
        <http://matplotlib.org/api/colors_api.html?highlight=color
        #module-matplotlib.colors>`_).
        or floats in [0, 1] defining the position in the palette.
    nborder_width : float or array of floats, optional (default: 0.5)
        Width of the border in percent of canvas size.
    esize : float, str, or array of floats, optional (default: 0.5)
        Width of the edges in percent of canvas length. Available string values
        are "betweenness" and "weight".
    ecolor : str, char, float or array, optional (default: "k")
        Edge color. If ecolor="groups", edges color will depend on the source
        and target groups, i.e. only edges from and toward same groups will
        have the same color.
    threshold : float, optional (default: 0.5)
        Size under which edges are not plotted.
    decimate_connections : int, optional (default: keep all connections)
        Plot only one connection every `decimate_connections`.
        Use -1 to hide all edges.
    spatial : bool, optional (default: True)
        If True, use the neurons' positions to draw them.
    restrict_sources : str, group, or list, optional (default: all)
        Only draw edges starting from a restricted set of source nodes.
    restrict_targets : str, group, or list, optional (default: all)
        Only draw edges ending on a restricted set of target nodes.
    restrict_nodes : str, group, or list, optional (default: plot all nodes)
        Only draw a subset of nodes.
    restrict_edges : list of edges, optional (default: all)
        Only draw a subset of edges.
    show_environment : bool, optional (default: True)
        Plot the environment if the graph is spatial.
    size : tuple of ints, optional (default: (600, 600))
        (width, height) tuple for the canvas size (in px).
    dpi : int, optional (default: 75)
        Resolution (dot per inch).
    colorbar : bool, optional (default: False)
        Whether to display a colorbar for the node colors or not.
    axis : matplotlib axis, optional (default: create new axis)
        Axis on which the network will be plotted.
    layout : str, optional (default: library-dependent or spatial positions)
        Name of a standard layout to structure the network. Available layouts
        are: "circular", "spring-block", "random". If no layout is
        provided and the network is spatial, then node positions will be
        used by default.
    show : bool, optional (default: True)
        Display the plot immediately.
    **kwargs : dict
        Optional keyword arguments.

        ================  ==================  =================================
              Name               Type            Purpose and possible values
        ================  ==================  =================================
                                              Desired node colormap (default is
        node_cmap         str                 "magma" for continuous variables
                                              and "Set1" for groups)
        ----------------  ------------------  ---------------------------------
        title             str                 Title of the plot
        ----------------  ------------------  ---------------------------------
        max_*             float               Maximum value for `nsize` or
                                              `esize`
        ----------------  ------------------  ---------------------------------
        min_*             float               Minimum value for `nsize` or
                                              `esize`
        ----------------  ------------------  ---------------------------------
        annotate          bool                Use annotations to show node
                                              information (default: True)
        ----------------  ------------------  ---------------------------------
                                              Information that will be displayed
        annotations       str or list         such as a node attribute or a list
                                              of values. (default: node id)
        ================  ==================  =================================
    '''
    import matplotlib.pyplot as plt

    # backend and axis
    try:
        import igraph
        igv = igraph.__version__
    except:
        igv = '1.0'

    ig_test = nngt.get_config("backend") == "igraph" and igv <= '0.9.6'

    if nngt.get_config("backend") == "graph-tool" or ig_test:
        mpl_backend = mpl.get_backend()

        if mpl_backend.startswith("Qt4"):
            if mpl_backend != "Qt4Cairo":
                plt.switch_backend("Qt4Cairo")
        elif mpl_backend.startswith("Qt5"):
            if mpl_backend != "Qt5Cairo":
                plt.switch_backend("Qt5Cairo")
        elif mpl_backend.startswith("GTK"):
            if mpl_backend != "GTK3Cairo":
                plt.switch_backend("GTK3Cairo")
        elif mpl_backend != "cairo":
            plt.switch_backend("cairo")

    if axis is None:
        size_inches = (size[0]/float(dpi), size[1]/float(dpi))
        fig, axis = plt.subplots(figsize=size_inches)

    axis.axis('off')

    # default plot
    if nngt.get_config("backend") == "nngt":
        draw_network(
            network, nsize=nsize, ncolor=ncolor, nshape=nshape,
            nborder_color=nborder_color, nborder_width=nborder_width,
            esize=esize, ecolor=ecolor, curved_edges=curved_edges,
            threshold=threshold, decimate_connections=decimate_connections,
            spatial=spatial, restrict_nodes=restrict_nodes,
            show_environment=show_environment, size=size, axis=axis,
            layout=layout, show=show, **kwargs)

        return

    # otherwise, preapre data
    restrict_nodes = _convert_to_nodes(restrict_nodes,
                                       "restrict_nodes", network)

    # shize and shape
    max_nsize = kwargs.get("max_nsize", 5)
    min_nsize = kwargs.get("min_nsize", None)

    max_esize = kwargs.get("max_esize", 2)
    min_esize = kwargs.get("min_esize", 0)

    markers, nsize, esize = _node_edge_shape_size(
        network, nshape, nsize, max_nsize, min_nsize, esize, max_esize,
        min_esize, restrict_nodes, restrict_edges, size, threshold)

    # node color information
    if ncolor is None:
        if network.structure is not None:
            ncolor = "group"
        else:
            ncolor = "r"

    discrete_colors, default_ncmap = _get_ncmap(network, ncolor)

    ncmap = get_cmap(kwargs.get("node_cmap", default_ncmap))

    node_color, nticks, ntickslabels, nlabel = _node_color(
        network, restrict_nodes, ncolor, discrete_colors=discrete_colors)

    # edge color
    ecolor = _edge_prop(network, ecolor)
    esize  = _edge_prop(network, esize)

    if nonstring_container(esize) and len(esize):
        esize *= max_esize / np.max(esize)
    
    # environment
    if spatial and network.is_spatial():
        if show_environment:
            nngt.geometry.plot.plot_shape(network.shape, axis=axis, show=False)

    # do the plot
    if nngt.get_config("backend") == "graph-tool":
        from graph_tool.draw import (graph_draw, sfdp_layout, random_layout)

        graph = network.graph

        # resize
        if nonstring_container(nsize):
            nsize *= 0.05

        nborder_width *= 0.1

        esize *= 0.02

        # positions
        pos = None

        if layout is None:
            if isinstance(network, nngt.SpatialGraph) and spatial:
                xy  = network.get_positions()
                pos = graph.new_vp("vector<double>", vals=xy)
            else:
                weights = (None if not network.is_weighted()
                           else graph.edge_properties['weight'])
                pos = sfdp_layout(graph, eweight=weights)
        elif layout == "random":
            pos = random_layout(graph)
        elif layout == "circular":
            pos = graph.new_vp("vector<double>",
                               vals=_circular_layout(network, nsize))
        elif nonstring_container(layout):
            assert np.shape(layout) == (network.node_nb(), 2), \
                "One position per node in the network is required."
            pos = graph.new_vp("vector<double>", vals=layout)
        else:
            # spring block
            weights = (None if not network.is_weighted()
                       else graph.edge_properties['weight'])
            pos = sfdp_layout(graph, eweight=weights)

        convert_shape = {
            "o": "circle",
            "v": "triangle",
            "^": "triangle",
            "s": "square",
            "p": "pentagon",
            "h": "hexagon",
            "H": "hexagon",
        }

        shape_dict = defaultdict(
            lambda k: "circle" if k not in convert_shape.values() else k)

        for k, v in convert_shape.items():
            shape_dict[k] = v

        vprops = {
            "shape": shape_dict[nshape],
            "fill_color": _to_gt_prop(graph, node_color, ncmap, color=True),
            "color": _to_gt_prop(graph, nborder_color, ncmap, color=True),
            "size": _to_gt_prop(graph, nsize, ncmap),
            "pen_width": _to_gt_prop(graph, nborder_width, ncmap),
        }

        if vprops["fill_color"] is None:
            vprops["fill_color"] = [0.640625, 0, 0, 0.9]

        eprops = None if network.edge_nb() == 0 else {
            "color": _to_gt_prop(graph, ecolor, palette_continuous(),
                                 ptype='edge', color=True),
            "pen_width": _to_gt_prop(graph, esize, None, ptype='edge'),
        }

        if restrict_edges is not None:
            efilt = network.graph.new_ep(
                "bool", vals=np.zeros(network.edge_nb(), dtype=bool))
            eids = [network.edge_id(e) for e in restrict_edges]

            efilt.a[eids] = 1

            network.graph.set_edge_filter(efilt)

        graph_draw(network.graph, pos=pos, vprops=vprops, eprops=eprops,
                   output_size=size, mplfig=axis)

        if restrict_edges is not None:
            # clear edge filter
            network.graph.set_edge_filter(None)
    elif nngt.get_config("backend") == "networkx":
        import networkx as nx

        pos = None

        if layout is None:
            if isinstance(network, nngt.SpatialGraph) and spatial:
                xy  = network.get_positions()
                pos = {i: coords for i, coords in enumerate(xy)}
        elif layout == "circular":
            pos = nx.circular_layout(network.graph)
        elif layout == "random":
            pos = nx.random_layout(network.graph)
        elif nonstring_container(layout):
            assert np.shape(layout) == (network.node_nb(), 2), \
                "One position per node in the network is required."
            pos = {i: coords for i, coords in enumerate(layout)}
        else:
            pos = nx.spring_layout(network.graph)

        # normalize sizes compared to igraph
        nsize = _increase_nx_size(nsize)

        nborder_width = _increase_nx_size(nborder_width, 2)

        edges = None if restrict_edges is None else list(restrict_edges)

        nx.draw_networkx(
            network.graph, pos=pos, ax=axis, nodelist=restrict_nodes,
            edgelist=edges, node_size=nsize, node_color=node_color,
            node_shape=nshape, linewidths=nborder_width, edge_color=ecolor,
            edge_cmap=palette_continuous(), cmap=ncmap,
            with_labels=show_labels, width=esize, edgecolors=nborder_color)
    elif nngt.get_config("backend") == "igraph":
        import igraph
        from igraph import Layout, PrecalculatedPalette

        pos = None

        if layout is None:
            if isinstance(network, nngt.SpatialGraph) and spatial:
                xy  = network.get_positions()
                pos = Layout(xy)
            else:
                pos = network.graph.layout_fruchterman_reingold()
        elif layout == "circular":
            pos = network.graph.layout_circle()
        elif layout == "random":
            pos = network.graph.layout_random()

        palette = PrecalculatedPalette(ncmap(np.linspace(0, 1, 256)))

        # convert color to igraph-format
        node_color = _to_ig_color(node_color)
        ecolor     = _to_ig_color(ecolor)

        convert_shape = {
            "o": "circle",
            "v": "triangle-down",
            "^": "triangle-up",
            "s": "rectangle",
        }

        shape_dict = defaultdict(
            lambda k: "circle" if k not in convert_shape.values() else k)

        for k, v in convert_shape.items():
            shape_dict[k] = v

        if nonstring_container(nsize):
            nsize = list(nsize)

        if nonstring_container(node_color):
            node_color = list(node_color)

        if nonstring_container(esize):
            esize = list(esize)

        if nonstring_container(ecolor):
            ecolor = list(ecolor)

        visual_style = {
            "vertex_size": nsize,
            "vertex_color": node_color,
            "vertex_shape": shape_dict[nshape],
            "edge_width": esize,
            "edge_color": ecolor,
            "layout": pos,
            "palette": palette,
        }

        graph = network.graph

        if restrict_edges is not None:
            eids  = [network.edge_id(e) for e in restrict_edges]
            graph = network.graph.subgraph_edges(eids, delete_vertices=False)

        if igv > '0.9.6':
            igraph.plot(graph, target=axis, **visual_style)
        else:
            graph_artist = GraphArtist(graph, axis, **visual_style)

            axis.artists.append(graph_artist)

    if "title" in kwargs:
        axis.set_title(kwargs["title"])

    if show:
        plt.show()


def chord_diagram(network, weights=True, names=None, order=None, width=0.1,
                  pad=2., gap=0.03, chordwidth=0.7, axis=None, colors=None,
                  cmap=None, alpha=0.7, use_gradient=False, chord_colors=None,
                  show=False, **kwargs):
    """
    Plot a chord diagram.

    Parameters
    ----------
    network : a :class:`nngt.Graph` object
        Network used to plot the chord diagram.
    weights : bool or str, optional (default: 'weight' attribute)
        Weights used to plot the connections.
    names : str or list of str, optional (default: no names)
        Names of the nodes that will be displayed, either a node attribute
        or a custom list (must be ordered following the nodes' indices).
    order : list, optional (default: order of the matrix entries)
        Order in which the arcs should be placed around the trigonometric
        circle.
    width : float, optional (default: 0.1)
        Width/thickness of the ideogram arc.
    pad : float, optional (default: 2)
        Distance between two neighboring ideogram arcs. Unit: degree.
    gap : float, optional (default: 0.03)
        Distance between the arc and the beginning of the cord.
    chordwidth : float, optional (default: 0.7)
        Position of the control points for the chords, controlling their shape.
    axis : matplotlib axis, optional (default: new axis)
        Matplotlib axis where the plot should be drawn.
    colors : list, optional (default: from `cmap`)
        List of user defined colors or floats.
    cmap : str or colormap object (default: viridis)
        Colormap that will be used to color the arcs and chords by default.
        See `chord_colors` to use different colors for chords.
    alpha : float in [0, 1], optional (default: 0.7)
        Opacity of the chord diagram.
    use_gradient : bool, optional (default: False)
        Whether a gradient should be use so that chord extremities have the
        same color as the arc they belong to.
    chord_colors : str, or list of colors, optional (default: None)
        Specify color(s) to fill the chords differently from the arcs.
        When the keyword is not used, chord colors default to the colomap given
        by `colors`.
        Possible values for `chord_colors` are:

        * a single color (do not use an RGB tuple, use hex format instead),
          e.g. "red" or "#ff0000"; all chords will have this color
        * a list of colors, e.g. ``["red", "green", "blue"]``, one per node
          (in this case, RGB tuples are accepted as entries to the list).
          Each chord will get its color from its associated source node, or
          from both nodes if `use_gradient` is True.
    show : bool, optional (default: False)
        Whether the plot should be displayed immediately via an automatic call
        to `plt.show()`.
    kwargs : keyword arguments
        Available kwargs are:

        ================  ==================  ===============================
              Name               Type           Purpose and possible values
        ================  ==================  ===============================
        fontcolor         str or list         Color of the names
        fontsize          int                 Size of the font for names
        rotate_names      (list of) bool(s)   Rotate names by 90°
        sort              str                 Either "size" or "distance"
        zero_entry_size   float               Size of zero-weight reciprocal
        ================  ==================  ===============================
    """
    ww = 'weight' if weights is True else weights
    nn = network.node_attributes[names] if isinstance(names, str) else names

    mat = network.adjacency_matrix(weights=ww)

    return _chord_diag(
        mat, nn, order=order, width=width, pad=pad, gap=gap,
        chordwidth=chordwidth, ax=axis, colors=colors, cmap=cmap, alpha=alpha,
        use_gradient=use_gradient, chord_colors=chord_colors, show=show,
        **kwargs)


# ----- #
# Tools #
# ----- #

def _norm_size(size, max_size, min_size):
    ''' Normalize the size array '''
    maxs = np.max(size)
    mins = np.min(size)

    if min_size is None or maxs == mins:
        return size * max_size / np.max(size)

    return min_size + (max_size - min_size) * (size - mins) / (maxs - mins)


def _node_edge_shape_size(network, nshape, nsize, max_nsize, min_nsize, esize,
                          max_esize, min_esize, restrict_nodes, edges, size,
                          threshold, simple_nodes=False):
    ''' Returns the shape and size of the nodes and edges '''
    n = network.node_nb() if restrict_nodes is None else len(restrict_nodes)
    e = len(edges) if edges is not None else network.edge_nb()

    # markers
    markers = nshape

    if nonstring_container(nshape):
        if isinstance(nshape[0], nngt.Group):
            # check disjunction
            for i, g in enumerate(nshape):
                for j in range(i + 1, len(nshape)):
                    if not set(g.ids).isdisjoint(nshape[j].ids):
                        raise ValueError("Groups passed to `nshape` "
                                         "must be disjoint.")

            mm = cycle(MarkerStyle.filled_markers)

            shapes  = np.full(n, "", dtype=object)

            if restrict_nodes is None:
                for g, m in zip(nshape, mm):
                    shapes[g.ids] = m
            else:
                converter = {n: i for i, n in enumerate(restrict_nodes)}
                for g, m in zip(nshape, mm):
                    ids = [converter[n]
                           for n in restrict_nodes.intersection(g.ids)]
                    shapes[ids] = m

            markers = list(shapes)
        elif len(nshape) == network.node_nb() and restrict_nodes is not None:
            markers = nshape[list(restrict_nodes)]
        elif len(nshape) != n:
            raise ValueError("When passing an array of markers to "
                             "`nshape`, one entry per node in the "
                             "network must be provided.")
    else:
        markers = [nshape for _ in range(n)]

    # size
    if isinstance(nsize, str):
        if e:
            nsize = _node_size(network, restrict_nodes, nsize)
            nsize = _norm_size(nsize, max_nsize, min_nsize)
        else:
            nsize = np.ones(n, dtype=float)
    elif isinstance(nsize, (float, int, np.number)):
        nsize = np.full(n, nsize, dtype=float)
    elif nonstring_container(nsize):
        if len(nsize) == n:
            nsize = _norm_size(nsize, max_nsize, min_nsize)
        elif len(nsize) == network.node_nb() and restrict_nodes is not None:
            nsize = np.asarray(nsize)[list(restrict_nodes)]
            nsize = _norm_size(nsize, max_nsize, min_nsize)
        else:
            raise ValueError("`nsize` must contain either one entry per node "
                             "or be the same length as `restrict_nodes`.")

    if e:
        if isinstance(esize, str):
            esize = _edge_size(network, edges, esize)
            esize = _norm_size(esize, max_esize, min_esize)
            esize[esize < threshold] = 0.
        else:
            esize = _norm_size(esize, max_esize, min_esize)
    else:
        esize = np.array([])

    return markers, nsize, esize


def _set_ax_lim(ax, xmax, xmin, ymax, ymin, height, width, xlims, ylims,
                max_nsize, fast):
    if xlims is not None:
        ax.set_xlim(*xlims)
    else:
        dx = 0.05*width if fast else 1.5*max_nsize
        ax.set_xlim(xmin - dx, xmax + dx)
    if ylims is not None:
        ax.set_ylim(*ylims)
    else:
        dy = 0.05*height if fast else 1.5*max_nsize
        ax.set_ylim(ymin - dy, ymax + dy)


def _node_size(network, restrict_nodes, nsize):
    restrict_nodes = None if restrict_nodes is None else list(restrict_nodes)

    n = network.node_nb() if restrict_nodes is None else len(restrict_nodes)

    size = np.ones(n, dtype=float)

    if nsize in network.node_attributes:
        size = network.get_node_attributes(nodes=restrict_nodes, name=nsize)
    if "degree" in nsize:
        deg_type = nsize[:nsize.index("-")]
        size = network.get_degrees(deg_type,
                                   nodes=restrict_nodes).astype(float)
        if np.isclose(size.min(), 0):
            size[np.isclose(size, 0)] = 0.5
        if size.max() > 15*size.min():
            size = np.power(size, 0.4)
    elif "strength" in nsize:
        deg_type = nsize[:nsize.index("-")]
        size = network.get_degrees(deg_type, weights='weight',
                                   nodes=restrict_nodes)
        if np.isclose(size.min(), 0):
            size[np.isclose(size, 0)] = 0.5
        if size.max() > 15*size.min():
            size = np.power(size, 0.4)
    elif nsize == "betweenness":
        betw = None

        if restrict_nodes is None:
            betw = network.get_betweenness("node").astype(float)
        else:
            betw = network.get_betweenness(
                "node").astype(float)[restrict_nodes]

        if network.is_connected("weak") == 1:
            size *= betw
            if size.max() > 15*size.min():
                min_size = size[size!=0].min()
                size[size == 0.] = min_size
                size = np.log(size)
                if size.min()<0:
                    size -= 1.1*size.min()
    elif nsize == "clustering":
        size *= nngt.analysis.local_clustering(network, nodes=restrict_nodes)
    elif nsize in nngt.analyze_graph:
        if restrict_nodes is None:
            size *= nngt.analyze_graph[nsize](network)
        else:
            size *= nngt.analyze_graph[nsize](network)[restrict_nodes]

    if np.any(size):
        size /= size.max()

    return size.astype(float)


def _edge_size(network, edges, esize):
    num_edges = len(edges) if edges is not None else network.edge_nb()

    size = np.repeat(1., num_edges)

    if num_edges:
        max_size = 1.

        if nonstring_container(esize):
            max_size = np.max(esize)
        elif esize == "betweenness":
            betw = network.get_betweenness("edge")

            max_size = np.max(betw)

            size = betw if restrict_nodes is None else betw[restrict_nodes]
        elif esize == "weight":
            size = network.get_weights(edges=edges)

            max_size = np.max(network.get_weights())

        if np.any(size):
            size /= max_size

    return size


def _node_color(network, restrict_nodes, ncolor, discrete_colors=False):
    '''
    Return an array of colors, a set of ticks, and a label for the colorbar
    of the nodes (if necessary).
    '''
    color        = ncolor
    nticks       = None
    ntickslabels = None
    nlabel       = ""

    n = network.node_nb() if restrict_nodes is None else len(restrict_nodes)

    if restrict_nodes is not None:
        restrict_nodes = list(set(restrict_nodes))

    if isinstance(ncolor, float):
        color = np.repeat(ncolor, n)
    elif isinstance(ncolor, str):
        if ncolor in ColorConverter.colors or ncolor.startswith("#"):
            color = np.repeat(ncolor, n)
        elif discrete_colors:
            unique = None
            values = None

            if ncolor == "group" or ncolor == "groups":
                if network.structure is not None:
                    unique = sorted(list(network.structure))

                    if restrict_nodes is None:
                        values = network.structure.get_group(list(range(n)))
                    else:
                        values = network.structure.get_group(restrict_nodes)
                else:
                    raise ValueError("Requested coloring by group but the "
                                     "graph has no groups.")
            else:
                values = network.get_node_attributes(
                    name=ncolor, nodes=restrict_nodes)

                unique = sorted(list(set(values)))

            c = np.linspace(0, 1, len(unique))

            cnvrt = {v: i for i, v in enumerate(unique)}

            color = np.array([c[cnvrt[v]] for v in values])

            nlabel       = "Neuron groups"
            nticks       = list(range(len(unique)))
            ntickslabels = [s.replace("_", " ") for s in unique]
        else:
            values = None

            if "degree" in ncolor:
                dtype   = ncolor[:ncolor.find("-")]
                values = network.get_degrees(dtype, nodes=restrict_nodes)
            elif ncolor == "betweenness":
                if restrict_nodes is None:
                    values = network.get_betweenness("node")
                else:
                    values = network.get_betweenness(
                        "node")[restrict_nodes]
            elif ncolor in network.node_attributes:
                values = network.get_node_attributes(
                    name=ncolor, nodes=restrict_nodes)
            elif ncolor == "clustering":
                values = nngt.analysis.local_clustering(
                    network, nodes=restrict_nodes)
            elif ncolor in nngt.analyze_graph:
                if restrict_nodes is None:
                    values = nngt.analyze_graph[ncolor](network)
                else:
                    values = nngt.analyze_graph[ncolor](
                        network)[restrict_nodes]
            else:
                raise RuntimeError("Invalid `ncolor`: {}.".format(ncolor))

            if values is not None:
                vmin, vmax = np.min(values), np.max(values)
                color = values

                nlabel = "Node " + ncolor.replace("_", " ")
                setval = set(values)
                if len(setval) <= 10:
                    nticks = list(setval)
                    nticks.sort()
                    ntickslabels = nticks
                else:
                    nticks       = np.linspace(vmin, vmax, 10)
                    ntickslabels = nticks
    else:
        nlabel  = "Custom node colors"
        uniques = np.unique(ncolor, axis=0)
        if len(uniques) <= 10:
            nticks = uniques
        else:
            nticks = np.linspace(np.min(ncolor), np.max(ncolor), 10)
        ntickslabels = nticks

    return color, nticks, ntickslabels, nlabel


def _edge_prop(network, value):
    prop = value

    enum = network.edge_nb()

    if isinstance(value, str) and value not in ColorConverter.colors:
        if value in network.edge_attributes:
            color = network.edge_attributes[value]
        elif value == "betweenness":
            prop = network.get_betweenness("edge")
        else:
            raise RuntimeError("Invalid `value`: {}.".format(value))

    return prop


def _discrete_cmap(N, base_cmap=None, discrete=False):
    '''
    Create an N-bin discrete colormap from the specified input map

    Parameters
    ----------
    N : number of values
    base_cmap : str, None, or cmap object
    clist : list of colors

    # Modified from Jake VanderPlas
    # License: BSD-style
    '''
    import matplotlib.pyplot as plt
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap, N)

    color_list = base(np.arange(N))
    cmap_name = base.name + str(N)
    try:
        return base.from_list(cmap_name, color_list, N)
    except:
        return ListedColormap(color_list, cmap_name, N=N)


def _convert_to_nodes(node_restriction, name, network):
    if nonstring_container(node_restriction):
        if isinstance(node_restriction[0], str):
            assert network.structure is not None, \
                "`" + name + "` can be string only for Network or graph " \
                "with a `structure`."
            ids = set()
            for name in node_restriction:
                ids.update(network.structure[name].ids)
            return ids
        elif isinstance(node_restriction[0], nngt.Group):
            ids = set()
            for g in node_restriction:
                ids.update(g.ids)
            return ids

        return set(node_restriction) 
    elif isinstance(node_restriction, str):
        assert network.is_network(), \
            "`" + name + "` can be string only for Network."
        return set(network.structure[node_restriction].ids)
    elif isinstance(node_restriction, nngt.Group):
        return set(node_restriction.ids)
    elif node_restriction is not None:
        raise ValueError(
            "Invalid `" + name + "`: '{}'".format(node_restriction))

    return node_restriction


def _custom_arrows(sources, targets, angle):
    r'''
    Create a curved arrow between `source` and `target` as the combination of
    the arc of a circle and a triangle.

    The initial and final angle $\alpha$ between the source-target line and
    the arrow is linked to the radius of the circle, $r$ and the distance $d$
    between the points:

    .. math:: r = \frac{d}{2 \cdot \tan(\alpha)}

    The beginning and the end of the arc are given through initial and final
    angles, respectively $\theta_1$ and $\theta_2$, which are given with
    respect to the y-axis; This leads to $\alpha = 0.5(\theta_1 - \theta_2)$.
    '''
    # compute the distances between the points
    pass
    #~ # compute the radius and the position of the center of the circle

    #~ #========Line
    #~ arc = Arc([centX,centY],radius,radius,angle=angle_,
          #~ theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=10,color=color_)
    #~ ax.add_patch(arc)


    #~ #========Create the arrow head
    #~ endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    #~ endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))

    #~ ax.add_patch(                    #Create triangle as arrow head
        #~ RegularPolygon(
            #~ (endX, endY),            # (x,y)
            #~ 3,                       # number of vertices
            #~ radius/9,                # radius
            #~ rad(angle_+theta2_),     # orientation
            #~ color=color_
        #~ )
    #~ )


def _to_ig_color(color):
    import igraph as ig

    if isinstance(color, str) and color not in ig.known_colors:
        color = str(ColorConverter.to_rgb(color))[1:-1]
    elif nonstring_container(color) and len(color):
        # need to convert floating point colors to [0, 255] integers
        if is_integer(color[0]) or isinstance(color[0], float):
            vmin = np.min(color)
            vmax = np.max(color)
            vint = vmax - vmin
            if vint > 0:
                color = [int(255 * (v - vmin) / vint) for v in color]
            else:
                color = [0]*len(color)
        else:
            for i, c in enumerate(color):
                if isinstance(color, str) and color not in ig.known_colors:
                    color[i] = str(ColorConverter.to_rgb(color))[1:-1]

    return color


def _increase_nx_size(size, factor=4):
    
    if isinstance(size, float) or is_integer(size):
        return factor*size
    elif nonstring_container(size) and len(size):
        if isinstance(size[0], float) or is_integer(size[0]):
            return factor*np.asarray(size)

    return size


def _to_gt_prop(graph, value, cmap, ptype='node', color=False):
    pmap = (graph.new_vertex_property if ptype == 'node'
            else graph.new_edge_property)
    
    if nonstring_container(value) and len(value):
        if isinstance(value[0], str):
            if color:
                # custom namedcolors
                return pmap("vector<double>",
                            vals=[ColorConverter.to_rgba(v) for v in value])
            else:
                return pmap("string", vals=value)
        elif nonstring_container(value[0]):
            # direct rgb(a) description
            return pmap("vector<double>", vals=value)

        # numbers
        if color:
            vmin, vmax = np.min(value), np.max(value)

            normalized = None

            if vmax - vmin > 0:
                normalized = (np.array(value) - vmin) / (vmax - vmin)
            else:
                return normalized

            return pmap("vector<double>", vals=[cmap(v) for v in normalized])

        return pmap("double", vals=value)


    return value


def _circular_layout(graph, max_nsize):
    # chose radius such that r*dtheta > max_nsize
    dtheta = 2*np.pi / graph.node_nb()

    r = 1.1*max_nsize / dtheta

    thetas = np.array([i*dtheta for i in range(graph.node_nb())])
    x = r*np.cos(thetas)
    y = r*np.sin(thetas)

    return np.array((x, y)).T


def _connectionstyle(axis, nsize, esize):
    def cs(posA, posB, *args, **kwargs):
        # Self-loops are scaled by node size
        vshift = 0.1*max(nsize, 2*esize)
        hshift = 0.7*vshift
        # this is called with _screen space_ values so covert back
        # to data space

        s1 = np.asarray([-hshift, vshift])
        s2 = np.asarray([hshift, vshift])

        p1 = axis.transData.inverted().transform(posA)
        p2 = axis.transData.inverted().transform(posA + s1)
        p3 = axis.transData.inverted().transform(posA + s2)

        path = [p1, p2, p3, p1]

        return mpl.path.Path(axis.transData.transform(path), [1, 2, 2, 2])

    return cs

def _split_edges_sizes(edges, esize, decimate_connections, ecolor=None,
                       strght_colors=None, loop_colors=None):
    strght_edges, self_loops = None, None
    strght_sizes, loop_sizes = None, None

    keep  = (esize > 0) if nonstring_container(esize) else True
    loops = (edges[:, 0] == edges[:, 1])

    strght = keep*(~loops)

    strght_edges = edges[strght]

    self_loops = set(edges[loops, 0])

    if ecolor is not None:
        if nonstring_container(ecolor):
            if decimate_connections < 1:
                strght_colors.extend(ecolor[strght])
            loop_colors.extend(ecolor[loops])
        else:
            if decimate_connections < 1:
                strght_colors.extend([ecolor]*len(strght_edges))
            loop_colors.extend([ecolor]*len(self_loops))

    if nonstring_container(esize):
        strght_sizes = esize[strght]
        loop_sizes = esize[loops]
    else:
        strght_sizes = np.full(len(strght_edges), esize)
        loop_sizes = np.full(len(self_loops), esize)

    if decimate_connections > 1:
        strght_edges = \
            strght_edges[::decimate_connections]

        if nonstring_container(esize):
            strght_sizes = \
                strght_sizes[::decimate_connections]

        if ecolor is not None:
            if nonstring_container(ecolor):
                strght_colors.extend(ecolor[strght][::decimate_connections])
            else:
                strght_colors.extend(
                    [ecolor] * (len(strght_edges) // decimate_connections))
    elif ecolor is not None:
        if nonstring_container(ecolor):
            strght_colors.extend(ecolor[strght])
        else:
            strght_colors.extend([ecolor]*len(strght_edges))

    return strght_edges, self_loops, strght_sizes, loop_sizes


def _get_ncmap(network, ncolor):
    ''' Return whether a discrete palette is used and the default cmap '''
    discrete_colors = False

    if isinstance(ncolor, str):
        if ncolor == "group" or ncolor == "groups":
            discrete_colors = True
        elif ncolor in network.node_attributes:
            discrete_colors = \
                network.get_attribute_type(ncolor, "node") == "string"

    default_ncmap = palette_discrete() if discrete_colors \
                    else palette_continuous()

    return discrete_colors, default_ncmap


def _plot_loop(i, s, pos, loop_sizes, nsize, max_nsize, xmax, xmin, ymax, ymin,
               height, width, ec, ealpha, eborder_width, eborder_color, fast,
               network, restrict_nodes):
    '''
    Draw self loops
    '''
    es = loop_sizes[i]
    dl = 0.03*max(height, width)
    ns = nsize[s]*dl/max_nsize if fast else nsize[s]

    # get the neighbours
    nn = network.neighbours(s)

    if restrict_nodes is not None:
        nn = nn.intersection(restrict_nodes)

        convert = {n: i for i, n in enumerate(restrict_nodes)}

        nn = {convert[n] for n in nn}

    nn = list(nn - {s})

    vec = pos[nn] - pos[s]
    norm = np.sqrt((vec*vec).sum(axis=1))
    vec = np.asarray([vec[i] / n for i, n in enumerate(norm)])

    dir = np.average(vec, axis=0)
    dir /= np.linalg.norm(dir)

    if fast:
        xy = pos[s] - ns*dir
        return Circle(xy, ns, fc="none", alpha=ealpha, linewidth=0.5*es, ec=ec)

    es = min(0.5*ns, es)
    xy = pos[s] - 0.75*ns*dir

    return Annulus(xy, 0.75*ns, 0.5*es, fc=ec, alpha=ealpha, lw=eborder_width,
                   ec=eborder_color)


class Annulus(Patch):
    """
    An elliptical annulus.
    """

    def __init__(self, xy, r, width, angle=0.0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            xy coordinates of annulus centre.
        r : float or (float, float)
            The radius, or semi-axes:
            - If float: radius of the outer circle.
            - If two floats: semi-major and -minor axes of outer ellipse.
        width : float
            Width (thickness) of the annular ring. The width is measured inward
            from the outer ellipse so that for the inner ellipse the semi-axes
            are given by ``r - width``. *width* must be less than or equal to
            the semi-minor axis.
        angle : float, default: 0
            Rotation angle in degrees (anti-clockwise from the positive
            x-axis). Ignored for circular annuli (i.e., if *r* is a scalar).
        **kwargs
            Keyword arguments control the `Patch` properties:
            %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)

        self.set_radii(r)
        self.center = xy
        self.width = width
        self.angle = angle
        self._path = None

    def __str__(self):
        if self.a == self.b:
            r = self.a
        else:
            r = (self.a, self.b)

        return "Annulus(xy=(%s, %s), r=%s, width=%s, angle=%s)" % \
                (*self.center, r, self.width, self.angle)

    def set_center(self, xy):
        """
        Set the center of the annulus.
        Parameters
        ----------
        xy : (float, float)
        """
        self._center = xy
        self._path = None
        self.stale = True

    def get_center(self):
        """Return the center of the annulus."""
        return self._center

    center = property(get_center, set_center)

    def set_width(self, width):
        """
        Set the width (thickness) of the annulus ring.
        The width is measured inwards from the outer ellipse.
        Parameters
        ----------
        width : float
        """
        if min(self.a, self.b) <= width:
            raise ValueError(
                'Width of annulus must be less than or equal semi-minor axis')

        self._width = width
        self._path = None
        self.stale = True

    def get_width(self):
        """Return the width (thickness) of the annulus ring."""
        return self._width

    width = property(get_width, set_width)

    def set_angle(self, angle):
        """
        Set the tilt angle of the annulus.
        Parameters
        ----------
        angle : float
        """
        self._angle = angle
        self._path = None
        self.stale = True

    def get_angle(self):
        """Return the angle of the annulus."""
        return self._angle

    angle = property(get_angle, set_angle)

    def set_semimajor(self, a):
        """
        Set the semi-major axis *a* of the annulus.
        Parameters
        ----------
        a : float
        """
        self.a = float(a)
        self._path = None
        self.stale = True

    def set_semiminor(self, b):
        """
        Set the semi-minor axis *b* of the annulus.
        Parameters
        ----------
        b : float
        """
        self.b = float(b)
        self._path = None
        self.stale = True

    def set_radii(self, r):
        """
        Set the semi-major (*a*) and semi-minor radii (*b*) of the annulus.
        Parameters
        ----------
        r : float or (float, float)
            The radius, or semi-axes:
            - If float: radius of the outer circle.
            - If two floats: semi-major and -minor axes of outer ellipse.
        """
        if np.shape(r) == (2,):
            self.a, self.b = r
        elif np.shape(r) == ():
            self.a = self.b = float(r)
        else:
            raise ValueError("Parameter 'r' must be one or two floats.")

        self._path = None
        self.stale = True

    def get_radii(self):
        """Return the semi-major and semi-minor radii of the annulus."""
        return self.a, self.b

    radii = property(get_radii, set_radii)

    def _transform_verts(self, verts, a, b):
        return Affine2D() \
            .scale(*self._convert_xy_units((a, b))) \
            .rotate_deg(self.angle) \
            .translate(*self._convert_xy_units(self.center)) \
            .transform(verts)

    def _recompute_path(self):
        # circular arc
        arc = Path.arc(0, 360)

        # annulus needs to draw an outer ring
        # followed by a reversed and scaled inner ring
        a, b, w = self.a, self.b, self.width
        v1 = self._transform_verts(arc.vertices, a, b)
        v2 = self._transform_verts(arc.vertices[::-1], a - w, b - w)
        v = np.vstack([v1, v2, v1[0, :], (0, 0)])
        c = np.hstack([arc.codes, Path.MOVETO,
                       arc.codes[1:], Path.MOVETO,
                       Path.CLOSEPOLY])
        self._path = Path(v, c)

    def get_path(self):
        if self._path is None:
            self._recompute_path()
        return self._path


class GraphArtist(Artist):
    """
    Matplotlib artist class that draws igraph graphs.

    Only Cairo-based backends are supported.

    Adapted from: https://stackoverflow.com/a/36154077/5962321
    """

    def __init__(self, graph, axis, palette=None, *args, **kwds):
        """Constructs a graph artist that draws the given graph within
        the given bounding box.

        `graph` must be an instance of `igraph.Graph`.
        `bbox` must either be an instance of `igraph.drawing.BoundingBox`
        or a 4-tuple (`left`, `top`, `width`, `height`). The tuple
        will be passed on to the constructor of `BoundingBox`.
        `palette` is an igraph palette that is used to transform
        numeric color IDs to RGB values. If `None`, a default grayscale
        palette is used from igraph.

        All the remaining positional and keyword arguments are passed
        on intact to `igraph.Graph.__plot__`.
        """
        from igraph import BoundingBox, palettes

        super().__init__()

        self.graph = graph
        self.palette = palette or palettes["gray"]
        self.bbox = BoundingBox(axis.bbox.bounds)
        self.args = args
        self.kwds = kwds

    def draw(self, renderer):
        from matplotlib.backends.backend_cairo import RendererCairo

        if not isinstance(renderer, RendererCairo):
            raise TypeError(
                "graph plotting is supported only on Cairo backends")

        self.graph.__plot__(renderer.gc.ctx, self.bbox, self.palette,
                            *self.args, **self.kwds)
