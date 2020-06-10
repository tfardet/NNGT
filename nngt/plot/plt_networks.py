#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
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

from itertools import cycle
from collections import defaultdict

import numpy as np
from matplotlib.artist import Artist
from matplotlib.patches import FancyArrowPatch, ArrowStyle, FancyArrow, Circle
from matplotlib.patches import Arc, RegularPolygon, PathPatch
from matplotlib.cm import get_cmap
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, Normalize, ColorConverter
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import nngt
from nngt.lib import POS, nonstring_container, is_integer
from .custom_plt import palette_continuous, palette_discrete, format_exponent



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

__all__ = ["draw_network", "library_draw"]


# ------- #
# Drawing #
# ------- #

def draw_network(network, nsize="total-degree", ncolor="group", nshape="o",
                 nborder_color="k", nborder_width=0.5, esize=1., ecolor="k",
                 ealpha=0.5, max_nsize=None, max_esize=2., curved_edges=False,
                 threshold=0.5, decimate_connections=None, spatial=True,
                 restrict_sources=None, restrict_targets=None,
                 restrict_nodes=None, show_environment=True, fast=False,
                 size=(600, 600), xlims=None, ylims=None, dpi=75, axis=None,
                 colorbar=False, layout=None, show=False, **kwargs):
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
    ncolor : float, array of floats or string, optional (default: 0.5)
        Color of the nodes; if a float in [0, 1], position of the color in the
        current palette, otherwise a string that correlates the color to a node
        attribute among "in/out/total-degree", "betweenness" or "group".
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
    max_esize : float, optional (default: 5.)
        If a custom property is entered as `esize`, this normalizes the edge
        width between 0. and `max_esize`.
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
    show_environment : bool, optional (default: True)
        Plot the environment if the graph is spatial.
    fast : bool, optional (default: False)
        Use a faster algorithm to plot the edges. This method leads to less
        pretty plots and zooming on the graph will make the edges start or
        ending in places that will differ more or less strongly from the actual
        node positions.
    size : tuple of ints, optional (default: (600,600))
        (width, height) tuple for the canvas size (in px).
    dpi : int, optional (default: 75)
        Resolution (dot per inch).
    axis : matplotlib axis, optional (default: create new axis)
        Axis on which the network will be plotted.
    colorbar : bool, optional (default: False)
        Whether to display a colorbar for the node colors or not.
    layout : str, optional (default: random or spatial positions)
        Name of a standard layout to structure the network. Available layouts
        are: "circular" or "random". If no layout is provided and the network
        is spatial, then node positions will be used by default.
    show : bool, optional (default: True)
        Display the plot immediately.
    **kwargs : dict
        Optional keyword arguments including `node_cmap` to set the
        nodes colormap (default is "magma" for continuous variables and
        "Set1" for groups) and the boolean `simple_nodes` to make node
        plotting faster.
    '''
    import matplotlib.pyplot as plt

    # figure and axes
    size_inches = (size[0]/float(dpi), size[1]/float(dpi))

    if axis is None:
        fig = plt.figure(facecolor='white', figsize=size_inches,
                         dpi=dpi)
        axis = fig.add_subplot(111, frameon=0, aspect=1)

    axis.set_axis_off()

    pos = None

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
             for node in range(n)],
            dtype=bool)
        adj_mat[remove] = 0

    if restrict_targets is not None:
        remove = np.array(
            [1 if node not in restrict_targets else 0
             for node in range(n)],
            dtype=bool)
        adj_mat[:, remove] = 0

    e = len(adj_mat.nonzero()[0])  # avoid calling `eliminate_zeros`

    # compute properties
    decimate_connections = 1 if decimate_connections is None\
                           else decimate_connections

    # get node and edge shape/size properties
    simple_nodes = kwargs.get("simple_nodes", False)

    max_nsize = 20 if simple_nodes else 5

    markers, nsize, esize = _node_edge_shape_size(
        network, nshape, nsize, max_nsize, esize, max_esize, restrict_nodes,
        size, threshold, simple_nodes=simple_nodes)

    # node color information
    default_ncmap = (palette_discrete() if ncolor == "group"
                     else palette_continuous())

    ncmap = get_cmap(kwargs.get("node_cmap", default_ncmap))
    node_color, nticks, ntickslabels, nlabel = \
        _node_color(network, restrict_nodes, ncolor)

    if nonstring_container(ncolor):
        assert len(ncolor) == n, "For color arrays, one " +\
            "color per node is required."
        ncolor = "custom"

    c = node_color

    if not nonstring_container(nborder_color):
        nborder_color = np.repeat(nborder_color, n)

    # check edge color
    group_based = False

    default_ecmap = (palette_discrete() if ecolor == "group"
                     else palette_continuous())

    if isinstance(ecolor, float):
        ecolor = np.repeat(ecolor, e)
    elif ecolor == "groups" or ecolor == "group":
        if not network.is_network():
            raise TypeError(
                "The graph must be a Network to use `ecolor='groups'`.")

        group_based = True
        ecolor      = {}

        for i, src in enumerate(network.population):
            if network.population[src].ids:
                idx1 = network.population[src].ids[0]
                for j, tgt in enumerate(network.population):
                    if network.population[tgt].ids:
                        idx2 = network.population[tgt].ids[0]
                        if src == tgt:
                            ecolor[(src, tgt)] = node_color[idx1]
                        else:
                            ecolor[(src, tgt)] = \
                                np.abs(0.8*node_color[idx1]
                                       - 0.2*node_color[idx2])

    # draw
    pos = np.zeros((n, 2))

    if layout == "circular":
        pos = _circular_layout(network, nsize)
    elif layout is None and spatial and network.is_spatial():
        if show_environment:
            nngt.geometry.plot.plot_shape(network.shape, axis=axis,
                                          show=False)

        neurons = None if restrict_nodes is None else list(restrict_nodes)

        pos = network.get_positions(neurons=neurons)
    else:
        pos[:, 0] = size[0]*(np.random.uniform(size=n)-0.5)
        pos[:, 1] = size[1]*(np.random.uniform(size=n)-0.5)

    # make nodes
    nodes = []

    if nonstring_container(c) and not isinstance(c[0], str):
        # make the colorbar for the nodes
        cmap = ncmap
        if colorbar:
            clist = np.unique(c, axis=0) if ncolor == "group" else None
            cnorm = None
            if ncolor.startswith("group"):
                cmap  = _discrete_cmap(len(nticks), ncmap, clist=clist)
                cnorm = Normalize(nticks[0]-0.5, nticks[-1] + 0.5)
            else:
                cnorm = Normalize(np.min(c), np.max(c))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
            c  = cnorm(c)
            if ncolor.startswith("group"):
                sm.set_array(nticks)
            else:
                sm.set_array(c)
            plt.subplots_adjust(right=0.95)
            divider = make_axes_locatable(axis)
            cax     = divider.append_axes("right", size="5%", pad=0.05)
            if ncolor.startswith("group"):
                cb = plt.colorbar(sm, ticks=nticks, cax=cax, shrink=0.8)
                cb.set_ticklabels(ntickslabels)
                if nlabel:
                    cb.set_label(nlabel)
            else:
                cb = plt.colorbar(sm, cax=cax, shrink=0.8)
        else:
            cmin, cmax = np.min(c), np.max(c)
            if cmin != cmax:
                c = (c - cmin)/(cmax - cmin)
        c = cmap(c)
    else:
        if not nonstring_container(c) and not isinstance(c, str):
            minc = np.min(node_color)

            c = np.array(
                [ncmap((node_color - minc)/(np.max(node_color) - minc))]*n)

    # plot nodes
    if simple_nodes:
        if nonstring_container(nshape):
            # matplotlib scatter does not support marker arrays
            if isinstance(nshape[0], nngt.NeuralGroup):
                for g in nshape:
                    ids = g.ids if restrict_nodes is None \
                          else list(set(g.ids).intersection(restrict_nodes))
                    axis.scatter(pos[ids, 0], pos[ids, 1], c=c[ids],
                                 s=0.5*np.array(nsize)[ids],
                                 marker=markers[ids[0]])
            else:
                ids = range(network.node_nb()) if restrict_nodes is None \
                      else restrict_nodes
                for i in ids:
                    axis.plot(pos[i, 0], pos[i, 1], c=c[i], ms=0.5*nsize[i],
                              marker=markers[ids[0]], ls="")
        else:
            axis.scatter(pos[:, 0], pos[:, 1], c=c, s=0.5*np.array(nsize),
                         marker=nshape)
    else:
        axis.set_aspect(1.)

        if network.is_network():
            for group in network.population.values():
                idx = group.ids if restrict_nodes is None \
                      else list(set(restrict_nodes).intersection(group.ids))
                for i, fc in zip(idx, c[idx]):
                    m = MarkerStyle(markers[i]).get_path()
                    transform = Affine2D().scale(
                        0.5*nsize[i]).translate(pos[i][0], pos[i][1])
                    patch = PathPatch(m.transformed(transform), facecolor=fc,
                                      edgecolor=nborder_color[i])
                    nodes.append(patch)
        else:
            for i, ci in enumerate(c):
                m = MarkerStyle(markers[i]).get_path()
                transform = Affine2D().scale(0.5*nsize[i]).translate(
                    pos[i][0], pos[i][1])
                patch = PathPatch(m.transformed(transform), facecolor=ci,
                                  edgecolor=nborder_color[i])
                nodes.append(patch)

        nodes = PatchCollection(nodes, match_original=True)
        nodes.set_zorder(2)
        axis.add_collection(nodes)

    if not show_environment or not spatial or not network.is_spatial():
        # axis.get_data()
        _set_ax_lim(axis, pos[:, 0], pos[:, 1], xlims, ylims)

    # use quiver to draw the edges
    if e and decimate_connections != -1:
        avg_size = np.average(nsize)
        arr_style = ArrowStyle.Simple(head_length=0.15*avg_size,
                                      head_width=0.1*avg_size,
                                      tail_width=0.05*avg_size)
        arrows = []
        if group_based:
            for src_name, src_group in network.population.items():
                for tgt_name, tgt_group in network.population.items():
                    s_ids        = src_group.ids
                    if restrict_sources is not None:
                        s_ids = list(set(restrict_sources).intersection(s_ids))
                    t_ids        = tgt_group.ids
                    if restrict_targets is not None:
                        t_ids = list(set(restrict_targets).intersection(t_ids))
                    if t_ids and s_ids:
                        s_min, s_max = np.min(s_ids), np.max(s_ids) + 1
                        t_min, t_max = np.min(t_ids), np.max(t_ids) + 1
                        edges        = np.array(
                            adj_mat[s_min:s_max, t_min:t_max].nonzero(),
                            dtype=int)
                        edges[0, :] += s_min
                        edges[1, :] += t_min
                        if nonstring_container(esize):
                            keep = (esize > 0)
                            edges = edges[:, keep]
                            esize = esize[keep]
                        if decimate_connections > 1:
                            edges = edges[:, ::decimate_connections]
                            if nonstring_container(esize):
                                esize = esize[::decimate_connections]
                        # plot
                        ec = default_ecmap(ecolor[(src_name, tgt_name)])
                        if fast:
                            dl       = 0.5*np.max(nsize)
                            arrow_x  = pos[edges[1], 0] - pos[edges[0], 0]
                            arrow_x -= np.sign(arrow_x) * dl
                            arrow_y  = pos[edges[1], 1] - pos[edges[0], 1]
                            arrow_x -= np.sign(arrow_y) * dl
                            axis.quiver(
                                pos[edges[0], 0], pos[edges[0], 1], arrow_x,
                                arrow_y, scale_units='xy', angles='xy',
                                scale=1, alpha=0.5, width=1.5e-3,
                                linewidths=0.5*esize, edgecolors=ec, zorder=1)
                        else:
                            for s, t in zip(edges[0], edges[1]):
                                xs, ys = pos[s, 0], pos[s, 1]
                                xt, yt = pos[t, 0], pos[t, 1]
                                dl     = 0.5*nsize[t]
                                dx     = xt-xs
                                dx -= np.sign(dx) * dl
                                dy     = yt-ys
                                dy -= np.sign(dy) * dl

                                if curved_edges:
                                    arrow = FancyArrowPatch(
                                        posA=(xs, ys), posB=(xt, yt),
                                        arrowstyle=arr_style,
                                        connectionstyle='arc3,rad=0.1',
                                        alpha=ealpha, fc=ec, lw=0.5)
                                    axis.add_patch(arrow)
                                else:
                                    arrows.append(FancyArrow(
                                        xs, ys, dx, dy, width=0.3*avg_size,
                                        head_length=0.7*avg_size,
                                        head_width=0.7*avg_size,
                                        length_includes_head=True,
                                        alpha=ealpha, fc=ec, lw=0.5))
        else:
            edges = []

            if e and decimate_connections != -1:
                # ~ edges = network.edges_array
                sources = (None if restrict_sources is None
                           else list(restrict_sources))

                targets = (None if restrict_targets is None
                           else list(restrict_targets))

                edges = network.get_edges(source_node=sources,
                                          target_node=targets)

                # keep only large edges
                if nonstring_container(esize):
                    keep = (esize > 0)
                    edges  = edges[keep]
                    if nonstring_container(ecolor):
                        ecolor = ecolor[keep]
                    esize = esize[keep]

                if decimate_connections > 1:
                    edges = edges[::decimate_connections]
                    if nonstring_container(esize):
                        esize = esize[::decimate_connections]
                    if nonstring_container(ecolor):
                        ecolor = ecolor[::decimate_connections]

                # keep only desired edges
                if None not in (restrict_sources, restrict_targets):
                    new_edges = []

                    for edge in edges:
                        s, t = edge

                        if s in restrict_sources and t in restrict_targets:
                            new_edges.append(edge)

                    edges = np.array(new_edges, dtype=int)

                    if restrict_nodes is not None:
                        nodes = list(restrict_nodes)
                        nodes.sort()

                        for i, node in enumerate(nodes):
                            edges[edges == node] = i
                elif restrict_sources is not None:
                    new_edges = []

                    for edge in edges:
                        s, _ = edge

                        if s in restrict_sources:
                            new_edges.append(edge)

                    edges = np.array(new_edges, dtype=int)
                elif restrict_targets is not None:
                    new_edges = []

                    for edge in edges:
                        _, t = edge

                        if t in restrict_targets:
                            new_edges.append(edge)

                    edges = np.array(new_edges, dtype=int)

            if isinstance(ecolor, str):
                ecolor = [ecolor for i in range(0, e, decimate_connections)]

            if len(edges) and fast:
                dl = 0.5*np.max(nsize) if not simple_nodes else 0.

                arrow_x  = pos[edges[:, 1], 0] - pos[edges[:, 0], 0]
                arrow_x -= np.sign(arrow_x) * dl
                arrow_y  = pos[edges[:, 1], 1] - pos[edges[:, 0], 1]
                arrow_x -= np.sign(arrow_y) * dl
                axis.quiver(pos[edges[:, 0], 0], pos[edges[:, 0], 1], arrow_x,
                            arrow_y, scale_units='xy', angles='xy', scale=1,
                            alpha=0.5, width=1.5e-3, linewidths=0.5*esize,
                            edgecolors=ecolor, zorder=1)
            elif len(edges):
                for i, (s, t) in enumerate(edges):
                    xs, ys = pos[s, 0], pos[s, 1]
                    xt, yt = pos[t, 0], pos[t, 1]

                    if curved_edges:
                        arrow = FancyArrowPatch(
                            posA=(xs, ys), posB=(xt, yt), arrowstyle=arr_style,
                            connectionstyle='arc3,rad=0.1',
                            alpha=ealpha, fc=ecolor[i], lw=0.5)
                        axis.add_patch(arrow)
                    else:
                        dl     = 0.5*nsize[t]
                        dx     = xt-xs
                        dx -= np.sign(dx) * dl
                        dy     = yt-ys
                        dy -= np.sign(dy) * dl
                        arrows.append(FancyArrow(
                            xs, ys, dx, dy, width=0.3*avg_size,
                            head_length=0.7*avg_size, head_width=0.7*avg_size,
                            length_includes_head=True, alpha=ealpha,
                            fc=ecolor[i], lw=0.5))

        if not fast:
            arrows = PatchCollection(arrows, match_original=True)
            arrows.set_zorder(1)
            axis.add_collection(arrows)

    if kwargs.get('tight', True):
        plt.tight_layout()
        plt.subplots_adjust(
            hspace=0., wspace=0., left=0., right=0.95 if colorbar else 1.,
            top=1., bottom=0.)

    if show:
        plt.show()


def library_draw(network, nsize="total-degree", ncolor="group", nshape="o",
                 nborder_color="k", nborder_width=0.5, esize=1., ecolor="k",
                 ealpha=0.5, max_nsize=5., max_esize=2., curved_edges=False,
                 threshold=0.5, decimate_connections=None, spatial=True,
                 restrict_sources=None, restrict_targets=None,
                 restrict_nodes=None, show_environment=True, size=(600, 600),
                 xlims=None, ylims=None, dpi=75, axis=None, colorbar=False,
                 show_labels=False, layout=None, show=False, **kwargs):
    '''
    Draw a given :class:`~nngt.Graph` using the underlying library's drawing
    functions.

    .. versionadded:: 2.0

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
        attribute among "in/out/total-degree", "betweenness" or "group".
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
    max_esize : float, optional (default: 5.)
        If a custom property is entered as `esize`, this normalizes the edge
        width between 0. and `max_esize`.
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
    show_environment : bool, optional (default: True)
        Plot the environment if the graph is spatial.
    fast : bool, optional (default: False)
        Use a faster algorithm to plot the edges. This method leads to less
        pretty plots and zooming on the graph will make the edges start or
        ending in places that will differ more or less strongly from the actual
        node positions.
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
        Optional keyword arguments including `node_cmap` to set the
        nodes colormap (default is "magma" for continuous variables and
        "Set1" for groups) and the boolean `simple_nodes` to make node
        plotting faster.
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # backend and axis
    if nngt.get_config("backend") in ("graph-tool", "igraph"):
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
            esize=esize, ecolor=ecolor, ealpha=ealpha, max_nsize=max_nsize,
            max_esize=max_esize, curved_edges=curved_edges,
            threshold=threshold, decimate_connections=decimate_connections,
            spatial=spatial, restrict_nodes=restrict_nodes,
            show_environment=show_environment, size=size, axis=axis,
            layout=layout, show=show, **kwargs)

    # otherwise, preapre data
    restrict_nodes = _convert_to_nodes(restrict_nodes,
                                       "restrict_nodes", network)

    # shize and shape
    markers, nsize, esize = _node_edge_shape_size(
        network, nshape, nsize, max_nsize, esize, max_esize, restrict_nodes,
        size, threshold)

    # node color information
    default_ncmap = (palette_discrete() if ncolor == "group"
                     else palette_continuous())

    ncmap = get_cmap(kwargs.get("node_cmap", default_ncmap))

    node_color, nticks, ntickslabels, nlabel = \
        _node_color(network, restrict_nodes, ncolor)

    # edge color
    ecolor = _edge_prop(network, ecolor)
    esize  = _edge_prop(network, esize)

    if nonstring_container(esize):
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
            nsize /= np.max(nsize)

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

        eprops = {
            "color": _to_gt_prop(graph, ecolor, palette_continuous(),
                                 ptype='edge', color=True),
            "pen_width": _to_gt_prop(graph, esize, None, ptype='edge'),
        }

        graph_draw(network.graph, pos=pos, vprops=vprops, eprops=eprops,
                   output_size=size, mplfig=axis)
    elif nngt.get_config("backend") == "networkx":
        import networkx as nx

        pos = None

        if layout is None:
            if isinstance(network, nngt.SpatialGraph) and spatial:
                xy  = network.get_positions()
                pos = {i: coords for coords in xy}
        elif layout == "circular":
            pos = nx.circular_layout(network.graph)
        elif layout == "random":
            pos = nx.random_layout(network.graph)
        else:
            pos = nx.spring_layout(network.graph)

        # normalize sizes compared to igraph
        nsize = _increase_nx_size(nsize)

        nborder_width = _increase_nx_size(nborder_width, 2)

        nx.draw_networkx(
            network.graph, pos=pos, ax=axis, nodelist=restrict_nodes,
            node_size=nsize, node_color=node_color, node_shape=nshape,
            linewidths=nborder_width, edge_color=ecolor,
            edge_cmap=palette_continuous(), cmap=ncmap,
            with_labels=show_labels, width=esize, edgecolors=nborder_color)
    elif nngt.get_config("backend") == "igraph":
        from igraph import Layout, PrecalculatedPalette

        pos = None

        if layout is None:
            if isinstance(network, nngt.SpatialGraph) and spatial:
                xy  = network.get_positions()
                pos = Layout(xy)
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

        visual_style = {
            "vertex_size": nsize,
            "vertex_color": node_color,
            "vertex_shape": shape_dict[nshape],
            "edge_width": esize,
            "edge_color": ecolor,
            "layout": pos,
            "palette": palette,
        }

        graph_artist = GraphArtist(network.graph, axis, **visual_style)

        axis.artists.append(graph_artist)

    if show:
        plt.show()


# ----- #
# Tools #
# ----- #

def _node_edge_shape_size(network, nshape, nsize, max_nsize, esize, max_esize,
                          restrict_nodes, size, threshold, simple_nodes=False):
    ''' Returns the shape and size of the nodes and edges '''
    n = network.node_nb() if restrict_nodes is None else len(restrict_nodes)
    e = network.edge_nb()

    # markers
    markers = nshape

    if nonstring_container(nshape):
        if isinstance(nshape[0], nngt.NeuralGroup):
            # check disjunction
            for i, g in enumerate(nshape):
                for j in range(i + 1, len(nshape)):
                    if not set(g.ids).isdisjoint(nshape[j].ids):
                        raise ValueError("Groups passed to `nshape` "
                                         "must be disjoint.")

            mm = cycle(MarkerStyle.filled_markers)

            shapes  = np.full(network.node_nb(), "", dtype=object)

            for g, m in zip(nshape, mm):
                shapes[g.ids] = m

            markers = list(shapes)
        elif len(nshape) != network.node_nb():
            raise ValueError("When passing an array of markers to "
                             "`nshape`, one entry per node in the "
                             "network must be provided.")
    else:
        markers = [nshape for _ in range(network.node_nb())]

    # size
    if isinstance(nsize, str):
        if e:
            nsize = _node_size(network, restrict_nodes, nsize)
            nsize *= max_nsize / np.max(nsize)
        else:
            nsize = np.ones(n, dtype=float)
    elif isinstance(nsize, (float, int, np.number)):
        nsize = np.full(n, nsize, dtype=float)
    elif nonstring_container(nsize):
        nsize *= max_nsize / np.max(nsize)

    nsize *= 0.01 * size[0]

    if isinstance(esize, str) and e:
        esize  = _edge_size(network, restrict_nodes, esize)
        esize *= max_esize
        esize[esize < threshold] = 0.

    esize *= 0.005 * size[0]  # border on each side (so 0.5 %)

    return markers, nsize, esize


def _set_ax_lim(ax, xdata, ydata, xlims, ylims):
    if xlims is not None:
        ax.set_xlim(*xlims)
    else:
        x_min, x_max = np.min(xdata), np.max(xdata)
        width = x_max - x_min
        ax.set_xlim(x_min - 0.05*width, x_max + 0.05*width)
    if ylims is not None:
        ax.set_ylim(*ylims)
    else:
        y_min, y_max = np.min(ydata), np.max(ydata)
        height = y_max - y_min
        ax.set_ylim(y_min - 0.05*height, y_max + 0.05*height)


def _node_size(network, restrict_nodes, nsize):
    restrict_nodes = None if restrict_nodes is None else list(restrict_nodes)

    n = network.node_nb() if restrict_nodes is None else len(restrict_nodes)

    size = np.ones(n, dtype=float)

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


def _edge_size(network, restrict_nodes, esize):
    edges, num_edges = None, None

    restrict_nodes = None if restrict_nodes is None else list(restrict_nodes)

    if restrict_nodes is None:
        num_edges = network.edge_nb()
    else:
        edges = network.get_edges(source_node=restrict_nodes,
                                  target_node=restrict_nodes)
        num_edges = len(edges)

    size = np.repeat(1., num_edges)

    if num_edges:
        max_size = None

        if esize == "betweenness":
            betw = network.get_betweenness("edge")

            max_size = np.max(betw)

            size = betw if restrict_nodes is None else betw[restrict_nodes]

        if esize == "weight":
            size = network.get_weights(edges=edges)

            max_size = np.max(network.get_weights())

        if np.any(size):
            size /= max_size

    return size


def _node_color(network, restrict_nodes, ncolor):
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
        restrict_nodes = set(restrict_nodes)

    if isinstance(ncolor, float):
        color = np.repeat(ncolor, n)
    elif isinstance(ncolor, str):
        if ncolor == "group" or ncolor == "groups":
            color = np.zeros(n)
            if hasattr(network, "population"):
                l = len(network.population)
                c = np.linspace(0, 1, l)
                tmp = 0
                for i, group in enumerate(network.population.values()):
                    if restrict_nodes is None:
                        color[group.ids] = c[i]
                    else:
                        ids = restrict_nodes.intersection(group.ids)
                        for j in range(len(ids)):
                            color[tmp + j] = c[i]
                        tmp += len(ids)

                nlabel       = "Neuron groups"
                nticks       = list(range(len(network.population)))
                ntickslabels = [s.replace("_", " ")
                                for s in network.population.keys()]
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
                        "node")[list(restrict_nodes)]
            elif ncolor in network.node_attributes:
                values = network.get_node_attributes(
                    name=ncolor, nodes=restrict_nodes)
            elif ncolor == "clustering" :
                values = nngt.analysis.local_clustering(
                    network, nodes=restrict_nodes)
            elif ncolor in nngt.analyze_graph:
                if restrict_nodes is None:
                    values = nngt.analyze_graph[ncolor](network)
                else:
                    values = nngt.analyze_graph[ncolor](
                        network)[list(restrict_nodes)]
            elif ncolor in ColorConverter.colors:
                color = np.repeat(ncolor, n)
            else:
                raise RuntimeError("Invalid `ncolor`: {}.".format(ncolor))

            if values is not None:
                vmin, vmax = np.min(values), np.max(values)
                #~ color = (values - vmin) / (vmax - vmin)
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


def _discrete_cmap(N, base_cmap=None, clist=None):
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
    color_list = base(np.linspace(0, 1, N)) if clist is None else clist
    cmap_name = base.name + str(N)
    try:
        return base.from_list(cmap_name, color_list, N)
    except:
        return ListedColormap(color_list, cmap_name, N=N)


def _convert_to_nodes(node_restriction, name, network):
    if nonstring_container(node_restriction):
        if isinstance(node_restriction[0], str):
            assert network.is_network(), \
                "`" + name + "` can be string only for Network."
            ids = set()
            for name in node_restriction:
                ids.update(network.population[name].ids)
            return ids
        elif isinstance(node_restriction[0], nngt.NeuralGroup):
            ids = set()
            for g in node_restriction:
                ids.update(g.ids)
            return ids

        return set(node_restriction) 
    elif isinstance(node_restriction, str):
        assert network.is_network(), \
            "`" + name + "` can be string only for Network."
        return set(network.population[node_restriction].ids)
    elif isinstance(node_restriction, nngt.NeuralGroup):
        return set(node_restriction.ids)
    elif node_restriction is not None:
        raise ValueError(
            "Invalid `" + name + "`: '{}'".format(node_restriction))

    return node_restriction


def _custom_arrows(sources, targets, angle):
    '''
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


def _circular_layout(graph, node_size):
    max_nsize = np.max(node_size)

    # chose radius such that r*dtheta > max_nsize
    dtheta = 2*np.pi / graph.node_nb()

    r = 1.1*max_nsize / dtheta

    thetas = np.array([i*dtheta for i in range(graph.node_nb())])
    x = r*np.cos(thetas)
    y = r*np.sin(thetas)

    return np.array((x, y)).T
