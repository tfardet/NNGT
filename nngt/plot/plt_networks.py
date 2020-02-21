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

import numpy as np
from matplotlib.patches import FancyArrowPatch, ArrowStyle, FancyArrow, Circle
from matplotlib.patches import Arc, RegularPolygon, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, Normalize, cnames, ColorConverter
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import nngt
from nngt.lib import POS, nonstring_container
from nngt.analysis import num_wcc
from .custom_plt import palette, format_exponent



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

__all__ = ["draw_network"]


# ------- #
# Drawing #
# ------- #

def draw_network(network, nsize="total-degree", ncolor="group", nshape="o",
                 nborder_color="k", nborder_width=0.5, esize=1., ecolor="k",
                 ealpha=0.5, max_nsize=5., max_esize=2., curved_edges=False,
                 threshold=0.5, decimate_connections=None, spatial=True,
                 restrict_sources=None, restrict_targets=None,
                 restrict_nodes=None, show_environment=True, fast=False,
                 size=(600, 600), xlims=None, ylims=None, dpi=75, axis=None,
                 colorbar=False, show=False, **kwargs):
    '''
    Draw a given graph/network.

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
    size : tuple of ints, optional (default: (600,600))
        (width, height) tuple for the canvas size (in px).
    dpi : int, optional (default: 75)
        Resolution (dot per inch).
    colorbar : bool, optional (default: False)
        Whether to display a colorbar for the node colors or not.
    show : bool, optional (default: True)
        Display the plot immediately.
    axis : matplotlib axis, optional (default: create new axis)
        Axis on which the network will be plotted.
    '''
    from matplotlib.cm import get_cmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    # figure and axes
    size_inches = (size[0]/float(dpi), size[1]/float(dpi))
    if axis is None:
        fig = plt.figure(facecolor='white', figsize=size_inches, dpi=dpi)
        axis = fig.add_subplot(111, frameon=0, aspect=1)
    axis.set_axis_off()
    pos, layout = None, None

    # restrict sources and targets
    if nonstring_container(restrict_sources):
        if isinstance(restrict_sources[0], str):
            assert network.is_network(), \
                "`restrict_sources` canbe string only for Network."
            sources = []
            for name in restrict_sources:
                sources.extend(network.population[name].ids)
            restrict_sources = sources
        elif isinstance(restrict_sources[0], nngt.NeuralGroup):
            sources = []
            for g in restrict_sources:
                sources.extend(g.ids)
            restrict_sources = sources
    elif isinstance(restrict_sources, str):
        assert network.is_network(), \
            "`restrict_sources` canbe string only for Network."
        restrict_sources = network.population[restrict_sources].ids
    elif isinstance(restrict_sources, nngt.NeuralGroup):
        restrict_sources = restrict_sources.ids
    elif restrict_sources is not None:
        raise ValueError(
            "Invalid restrict_sources: '{}'".format(restrict_sources))

    if nonstring_container(restrict_targets):
        if isinstance(restrict_targets[0], str):
            assert network.is_network(), \
                "`restrict_targets` canbe string only for Network."
            targets = []
            for name in restrict_targets:
                targets.extend(network.population[name].ids)
            restrict_targets = targets
        elif isinstance(restrict_targets[0], nngt.NeuralGroup):
            targets = []
            for g in restrict_targets:
                targets.extend(g.ids)
            restrict_targets = targets
    elif isinstance(restrict_targets, str):
        assert network.is_network(), \
            "`restrict_sources` canbe string only for Network."
        restrict_targets = network.population[restrict_targets].ids
    elif isinstance(restrict_targets, nngt.NeuralGroup):
        restrict_targets = restrict_targets.ids
    elif restrict_targets is not None:
        raise ValueError(
            "Invalid restrict_sources: '{}'".format(restrict_targets))

    # restrict nodes
    if nonstring_container(restrict_nodes):
        if isinstance(restrict_nodes[0], str):
            assert network.is_network(), \
                "`restrict_targets` canbe string only for Network."
            targets = []
            for name in restrict_nodes:
                targets.extend(network.population[name].ids)
            restrict_nodes = targets
        elif isinstance(restrict_nodes[0], nngt.NeuralGroup):
            nodes = []
            for g in restrict_nodes:
                nodes.extend(g.ids)
            restrict_nodes = nodes
    elif isinstance(restrict_nodes, str):
        assert network.is_network(), \
            "`restrict_sources` canbe string only for Network."
        restrict_nodes = network.population[restrict_nodes].ids
    elif isinstance(restrict_nodes, nngt.NeuralGroup):
        restrict_nodes = restrict_nodes.ids
    elif restrict_nodes is not None:
        raise ValueError(
            "Invalid restrict_sources: '{}'".format(restrict_nodes))

    if restrict_nodes is not None and restrict_sources is not None:
        restrict_sources = list(
            set(restrict_nodes).intersection(restrict_sources))
    elif restrict_nodes is not None:
        restrict_sources = restrict_nodes

    if restrict_nodes is not None and restrict_targets is not None:
        restrict_targets = list(
            set(restrict_nodes).intersection(restrict_targets))
    elif restrict_nodes is not None:
        restrict_targets = restrict_nodes

    # get nodes and edges
    n = network.node_nb()
    adj_mat = network.adjacency_matrix(weights=None)
    if restrict_sources is not None:
        adj_mat = adj_mat[restrict_sources, :]
    if restrict_targets is not None:
        adj_mat = adj_mat[:, restrict_targets]
    e = adj_mat.nnz

    # compute properties
    decimate_connections = 1 if decimate_connections is None\
                           else decimate_connections

    markers = nshape
    if nonstring_container(nshape):
        if isinstance(nshape[0], nngt.NeuralGroup):
            # check disjunction
            for i, g in enumerate(nshape):
                for j in range(i + 1, len(nshape)):
                    if not set(g.ids).isdisjoint(nshape[j].ids):
                        raise ValueError("Groups passed to `nshape` must be "
                                         "disjoint.")

            mm = cycle(MarkerStyle.filled_markers)
            shapes  = np.full(network.node_nb(), "", dtype=object)
            for g, m in zip(nshape, mm):
                shapes[g.ids] = m
            markers = list(shapes)
        elif len(nshape) != network.node_nb():
            raise ValueError("When passing an array of markers to `nshape`, "
                             "one entry per node in the network must be "
                             "provided.")
    else:
        markers = [nshape for _ in range(network.node_nb())]

    if isinstance(nsize, str):
        if e:
            nsize = _node_size(network, nsize)
            nsize *= max_nsize
        else:
            nsize = np.ones(n, dtype=float)
    elif isinstance(nsize, (float, int, np.number)):
        nsize = np.repeat(nsize, n)

    nsize *= 0.01 * size[0]

    if isinstance(esize, str) and e:
        # @todo check why this "if" is here
        # ~ if isinstance(ecolor, str):
            # ~ raise RuntimeError("Cannot use esize='{}' ".format(esize) +\
                               # ~ "and ecolor='{}'.".format(ecolor))
        esize  = _edge_size(network, esize)
        esize *= max_esize
        esize[esize < threshold] = 0.
    #~ elif isinstance(esize, float):
        #~ esize = np.repeat(esize, e)
    esize *= 0.005 * size[0]  # border on each side (so 0.5 %)

    # node color information
    ncmap = get_cmap(kwargs.get("node_cmap", palette()))
    node_color, nticks, ntickslabels, nlabel = _node_color(network, ncolor)

    if nonstring_container(ncolor):
        assert len(ncolor) == n, "For color arrays, one " +\
            "color per node is required."
        ncolor = "custom"
    c = node_color

    if not nonstring_container(nborder_color):
        nborder_color = np.repeat(nborder_color, n)

    # check edge color
    group_based = False
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
    pos = np.zeros((network.node_nb(), 2))
    if spatial and network.is_spatial():
        if show_environment:
            nngt.geometry.plot.plot_shape(network.shape, axis=axis, show=False)
        pos = network.get_positions()
    else:
        pos[:, 0] = size[0]*(np.random.uniform(size=n)-0.5)
        pos[:, 1] = size[1]*(np.random.uniform(size=n)-0.5)

    # make nodes
    nodes = []
    if nonstring_container(c):
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
        if not isinstance(c, str):
            minc = np.min(node_color)
            c    = ncmap((node_color - minc)/(np.max(node_color) - minc))
        c = np.array([c for _ in range(n)])

    # plot nodes
    if kwargs.get("simple_nodes", False):
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
            clist = c if restrict_nodes is None else c[restrict_nodes]
            xlist = pos[:, 0] if restrict_nodes is None \
                    else pos[restrict_nodes, 0]
            ylist = pos[:, 1] if restrict_nodes is None \
                    else pos[restrict_nodes, 1]

            axis.scatter(xlist, ylist, c=clist, s=0.5*np.array(nsize),
                         marker=nshape)
    else:
        if network.is_network():
            for group in network.population.values():
                idx = group.ids if restrict_nodes is None \
                      else list(set(restrict_nodes).intersection(group.ids))
                for i, fc in zip(idx, c[idx]):
                    m = MarkerStyle(markers[i]).get_path()
                    transform = Affine2D().scale(0.5*nsize[i]).translate(pos[i][0], pos[i][1])
                    patch = PathPatch(m.transformed(transform), facecolor=fc,
                                      edgecolor=nborder_color[i])
                    nodes.append(patch)
        else:
            for i, ci in enumerate(c):
                m = MarkerStyle(nshape[i]).get_path()
                transform = Affine2D().scale(0.5*nsize[i]).translate(pos[i][0], pos[i][1])
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
        adj_mat = network.adjacency_matrix(weights=None)
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
                        ec = palette(ecolor[(src_name, tgt_name)])
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
            edges = np.array(adj_mat.nonzero(), dtype=int)
            s_min, s_max, t_min, t_max = 0, n, 0, n
            if restrict_sources is not None:
                s_min = np.min(restrict_sources)
                s_max = np.min(np.max(restrict_sources) + 1, n)
            if restrict_targets is not None:
                t_min = np.min(restrict_targets)
                t_max = np.min(np.max(restrict_targets) + 1, n)
            edges = np.array(
                adj_mat[s_min:s_max, t_min:t_max].nonzero(), dtype=int)
            edges[0, :] += s_min
            edges[1, :] += t_min
            # keep only large edges
            if nonstring_container(esize):
                keep = (esize > 0)
                edges  = edges[:, keep]
                if nonstring_container(ecolor):
                    ecolor = ecolor[keep]
                esize = esize[keep]
            if decimate_connections > 1:
                edges = edges[:, ::decimate_connections]
                if nonstring_container(esize):
                    esize = esize[::decimate_connections]
                if nonstring_container(ecolor):
                    ecolor = ecolor[::decimate_connections]
            if isinstance(ecolor, str):
                ecolor = [ecolor for i in range(0, e, decimate_connections)]

            if fast:
                dl       = 0.5*np.max(nsize)
                arrow_x  = pos[edges[1], 0] - pos[edges[0], 0]
                arrow_x -= np.sign(arrow_x) * dl
                arrow_y  = pos[edges[1], 1] - pos[edges[0], 1]
                arrow_x -= np.sign(arrow_y) * dl
                axis.quiver(pos[edges[0], 0], pos[edges[0], 1], arrow_x,
                            arrow_y, scale_units='xy', angles='xy', scale=1,
                            alpha=0.5, width=1.5e-3, linewidths=0.5*esize,
                            edgecolors=ecolor, zorder=1)
            else:
                for i, (s, t) in enumerate(zip(edges[0], edges[1])):
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


#-----------------------------------------------------------------------------#
# Tools
#------------------------
#

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


def _node_size(network, nsize):
    size = np.ones(network.node_nb(), dtype=float)
    if "degree" in nsize:
        deg_type = nsize[:nsize.index("-")]
        size = network.get_degrees(deg_type).astype(float)
        if np.isclose(size.min(), 0):
            size[np.isclose(size, 0)] = 0.5
        if size.max() > 15*size.min():
            size = np.power(size, 0.4)
    elif nsize == "betweenness":
        betw = network.betweenness_list("node").astype(float)
        if num_wcc(network) == 1:
            size *= betw
            if size.max() > 15*size.min():
                min_size = size[size!=0].min()
                size[size == 0.] = min_size
                size = np.log(size)
                if size.min()<0:
                    size -= 1.1*size.min()
    elif nsize == "clustering":
        size *= nngt.analysis.local_clustering(network)
    elif nsize in nngt.analyze_graph:
        size *= nngt.analyze_graph[nsize](network)
    size /= size.max()
    return size.astype(float)


def _edge_size(network, esize):
    size = np.repeat(1., network.edge_nb())
    if esize == "betweenness":
        size = network.betweenness_list("edge")
    if esize == "weight":
        size = network.get_weights()
    size /= size.max()
    return size


def _node_color(network, ncolor):
    '''
    Return an array of colors, a set of ticks, and a label for the colorbar
    of the nodes (if necessary).
    '''
    color        = ncolor
    nticks       = None
    ntickslabels = None
    nlabel       = ""
    if isinstance(ncolor, np.float):
        color = np.repeat(ncolor, network.node_nb())
    elif isinstance(ncolor, str):
        if ncolor == "group" or ncolor == "groups":
            color = np.zeros(network.node_nb())
            if hasattr(network, "population"):
                l = len(network.population)
                c = np.linspace(0, 1, l)
                for i, group in enumerate(network.population.values()):
                    color[group.ids] = c[i]

                nlabel       = "Neuron groups"
                nticks       = list(range(len(network.population)))
                ntickslabels = [s.replace("_", " ")
                                for s in network.population.keys()]
        else:
            values = None
            if "degree" in ncolor:
                dtype   = ncolor[:ncolor.find("-")]
                values = network.get_degrees(dtype)
            elif ncolor == "betweenness":
                values = network.get_betweenness("node")
            elif ncolor in network.nodes_attributes:
                values = network.get_node_attributes(name=ncolor)
            elif ncolor == "clustering" :
                values = nngt.analysis.local_clustering(network)
            elif ncolor in nngt.analyze_graph:
                values = nngt.analyze_graph[ncolor](network)
            elif ncolor not in cnames and ncolor not in ColorConverter.colors:
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
