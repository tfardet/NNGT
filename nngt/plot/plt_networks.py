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

import numpy as np

from .custom_plt import palette, format_exponent
from nngt.lib import POS, nonstring_container
from nngt.analysis import num_wcc



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
                 max_nsize=5., max_esize=2., threshold=0.5,
                 decimate=None, spatial=True, size=(600,600), xlims=None,
                 ylims=None, dpi=75, axis=None, show=False, **kwargs):
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
    nshape : char or array of chars, optional (default: "o")
        Shape of the nodes (see `Matplotlib markers <http://matplotlib.org/api/
        markers_api.html?highlight=marker#module-matplotlib.markers>`_).
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
    ecolor : char, float or array, optional (default: "k")
        Edge color.
    max_esize : float, optional (default: 5.)
        If a custom property is entered as `esize`, this normalizes the edge
        width between 0. and `max_esize`.
    decimate : int, optional (default: keep all connections)
        Plot only one connection every `decimate`.
    spatial : bool, optional (default: True)
        If True, use the neurons' positions to draw them.
    size : tuple of ints, optional (default: (600,600))
        (width, height) tuple for the canvas size (in px).
    dpi : int, optional (default: 75)
        Resolution (dot per inch).
    '''
    import matplotlib.pyplot as plt
    size_inches = (size[0]/float(dpi), size[1]/float(dpi))
    if axis is None:
        fig = plt.figure(facecolor='white', figsize=size_inches, dpi=dpi)
        axis = fig.add_subplot(111, frameon=0, aspect=1)
    axis.set_axis_off()
    pos, layout = None, None
    n = network.node_nb()
    e = network.edge_nb()
    # compute properties
    decimate = 1 if decimate is None else decimate
    if isinstance(nsize, str):
        if e:
            nsize = _node_size(network, nsize)
            nsize *= max_nsize
        else:
            nsize = np.ones(n, dtype=float)
    elif isinstance(nsize, float):
        nsize = np.repeat(nsize, n)
    nsize *= 0.01 * size[0]
    if isinstance(esize, str) and e:
        esize = _edge_size(network, esize)
        esize *= max_esize
        esize[esize < threshold] = 0.
    #~ elif isinstance(esize, float):
        #~ esize = np.repeat(esize, e)
    esize *= 0.005 * size[0]  # border on each side (so 0.5 %)
    ncolor = _node_color(network, ncolor)
    c = ncolor
    # remove the edges
    if isinstance(nborder_color, float):
        nborder_color = np.repeat(nborder_color, n)
    if isinstance(ecolor, float):
        ecolor = np.repeat(ecolor, e)
    # draw
    pos = np.zeros((n, 2))
    if spatial and network.is_spatial():
        pos = network.get_positions()
    else:
        pos[:,0] = size[0]*(np.random.uniform(size=n)-0.5)
        pos[:,1] = size[1]*(np.random.uniform(size=n)-0.5)
    if hasattr(network, "population"):
        for group in network.population.values():
            idx = group.ids
            if nonstring_container(ncolor):
                c = palette(ncolor[idx[0]])
            # scatter required because of different markersize
            axis.scatter(pos[idx,0], pos[idx,1], s=nsize, marker=nshape,
                         c=c, edgecolors=nborder_color, zorder=2)
    else:
        if not isinstance(c, str):
            c = palette(ncolor)
        axis.scatter(pos[:,0], pos[:,1], s=nsize, marker=nshape,
                     c=c, edgecolors=nborder_color, zorder=2)
    _set_ax_lim(axis, pos[:,0], pos[:,1], xlims, ylims)
    # use quiver to draw the edges
    if e:
        adj_mat = network.adjacency_matrix(weights=None)
        edges = np.array(adj_mat.nonzero())
        if nonstring_container(esize):
            edges = edges[:, esize > 0]
            esize = esize[esize > 0]
        if decimate > 1:
            edges = edges[:, ::decimate]
            if nonstring_container(esize):
                esize = esize[::decimate]
        arrow_x = pos[edges[1], 0] - pos[edges[0], 0]
        arrow_y = pos[edges[1], 1] - pos[edges[0], 1]
        axis.quiver(pos[edges[0], 0], pos[edges[0], 1], arrow_x, arrow_y,
                  scale_units='xy', angles='xy', scale=1, alpha=0.5,
                  width=1.5e-3, linewidths=esize, edgecolors=ecolor, zorder=1)
    if kwargs.get('tight', True):
        plt.tight_layout()
        plt.subplots_adjust(
            hspace=0., wspace=0., left=0., right=1., top=1., bottom=0.)
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
        if size.max() > 15*size.min():
            size = np.power(size, 0.4)
    if nsize == "betweenness":
        betw = network.betweenness_list("node").astype(float)
        if num_wcc(network) == 1:
            size *= betw
            if size.max() > 15*size.min():
                min_size = size[size!=0].min()
                size[size == 0.] = min_size
                size = np.log(size)
                if size.min()<0:
                    size -= 1.1*size.min()
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
    color = ncolor
    if issubclass(float, ncolor.__class__):
        color = np.repeat(ncolor, n)
    elif ncolor == "group":
        color = np.zeros(network.node_nb())
        if hasattr(network, "population"):
            l = len(network.population)
            c = np.linspace(0,1,l)
            for i,group in enumerate(network.population.values()):
                color[group.ids] = c[i]
    return color
