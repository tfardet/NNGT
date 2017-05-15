#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Tools to plot networks """

from graph_tool.centrality import betweenness
import graph_tool.draw as gplot
 
import gtk

import matplotlib.pyplot as plt
import numpy as np

from .custom_plt import palette, format_exponent
from ..globals import POS 



'''
Plotting objectives
===================

Do it myself > simply implement the spring-block minimization.
Plot using matplotlib.

If edges have varying size, plot only those that are visible (size > min)

'''

__all__ = [ "draw_network" ]


#-----------------------------------------------------------------------------#
# Drawing
#------------------------
#

def draw_network(network, nsize="total-degree", ncolor="group", nshape="o",
                 nborder_color="k", nborder_width=0.5, esize=1., ecolor="k",
                 spatial=True, size=(600,600), dpi=75):
    '''
    Draw a given graph/network.

    Parameters
    ----------
    network : :class:`~nngt.Graph` or subclass
        The graph/network to plot.
    nsize : float, array of float or string, optional (default: "total-degree")
        Size of the nodes; if a number, percentage of the canvas length,
        otherwize a string that correlates the size to a node attribute among
        "in/out/total-degree", "betweenness".
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
    esize : float or array of floats, optional (default: 0.5)
        Width of the edges in percent of canvas size.
    ecolor : char, float or array, optional (default: "k")
        Edge color.
    spatial : bool, optional (default: True)
        If True, use the neurons' positions to draw them.
    size : tuple of ints, optional (default: (600,600))
        (width, height) tuple for the canvas size (in px).
    dpi : int, optional (default: 75)
        Resolution (dot per inch).
    '''
    size_inches = (size[0]/float(dpi), size[1]/float(dpi))
    fig = plt.figure(facecolor='white', figsize=size_inches, dpi=dpi)
    ax = fig.add_subplot(111, frameon=0, aspect=1)
    ax.set_axis_off()
    pos, layout = None, None
    n = network.node_nb()
    e = network.edge_nb()
    # compute properties
    if isinstance(nsize, str):
        if e:
            nsize = _node_size(network, nsize)
            nsize *= 0.05*size[0]
    elif isinstance(nsize, float):
        nsize = np.repeat(nsize, n)
    if isinstance(esize, str):
        if e:
            esize = _edge_size(network, esize)
            esize *= 0.01*size[0]
    elif isinstance(esize, float):
        esize = np.repeat(esize, e)
        esize = network.new_edge_property("double",esize)
    ncolor = _node_color(network, ncolor)        
    if isinstance(nborder_color, float):
        nborder_color = np.repeat(nborder_color, n)
    if isinstance(ecolor, float):
        ecolor = np.repeat(ecolor, e)
    # draw
    pos = np.zeros((n, 2))
    if spatial and network.is_spatial():
        pos = network.position
    else:
        pos[:,0] = size[0]*(np.random.uniform(size=n)-0.5)
        pos[:,1] = size[1]*(np.random.uniform(size=n)-0.5)
    if hasattr(network, "population"):
        for group in network.population.itervalues():
            idx = group.id_list
            ax.scatter(pos[idx,0], pos[idx,1], s=nsize, marker=nshape,
                    c=palette(ncolor[idx[0]]), edgecolors=nborder_color)
    else:
        ax.scatter(pos[:,0], pos[:,1], s=nsize, marker=nshape,
                   c=palette(ncolor), edgecolors=nborder_color)
    _set_ax_lim(ax, pos[:,0], pos[:,1])
    # use quiver to draw the edges
    adj_mat = network.adjacency_matrix()
    edges = adj_mat.nonzero()
    arrow_x = pos[edges[1], 0] - pos[edges[0], 0]
    arrow_y = pos[edges[1], 1] - pos[edges[0], 1]
    ax.quiver(pos[edges[0], 0], pos[edges[0], 1], arrow_x, arrow_y,
              scale_units='xy', angles='xy', scale=1, alpha=0.5)
    plt.tight_layout()
    plt.subplots_adjust(
        hspace=0., wspace=0., left=0., right=1., top=1., bottom=0.)
    plt.show()


#-----------------------------------------------------------------------------#
# Tools
#------------------------
#

def _set_ax_lim(ax, xdata, ydata):
    x_min, x_max = np.min(xdata), np.max(xdata)
    y_min, y_max = np.min(ydata), np.max(ydata)
    if x_min > 0:
        ax.set_xlim(left=0.95*x_min)
    else:
        ax.set_xlim(left=1.05*x_min)
    if y_min > 0:
        ax.set_ylim(bottom=0.95*y_min)
    else:
        ax.set_ylim(bottom=1.05*y_min)
    if x_max < 0:
        ax.set_xlim(right=0.95*x_max)
    else:
        ax.set_xlim(right=1.05*x_max)
    if y_max < 0:
        ax.set_ylim(top=0.95*y_max)
    else:
        ax.set_ylim(top=1.05*y_max)

def _node_size(network, nsize):
    size = np.ones(network.node_nb())
    if "degree" in nsize:
        deg_type = nsize[:nsize.index("-")]
        size = network.get_degrees(deg_type)
        if size.max() > 15*size.min():
            size = np.power(size, 0.4)
    if nsize == "betweenness":
        size = network.betweenness_list("node")
        if size.max() > 15*size.min():
            min_size = size[size!=0].min()
            size[size == 0.] = min_size
            size = np.log(size)
            if size.min()<0:
                size -= 1.1*size.min()
    size /= size.max()/1.5
    return size

def _edge_size(network, esize):
    size = network.new_edge_property("double",1.)
    if esize == "betweenness":
        size = network.betweenness_list("edge")
        size /= size.max()
    return size

def _node_color(network, ncolor):
    color = None
    if issubclass(float, ncolor.__class__):
        color = np.repeat(ncolor, n)
    elif ncolor == "group":
        color = np.zeros(network.node_nb())
        if hasattr(network, "population"):
            l = len(network.population)
            c = np.linspace(0,1,l)
            for i,group in enumerate(network.population.values()):
                color[group.id_list] = c[i]
    return color
