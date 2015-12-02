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
    nsize : float, array of floats or string, optional (default: "total-degree")
        Size of the nodes; if a number, percentage of the canvas length,
        otherwize a string that correlates the size to a node attribute among
        "in/out/total-degree", "betweenness".
    ncolor : float, array of floats or string, optional (default: 0.5)
        Color of the nodes; if a float in [0, 1], position of the color in the
        current palette, otherwise a string that correlates the color to a node
        attribute among "in/out/total-degree", "betweenness" or "group".
    nshape : char or array of chars, optional (default: "o")
        Shape of the nodes (see `Matplotlib markers <http://matplotlib.org/api/markers_api.html?highlight=marker#module-matplotlib.markers>`_).
    nborder_color : char, array of char, float or array of float, optional (default: "k")
        Color of the node's border using predefined `Matplotlib colors <http://matplotlib.org/api/colors_api.html?highlight=color#module-matplotlib.colors>`_).
        or floats in [0, 1] defining the position in the palette.
    nborder_width : float or array of floats, optional (default: 0.5)
        Width of the border in percent of canvas size.
    esize : float or array of floats, optional (default: 0.5)
        Width of the edges in percent of canvas size.
    ecolor : char, array of char, float or array of float, optional (default: "k")
        Edge color.
    spatial : bool, optional (default: True)
        If True, use the neurons' positions to draw them.
    size : tuple of ints, optional (default: (600,600))
        (width, height) tuple for the canvas size (in px).
    dpi : int, optional (default: 75)
        Resolution (dot per inch).
    '''
    pos,layout = None,None
    n = network.node_nb()
    e = network.edge_nb()
    # compute properties
    if issubclass(str,nsize.__class__):
        if e:
            nsize = _node_size(network, nsize)
            nsize.a *= 0.01*size[0]
    elif issubclass(float, nsize.__class__):
        nsize = np.repeat(nsize, n)
    if issubclass(str,esize.__class__):
        if e:
            esize = _edge_size(network, esize)
            esize.a *= 0.01*size[0]
    elif issubclass(float, esize.__class__):
        esize = np.repeat(esize, e)
        esize = network.graph.new_edge_property("double",esize)
    ncolor = _node_color(network, ncolor)        
    if issubclass(float, nborder_color.__class__):
        nborder_color = np.repeat(nborder_color, n)
    if issubclass(float, ecolor.__class__):
        ecolor = np.repeat(ecolor, e)
    # draw
    pos = np.zeros((n,2))
    if not e:
        nsize = 0.02*size[0]
        esize = 0.01*size[0]
        if spatial and network.is_spatial():
            pos = network[POS]
        else:
            pos[:,0] = size[0]*(np.random.uniform(size=n)-0.5)
            pos[:,1] = size[1]*(np.random.uniform(size=n)-0.5)
    elif spatial and network.is_spatial():
        pos = network[POS]
        pos = network.graph.new_vertex_property("vector<double>",pos)
    else:
        ebetw = network.graph.betweenness_list(as_prop=True)[1]
        pos = gplot.sfdp_layout(network.graph, eweight=ebetw)
    if not e:
        size_inches = (size[0]/float(dpi),size[1]/float(dpi))
        fig = plt.figure(facecolor='white', figsize=size_inches, dpi=dpi)
        ax = fig.add_subplot(111, frameon=0, aspect=1)
        fig.facecolor = "white"
        fig.figsize=size
        ax.set_axis_off()
        if hasattr(network, "population"):
            for group in network.population.itervalues():
                idx = group.id_list
                ax.scatter(pos[idx,0], pos[idx,1], s=nsize,
                           color=palette(ncolor[idx[0]]))
        else:
            ax.scatter(pos[:,0], pos[:,1], s=nsize)
        ax.set_xlim([-0.51*size[0],0.51*size[0]])
        ax.set_ylim([-0.51*size[1],0.51*size[1]])
        plt.show()
    elif spatial and network.is_spatial():
        gplot.graph_draw(network.graph, pos=pos, vertex_color=nborder_color,
            vertex_fill_color=ncolor, vertex_size=nsize, edge_color=ecolor,
            edge_pen_width=esize, output_size=size)
    else:
        gplot.graph_draw(network.graph, pos=pos, vertex_color=nborder_color,
            vertex_fill_color=ncolor, vertex_size=nsize, edge_color=ecolor,
            edge_pen_width=esize, output_size=size)


#-----------------------------------------------------------------------------#
# Tools
#------------------------
#

def _node_size(network, nsize):
    size = network.graph.new_vertex_property("double",1.)
    w = network.graph.edge_properties['weight'] if network.is_weighted() else None
    if "degree" in nsize:
        deg_type = nsize[:nsize.index("-")]
        size = network.graph.degree_property_map(deg_type, weight=w)
        if size.a.max() > 15*size.a.min():
            size.a = np.power(size.a,0.4)
    if nsize == "betweenness":
        size = network.graph.betweenness_list(as_prop=True)[0]
        if size.a.max() > 15*size.a.min():
            min_size = size.a[size.a!=0].min()
            size.a[size.a == 0.] = min_size
            size.a = np.log(size.a)
            if size.a.min()<0:
                size.a -= 1.1*size.a.min()
    size.a /=  size.a.max()/1.5
    return size

def _edge_size(network, esize):
    size = network.graph.new_edge_property("double",1.)
    if esize == "betweenness":
        size = network.graph.betweenness_list(as_prop=True)[1]
        size.a /= size.a.max()
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
    color = network.graph.new_vertex_property("double",color)
    return color
