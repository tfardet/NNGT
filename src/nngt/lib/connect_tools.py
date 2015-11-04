#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

from ..core.graph_objects import graph_lib


__all__ = [
    "_compute_connections",
    "price_network",
    "circular_graph"
]


#
#---
# Simple tools
#------------------------

def _compute_connections(nodes, density, edges, avg_deg, nw=None):
    if nw is not None:
        return int(nodes*nw[0]*(1+nw[1]))
    elif edges > 0:
        return edges
    elif density > 0.:
        return int(density * nodes**2)
    else:
        return int(avg_deg * nodes)



#
#---
# Graph model generation
#------------------------

def price_network():
    #@todo: do it for other libraries
    pass

def circular_graph():
    pass


if graph_lib == "graph_tool":
    from graph_tool.generation import price_network, circular_graph
