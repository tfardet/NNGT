#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

import numpy as np

from ..core.graph_objects import graph_lib


__all__ = [
    "_compute_connections",
    "_erdos_renyi",
    "_newman_watts",
    "price_network"
]

MAXTESTS = 1000 # ensure that generation will finish
EPS = 0.00001


#
#---
# Simple tools
#------------------------

def _compute_connections(nodes, density, edges, avg_deg):
    if edges > 0:
        return edges
    elif density > 0.:
        return int(density * nodes**2)
    else:
        return int(avg_deg * nodes)

def _directed_recip_edges(edges, directed, reciprocity):
    frac_recip = 0.
    pre_recip_edges = edges
    if not directed:
        edges = int(edges/2)
    elif reciprocity > 0.:
        frac_recip = reciprocity/(2.0-reciprocity)
        pre_recip_edges = edges / (1+frac_recip)
    return edges, pre_recip_edges

def _unique_rows(array):
    c = array
    b = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
    return np.unique(b).view(c.dtype).reshape(-1, c.shape[1])

def _no_self_loops(array):
    return array[array[:,0] != array[:,1],:]


#
#---
# Graph model generation
#------------------------

def _erdos_renyi(nodes, density, edges, avg_deg, reciprocity,
                 directed, multigraph):
    '''
    Returns a numpy array of dimension (2,edges) that describes the edge list
    of an Erdos-Renyi graph.
    @todo: perform all the calculations here
    '''

    np.random.seed()
    edges = _compute_connections(nodes, density, edges, avg_deg)
    edges, pre_recip_edges = _directed_recip_edges(edges, directed,
                                                   reciprocity)
    
    ia_edges = np.zeros((edges,2))
    num_test,num_ecurrent = 0,0 # number of tests and current number of edges
    
    while num_ecurrent != pre_recip_edges and num_test < MAXTESTS:
        ia_edges_tmp = np.random.randint(0, nodes,
                            (pre_recip_edges-num_ecurrent,2))
        ia_edges_tmp = _no_self_loops(ia_edges_tmp)
        num_added = ia_edges_tmp.shape[0]
        ia_edges[num_ecurrent:num_ecurrent+num_added,:] = ia_edges_tmp
        if not multigraph:
            ia_edges_tmp = _unique_rows(ia_edges[:num_ecurrent+num_added,:])
            num_ecurrent = ia_edges_tmp.shape[0]
            ia_edges[:num_ecurrent,:] = ia_edges_tmp
        num_test += 1
        
    if directed and reciprocity > 0:
        while num_ecurrent != edges and num_test < MAXTESTS:
            ia_indices = np.random.randint(0, num_ecurrent, edges-num_ecurrent)
            ia_edges[num_ecurrent:,:] = ia_edges[ia_indices,:]
            if not multigraph:
                ia_edges_tmp = _unique_rows(ia_edges)
                num_ecurrent = ia_edges_tmp.shape[0]
                ia_edges[num_ecurrent,:] = ia_edges_tmp
            num_test += 1
    return ia_edges.astype(int)

def _circular_graph(nodes, coord_nb):
    '''
    Connect every node `i` to its `coord_nb` nearest neighbours on a circle
    '''
    ia_edges = np.zeros((nodes*coord_nb,2))
    ia_edges[:,0] = np.repeat(np.arange(0,nodes).astype(int),coord_nb)
    dist = coord_nb/2.
    neg_dist = -int(np.floor(dist))
    pos_dist = 1-neg_dist if dist-np.floor(dist) < EPS else 2-neg_dist
    ia_base = np.concatenate((np.arange(neg_dist,0),np.arange(1,pos_dist)))
    ia_edges[:,1] = np.tile(ia_base, nodes)+ia_edges[:,0]
    ia_edges[ia_edges[:,1]<-0.5,1] += nodes
    ia_edges[ia_edges[:,1]>nodes-0.5,1] -= nodes
    return ia_edges

def _newman_watts(coord_nb, proba_shortcut, nodes, density, edges, avg_deg,
                  reciprocity, directed, multigraph):
    '''
    Returns a numpy array of dimension (2,edges) that describes the edge list
    of a Newmaan-Watts graph.
    '''
    np.random.seed()
    circular_edges = nodes*coord_nb
    edges = int(circular_edges*(1+proba_shortcut))
    edges, circular_edges = (edges, circular_edges if directed
                             else (int(edges/2), int(circular_edges/2)))
    # generate the initial circular graph
    ia_edges = np.zeros((edges,2))
    ia_edges[:circular_edges,:] = _circular_graph(nodes, coord_nb)
    # add the random connections
    num_test, num_ecurrent = 0, circular_edges
    while num_ecurrent != edges and num_test < MAXTESTS:
        ia_edges_tmp = np.random.randint(0,nodes, (edges-num_ecurrent,2))
        ia_edges_tmp = _no_self_loops(ia_edges_tmp)
        num_added = ia_edges_tmp.shape[0]
        ia_edges[num_ecurrent:num_ecurrent+num_added,:] = ia_edges_tmp
        if not multigraph:
            ia_edges_tmp = _unique_rows(ia_edges[:num_ecurrent+num_added,:])
            num_ecurrent = ia_edges_tmp.shape[0]
            ia_edges[:num_ecurrent,:] = ia_edges_tmp
        num_test += 1
    return ia_edges

def price_network():
    #@todo: do it for other libraries
    passt

if graph_lib == "graph_tool":
    from graph_tool.generation import price_network
