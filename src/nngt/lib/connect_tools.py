#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

import warnings
import numpy as np

from ..core.graph_objects import graph_lib


__all__ = [
    "_compute_connections",
    "_erdos_renyi",
    "_random_scale_free",
    "_price_scale_free",
    "_newman_watts",
    "price_network"
]

MAXTESTS = 1000 # ensure that generation will finish
EPS = 0.00001


#
#---
# Simple tools
#------------------------

def _compute_connections(num_source, num_target, density, edges, avg_deg,
                         directed, reciprocity):
    pre_recip_edges = 0
    if edges > 0:
        pre_recip_edges = int(edges)
    elif density > 0.:
        pre_recip_edges = int(density * num_source * num_target)
    else:
        pre_recip_edges = int(avg_deg * num_source)
    dens = pre_recip_edges / float(num_source * num_target)
    edges = pre_recip_edges
    if not directed:
        pre_recip_edges = edges = int(edges/2)
    elif reciprocity > max(0,(2.-1./dens)):
        frac_recip = ((reciprocity - 1. + np.sqrt(1.+dens*(reciprocity-2.))) /
                      (2. - reciprocity))
        if frac_recip < 1.:
            pre_recip_edges = int(edges/(1+frac_recip))
        else:
            warnings.warn("Such reciprocity cannot attained, request ignored.")
    elif reciprocity > 0.:
        warnings.warn("Reciprocity cannot be lower than 2-1/density.")
    return edges, pre_recip_edges

def _unique_rows(array):
    b = np.ascontiguousarray(array).view(np.dtype((np.void,
        array.dtype.itemsize * array.shape[1])))
    return np.unique(b).view(array.dtype).reshape(-1, array.shape[1])

def _no_self_loops(array):
    return array[array[:,0] != array[:,1],:]


#
#---
# Graph model generation
#------------------------

def _erdos_renyi(source_ids, target_ids, density, edges, avg_deg, reciprocity,
                 directed, multigraph):
    '''
    Returns a numpy array of dimension (2,edges) that describes the edge list
    of an Erdos-Renyi graph.
    @todo: perform all the calculations here
    '''

    np.random.seed()
    source_ids, target_ids = np.array(source_ids), np.array(target_ids)
    num_source, num_target = len(source_ids), len(target_ids)
    edges, pre_recip_edges = _compute_connections(num_source, num_target,
                                density, edges, avg_deg, directed, reciprocity)
    b_one_pop = (False if num_source != num_target else
                           not np.all(source_ids-target_ids))
    
    ia_edges = np.zeros((edges,2))
    num_test, num_ecurrent = 0, 0 # number of tests and current number of edges
    
    while num_ecurrent != pre_recip_edges and num_test < MAXTESTS:
        ia_sources = source_ids[np.random.randint(0, num_source,
                                                pre_recip_edges-num_ecurrent)]
        ia_targets = target_ids[np.random.randint(0, num_target,
                                                pre_recip_edges-num_ecurrent)]
        ia_edges_tmp = np.array([ia_sources,ia_targets]).T
        if b_one_pop:
            ia_edges_tmp = _no_self_loops(ia_edges_tmp)
        num_added = ia_edges_tmp.shape[0]
        ia_edges[num_ecurrent:num_ecurrent+num_added,:] = ia_edges_tmp
        num_ecurrent += num_added
        if not multigraph:
            ia_edges_tmp = _unique_rows(ia_edges[:num_ecurrent+num_added,:])
            num_ecurrent = ia_edges_tmp.shape[0]
            ia_edges[:num_ecurrent,:] = ia_edges_tmp
        num_test += 1
    
    if directed and reciprocity > 0:
        while num_ecurrent != edges and num_test < MAXTESTS:
            ia_indices = np.random.randint(0, pre_recip_edges,
                                           edges-num_ecurrent)
            ia_edges[num_ecurrent:,:] = ia_edges[ia_indices,::-1]
            num_ecurrent = edges
            if not multigraph:
                ia_edges_tmp = _unique_rows(ia_edges)
                num_ecurrent = ia_edges_tmp.shape[0]
                ia_edges[:num_ecurrent,:] = ia_edges_tmp
            num_test += 1
    return ia_edges.astype(int)

def _random_scale_free():
    pass

def _price_scale_free():
    pass

def _circular_graph(node_ids, coord_nb):
    '''
    Connect every node `i` to its `coord_nb` nearest neighbours on a circle
    '''
    nodes = len(node_ids)
    ia_sources, ia_targets = np.zeros(nodes*coord_nb), np.zeros(nodes*coord_nb)
    ia_sources = np.repeat(np.arange(0,nodes).astype(int),coord_nb)
    dist = coord_nb/2.
    neg_dist = -int(np.floor(dist))
    pos_dist = 1-neg_dist if dist-np.floor(dist) < EPS else 2-neg_dist
    ia_base = np.concatenate((np.arange(neg_dist,0),np.arange(1,pos_dist)))
    ia_targets = np.tile(ia_base, nodes)+ia_sources
    ia_targets[ia_targets<-0.5] += nodes
    ia_targets[ia_targets>nodes-0.5] -= nodes
    return np.array([node_ids[ia_sources], node_ids[ia_targets]]).T

def _newman_watts(node_ids, coord_nb, proba_shortcut, density, edges, avg_deg,
                  reciprocity, directed, multigraph):
    '''
    Returns a numpy array of dimension (2,edges) that describes the edge list
    of a Newmaan-Watts graph.
    '''
    np.random.seed()
    node_ids = np.array(node_ids)
    nodes = len(node_ids)
    circular_edges = nodes*coord_nb
    edges = int(circular_edges*(1+proba_shortcut))
    edges, circular_edges = (edges, circular_edges if directed
                             else (int(edges/2), int(circular_edges/2)))
    # generate the initial circular graph
    ia_edges = np.zeros((edges,2))
    ia_edges[:circular_edges,:] = _circular_graph(node_ids, coord_nb)
    # add the random connections
    num_test, num_ecurrent = 0, circular_edges
    while num_ecurrent != edges and num_test < MAXTESTS:
        ia_sources = node_ids[np.random.randint(0, nodes, edges-num_ecurrent)]
        ia_targets = node_ids[np.random.randint(0, nodes, edges-num_ecurrent)]
        ia_edges_tmp = _no_self_loops(np.array([ia_sources,ia_targets]).T)
        num_added = ia_edges_tmp.shape[0]
        ia_edges[num_ecurrent:num_ecurrent+num_added,:] = ia_edges_tmp
        num_ecurrent += num_added
        if not multigraph:
            ia_edges_tmp = _unique_rows(ia_edges[:num_ecurrent+num_added,:])
            num_ecurrent = ia_edges_tmp.shape[0]
            ia_edges[:num_ecurrent,:] = ia_edges_tmp
        num_test += 1
    return ia_edges

def price_network():
    #@todo: do it for other libraries
    pass

if graph_lib == "graph_tool":
    from graph_tool.generation import price_network
