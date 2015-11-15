#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

import warnings
import numpy as np
from numpy.random import randint

from ..core.graph_objects import graph_lib
from ..lib.errors import InvalidArgument


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

def _filter(ia_edges, ia_edges_tmp, num_ecurrent, b_one_pop, multigraph):
    if b_one_pop:
        ia_edges_tmp = _no_self_loops(ia_edges_tmp)
    num_added = ia_edges_tmp.shape[0]
    ia_edges[num_ecurrent:num_ecurrent+num_added,:] = ia_edges_tmp
    num_ecurrent += num_added
    if not multigraph:
        ia_edges_tmp = _unique_rows(ia_edges[:num_ecurrent+num_added,:])
        num_ecurrent = ia_edges_tmp.shape[0]
        ia_edges[:num_ecurrent,:] = ia_edges_tmp
    return ia_edges, num_ecurrent


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
    
    ia_edges = np.zeros((edges,2), dtype=int)
    num_test, num_ecurrent = 0, 0 # number of tests and current number of edges
    
    while num_ecurrent != pre_recip_edges and num_test < MAXTESTS:
        ia_sources = source_ids[randint(0, num_source,
                                        pre_recip_edges-num_ecurrent)]
        ia_targets = target_ids[randint(0, num_target,
                                        pre_recip_edges-num_ecurrent)]
        ia_edges_tmp = np.array([ia_sources,ia_targets]).T
        ia_edges, num_ecurrent = _filter(ia_edges, ia_edges_tmp, num_ecurrent,
                                         b_one_pop, multigraph)
        num_test += 1
    
    if directed and reciprocity > 0:
        while num_ecurrent != edges and num_test < MAXTESTS:
            ia_indices = randint(0, pre_recip_edges,
                                           edges-num_ecurrent)
            ia_edges[num_ecurrent:,:] = ia_edges[ia_indices,::-1]
            num_ecurrent = edges
            if not multigraph:
                ia_edges_tmp = _unique_rows(ia_edges)
                num_ecurrent = ia_edges_tmp.shape[0]
                ia_edges[:num_ecurrent,:] = ia_edges_tmp
            num_test += 1
    return ia_edges

def _random_scale_free(source_ids, target_ids, in_exp, out_exp, density,
                       edges, avg_deg, reciprocity, directed, multigraph):
    ''' Connect the nodes with power law distributions '''
    np.random.seed()
    source_ids, target_ids = np.array(source_ids), np.array(target_ids)
    num_source, num_target = len(source_ids), len(target_ids)
    edges, pre_recip_edges = _compute_connections(num_source, num_target,
                                density, edges, avg_deg, directed, reciprocity)
    b_one_pop = (False if num_source != num_target else
                           not np.all(source_ids-target_ids))
    
    ia_edges = np.zeros((edges,2),dtype=int)
    num_ecurrent, num_test = 0, 0

    # lists containing the in/out-degrees for all nodes
    ia_in_deg = np.random.pareto(in_exp,num_target)+1
    ia_out_deg = np.random.pareto(out_exp,num_source)+1
    sum_in, sum_out = np.sum(ia_in_deg), np.sum(ia_out_deg)
    ia_in_deg = np.around(np.multiply(pre_recip_edges/sum_in,
                                      ia_in_deg)).astype(int)
    ia_out_deg = np.around(np.multiply(pre_recip_edges/sum_out,
                                       ia_out_deg)).astype(int)
    sum_in, sum_out = np.sum(ia_in_deg), np.sum(ia_out_deg)
    while sum_in != pre_recip_edges or sum_out != pre_recip_edges:
        diff_in = sum_in-pre_recip_edges
        diff_out = sum_out-pre_recip_edges
        idx_correct_in = randint(0,num_source,np.abs(diff_in))
        idx_correct_out = randint(0,num_target,np.abs(diff_out))
        ia_in_deg[idx_correct_in] -= 1*np.sign(diff_in)
        ia_out_deg[idx_correct_out] -= 1*np.sign(diff_out)
        sum_in, sum_out = np.sum(ia_in_deg), np.sum(ia_out_deg)
        ia_in_deg[ia_in_deg<0] = 0
        ia_out_deg[ia_out_deg<0] = 0
    # make the edges
    ia_sources = np.repeat(source_ids,ia_out_deg)
    ia_targets = np.repeat(target_ids,ia_in_deg)
    np.random.shuffle(ia_targets)
    ia_edges_tmp = np.array([ia_sources,ia_targets]).T
    ia_edges, num_ecurrent = _filter(ia_edges, ia_edges_tmp, num_ecurrent,
                                     b_one_pop, multigraph)
        
    while num_ecurrent != pre_recip_edges and num_test < MAXTESTS:
        num_desired = pre_recip_edges-num_ecurrent
        ia_sources_tmp = ia_sources[randint(0,pre_recip_edges,num_desired)]
        ia_targets_tmp = ia_targets[randint(0,pre_recip_edges,num_desired)]
        ia_edges_tmp = np.array([ia_sources_tmp,ia_targets_tmp]).T
        ia_edges, num_ecurrent = _filter(ia_edges, ia_edges_tmp, num_ecurrent,
                                         b_one_pop, multigraph)
        num_test += 1
    
    if directed and reciprocity > 0:
        while num_ecurrent != edges and num_test < MAXTESTS:
            ia_indices = randint(0, pre_recip_edges,
                                           edges-num_ecurrent)
            ia_edges[num_ecurrent:,:] = ia_edges[ia_indices,::-1]
            num_ecurrent = edges
            if not multigraph:
                ia_edges_tmp = _unique_rows(ia_edges)
                num_ecurrent = ia_edges_tmp.shape[0]
                ia_edges[:num_ecurrent,:] = ia_edges_tmp
            num_test += 1
    return ia_edges

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

def _newman_watts(source_ids, target_ids, coord_nb, proba_shortcut, density,
                  edges, avg_deg, reciprocity, directed, multigraph):
    '''
    Returns a numpy array of dimension (2,edges) that describes the edge list
    of a Newmaan-Watts graph.
    '''
    np.random.seed()
    node_ids, target_ids = np.array(source_ids), np.array(target_ids)
    nodes = len(node_ids)
    circular_edges = nodes*coord_nb
    edges = int(circular_edges*(1+proba_shortcut))
    edges, circular_edges = (edges, circular_edges if directed
                             else (int(edges/2), int(circular_edges/2)))
    b_one_pop = (False if node_ids != num_target else
                           not np.all(node_ids-target_ids))
    if not b_one_pop:
        raise InvalidArgument("This graph model can only be used if source \
                              and target populations are the same")
    # generate the initial circular graph
    ia_edges = np.zeros((edges,2))
    ia_edges[:circular_edges,:] = _circular_graph(node_ids, coord_nb)
    # add the random connections
    num_test, num_ecurrent = 0, circular_edges
    while num_ecurrent != edges and num_test < MAXTESTS:
        ia_sources = node_ids[randint(0, nodes, edges-num_ecurrent)]
        ia_targets = node_ids[randint(0, nodes, edges-num_ecurrent)]
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
