#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

import warnings
import numpy as np
import scipy.sparse as ssp
from numpy.random import randint

import nngt
from nngt.globals import config
from nngt.lib.errors import InvalidArgument


__all__ = [
    "_compute_connections",
    "_distance_rule",
    "_erdos_renyi",
    "_filter",
    "_fixed_degree",
    "_gaussian_degree",
    "_newman_watts",
    "_no_self_loops",
    "_price_scale_free",
    "_random_scale_free",
    "_set_options",
    "_unique_rows",
    "price_network",
]

MAXTESTS = 1000 # ensure that generation will finish
EPS = 0.00001


#-----------------------------------------------------------------------------#
# Simple tools
#------------------------
#

def _set_options(graph, weighted, population, shape, positions):
    if weighted:
        graph.set_weights()
    if issubclass(graph.__class__, nngt.Network):
        Connections.delays(graph)
    elif population is not None:
        nngt.Network.make_network(graph, population)
    if shape is not None:
        nngt.SpatialGraph.make_spatial(graph, shape, positions)

def _compute_connections(num_source, num_target, density, edges, avg_deg,
                         directed, reciprocity):
    pre_recip_edges = 0
    if avg_deg > 0:
        pre_recip_edges = int(avg_deg * num_source)
    elif edges > 0:
        pre_recip_edges = int(edges)
    else:
        pre_recip_edges = int(density * num_source * num_target)
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

def _unique_rows(arr):
    b = np.ascontiguousarray(arr).view(np.dtype((np.void,
        arr.dtype.itemsize * arr.shape[1])))
    return np.unique(b).view(arr.dtype).reshape(-1,arr.shape[1]).astype(int)

def _no_self_loops(array):
    return array[array[:,0] != array[:,1],:].astype(int)

def _filter(ia_edges, ia_edges_tmp, num_ecurrent, b_one_pop, multigraph):
    '''
    Filter the edges: remove self loops and multiple connections if the graph
    is not a multigraph.
    '''
    if b_one_pop:
        ia_edges_tmp = _no_self_loops(ia_edges_tmp)
    num_added = ia_edges_tmp.shape[0]
    ia_edges[num_ecurrent:num_ecurrent+num_added,:] = ia_edges_tmp
    num_ecurrent += num_added
    if not multigraph:
        ia_edges_tmp = _unique_rows(ia_edges[:num_ecurrent,:])
        num_ecurrent = ia_edges_tmp.shape[0]
        ia_edges[:num_ecurrent,:] = ia_edges_tmp
    return ia_edges, num_ecurrent


#-----------------------------------------------------------------------------#
# Graph model generation
#------------------------
#

def _fixed_degree(source_ids, target_ids, degree, degree_type, reciprocity,
                  directed, multigraph, existing_edges=None):
    np.random.seed()
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
    num_source, num_target = len(source_ids), len(target_ids)
    # type of degree
    b_out = (degree_type == "out")
    b_total = (degree_type == "total")
    # edges
    edges = num_target*degree if degree_type == "out" else num_source*degree
    b_one_pop = (False if num_source != num_target else
                           not np.all(source_ids-target_ids))
    if not b_one_pop and not multigraph:
        b_d = (edges > num_source*num_target)
        b_nd = (edges > int(0.5*num_source*num_target))
        if (not directed and b_nd) or (directed and b_d):
            raise InvalidArgument("Required degree is too high")
    elif b_one_pop and not multigraph:
        b_d = (edges > num_source*(num_target-1))
        b_nd = (edges > int((0.5*num_source-1)*num_target))
        if (not directed and b_nd) or (directed and b_d):
            raise InvalidArgument("Required degree is too high")
    
    existing = 0 if existing_edges is None else existing_edges.shape[0]
    ia_edges = np.zeros((existing+edges, 2), dtype=int)
    if existing:
        ia_edges[:existing,:] = existing_edges
    idx = 0 if b_out else 1 # differenciate source / target
    variables = source_ids if b_out else target_ids # nodes picked randomly
    
    for i,v in enumerate(target_ids):
        edges_i, ecurrent, variables_i = np.zeros((degree,2)), 0, []
        if existing_edges is not None:
            with_v = np.where(ia_edge[:,idx] == v)
            variables_i.extend(ia_edge[with_v:int(not idx)])
            ecurrent = len(variables_i)
        ia_edges[i*degree:(i+1)*degree, idx] = v
        rm = np.argwhere(variables == v)[0]
        rm = rm[0] if len(rm) else -1
        var_tmp = ( np.array(variables, copy=True) if rm == -1 else
                    np.concatenate((variables[:rm], variables[rm+1:])) )
        num_var_i = len(var_tmp)
        while ecurrent != degree:
            var = var_tmp[randint(0,num_var_i,degree-ecurrent)]
            variables_i.extend(var)
            if not multigraph:
                variables_i = list(set(variables_i))
            ecurrent = len(variables_i)
        ia_edges[i*degree:(i+1)*degree, int(not idx)] = variables_i
    if not directed:
        ia_edges = np.concatenate((ia_edges, ia_edges[:,::-1]))
        ia_edges = _unique_rows(ia_edges)
    return ia_edges


def _gaussian_degree(source_ids, target_ids, avg, std, degree_type,
                     reciprocity, directed, multigraph, existing_edges=None):
    ''' Connect nodes with a Gaussian distribution '''
    np.random.seed()
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
    num_source, num_target = len(source_ids), len(target_ids)
    # type of degree
    b_out = (degree_type == "out")
    b_total = (degree_type == "total")
    # edges
    num_node = num_source if degree_type == "in" else num_target
    lst_deg = np.around( np.maximum(
                         np.random.normal(avg, std, num_node),0.) ).astype(int)
    edges = np.sum(lst_deg)
    b_one_pop = (False if num_source != num_target else
                           not np.all(source_ids-target_ids))
    if not b_one_pop and not multigraph:
        if edges > num_source*num_target:
            raise InvalidArgument("Required degree is too high")
    elif b_one_pop and not multigraph:
        if edges > num_source*(num_target-1):
            raise InvalidArgument("Required degree is too high")
    
    num_etotal = 0 if existing_edges is None else existing_edges.shape[0]
    ia_edges = np.zeros((num_etotal+edges, 2), dtype=int)
    if num_etotal:
        ia_edges[:num_etotal,:] = existing_edges
    idx = 0 if b_out else 1 # differenciate source / target
    variables = source_idx if b_out else target_ids # nodes picked randomly
    
    for i,v in enumerate(target_ids):
        degree_i = lst_deg[i]
        edges_i, ecurrent, variables_i = np.zeros((degree_i,2)), 0, []
        if existing_edges is not None:
            with_v = np.where(ia_edge[:,idx] == v)
            variables_i.extend(ia_edge[with_v:int(not idx)])
            ecurrent = len(variables_i)
        rm = np.argwhere(variables == v)[0]
        rm = rm[0] if len(rm) else -1
        var_tmp = ( np.array(variables, copy=True) if rm == -1 else
                    np.concatenate((variables[:rm], variables[rm+1:])) )
        num_var_i = len(var_tmp)
        ia_edges[num_etotal:num_etotal+degree_i, idx] = v
        while len(variables_i) != degree_i:
            var = var_tmp[randint(0, num_var_i, degree_i-ecurrent)]
            variables_i.extend(var)
            if not multigraph:
                variables_i = list(set(variables_i))
            ecurrent = len(variables_i)
        ia_edges[num_etotal:num_etotal+ecurrent, int(not idx)] = variables_i
        num_etotal += ecurrent
    if not directed:
        ia_edges = np.concatenate((ia_edges, ia_edges[:,::-1]))
        ia_edges = _unique_rows(ia_edges)
    return ia_edges
    

def _random_scale_free(source_ids, target_ids, in_exp, out_exp, density,
                       edges, avg_deg, reciprocity, directed, multigraph,
                       **kwargs):
    ''' Connect the nodes with power law distributions '''
    np.random.seed()
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
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
        idx_correct_in = randint(0,num_target,np.abs(diff_in))
        idx_correct_out = randint(0,num_source,np.abs(diff_out))
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
    if not directed:
        ia_edges = np.concatenate((ia_edges, ia_edges[:,::-1]))
        ia_edges = _unique_rows(ia_edges)
    return ia_edges
    

def _erdos_renyi(source_ids, target_ids, density, edges, avg_deg, reciprocity,
                 directed, multigraph, **kwargs):
    '''
    Returns a numpy array of dimension (2,edges) that describes the edge list
    of an Erdos-Renyi graph.
    @todo: perform all the calculations here
    '''
    np.random.seed()
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
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
    if not directed:
        ia_edges = np.concatenate((ia_edges, ia_edges[:,::-1]))
        ia_edges = _unique_rows(ia_edges)
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
    return np.array([node_ids[ia_sources], node_ids[ia_targets]]).astype(int).T

def _newman_watts(source_ids, target_ids, coord_nb, proba_shortcut,
                  directed, multigraph, **kwargs):
    '''
    Returns a numpy array of dimension (num_edges,2) that describes the edge 
    list of a Newmaan-Watts graph.
    '''
    np.random.seed()
    node_ids = np.array(source_ids, dtype=int)
    target_ids = np.array(target_ids, dtype=int)
    nodes = len(node_ids)
    circular_edges = nodes*coord_nb
    num_edges = int(circular_edges*(1+proba_shortcut))
    num_edges, circular_edges = (num_edges, circular_edges if directed
                             else (int(num_edges/2), int(circular_edges/2)))
    b_one_pop = (False if len(target_ids) != nodes else
                           not np.all(node_ids-target_ids))
    if not b_one_pop:
        raise InvalidArgument("This graph model can only be used if source \
                              and target populations are the same")
    # generate the initial circular graph
    ia_edges = np.zeros((num_edges,2),dtype=int)
    ia_edges[:circular_edges,:] = _circular_graph(node_ids, coord_nb)
    # add the random connections
    num_test, num_ecurrent = 0, circular_edges
    while num_ecurrent != num_edges and num_test < MAXTESTS:
        ia_sources = node_ids[randint(0, nodes, num_edges-num_ecurrent)]
        ia_targets = node_ids[randint(0, nodes, num_edges-num_ecurrent)]
        ia_edges_tmp = np.array([ia_sources,ia_targets]).T
        ia_edges, num_ecurrent = _filter(ia_edges, ia_edges_tmp, num_ecurrent,
                                         b_one_pop, multigraph)
        num_test += 1
    ia_edges = _no_self_loops(ia_edges)
    if not directed:
        ia_edges = np.concatenate((ia_edges, ia_edges[:,::-1]))
        ia_edges = _unique_rows(ia_edges)
    return ia_edges


def _distance_rule(source_ids, target_ids, density, edges, avg_deg, scale,
                   rule, shape, positions, directed, multigraph, **kwargs):
    '''
    Returns a distance-rule graph
    '''
    np.random.seed()
    def exp_rule(pos_src, pos_target):
        dist = np.linalg.norm(pos_src-pos_target,axis=0)
        return np.exp(np.divide(dist,-scale))
    def lin_rule(pos_src, pos_target):
        dist = np.linalg.norm(pos_src-pos_target,axis=0)
        return np.divide(scale-dist,scale).clip(min=0.)
    dist_test = exp_rule if rule == "exp" else lin_rule
    # compute the required values
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
    num_source, num_target = len(source_ids), len(target_ids)
    edges, _ = _compute_connections(num_source, num_target,
                             density, edges, avg_deg, directed, reciprocity=-1)
    b_one_pop = (False if num_source != num_target else
                           not np.all(source_ids-target_ids))
    # create the edges
    ia_edges = np.zeros((edges,2), dtype=int)
    num_test, num_ecurrent = 0, 0
    while num_ecurrent != edges and num_test < MAXTESTS:
        num_create = edges-num_ecurrent
        ia_sources = source_ids[randint(0, num_source, num_create)]
        ia_targets = target_ids[randint(0, num_target, num_create)]
        test = dist_test(positions[:,ia_sources],positions[:,ia_targets])
        ia_valid = np.greater(test,np.random.uniform(size=num_create))
        ia_edges_tmp = np.array([ia_sources[ia_valid],ia_targets[ia_valid]]).T
        ia_edges, num_ecurrent = _filter(ia_edges, ia_edges_tmp, num_ecurrent,
                                         b_one_pop, multigraph)
        num_test += 1
    if num_test == MAXTESTS:
        ia_edges = ia_edges[:num_ecurrent,:]
        warnings.warn("Maximum number of tests reached, stopped  generation \
with {} edges.".format(num_ecurrent), RuntimeWarning)
    if not directed:
        ia_edges = np.concatenate((ia_edges, ia_edges[:,::-1]))
        ia_edges = _unique_rows(ia_edges)
    return ia_edges

def price_network():
    #@todo: do it for other libraries
    pass

if config["graph_library"] == "graph-tool":
    from graph_tool.generation import price_network
    
