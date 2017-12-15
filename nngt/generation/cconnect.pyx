#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#!/usr/bin/env cython
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

import warnings

from cython.parallel import parallel, prange
from .cconnect cimport *
cimport numpy as np

import numpy as np
import scipy.sparse as ssp
from numpy.random import randint

import nngt
from nngt.lib import InvalidArgument
from nngt.lib.connect_tools import *


__all__ = [
    "_distance_rule",
    "_erdos_renyi",
    "_filter",
    "_fixed_degree",
    "_gaussian_degree",
    "_newman_watts",
    "_no_self_loops",
    "_price_scale_free",
    "_random_scale_free",
    "_unique_rows",
]


cdef int MAXTESTS = 1000 # ensure that generation will finish
cdef float EPS = 0.00001

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.uint


#-----------------------------------------------------------------------------#
# Simple tools
#------------------------
#

cdef bytes _to_bytes(string):
    ''' Convert string to bytes '''
    if not isinstance(string, bytes):
        return bytes(string, "UTF-8")
    return string

def _unique_rows(arr):
    b = np.ascontiguousarray(arr).view(np.dtype((np.void,
        arr.dtype.itemsize * arr.shape[1])))
    return np.unique(b).view(arr.dtype).reshape(-1,arr.shape[1])

def _no_self_loops(array):
    return array[array[:,0] != array[:,1],:]

def _filter(ia_edges, ia_edges_tmp, num_ecurrent, b_one_pop,
                            multigraph):
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

def _fixed_degree(np.ndarray[size_t, ndim=1] source_ids,
                  np.ndarray[size_t, ndim=1] target_ids, size_t degree=-1,
                  degree_type="in", float reciprocity=-1, bool directed=True,
                  bool multigraph=False, existing_edges=None, **kwargs):
    ''' Generation of the edges through the C++ function '''
    cdef:
        # type of degree
        bool b_out = (degree_type == "out")
        bool b_total = (degree_type == "total")
        size_t num_source = source_ids.shape[0]
        size_t num_target = target_ids.shape[0]
        size_t edges = (num_target * degree
                        if degree_type == "out" else num_source * degree)
        bool b_one_pop = _check_num_edges(
            source_ids, target_ids, edges, directed, multigraph)
        unsigned int existing = \
            0 if existing_edges is None else existing_edges.shape[0]
        np.ndarray[size_t, ndim=2, mode="c"] ia_edges = np.zeros(
            (existing+edges, 2), dtype=DTYPE)
    if existing:
        ia_edges[:existing,:] = existing_edges
    cdef:
        unsigned int idx = 0 if b_out else 1 # differenciate source / target
        unsigned int omp = nngt._config["omp"]
        unsigned int num_node = num_target if b_out else num_source
        long msd = np.random.randint(0, edges + 1)
        vector[size_t] degrees = np.repeat(degree, num_node)
        vector[ vector[size_t] ] old_edges = vector[ vector[size_t] ]()

    if b_out:
        _gen_edges(&ia_edges[0,0], source_ids, degrees, target_ids, old_edges,
            idx, multigraph, directed, msd, omp)
    else:
        _gen_edges(&ia_edges[0,0], target_ids, degrees, source_ids, old_edges,
            idx, multigraph, directed, msd, omp)
    return ia_edges


def _gaussian_degree(np.ndarray[size_t, ndim=1] source_ids,
                     np.ndarray[size_t, ndim=1] target_ids, float avg=-1,
                     float std=-1, degree_type="in", float reciprocity=-1,
                     bool directed=True, bool multigraph=False,
                     existing_edges=None, **kwargs):
    '''
    Connect nodes with a Gaussian distribution (generation through C++
    function.
    '''
    cdef:
        # type of degree
        b_out = (degree_type == "out")
        b_total = (degree_type == "total")
        size_t num_source = source_ids.shape[0]
        size_t num_target = target_ids.shape[0]
        unsigned int idx = 0 if b_out else 1 # differenciate source / target
        unsigned int omp = nngt._config["omp"]
        unsigned int num_node = num_target if b_out else num_source
        vector[size_t] degrees = np.around(np.maximum(
            np.random.normal(avg, std, num_node), 0.)).astype(DTYPE)
        vector[ vector[size_t] ] old_edges = vector[ vector[size_t] ]()

    # edges
    edges = np.sum(degrees)
    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)

    cdef:
        long msd = np.random.randint(0, edges + 1)
        unsigned int existing = \
            0 if existing_edges is None else existing_edges.shape[0]
        np.ndarray[size_t, ndim=2, mode="c"] ia_edges = np.zeros(
            (existing + edges, 2), dtype=DTYPE)
    if existing:
        ia_edges[:existing,:] = existing_edges

    if b_out:
        _gen_edges(&ia_edges[0,0], source_ids, degrees, target_ids, old_edges,
            idx, multigraph, directed, msd, omp)
    else:
        _gen_edges(&ia_edges[0,0], target_ids, degrees, source_ids, old_edges,
            idx, multigraph, directed, msd, omp)
    return ia_edges
    

def _random_scale_free(source_ids, target_ids, in_exp=-1, out_exp=-1,
                       density=-1, edges=-1, avg_deg=-1, reciprocity=-1,
                       directed=True, multigraph=False, **kwargs):
    ''' Connect the nodes with power law distributions '''
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
    num_source, num_target = len(source_ids), len(target_ids)
    edges, pre_recip_edges = _compute_connections(num_source, num_target,
                                density, edges, avg_deg, directed, reciprocity)
    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)
    
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
    return ia_edges
    

def _erdos_renyi(source_ids, target_ids, float density=-1, int edges=-1,
                 float avg_deg=-1, float reciprocity=-1, bool directed=True,
                 bool multigraph=False, **kwargs):
    '''
    Returns a numpy array of dimension (2,edges) that describes the edge list
    of an Erdos-Renyi graph.
    @todo: perform all the calculations here
    '''
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
    num_source, num_target = len(source_ids), len(target_ids)
    edges, pre_recip_edges = _compute_connections(num_source, num_target,
                                density, edges, avg_deg, directed, reciprocity)
    
    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)
    
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


def _newman_watts(source_ids, target_ids, int coord_nb=-1,
                  float proba_shortcut=-1, directed=True, multigraph=False,
                  **kwargs):
    '''
    Returns a numpy array of dimension (num_edges,2) that describes the edge 
    list of a Newmaan-Watts graph.
    '''
    node_ids = np.array(source_ids, dtype=int)
    target_ids = np.array(target_ids, dtype=int)
    nodes = len(node_ids)
    circular_edges = nodes*coord_nb
    num_edges = int(circular_edges*(1+proba_shortcut))
    num_edges, circular_edges = (num_edges, circular_edges if directed
                             else (int(num_edges/2), int(circular_edges/2)))
    
    b_one_pop = _check_num_edges(
        source_ids, target_ids, num_edges, directed, multigraph)
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
    return ia_edges


def _distance_rule(np.ndarray[size_t, ndim=1] source_ids,
                   np.ndarray[size_t, ndim=1] target_ids,
                   float density=-1, int edges=-1, float avg_deg=-1,
                   float scale=-1, float norm=-1, str rule="exp", shape=None,
                   np.ndarray[float, ndim=2] positions=np.array([[0], [0]]),
                   bool directed=True, bool multigraph=False,
                   num_neurons=None, distance=None, **kwargs):
    '''
    Returns a distance-rule graph
    '''
    distance = [] if distance is None else distance
    if num_neurons is None:
        num_neurons = len(set(np.concatenate((source_ids, target_ids))))
    cdef:
        size_t cnum_neurons = num_neurons
        size_t num_source = source_ids.shape[0]
        size_t num_target = target_ids.shape[0]
        string crule = _to_bytes(rule)
        unsigned int omp = nngt._config["omp"]
        vector[ vector[size_t] ] old_edges = vector[ vector[size_t] ]()
        vector[ vector[size_t] ] targets
        vector[float] x = positions[0]
        vector[float] y = positions[1]
        float cscale = scale
        float cnorm = norm
    # compute the required values
    edge_num, _ = _compute_connections(
        num_source, num_target, density, edges, avg_deg, directed)
    existing = 0  # for now
    #~ b_one_pop = _check_num_edges(
        #~ source_ids, target_ids, edges, directed, multigraph)
    # for each node, check the neighbours that are in an area where
    # connections can be made: +/- scale for lin, +/- 10*scale for exp
    lim = scale if rule == 'lin' else 10*scale
    for i in source_ids:
        keep  = (np.abs(positions[0, target_ids] - positions[0, i]) < lim)
        keep *= (np.abs(positions[1, target_ids] - positions[1, i]) < lim)
        targets.push_back(target_ids[keep].tolist())
    # create the edges
    cdef:
        long msd = np.random.randint(0, edge_num + 1)
        #~ float area = shape.area * conversion_factor
        size_t cedges = edge_num
        np.ndarray[size_t, ndim=2, mode="c"] ia_edges = np.zeros(
            (existing + edge_num, 2), dtype=DTYPE)
        vector[float] dist = vector[float]()

    _cdistance_rule(&ia_edges[0,0], source_ids, targets, crule, cscale, cnorm,
                    x, y, cnum_neurons, cedges, old_edges, dist, multigraph,
                    msd, omp)
    distance.extend(dist)
    assert np.all(np.greater(distance, 0.)), "Negative distance detected."
    return ia_edges


def price_network():
    #@todo: do it for other libraries
    pass
