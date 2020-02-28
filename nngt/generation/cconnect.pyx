#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#!/usr/bin/env cython
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

import warnings

from cython.parallel import parallel, prange
from .cconnect cimport *
cimport numpy as cnp

import numpy as np
import scipy.sparse as ssp
from numpy.random import randint, get_state

import nngt
from nngt.lib import InvalidArgument
from nngt.lib.connect_tools import *


__all__ = [
    "_all_to_all",
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


# ---------------------- #
# Graph model generation #
# ---------------------- #

def _all_to_all(cnp.ndarray[size_t, ndim=1] source_ids,
                cnp.ndarray[size_t, ndim=1] target_ids,
                bool directed=True, bool multigraph=False,
                list distance=None, **kwargs):
    ''' Generation of a fully connected graph '''
    cdef:
        size_t num_sources = len(source_ids)
        size_t num_targets = len(target_ids)
        cnp.ndarray[float, ndim=2, mode="c"] vectors
        cnp.ndarray[float, ndim=1] x, y
        size_t s
    # find common nodes
    edges  = None
    common = set(source_ids).intersection(target_ids)

    if common:
        num_edges     = num_sources*num_targets - len(common)
        edges         = np.empty((num_edges, 2), dtype=DTYPE)
        current_edges = 0
        next_enum     = 0
        for s in source_ids:
            if s in common:
                idx       = np.where(target_ids == s)[0][0]
                tgts      = target_ids[np.arange(num_targets) != idx]
                next_enum = current_edges + num_targets - 1
                edges[current_edges:next_enum, 0] = s
                edges[current_edges:next_enum, 1] = tgts
                current_edges = next_enum
            else:
                next_enum = current_edges + num_targets
                edges[current_edges:next_enum, 0] = s
                edges[current_edges:next_enum, 1] = target_ids
                current_edges = next_enum
    else:
        edges       = np.empty((num_sources*num_targets, 2), dtype=DTYPE)
        edges[:, 0] = np.repeat(source_ids, num_targets)
        edges[:, 1] = np.tile(target_ids, num_sources)

    if distance is not None:
        pos       = kwargs['positions']
        x, y      = pos[0], pos[1]
        vectors   = np.array((x[edges[:, 1]] - x[edges[:, 0]],
                              y[edges[:, 1]] - y[edges[:, 0]]))
        distance.extend(np.linalg.norm(vectors, axis=0))

    return edges


def _fixed_degree(cnp.ndarray[size_t, ndim=1] source_ids,
                  cnp.ndarray[size_t, ndim=1] target_ids, degree=-1,
                  degree_type="in", float reciprocity=-1, bool directed=True,
                  bool multigraph=False, existing_edges=None, **kwargs):
    ''' Generation of the edges through the C++ function '''
    degree = int(degree)
    assert degree >= 0, "A positive value is required for `degree`."

    cdef:
        # type of degree
        bool b_out = (degree_type == "out")
        bool b_total = (degree_type == "total")
        size_t num_source = source_ids.shape[0]
        size_t num_target = target_ids.shape[0]
        size_t edges = (num_source * degree
                        if degree_type == "out" else num_target * degree)
        bool b_one_pop = _check_num_edges(
            source_ids, target_ids, edges, directed, multigraph)
        unsigned int existing = \
            0 if existing_edges is None else existing_edges.shape[0]
        cnp.ndarray[size_t, ndim=2, mode="c"] ia_edges = np.zeros(
            (existing+edges, 2), dtype=DTYPE)
    if existing:
        ia_edges[:existing,:] = existing_edges
    cdef:
        unsigned int idx = 0 if b_out else 1 # differenciate source / target
        unsigned int omp = nngt._config["omp"]
        unsigned int num_degrees = num_source if b_out else num_target
        long msd = np.random.randint(0, edges + 1)
        vector[unsigned int] degrees = np.repeat(degree, num_degrees)
        vector[ vector[size_t] ] old_edges = vector[ vector[size_t] ]()

    if b_out:
        _gen_edges(&ia_edges[0,0], source_ids, degrees, target_ids, old_edges,
            idx, multigraph, directed, msd, omp)
    else:
        _gen_edges(&ia_edges[0,0], target_ids, degrees, source_ids, old_edges,
            idx, multigraph, directed, msd, omp)

    return ia_edges


def _gaussian_degree(cnp.ndarray[size_t, ndim=1] source_ids,
                     cnp.ndarray[size_t, ndim=1] target_ids, float avg=-1,
                     float std=-1, degree_type="in", float reciprocity=-1,
                     bool directed=True, bool multigraph=False,
                     existing_edges=None, **kwargs):
    '''
    Connect nodes with a Gaussian distribution (generation through C++
    function.
    '''
    # switch values to float
    avg = float(avg)
    std = float(std)
    assert avg >= 0, "A positive value is required for `avg`."
    assert std >= 0, "A positive value is required for `std`."

    cdef:
        # type of degree
        b_out = (degree_type == "out")
        b_total = (degree_type == "total")
        size_t num_source = source_ids.shape[0]
        size_t num_target = target_ids.shape[0]
        unsigned int idx = 0 if b_out else 1 # differenciate source / target
        unsigned int omp = nngt._config["omp"]
        unsigned int num_degrees = num_source if b_out else num_target
        vector[unsigned int] degrees = np.around(np.maximum(
            np.random.normal(avg, std, num_degrees), 0.)).astype(DTYPE)
        vector[ vector[size_t] ] old_edges = vector[ vector[size_t] ]()

    # edges
    edges = np.sum(degrees)
    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)

    cdef:
        long msd = np.random.randint(0, edges + 1)
        unsigned int existing = \
            0 if existing_edges is None else existing_edges.shape[0]
        cnp.ndarray[size_t, ndim=2, mode="c"] ia_edges = np.zeros(
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

    get_state()
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


def _distance_rule(cnp.ndarray[size_t, ndim=1] source_ids,
                   cnp.ndarray[size_t, ndim=1] target_ids, float density=-1.,
                   int edges=-1, float avg_deg=-1., float scale=-1.,
                   str rule="exp", float max_proba=-1., shape=None,
                   cnp.ndarray[float, ndim=2] positions=np.array([[0], [0]]),
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
        size_t s, i
        string crule = _to_bytes(rule)
        unsigned int omp = nngt._config["omp"]
        vector[ vector[size_t] ] old_edges = vector[ vector[size_t] ]()
        vector[ vector[size_t] ] local_targets
        cnp.ndarray[long, ndim=1] loc_tgts
        list sources = []
        list targets = []
        vector[float] x = positions[0]
        vector[float] y = positions[1]
        float cscale = scale
    # compute the required values
    edge_num = 0
    if max_proba <= 0.:
        edge_num, _ = _compute_connections(
            num_source, num_target, density, edges, avg_deg, directed)
    else:
        raise RuntimeError(
            "Not working with max_proba and multithreading yet.")
    existing = 0  # for now
    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)
    # for each node, check the neighbours that are in an area where
    # connections can be made: +/- scale for lin, +/- 10*scale for exp
    lim = scale if rule == 'lin' else 10*scale
    for i in source_ids:
        keep  = (np.abs(positions[0, target_ids] - positions[0, i]) < lim)
        keep *= (np.abs(positions[1, target_ids] - positions[1, i]) < lim)
        if b_one_pop:
            idx = np.where(target_ids == i)[0][0]
            keep[idx] = 0
        local_targets.push_back(target_ids[keep].tolist())
    # create the edges
    cdef:
        long msd = np.random.randint(0, edge_num + 1)
        #~ float area = shape.area * conversion_factor
        size_t cedges = edge_num
        cnp.ndarray[size_t, ndim=2, mode="c"] ia_edges = np.zeros(
            (existing + edge_num, 2), dtype=DTYPE)
        vector[float] dist = vector[float]()

    if max_proba <= 0.:
        _cdistance_rule(&ia_edges[0,0], source_ids, local_targets, crule,
                        cscale, 1., x, y, cnum_neurons, cedges, old_edges,
                        dist, multigraph, msd, omp)
        distance.extend(dist)
        return ia_edges
    else:
        for i, s in enumerate(source_ids):
            loc_tgts = np.array(local_targets[i], dtype=int)
            if len(loc_tgts):
                dist_tmp = []
                test = max_proba_dist_rule(
                    rule, scale, max_proba, positions[:, s],
                    positions[:, loc_tgts], dist=dist_tmp)
                test = np.greater(test, np.random.uniform(size=len(test)))
                added = np.sum(test)
                sources.extend((s for i in range(added)))
                targets.extend(loc_tgts[test])
                distance.extend(np.array(dist_tmp)[test])
        return np.array([sources, targets]).T


def price_network():
    #@todo: do it for other libraries
    pass
