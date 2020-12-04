#cython: cdivision=True, language_level=3
#!/usr/bin/env cython
#-*- coding:utf-8 -*-

""" Cython interface to C++ parallel generation tools for NNGT """

import warnings

from .cconnect cimport *
cimport numpy as cnp

import numpy as np
import scipy.sparse as ssp
from numpy.random import randint, get_state

import nngt
from nngt.lib import InvalidArgument
from nngt.lib.connect_tools import (_check_num_edges, _compute_connections,
                                    _set_degree_type, max_proba_dist_rule)


__all__ = [
    "_all_to_all",
    "_distance_rule",
    "_fixed_degree",
    "_from_degree_list",
    "_gaussian_degree",
]


cdef unsigned int MAXTESTS = 1000 # ensure that generation will finish
cdef float EPS = 0.00001

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int64

ctypedef cnp.uint8_t uint8
ctypedef cnp.int64_t int64


# ------------ #
# Simple tools #
# ------------ #

cdef bytes _to_bytes(string):
    ''' Convert string to bytes '''
    if not isinstance(string, bytes):
        return bytes(string, "UTF-8")

    return string


def _no_self_loops(cnp.ndarray[int64_t, ndim=2] array):
    '''
    Remove self-loops
    '''
    cdef cnp.ndarray[bool, ndim=1] test = array[:, 0] != array[:, 1]

    return array[test], test


def _filter(cnp.ndarray[int64, ndim=2] ia_edges,
            cnp.ndarray[int64, ndim=2] ia_edges_tmp,
            size_t num_ecurrent, set edges_hash, bool b_one_pop,
            bool multigraph, bool directed=True, set recip_hash=None,
            cnp.ndarray distance=None, cnp.ndarray dist_tmp=None):
    '''
    Filter the edges: remove self loops and multiple connections if the graph
    is not a multigraph.
    '''
    if b_one_pop:
        ia_edges_tmp, test = _no_self_loops(ia_edges_tmp)

        if dist_tmp is not None:
            dist_tmp = dist_tmp[test]

    cdef:
        int64_t[:] e
        float d
        tuple tpl_e

    if not multigraph:
        num_ecurrent = len(edges_hash)

        if distance is not None:
            for e, d in zip(ia_edges_tmp, dist_tmp):
                tpl_e = tuple(e)

                if tpl_e not in edges_hash:
                    if directed or tpl_e not in recip_hash:
                        ia_edges[num_ecurrent] = e
                        distance.append(d)
                        edges_hash.add(tpl_e)

                        if not directed:
                            recip_hash.add(tpl_e[::-1])

                        num_ecurrent += 1
        else:
            for e in ia_edges_tmp:
                tpl_e = tuple(e)

                if tpl_e not in edges_hash:
                    if directed or tpl_e not in recip_hash:
                        ia_edges[num_ecurrent] = e
                        edges_hash.add(tpl_e)

                        if not directed:
                            recip_hash.add(tpl_e[::-1])

                        num_ecurrent += 1
    else:
        num_added = len(ia_edges_tmp)
        ia_edges[num_ecurrent:num_ecurrent + num_added, :] = ia_edges_tmp
        num_ecurrent += num_added

        if distance is not None:
            distance.extend(dist_tmp)

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
        size_t s, next_enum

    # find common nodes
    edges  = None
    common = set(source_ids).intersection(target_ids)

    num_edges = int(0.5*(1 + directed)*num_sources*num_targets)

    if common:
        num_edges -= int(0.5*(1 + directed)*len(common))

        edges         = np.full((num_edges, 2), -1, dtype=DTYPE)
        current_edges = 0
        next_enum     = 0

        for s in source_ids:
            if s in common:
                if directed:
                    next_enum = current_edges + num_targets - 1

                    edges[current_edges:next_enum, 1] = \
                        target_ids[target_ids != s]
                else:
                    tgts = [t for t in target_ids if t not in common or t > s]

                    next_enum = current_edges + len(tgts)

                    edges[current_edges:next_enum, 1] = tgts

                edges[current_edges:next_enum, 0] = s

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

    if distance is not None and 'positions' in kwargs:
        pos       = kwargs['positions']
        x, y      = pos[0], pos[1]
        vectors   = np.array((x[edges[:, 1]] - x[edges[:, 0]],
                              y[edges[:, 1]] - y[edges[:, 0]]))
        distance.extend(np.linalg.norm(vectors, axis=0))

    return edges


def _total_degree_list(cnp.ndarray[int64, ndim=1] source_ids,
                       cnp.ndarray[int64, ndim=1] target_ids,
                       cnp.ndarray[int64, ndim=1] degree_list,
                       bool directed=True, bool multigraph=False, **kwargs):
    ''' Called from _from_degree_list '''
    cdef:
        size_t num_source = len(source_ids)
        size_t num_target = len(target_ids)
        size_t edges = 0.5*np.sum(degree_list)

    # check if the sequence is not obviously impossible
    if not multigraph:
        assert np.all(np.less_equal(degree_list, (num_target - 1))), \
            "Some degrees are higher than the number of available targets."

    if np.sum(degree_list) % 2 != 0:
        raise ValueError("The sum of the degrees must ben even.")

    # check one pop
    cdef:
        bool b_one_pop
        set source_set, target_set

    b_one_pop, source_set, target_set = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph, return_sets=True)

    cdef:
        size_t i, k, num_tests, num_choice, new_etot
        size_t ecurrent = 0
        set edges_hash = set()
        set recip_hash = set()
        set ids
        int64[:] sorted_degrees = np.sort(degree_list)[::-1]
        int64[:] data
        cnp.ndarray[int64, ndim=1] incr, decr
        cnp.ndarray[int64, ndim=1] deg_tmp, unfinished, even_deg, add_node
        cnp.ndarray[int64, ndim=2] chosen, new_edges
        cnp.ndarray[int64, ndim=2] ia_edges = np.full((edges, 2), -1,
                                                      dtype=np.int64)

    rng = nngt._rng

    if b_one_pop:
        # check that the degree sequence is graphic (i.e. can be realized)
        for k in range(1, num_source + 1):
            partial_sum = np.sum(sorted_degrees[:k])
            capped_deg  = np.minimum(sorted_degrees[k:], k)

            if not partial_sum <= k*(k-1) + np.sum(capped_deg):
                raise ValueError("The degree sequence provided is not "
                                 "graphical and cannot be realized.")

        num_tests = 0

        while ecurrent < edges and num_tests < MAXTESTS:
            # initial procedure
            remaining = max(int(0.5*(edges - ecurrent)), 1)

            data = np.repeat(source_ids, degree_list)
            sources = rng.choice(data, remaining, replace=False)

            deg_tmp = degree_list.copy()
            np.add.at(deg_tmp, sources, -1)

            data = np.repeat(source_ids, deg_tmp)
            targets = rng.choice(data, remaining, replace=False)

            new_edges = np.array((sources, targets), dtype=DTYPE).T

            ia_edges, new_etot =  _filter(
                ia_edges, new_edges, ecurrent, edges_hash, b_one_pop,
                multigraph, directed=directed, recip_hash=recip_hash)

            np.add.at(degree_list, np.ravel(ia_edges[ecurrent:new_etot]), -1)

            ecurrent = new_etot

            # trying to correct things if initial procedure did not converge
            if num_tests >= 0.5*MAXTESTS:
                nnz_degree = degree_list != 0
                unfinished = np.where(nnz_degree)[0]
                num_choice = max(int(0.5*len(unfinished)), 1)

                # pick random edges
                ids    = set(rng.choice(ecurrent, num_choice, replace=False))
                chosen = np.array([ia_edges[i] for i in ids], dtype=np.int64)

                # if unfinished is odd then we need to readd a node that has
                # a remaining degree of 2
                if len(unfinished) % 2 == 1:
                    even_deg = np.where((degree_list % 2 == 0)*nnz_degree)[0]
                    add_node = rng.choice(even_deg, 1)
                    unfinished = np.array(
                        list(unfinished) + list(add_node), dtype=np.int64)

                # try to pair them differently
                rng.shuffle(unfinished)

                new_edges = np.array(
                    [(u, v) for u, v in zip(unfinished, chosen.ravel())],
                    dtype=np.int64)

                # randomize direction if directed
                if directed:
                    num_reverse = rng.binomial(2*num_choice, 0.5)
                    reverse     = rng.choice(2*num_choice, num_reverse,
                                             replace=False)

                    new_edges[reverse, 0], new_edges[reverse, 1] = \
                        new_edges[reverse, 1], new_edges[reverse, 0]

                # check that new_edges are indeed new
                skip = False

                for e in new_edges:
                    if e[0] == e[1]:
                        skip = True
                        break

                    tpl_e = tuple(e)
                        
                    if tpl_e in edges_hash:
                        skip = True
                        break

                    if not directed and tpl_e in recip_hash:
                        skip = True
                        break

                if skip:
                    num_tests += 1
                    continue

                # remove old ones from edges_hash
                edges_hash -= {tuple(e) for e in chosen}

                if not directed:
                    recip_hash -= {tuple(e) for e in chosen[:, ::-1]}

                # remove chosen edges from existing edges
                ia_edges[:ecurrent - num_choice] = np.array(
                    [e for i, e in enumerate(ia_edges[:ecurrent])
                     if i not in ids])

                ecurrent -= num_choice

                ia_edges, new_etot =  _filter(
                    ia_edges, new_edges, ecurrent, edges_hash, b_one_pop,
                    multigraph, directed=directed, recip_hash=recip_hash)

                decr  = new_edges.ravel()
                incr  = chosen.ravel()

                np.add.at(degree_list, incr, 1)
                np.add.at(degree_list, decr, -1)
                
                ecurrent += len(new_edges)
                
            num_tests += 1

        if num_tests == MAXTESTS:
            raise RuntimeError("Graph generation did not converge.")

        return ia_edges

    raise NotImplementedError("not available if sources != targets.")


def _from_degree_list(cnp.ndarray[size_t, ndim=1] source_ids,
                      cnp.ndarray[size_t, ndim=1] target_ids, degrees,
                      degree_type="in", bool directed=True,
                      bool multigraph=False, existing_edges=None, **kwargs):
    ''' Generation of the degree list through the C++ function '''

    assert len(degrees) == len(source_ids), \
        "One degree per source neuron must be provided."

    degree_type = _set_degree_type(degree_type)

    cdef:
        # type of degree
        bool b_out = (degree_type == "out")
        bool b_total = (degree_type == "total")
        bool use_directed = directed + b_total
        size_t num_source = source_ids.shape[0]
        size_t edges = np.sum(degrees)
        bool b_one_pop = _check_num_edges(
            source_ids, target_ids, edges, directed, multigraph)
        int64_t[:, :] ia_edges = np.full((edges, 2), -1, dtype=DTYPE)

        vector[size_t] sources = source_ids
        vector[size_t] targets = target_ids

        unsigned int idx = 0 if b_out else 1  # differenciate source / target
        unsigned int omp = nngt._config["omp"]
        cnp.ndarray[int64, ndim=1] degree_list, source64, target64
        vector[unsigned int] cdegrees = degrees
        vector[ vector[size_t] ] old_edges = vector[ vector[size_t] ]()
        vector[long] seeds = _random_init(omp)

    # total-degree or undirected case
    if b_total or not directed:
        degree_list = np.array(degrees, dtype=np.int64)
        source64 = np.array(source_ids, dtype=np.int64)
        target64 = np.array(target_ids, dtype=np.int64)

        return _total_degree_list(source64, target64, degree_list,
                                  directed=directed, multigraph=multigraph)

    if existing_edges is not None:
        old_edges.push_back(list(existing_edges[:, 0]))
        old_edges.push_back(list(existing_edges[:, 1]))

    # directed case for in/out-degrees
    _gen_edges(&ia_edges[0,0], source_ids, cdegrees, target_ids, old_edges,
               idx, multigraph, use_directed, seeds)

    return ia_edges


def _fixed_degree(cnp.ndarray[size_t, ndim=1] source_ids,
                  cnp.ndarray[size_t, ndim=1] target_ids, degree=-1,
                  degree_type="in", float reciprocity=-1, bool directed=True,
                  bool multigraph=False, existing_edges=None, **kwargs):
    ''' Generation of the edges through the C++ function '''
    degree = int(degree)

    assert degree >= 0, "A positive value is required for `degree`."

    degrees = np.repeat(degree, len(source_ids))

    return _from_degree_list(source_ids, target_ids, degrees,
                             degree_type=degree_type, directed=directed,
                             multigraph=multigraph,
                             existing_edges=existing_edges)


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

    degree_type = _set_degree_type(degree_type)

    rng = nngt._rng

    cdef:
        # type of degree
        bool b_total = (degree_type == "total")
        size_t num_source = source_ids.shape[0]
        uint64_t[:] degrees = np.around(np.maximum(
            rng.normal(avg, std, num_source), 0)).astype(np.uint64)
        size_t edges = np.sum(degrees)
        int idx

    if b_total or not directed:
        # check that the sum of the degrees is even
        if edges % 2 != 0:
            idx = -1

            # correct if its not the case
            while idx < 0 or degrees[idx] == 0:
                idx = rng.choice(num_source)
                if degrees[idx] > 0:
                    degrees[idx] -= 1

    return _from_degree_list(source_ids, target_ids, degrees,
        degree_type=degree_type, directed=directed, multigraph=multigraph,
        existing_edges=existing_edges)


def _distance_rule(cnp.ndarray[size_t, ndim=1] source_ids,
                   cnp.ndarray[size_t, ndim=1] target_ids, density=None,
                   edges=None, avg_deg=None, float scale=-1., str rule="exp",
                   float max_proba=-1., shape=None,
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
        size_t s, i, edge_num
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
    if max_proba <= 0.:
        edge_num, _ = _compute_connections(
            num_source, num_target, density, edges, avg_deg, directed)
    else:
        raise RuntimeError(
            "Not working with max_proba and multithreading yet.")

    existing = 0  # for now
    b_one_pop = _check_num_edges(
        source_ids, target_ids, edge_num, directed, multigraph)

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
        size_t cedges = edge_num
        int64_t[:, :] ia_edges = np.full((existing + edge_num, 2), -1,
                                         dtype=DTYPE)
        vector[float] dist = vector[float]()
        vector[long] seeds = _random_init(omp)

    if max_proba <= 0.:
        _cdistance_rule(&ia_edges[0,0], source_ids, local_targets, crule,
                        cscale, 1., x, y, cnum_neurons, cedges, old_edges,
                        dist, multigraph, directed, seeds)
        distance.extend(dist)
        return ia_edges

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


# ------------ #
# Random seeds #
# ------------ #


def _random_init(omp):
    '''
    Init the local random seeds
    '''
    # compute local random seeds
    seeds = None

    if not nngt._seeded_local:
        # generate the local seeds for the first time
        msd   = nngt._config["msd"]
        seeds = [msd + i + 1 for i in range(omp)]

        nngt._config["seeds"] = seeds
    elif nngt._used_local:
        # the initial seeds have already been used so we generate new ones
        msd   = np.random.randint(0, 2**31 - omp - 1)
        seeds = [msd + i + 1 for i in range(omp)]
    else:
        seeds = nngt.get_config('seeds')

    assert len(seeds) == omp, "Wrong number of seeds, need one per thread."

    nngt._seeded_local = True
    nngt._used_local   = True

    return seeds
