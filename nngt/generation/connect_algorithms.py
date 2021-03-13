#-*- coding:utf-8 -*-
#
# generation/connect_algorithms.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Generation tools for NNGT """

import logging
import numpy as np
import scipy.sparse as ssp
from numpy.random import randint

import nngt
from nngt.lib import InvalidArgument
from nngt.lib.connect_tools import *


__all__ = [
    "_all_to_all",
    "_circular",
    "_distance_rule",
    "_erdos_renyi",
    "_fixed_degree",
    "_from_degree_list",
    "_gaussian_degree",
    "_newman_watts",
    "_price_scale_free",
    "_random_scale_free",
    "_unique_rows",
    "_watts_strogatz",
]

logger = logging.getLogger(__name__)

MAXTESTS = 1000 # ensure that generation will finish
EPS = 0.00001


# ---------------------- #
# Graph model generation #
# ---------------------- #

def _all_to_all(source_ids, target_ids, directed=True, multigraph=False,
                distance=None, **kwargs):
    num_sources, num_targets = len(source_ids), len(target_ids)

    # find common nodes
    edges  = None
    common = set(source_ids).intersection(target_ids)

    num_edges = int(0.5*(1 + directed)*num_sources*num_targets)

    if common:
        num_edges -= int(0.5*(1 + directed)*len(common))

        edges = np.full((num_edges, 2), -1, dtype=int)

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
        edges       = np.empty((num_edges, 2), dtype=int)
        edges[:, 0] = np.repeat(source_ids, num_targets)
        edges[:, 1] = np.tile(target_ids, num_sources)

    if distance is not None and 'positions' in kwargs:
        pos       = kwargs['positions']
        x, y      = pos[0], pos[1]
        vectors   = np.array((x[edges[:, 1]] - x[edges[:, 0]],
                              y[edges[:, 1]] - y[edges[:, 0]]))
        distance.extend(np.linalg.norm(vectors, axis=0))

    return edges


def _total_degree_list(source_ids, target_ids, degree_list, directed=True,
                       multigraph=False, **kwargs):
    ''' Called from _from_degree_list '''
    degree_list = np.array(degree_list, dtype=int)

    num_source, num_target = len(source_ids), len(target_ids)
    edges = int(0.5*np.sum(degree_list))

    # check if the sequence is not obviously impossible
    if not multigraph:
        assert np.all(degree_list <= (num_target - 1)), \
            "Some degrees are higher than the number of available targets."

    if np.sum(degree_list) % 2 != 0:
        raise ValueError("The sum of the degrees must ben even.")

    # check one pop
    b_one_pop, source_set, target_set = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph, return_sets=True)

    ecurrent = 0
    edges_hash = set()
    recip_hash = set()

    ia_edges = np.full((edges, 2), -1, dtype=int)

    rng = nngt._rng

    if b_one_pop:
        # check that the degree sequence is graphic (i.e. can be realized)
        sorted_degrees = np.sort(degree_list)[::-1]
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

            new_edges = np.array((sources, targets), dtype=int).T

            ia_edges, new_etot =  _filter(
                ia_edges, new_edges, ecurrent, edges_hash, b_one_pop,
                multigraph, directed=directed, recip_hash=recip_hash)

            np.add.at(degree_list, ia_edges[ecurrent:new_etot].ravel(), -1)

            ecurrent = new_etot

            # trying to correct things if initial procedure did not converge
            if num_tests >= 0.5*MAXTESTS:
                nnz_degree = degree_list != 0
                unfinished = np.where(nnz_degree)[0]
                num_choice = max(int(0.5*len(unfinished)), 1)

                # pick random edges
                ids    = set(rng.choice(ecurrent, num_choice, replace=False))
                chosen = np.array([ia_edges[i] for i in ids], dtype=int)

                # if unfinished is odd then we need to readd a node that has
                # a remaining degree of 2
                if len(unfinished) % 2 == 1:
                    even_deg = np.where((degree_list % 2 == 0)*nnz_degree)[0]
                    add_node = rng.choice(even_deg, 1)
                    unfinished = list(unfinished) + list(add_node)

                # try to pair them differently
                rng.shuffle(unfinished)

                new_edges = np.array(
                    [(u, v) for u, v in zip(unfinished, chosen.ravel())],
                    dtype=int)

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
                        
                    if tuple(e) in edges_hash:
                        skip = True
                        break

                    if not directed and tuple(e) in recip_hash:
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

                decr  = list(new_edges.ravel())
                incr  = list(chosen.ravel())

                np.add.at(degree_list, incr, 1)
                np.add.at(degree_list, decr, -1)
                
                ecurrent += len(new_edges)

            num_tests += 1

        if num_tests == MAXTESTS:
            raise RuntimeError("Graph generation did not converge.")

        return ia_edges

    raise NotImplementedError("not available if sources != targets.")
    

def _from_degree_list(source_ids, target_ids, degree_list, degree_type="in",
                      directed=True, multigraph=False, existing_edges=None,
                      **kwargs):
    assert len(degree_list) == len(source_ids), \
        "One degree per source neuron must be provided."

    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
    num_source, num_target = len(source_ids), len(target_ids)

    # type of degree
    degree_type = _set_degree_type(degree_type)

    b_out = (degree_type == "out")
    b_total = (degree_type == "total")

    # directed case for in/out-degree
    if directed and not b_total:
        edges = np.sum(degree_list)

        b_one_pop = _check_num_edges(
            source_ids, target_ids, edges, directed, multigraph)

        num_einit  = 0 if existing_edges is None else existing_edges.shape[0]
        num_etotal = num_einit

        ia_edges = np.full((num_einit + edges, 2), -1, dtype=int)

        if num_einit:
            ia_edges[:num_einit, :] = existing_edges

        idx = 0 if b_out else 1  # differenciate source / target

        # keep track of degrees for undirected case
        target_degrees  = {}
        current_targets = {u: [] for u in source_ids}

        for u, d in zip(source_ids, degree_list):
            if existing_edges is not None:
                current_targets[u].extend(
                    existing_edges[existing_edges[:, idx] == u, 1 - idx])

            if not multigraph and (num_target - len(current_targets[u]) < d):
                raise RuntimeError(
                    ("Node {node} already has a degree {d0} which means that "
                     "there are not enough targets left to add {d} edges."
                    ).format(node=u, d0=len(current_targets[u]), target=d))

            target_degrees[u] = d

        rng = nngt._rng

        # create new connections
        for v, degree_v in target_degrees.items():
            existing_v = current_targets[v]
            targets_v  = [] # where the targets will be stored

            # set targets
            rm = np.where(target_ids == v)[0]
            rm = rm[0] if len(rm) else -1

            tgts = (target_ids if rm == -1 else
                    np.concatenate((target_ids[:rm], target_ids[rm+1:])))

            # select targets
            while len(targets_v) < degree_v:
                var = rng.choice(tgts, degree_v - len(targets_v),
                                 replace=multigraph)

                targets_v.extend(var)

                if not multigraph:
                    s = set(existing_v + targets_v)

                    targets_v  = list(s.difference(existing_v))
                    existing_v = list(s)


            ia_edges[num_etotal:num_etotal + degree_v, idx] = v
            ia_edges[num_etotal:num_etotal + degree_v, 1 - idx] = targets_v

            num_etotal += degree_v

        return ia_edges[num_einit:]

    # unidrected or total-degree cases
    return _total_degree_list(source_ids, target_ids, degree_list,
                              directed=directed, multigraph=multigraph)


def _fixed_degree(source_ids, target_ids, degree, degree_type="in",
                  reciprocity=-1, directed=True, multigraph=False,
                  existing_edges=None, **kwargs):
    degree = int(degree)
    assert degree >= 0, "A positive value is required for `degree`."

    num_source = len(source_ids)

    # edges (we set the in, out, or total degree of the source neurons)
    lst_deg = np.full(num_source, degree, dtype=int)

    return _from_degree_list(
        source_ids, target_ids, lst_deg, degree_type=degree_type,
        directed=directed, multigraph=multigraph,
        existing_edges=existing_edges, **kwargs)


def _gaussian_degree(source_ids, target_ids, avg, std, degree_type="in",
                     directed=True, multigraph=False, existing_edges=None,
                     **kwargs):
    ''' Connect nodes with a Gaussian distribution '''
    # switch values to float
    avg = float(avg)
    std = float(std)

    assert avg >= 0, "A positive value is required for `avg`."
    assert std >= 0, "A positive value is required for `std`."

    num_source = len(source_ids)

    # type of degree
    degree_type = _set_degree_type(degree_type)

    b_total = (degree_type == "total")

    # edges (we set the in, out, or total degree of the source neurons)
    rng = nngt._rng

    degrees = np.around(
        np.maximum(rng.normal(avg, std, num_source), 0.)).astype(int)

    if b_total or not directed:
        # check that the sum of the degrees is even
        if np.sum(degrees) % 2 != 0:
            idx = -1

            # correct if its not the case
            while idx < 0 or degrees[idx] == 0:
                idx = rng.choice(num_source)
                if degrees[idx] > 0:
                    degrees[idx] -= 1

    return _from_degree_list(
        source_ids, target_ids, degrees, degree_type=degree_type,
        directed=directed, multigraph=multigraph,
        existing_edges=existing_edges, **kwargs)


def _random_scale_free(source_ids, target_ids, in_exp, out_exp, density=None,
                       edges=None, avg_deg=None, reciprocity=-1,
                       directed=True, multigraph=False, **kwargs):
    ''' Connect the nodes with power law distributions '''
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
    num_source, num_target = len(source_ids), len(target_ids)
    edges, pre_recip_edges = _compute_connections(num_source, num_target,
                                density, edges, avg_deg, directed, reciprocity)
    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)

    arr_edges = np.full((edges, 2), -1, dtype=int)
    num_ecurrent, num_test = 0, 0
    edges_hash = set()
    recip_hash = None if not directed else set()

    # lists containing the in/out-degrees for all nodes
    in_deg = np.random.pareto(in_exp, num_target) + 1
    out_deg = np.random.pareto(out_exp, num_source) + 1

    sum_in, sum_out = np.sum(in_deg), np.sum(out_deg)

    in_deg = np.around(np.multiply(
        pre_recip_edges / sum_in, in_deg)).astype(int)
    out_deg = np.around(np.multiply(
        pre_recip_edges / sum_out, out_deg)).astype(int)

    sum_in, sum_out = np.sum(in_deg), np.sum(out_deg)

    # correct
    while sum_in != pre_recip_edges or sum_out != pre_recip_edges:
        diff_in  = sum_in - pre_recip_edges
        diff_out = sum_out - pre_recip_edges

        idx_correct_in = randint(0, num_target, np.abs(diff_in))
        idx_correct_out = randint(0, num_source, np.abs(diff_out))
        in_deg[idx_correct_in] -= 1*np.sign(diff_in)
        out_deg[idx_correct_out] -= 1*np.sign(diff_out)
        in_deg[in_deg < 0] = 0
        out_deg[out_deg < 0] = 0
        sum_in, sum_out = np.sum(in_deg), np.sum(out_deg)

    # make the edges
    sources = np.repeat(source_ids, out_deg)
    targets = np.repeat(target_ids, in_deg)

    np.random.shuffle(targets)

    arr_edges_tmp = np.array((sources, targets)).T

    arr_edges, num_ecurrent = _filter(arr_edges, arr_edges_tmp, num_ecurrent,
                                     edges_hash, b_one_pop, multigraph,
                                     directed=directed, recip_hash=recip_hash)

    while num_ecurrent != pre_recip_edges and num_test < MAXTESTS:
        num_desired = pre_recip_edges-num_ecurrent
        sources_tmp = np.random.choice(sources, num_desired)
        targets_tmp = np.random.choice(targets, num_desired)
        arr_edges_tmp = np.array([sources_tmp, targets_tmp]).T
        arr_edges, num_ecurrent = _filter(
            arr_edges, arr_edges_tmp, num_ecurrent, edges_hash, b_one_pop,
            multigraph, directed=directed, recip_hash=recip_hash)
        num_test += 1

    if directed and reciprocity > 0:
        while num_ecurrent != edges and num_test < MAXTESTS:
            keep = np.random.choice(num_ecurrent, edges-num_ecurrent,
                                    replace=multigraph)
            arr_edges[num_ecurrent:] = arr_edges[keep]
            num_ecurrent = edges
            if not multigraph:
                arr_edges_tmp = _unique_rows(arr_edges)
                num_ecurrent = arr_edges_tmp.shape[0]
                arr_edges[:num_ecurrent,:] = arr_edges_tmp
            num_test += 1

    return arr_edges


def _erdos_renyi(source_ids, target_ids, density=None, edges=None,
                 avg_deg=None, reciprocity=-1, directed=True, multigraph=False,
                 **kwargs):
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

    ia_edges = np.full((edges, 2), -1, dtype=int)
    num_test, num_ecurrent = 0, 0 # number of tests and current number of edges
    edges_hash = set()
    recip_hash = None if directed else set()

    while num_ecurrent != pre_recip_edges and num_test < MAXTESTS:
        ia_sources = source_ids[randint(0, num_source,
                                        pre_recip_edges-num_ecurrent)]
        ia_targets = target_ids[randint(0, num_target,
                                        pre_recip_edges-num_ecurrent)]
        ia_edges_tmp = np.array([ia_sources,ia_targets]).T
        ia_edges, num_ecurrent = _filter(
            ia_edges, ia_edges_tmp, num_ecurrent, edges_hash, b_one_pop,
            multigraph, directed=directed, recip_hash=recip_hash)
        num_test += 1

    if directed and reciprocity > 0:
        while num_ecurrent != edges and num_test < MAXTESTS:
            ia_indices = randint(0, pre_recip_edges,
                                           edges-num_ecurrent)
            ia_edges[num_ecurrent:] = ia_edges[ia_indices]
            num_ecurrent = edges
            if not multigraph:
                ia_edges_tmp = _unique_rows(ia_edges)
                num_ecurrent = ia_edges_tmp.shape[0]
                ia_edges[:num_ecurrent,:] = ia_edges_tmp
            num_test += 1

    return ia_edges


def _price_scale_free(ids, m, c, gamma, reciprocity, directed, multigraph,
                      **kwargs):
    '''
    Generate a Price network.
    '''
    ids = np.array(ids).astype(int)

    num_nodes = len(ids)

    assert isinstance(m, int) and m >= 0, "`m` must be a positive integer."

    if directed:
        assert c > 0, "`c` > 0 is required for directed graphs."
    else:
        assert c > -1, "`c` > -1 is required for undirected graphs."

    assert 0 <= reciprocity <= 1, "`reciprocity` must be in [0, 1]."

    in_degrees = np.zeros(num_nodes)

    edges = []
    edges_hash = set()
    recip_hash = None if directed else set()

    rng = nngt._rng

    for i, n in zip(range(1, num_nodes), ids[1:]):
        proba = None if i == 1 else np.power(in_degrees[:i], gamma) + c

        if proba is not None:
            proba /= proba.sum()

        edges_i = []

        m_i = min(i, m)

        while len(edges_i) < m_i:
            targets = rng.choice(i, size=m_i, replace=False, p=proba)

            for t in targets:
                e = (n, ids[t])

                if e not in edges_hash:
                    if directed:
                        edges_i.append(e)
                        in_degrees[t] += 1
                    elif e not in recip_hash:
                        edges_i.append(e)
                        in_degrees[i] += 1
                        in_degrees[t] += 1

        edges.extend(edges_i)

        if directed and reciprocity > 0:
            make_recip = rng.random(m) < reciprocity

            for b, e in zip(make_recip, edges_i):
                if b:
                    edges.append(e[::-1])
                    in_degrees[i] += 1

    return edges


def _circular(source_ids, target_ids, coord_nb, reciprocity=1, directed=True,
              reciprocity_choice="random", **kwargs):
    '''
    Circular graph.

    Note
    ----
    `source_ids` and `target_ids` are only there for compatibility with the
    connect functions, this algorithm requires a single population.
    This check (if necessary) is performed above.
    '''
    if reciprocity_choice not in ("random", "closest", "closest-ordered"):
        # note: only "random" and "closest" are publicly advertised,
        # "closest-ordered" is only for internal use in lattice_rewire.
        raise ValueError("Valid entries for `reciprocity_choice` are "
                         "'random' and 'closest'.")

    if coord_nb % 2:
        raise ValueError("`coord_nb` must be even.")

    if reciprocity == 1 or not directed:
        return _circular_full(source_ids, coord_nb, directed)
    elif directed:
        return _circular_directed_recip(source_ids, coord_nb, reciprocity,
                                        reciprocity_choice=reciprocity_choice)

    raise ValueError("For undirected graphs, reciprocity is 1 by definition.")


def _circular_directed_recip(node_ids, coord_nb, reciprocity,
                             reciprocity_choice="random", **kwargs):
    ''' Circular graph with given reciprocity '''
    nodes    = len(node_ids)
    edges    = int(np.round(0.5 * nodes * coord_nb *
                   (1 + reciprocity / (2 - reciprocity))))
    init_deg = int(0.5*coord_nb)

    # sources and targets
    sources = np.full(edges, -1, dtype=int)
    targets = np.full(edges, -1, dtype=int)

    # set non-reciprocal edges using the full undirected circular graph
    init_edges = _circular_full(node_ids, coord_nb, False)
    num_init   = len(init_edges)

    sources[:num_init] = init_edges[:, 0]
    targets[:num_init] = init_edges[:, 1]

    # then we randomize the direction of these E_init edges
    # this is equivalent to reversing E edges with E from Binom(E_init, 0.5)
    rng = nngt._rng
    E   = rng.binomial(num_init, 0.5)

    chosen = rng.choice(num_init, E, replace=False)

    sources[chosen], targets[chosen] = targets[chosen], sources[chosen]

    # set reciprocal edges
    num_recip = edges - nodes*init_deg

    if num_recip:
        if reciprocity_choice == "random":
            chosen = rng.choice(nodes*init_deg, size=num_recip, replace=False)

            sources[-num_recip:] = targets[chosen]
            targets[-num_recip:] = sources[chosen]
        elif reciprocity_choice == "closest":
            # closest connections are the first ones, so if
            # num_recip = k*nodes + l then we reverse all first k*nodes
            # then we randomly chose the remaining l
            remainder = num_recip % nodes
            rounds    = num_recip - remainder
            
            sources[-num_recip:-num_recip + rounds] = targets[:rounds]
            targets[-num_recip:-num_recip + rounds] = sources[:rounds]

            # chose randomly the remaining connections
            stop = min(rounds + nodes, edges)
            chosen = rng.choice([i for i in range(rounds, stop)],
                                size=remainder, replace=False)

            sources[-num_recip + rounds:] = targets[chosen]
            targets[-num_recip + rounds:] = sources[chosen]
        else:
            # deterministic (ordered) closest connections, only for the
            # lattice rewiring (see function _lattice_shuffle_eattr)
            # closest connections are the first ones
            remainder = num_recip % nodes
            rounds    = num_recip - remainder

            start = edges - num_recip
            stop  = edges - num_recip + rounds

            sources[start:stop] = targets[:rounds]
            targets[start:stop] = sources[:rounds]

            # chose randomly the remaining connections
            start = stop
            stop  = min(rounds + nodes, edges)

            chosen = rng.choice(list(range(rounds, stop)), size=remainder,
                                replace=False)

            sources[start:] = targets[chosen]
            targets[start:] = sources[chosen]

    return np.array([sources, targets], dtype=int).T


def _circular_full(node_ids, coord_nb, directed, **kwargs):
    ''' Create a circular graph with all possible edges '''
    nodes = len(node_ids)

    dist = int(0.5*coord_nb)

    out_deg = coord_nb if directed else dist

    # create the graph using a continuous range from zero
    sources = np.tile(np.arange(0, nodes).astype(int), out_deg)

    # prepare conversion dict if nodes do not start from zero or are not
    # contiguous. Initial generation assumes continuous range from zero
    convertor = {i: n for i, n in enumerate(node_ids)}

    # create the connection mask
    start = -dist if directed else 0
    stop  = dist + 1

    conn_mask = np.concatenate((np.arange(start, 0), np.arange(1, stop)))

    # make the targets and put them back into [0, nodes - 1]
    targets   = np.repeat(conn_mask, nodes) + sources

    targets[targets < 0] += nodes
    targets[targets >= nodes] -= nodes

    # convert back to ids
    sources = np.array([convertor[i] for i in sources])
    targets = np.array([convertor[i] for i in targets])

    return np.array((sources, targets), dtype=int).T


def _newman_watts(source_ids, target_ids, coord_nb, proba_shortcut,
                  reciprocity_circular=1, reciprocity_choice_circular="random",
                  edges=None, directed=True, multigraph=False, **kwargs):
    '''
    Returns a numpy array of dimension (num_edges,2) that describes the edge
    list of a Newman-Watts graph.

    Note
    ----
    `source_ids` and `target_ids` are only there for compatibility with the
    connect functions, this algorithm requires a single population.
    This check (if necessary) is performed above.
    '''
    nodes      = len(source_ids)
    source_ids = np.array(source_ids, dtype=int)
    target_ids = np.array(target_ids, dtype=int)

    # check the number of edges
    direct_factor  = 0.5*(1 + directed)
    recip_factor   = 1 + reciprocity_circular / (2 - reciprocity_circular)
    circular_edges = int(np.round(
        0.5 * direct_factor * nodes * coord_nb * recip_factor))

    if edges is None:
        rng   = nngt._rng
        edges = circular_edges + rng.binomial(circular_edges, proba_shortcut)
    elif edges < circular_edges:
        raise ValueError("`edges` must be greater or equal to "
                         "{} given current arguments.".format(circular_edges))

    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)

    if not b_one_pop:
        raise InvalidArgument("This graph model can only be used if source "
                              "and target populations are the same.")

    # generate the initial circular graph
    ia_edges = np.full((edges, 2), -1, dtype=int)

    ia_edges[:circular_edges, :] = _circular(
        source_ids, source_ids, coord_nb, reciprocity_circular, directed,
        reciprocity_choice=reciprocity_choice_circular)

    # add the random connections
    num_test, num_ecurrent = 0, circular_edges
    edges_hash = set(tuple(e) for e in ia_edges[:circular_edges])
    recip_hash = None

    if not directed:
        recip_hash = set((tuple(e) for e in ia_edges[:circular_edges, ::-1]))

    rng = nngt._rng

    while num_ecurrent != edges and num_test < MAXTESTS:
        todo   = edges - num_ecurrent
        chosen = rng.choice(source_ids, int(2*todo))

        ia_edges, num_ecurrent = _filter(
            ia_edges, chosen.reshape(todo, 2), num_ecurrent, edges_hash,
            b_one_pop, multigraph, directed=directed, recip_hash=recip_hash)

        num_test += 1

    return ia_edges


def _watts_strogatz(
        source_ids, target_ids, coord_nb, proba_shortcut,
        reciprocity_circular=1, reciprocity_choice_circular="random",
        shuffle="random", directed=True, multigraph=False, **kwargs):
    '''
    Returns a numpy array of dimension (num_edges,2) that describes the edge
    list of a Watts-Strogatz graph.

    Note
    ----
    `source_ids` and `target_ids` are only there for compatibility with the
    connect functions, this algorithm requires a single population.
    This check (if necessary) is performed above.
    '''
    assert shuffle in ('sources', 'targets', 'random'), \
        "Shuffle must be either 'sources', 'targets', or 'random'."

    nodes      = len(source_ids)
    source_ids = np.array(source_ids, dtype=int)
    target_ids = np.array(target_ids, dtype=int)

    # check the number of edges
    direct_factor  = 0.5*(1 + directed)
    recip_factor   = 0.5*(1 + reciprocity_circular)

    deg   = int(coord_nb * recip_factor * direct_factor)
    edges = nodes * deg

    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)

    if not b_one_pop:
        raise InvalidArgument("This graph model can only be used if source "
                              "and target populations are the same.")

    # generate the initial circular graph
    ia_edges = _circular(
        source_ids, source_ids, coord_nb, reciprocity_circular, directed,
        reciprocity_choice=reciprocity_choice_circular)

    # randomize some of the outgoing connections
    node_set = set(source_ids)

    rng = nngt._rng

    edge_hash = set(tuple(e) for e in ia_edges)

    if not directed:
        edge_hash.update([tuple(e) for e in ia_edges[:, ::-1]])

    # compute how many we rewire in total, choose them
    rewire = rng.binomial(edges, proba_shortcut)
    chosen = rng.choice(edges, rewire, replace=False).tolist()

    # rewire
    keep = [0]*rewire

    if shuffle == "random":
        keep = rng.integers(1, size=rewire, endpoint=True).tolist()
    elif shuffle == "sources":
        keep = [1]*rewire

    new_targets = rng.integers(nodes, size=rewire).tolist()

    max_tests   = 50*rewire
    num_test    = 0
    num_rewired = 0

    while num_rewired < rewire and num_test < max_tests:
        # if used all previous chosen, regenerate some
        if not chosen:
            remaining = rewire - num_rewired

            chosen = rng.choice(edges, remaining, replace=False).tolist()

            keep = [0]*remaining
            if shuffle == "random":
                keep = rng.integers(1, size=remaining, endpoint=True).tolist()
            elif shuffle == "sources":
                keep = [1]*remaining

            new_targets = rng.integers(nodes, size=remaining).tolist()

        # try to rewire
        idx1 = chosen.pop()
        idx2 = keep.pop()

        old_edge = tuple(ia_edges[idx1])

        s = old_edge[idx2]
        t = new_targets.pop()
        e = (s, t) if idx2 == 0 else (t, s)

        mtests = 50*nodes
        ntest  = 0

        while (s == t or e in edge_hash) and ntest < mtests:
            t = rng.integers(nodes)
            e = (s, t) if idx2 == 0 else (t, s)
            ntest += 1

        # update test here to count skips
        num_test += 1

        # chosen node may already reached max degree, skip
        if ntest == mtests:
            continue

        # rewire edge if successful
        ia_edges[idx1, 1 - idx2] = t

        # add new edge and remove old one
        edge_hash.add(e)

        edge_hash -= {old_edge}

        if not directed:
            edge_hash.add(e[::-1])

            edge_hash -= {old_edge[::-1]}

        num_rewired += 1

    if max_tests and num_test == max_tests:
        raise RuntimeError("Algorithm did not converge.")

    return ia_edges


def _distance_rule(source_ids, target_ids, density=None, edges=None,
                   avg_deg=None, scale=-1, rule="exp", max_proba=-1,
                   shape=None, positions=None, directed=True, multigraph=False,
                   distance=None, **kwargs):
    '''
    Returns a distance-rule graph
    '''
    distance = [] if distance is None else distance
    edges_hash = set()
    recip_hash = None if directed else set()

    # compute the required values
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)

    num_source, num_target = len(source_ids), len(target_ids)
    num_edges = 0

    if max_proba <= 0:
        num_edges, _ = _compute_connections(
            num_source, num_target, density, edges, avg_deg, directed,
            reciprocity=-1)

    b_one_pop = _check_num_edges(
        source_ids, target_ids, num_edges, directed, multigraph)

    num_neurons = len(set(np.concatenate((source_ids, target_ids))))

    # for each node, check the neighbours that are in an area where
    # connections can be made: +/- scale for lin, +/- 10*scale for exp.
    # Get the sources and associated targets for each MPI process
    list_targets = []
    lim = scale if rule == 'lin' else 10*scale

    for s in source_ids:
        keep  = (np.abs(positions[0, target_ids] - positions[0, s]) < lim)
        keep *= (np.abs(positions[1, target_ids] - positions[1, s]) < lim)
        if b_one_pop:
            idx = np.where(target_ids == s)[0][0]
            keep[idx] = 0
        list_targets.append(target_ids[keep])

    # the number of trials should be done depending on the number of
    # neighbours that each node has, so compute this number
    tot_neighbours = 0
    neigh_norm     = 1.

    for tgt_list in list_targets:
        tot_neighbours += len(tgt_list)

    if max_proba <= 0:
        assert tot_neighbours > num_edges, \
            "Scale is too small: there are not enough close neighbours to " +\
            "create the required number of connections. Increase `scale` " +\
            "or `neuron_density`."

        neigh_norm = 1. / tot_neighbours

    # try to create edges until num_edges is attained
    ia_edges = None
    num_ecurrent = 0

    if max_proba <= 0:
        ia_edges = np.full((num_edges, 2), -1, dtype=int)
        while num_ecurrent < num_edges:
            trials = []
            for tgt_list in list_targets:
                trials.append(max(
                    int(len(tgt_list)*(num_edges - num_ecurrent)*neigh_norm),
                    1))
            edges_tmp = [[], []]
            dist = []
            dist_tmp = []
            total_trials = int(np.sum(trials))
            local_targets = []
            local_sources = []
            current_pos = 0
            for s, tgts, num_try in zip(source_ids, list_targets, trials):
                if len(tgts):
                    t = np.random.randint(0, len(tgts), num_try)
                    local_targets.extend(tgts[t])
                    local_sources.extend((s for _ in range(num_try)))
            local_targets = np.array(local_targets, dtype=int)
            local_sources = np.array(local_sources, dtype=int)
            test = dist_rule(rule, scale, positions[:, local_sources],
                             positions[:, local_targets], dist=dist_tmp)
            test = np.greater(test, np.random.uniform(size=len(test)))
            edges_tmp[0].extend(local_sources[test])
            edges_tmp[1].extend(local_targets[test])
            dist = np.array(dist_tmp)[test]

            edges_tmp = np.array(edges_tmp).T

            # assess the current number of edges
            # if we're at the end, we'll make too many edges, so we keep only
            # the necessary fraction that we pick randomly
            num_desired = num_edges - num_ecurrent
            if num_desired < len(edges_tmp):
                chosen = {}
                while len(chosen) != num_desired:
                    idx = np.random.randint(
                        0, len(edges_tmp), num_desired - len(chosen))
                    for i in idx:
                        chosen[i] = None
                idx = list(chosen.keys())
                edges_tmp = edges_tmp[idx]
                dist = dist[idx]

            ia_edges, num_ecurrent = _filter(
                ia_edges, edges_tmp, num_ecurrent, edges_hash, b_one_pop,
                multigraph, directed=directed, recip_hash=recip_hash,
                distance=distance, dist_tmp=dist)
    else:
        sources, targets = [], []
        for i, s in enumerate(source_ids):
            local_tgts = np.array(list_targets[i], dtype=int)
            if len(local_tgts):
                dist_tmp = []
                test = max_proba_dist_rule(
                    rule, scale, max_proba, positions[:, s],
                    positions[:, local_tgts], dist=dist_tmp)
                test = np.greater(test, np.random.uniform(size=len(test)))
                added = np.sum(test)
                sources.extend((s for _ in range(added)))
                targets.extend(local_tgts[test])
                distance.extend(np.array(dist_tmp)[test])
        ia_edges = np.array([sources, targets]).T

    return ia_edges
