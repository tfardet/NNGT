#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Generation tools for NNGT """

import logging
import numpy as np
import scipy.sparse as ssp
from scipy.spatial.distance import cdist
from numpy.random import randint

import nngt
from nngt.lib import InvalidArgument
from nngt.lib.connect_tools import *


__all__ = [
    "_all_to_all",
    "_distance_rule",
    "_erdos_renyi",
    "_fixed_degree",
    "_from_degree_list",
    "_gaussian_degree",
    "_newman_watts",
    "_price_scale_free",
    "_random_scale_free",
    "_unique_rows",
    "price_network",
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
    if common:
        num_edges     = num_sources*num_targets - len(common)
        edges         = np.empty((num_edges, 2))
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
        edges       = np.empty((num_sources*num_targets, 2))
        edges[:, 0] = np.repeat(source_ids, num_targets)
        edges[:, 1] = np.tile(target_ids, num_sources)

    if distance is not None:
        pos       = kwargs['positions']
        x, y      = pos[0], pos[1]
        vectors   = np.array((x[edges[:, 1]] - x[edges[:, 0]],
                              y[edges[:, 1]] - y[edges[:, 0]]))
        distance.extend(np.linalg.norm(vectors, axis=0))

    return edges


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

    if b_total:
        raise NotImplementedError("Total degree is not supported yet.")

    if not directed:
        raise NotImplementedError("This function is not yet implemented for "
                                  "undirected graphs.")

    edges = np.sum(degree_list)
    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)

    num_etotal = 0 if existing_edges is None else existing_edges.shape[0]

    ia_edges = np.zeros((num_etotal + edges, 2), dtype=int)

    if num_etotal:
        ia_edges[:num_etotal,:] = existing_edges

    idx = 0 if b_out else 1  # differenciate source / target

    for i, v in enumerate(source_ids):
        degree_i = degree_list[i]
        edges_i, ecurrent, variables_i = np.zeros((degree_i, 2)), 0, []

        if existing_edges is not None:
            with_v = np.where(ia_edge[:,idx] == v)
            variables_i.extend(ia_edge[with_v:int(not idx)])
            ecurrent = len(variables_i)

        rm = np.where(target_ids == v)[0]
        rm = rm[0] if len(rm) else -1
        var_tmp = (target_ids if rm == -1 else
                   np.concatenate((target_ids[:rm], target_ids[rm+1:])))

        num_var_i = len(var_tmp)
        ia_edges[num_etotal:num_etotal+degree_i, idx] = v

        while len(variables_i) != degree_i:
            var = np.random.choice(var_tmp, degree_i-ecurrent,
                                   replace=multigraph)
            variables_i.extend(var)

            if not multigraph:
                variables_i = list(set(variables_i))

            ecurrent = len(variables_i)

        ia_edges[num_etotal:num_etotal+ecurrent, 1 - idx] = variables_i
        num_etotal += ecurrent

    return ia_edges


def _fixed_degree(source_ids, target_ids, degree=-1, degree_type="in",
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


def _gaussian_degree(source_ids, target_ids, avg=-1, std=-1, degree_type="in",
                     directed=True, multigraph=False, existing_edges=None,
                     **kwargs):
    ''' Connect nodes with a Gaussian distribution '''
    # switch values to float
    avg = float(avg)
    std = float(std)

    assert avg >= 0, "A positive value is required for `avg`."
    assert std >= 0, "A positive value is required for `std`."

    num_source = len(source_ids)

    # edges (we set the in, out, or total degree of the source neurons)
    lst_deg = np.around(
        np.maximum(np.random.normal(avg, std, num_source), 0.)).astype(int)

    return _from_degree_list(
        source_ids, target_ids, lst_deg, degree_type=degree_type,
        directed=directed, multigraph=multigraph,
        existing_edges=existing_edges, **kwargs)


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
    edges_hash = set()

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
    ia_sources = np.repeat(source_ids, ia_out_deg)
    ia_targets = np.repeat(target_ids, ia_in_deg)
    np.random.shuffle(ia_targets)
    ia_edges_tmp = np.array([ia_sources,ia_targets]).T
    ia_edges, num_ecurrent = _filter(ia_edges, ia_edges_tmp, num_ecurrent,
                                     edges_hash, b_one_pop, multigraph)

    while num_ecurrent != pre_recip_edges and num_test < MAXTESTS:
        num_desired = pre_recip_edges-num_ecurrent
        ia_sources_tmp = np.random.choice(ia_sources, num_desired)
        ia_targets_tmp = np.random.choice(ia_targets, num_desired)
        ia_edges_tmp = np.array([ia_sources_tmp, ia_targets_tmp]).T
        ia_edges, num_ecurrent = _filter(ia_edges, ia_edges_tmp, num_ecurrent,
                                         edges_hash, b_one_pop, multigraph)
        num_test += 1

    if directed and reciprocity > 0:
        while num_ecurrent != edges and num_test < MAXTESTS:
            keep = np.random.choice(num_ecurrent, edges-num_ecurrent,
                                    replace=multigraph)
            ia_edges[num_ecurrent:] = ia_edges[keep]
            num_ecurrent = edges
            if not multigraph:
                ia_edges_tmp = _unique_rows(ia_edges)
                num_ecurrent = ia_edges_tmp.shape[0]
                ia_edges[:num_ecurrent,:] = ia_edges_tmp
            num_test += 1

    return ia_edges


def _erdos_renyi(source_ids, target_ids, density=-1, edges=-1, avg_deg=-1,
                 reciprocity=-1, directed=True, multigraph=False, **kwargs):
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
    edges_hash = set()

    while num_ecurrent != pre_recip_edges and num_test < MAXTESTS:
        ia_sources = source_ids[randint(0, num_source,
                                        pre_recip_edges-num_ecurrent)]
        ia_targets = target_ids[randint(0, num_target,
                                        pre_recip_edges-num_ecurrent)]
        ia_edges_tmp = np.array([ia_sources,ia_targets]).T
        ia_edges, num_ecurrent = _filter(ia_edges, ia_edges_tmp, num_ecurrent,
                                         edges_hash, b_one_pop, multigraph)
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


def _price_scale_free():
    pass


def _circular_directed_recip(node_ids, coord_nb, reciprocity):
    ''' Circular graph with given reciprocity '''
    nodes    = len(node_ids)
    edges    = int(0.5*nodes*coord_nb*(1 + reciprocity))
    init_deg = int(0.5*coord_nb)

    # sources and targets
    sources = np.zeros(edges, dtype=int)
    targets = np.zeros(edges, dtype=int)

    # set non-reciprocal edges using the full undirected circular graph
    init_edges = _circular_full(node_ids, coord_nb, False)
    num_init   = len(init_edges)

    sources[:num_init] = init_edges[:, 0]
    targets[:num_init] = init_edges[:, 1]

    # then we randomize the direction of these E_init edges
    # this is equivalent to reversing E edges with E from Binom(E_init, 0.5)
    rng = np.random.default_rng()
    E   = rng.binomial(num_init, 0.5)

    chosen = rng.choice(num_init, E, replace=False)

    sources[chosen], targets[chosen] = targets[chosen], sources[chosen]

    # set reciprocal edges
    num_recip = edges - nodes*init_deg

    if num_recip:
        chosen = rng.choice(
            [i for i in range(nodes*init_deg)], num_recip, replace=False)

        sources[-num_recip:] = targets[chosen]
        targets[-num_recip:] = sources[chosen]

    return np.array([sources, targets], dtype=int).T


def _circular_full(node_ids, coord_nb, directed):
    ''' Create a circular graph with all possible edges '''
    nodes = len(node_ids)

    dist = int(0.5*coord_nb)

    out_deg = coord_nb if directed else dist

    sources = np.repeat(np.arange(0, nodes).astype(int), out_deg)

    # create the connection mask
    start = -dist if directed else 0
    stop  = dist + 1

    conn_mask = np.concatenate((np.arange(start, 0), np.arange(1, stop)))

    # make the targets and put them back into [0, nodes - 1]
    targets   = np.tile(conn_mask, nodes) + sources

    targets[targets < 0] += nodes
    targets[targets >= nodes] -= nodes

    return np.array((sources, targets), dtype=int).T


def _newman_watts(node_ids, coord_nb, proba_shortcut, reciprocity_circular,
                  directed=True, multigraph=False, **kwargs):
    '''
    Returns a numpy array of dimension (num_edges,2) that describes the edge
    list of a Newman-Watts graph.
    '''
    nodes      = len(node_ids)
    source_ids = np.array(node_ids, dtype=int)
    target_ids = np.array(node_ids, dtype=int)

    # check the number of edges
    direct_factor  = 0.5*(1 + directed)
    recip_factor   = 0.5*(1 + reciprocity_circular)
    circular_edges = int(nodes * coord_nb * recip_factor * direct_factor)

    num_edges = int(circular_edges*(1 + proba_shortcut))

    b_one_pop = _check_num_edges(
        source_ids, target_ids, num_edges, directed, multigraph)

    if not b_one_pop:
        raise InvalidArgument("This graph model can only be used if source "
                              "and target populations are the same.")

    # generate the initial circular graph
    ia_edges = np.zeros((num_edges, 2), dtype=int)

    if reciprocity_circular == 1 or not directed:
        ia_edges[:circular_edges, :] = _circular_full(
            node_ids, coord_nb, directed)
    elif directed:
        ia_edges[:circular_edges, :] = _circular_directed_recip(
            node_ids, coord_nb, reciprocity_circular)
    else:
        raise ValueError("`reciprocity_circular` is 1 by definition for an "
                         "undirected Newman-Watts graph.")

    # add the random connections
    num_test, num_ecurrent = 0, circular_edges
    edges_hash = set(tuple(e) for e in ia_edges[:circular_edges])

    rng = np.random.default_rng()

    while num_ecurrent != num_edges and num_test < MAXTESTS:
        todo   = num_edges - num_ecurrent
        chosen = rng.choice(node_ids, int(2*todo))

        ia_edges, num_ecurrent = _filter(
            ia_edges, chosen.reshape(todo, 2), num_ecurrent, edges_hash,
            b_one_pop, multigraph)

        num_test += 1

    ia_edges = _no_self_loops(ia_edges)

    return ia_edges


def _distance_rule(source_ids, target_ids, density=-1, edges=-1, avg_deg=-1,
                   scale=-1, rule="exp", max_proba=-1, shape=None,
                   positions=None, directed=True, multigraph=False,
                   distance=None, **kwargs):
    '''
    Returns a distance-rule graph
    '''
    distance = [] if distance is None else distance
    edges_hash = set()
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
        ia_edges = np.zeros((num_edges, 2), dtype=int)
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
                multigraph, distance=distance, dist_tmp=dist)
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


def price_network(*args, **kwargs):
    #@todo: do it for other libraries
    raise NotImplementedError("Not implemented except for graph-tool.")


if nngt.get_config("backend") == "graph-tool":
    from graph_tool.generation import price_network as _pn

    def price_network(*args, **kwargs):
        weighted   = kwargs.get("weighted", True)
        directed   = kwargs.get("directed", True)
        population = kwargs.get("population", None)
        shape      = kwargs.get("shape", None)
        for k in ("weighted", "directed", "population", "shape"):
            try:
                del kwargs[k]
            except KeyError:
                pass
        g = _pn(*args, **kwargs)
        return Graph.from_library(
            g, weighted=weighted, directed=directed, population=population,
            shape=shape, **kwargs)
