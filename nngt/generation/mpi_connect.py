#-*- coding:utf-8 -*-
#
# generation/mpi_connect.py
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

""" Generation tools for NNGT using MPI """

import warnings
import numpy as np
import scipy.sparse as ssp
from scipy.spatial.distance import cdist
from numpy.random import randint

from mpi4py import MPI

import nngt
from nngt.lib import InvalidArgument
from nngt.lib.connect_tools import *
from . import connect_algorithms
from .connect_algorithms import *


__all__ = connect_algorithms.__all__


def _from_degree_list(source_ids, target_ids, degrees, degree_type="in",
                      directed=True, multigraph=False, existing_edges=None,
                      **kwargs):
    ''' Connect nodes from a list of degrees '''
    if not directed:
        raise NotImplementedError("This function is not yet implemented for "
                                  "undirected graphs in MPI.")

    # type of degree
    degree_type = _set_degree_type(degree_type)

    b_out = (degree_type == "out")
    b_total = (degree_type == "total")

    if b_total:
        raise NotImplementedError(
            "Total degree is not supported yet with MPI.")

    # mpi-related stuff
    comm, size, rank = _mpi_and_random_init()

    # use only local sources and degrees unless already_local=True
    if not kwargs.get("already_local", False):
        source_ids = np.array(source_ids, dtype=int)[rank::size]

        if rank != 0:
            degrees = None

        degrees = comm.bcast(degrees, root=0)
        degrees = degrees[rank::size]

    source_ids = np.array(source_ids, dtype=int)
    target_ids = np.array(target_ids, dtype=int)

    num_source = len(source_ids)

    # compute the local number of edges
    edges = np.sum(degrees)

    b_one_pop = _check_num_edges(
        source_ids, target_ids, edges, directed, multigraph)

    num_etotal = 0
    ia_edges   = np.zeros((edges, 2), dtype=int)
    idx        = 0 if b_out else 1  # differenciate source / target
    max_degree = np.inf if multigraph else len(target_ids)

    for i, v in enumerate(source_ids):
        degree_i = degrees[i]
        edges_i, ecurrent, variables_i = np.zeros((degree_i, 2)), 0, []

        if existing_edges is not None:
            with_v = np.where(existing_edges[:, idx] == v)
            variables_i.extend(existing_edges[with_v, 1 - idx])
            degree_i += len(variables_i)
            assert degree_i < max_degree, "Required degree is greater that " +\
                "maximum possible degree {}.".format(max_degree)

        rm = np.where(target_ids == v)[0]
        rm = rm[0] if len(rm) else -1
        var_tmp = (np.array(target_ids, copy=True) if rm == -1 else
                   np.concatenate((target_ids[:rm], target_ids[rm+1:])))

        num_var_i = len(var_tmp)
        ia_edges[num_etotal:num_etotal + degree_i, idx] = v

        while len(variables_i) != degree_i:
            variables_i.extend(np.random.choice(var_tmp, degree_i-ecurrent,
                                                replace=multigraph))

            if not multigraph:
                variables_i = list(set(variables_i))

            ecurrent = len(variables_i)

        ia_edges[num_etotal:num_etotal+ecurrent, int(not idx)] = variables_i
        num_etotal += ecurrent

    comm.Barrier()

    _finalize_random(rank)

    # the 'nngt' backend is made to be distributed, but the others are not
    if nngt.get_config("backend") == "nngt":
        return ia_edges
    else:
        # all the data is gather on the root processus
        ia_edges  = comm.gather(ia_edges, root=0)
        if rank == 0:
            ia_edges = np.concatenate(ia_edges, axis=0)
            return ia_edges
        else:
            return None


def _fixed_degree(source_ids, target_ids, degree, degree_type="in",
                  reciprocity=-1, directed=True, multigraph=False,
                  existing_edges=None, **kwargs):
    ''' Connect nodes with a delta distribution '''
    # mpi-related stuff (finalized in _from_degree_list)
    comm, size, rank = _mpi_and_random_init()

    assert degree >= 0, "A positive value is required for `degree`."

    # use only local sources
    source_ids = np.array(source_ids, dtype=int)[rank::size]
    target_ids = np.array(target_ids, dtype=int)
        
    num_source = len(source_ids)

    # compute the local number of edges
    lst_deg = np.full(num_source, degree, dtype=int)

    # !IMPORTANT! use `already_local` to tell _from_degree_list that only
    # local sources and degrees have been sent
    return _from_degree_list(
        source_ids, target_ids, lst_deg, degree_type=degree_type,
        directed=directed, multigraph=multigraph,
        existing_edges=existing_edges, already_local=True, **kwargs)


def _gaussian_degree(source_ids, target_ids, avg=-1, std=-1, degree_type="in",
                     reciprocity=-1, directed=True, multigraph=False,
                     existing_edges=None, **kwargs):
    ''' Connect nodes with a Gaussian distribution '''
    # mpi-related stuff (finalized in _from_degree_list)
    comm, size, rank = _mpi_and_random_init()

    # switch values to float
    avg = float(avg)
    std = float(std)

    assert avg >= 0, "A positive value is required for `avg`."
    assert std >= 0, "A positive value is required for `std`."

    # use only local sources
    source_ids = np.array(source_ids, dtype=int)[rank::size]
    target_ids = np.array(target_ids, dtype=int)
        
    num_source = len(source_ids)

    # compute the local number of edges
    lst_deg = np.around(
        np.maximum(np.random.normal(avg, std, num_source), 0.)).astype(int)

    # !IMPORTANT! use `already_local` to tell _from_degree_list that only
    # local sources and degrees have been sent
    return _from_degree_list(
        source_ids, target_ids, lst_deg, degree_type=degree_type,
        directed=directed, multigraph=multigraph,
        existing_edges=existing_edges, already_local=True, **kwargs)


def _distance_rule(source_ids, target_ids, density=-1, edges=-1, avg_deg=-1,
                   scale=-1, rule="exp", max_proba=-1., shape=None,
                   positions=None, directed=True, multigraph=False,
                   distance=None, **kwargs):
    '''
    Returns a distance-rule graph
    '''
    assert max_proba <= 0, "MPI distance_rule cannot use `max_proba` yet."

    distance     = [] if distance is None else distance
    distance_tmp = []
    edges_hash   = set()

    # mpi-related stuff
    comm, size, rank = _mpi_and_random_init()

    # compute the required values
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
    num_source, num_target = len(source_ids), len(target_ids)
    num_edges, _ = _compute_connections(
        num_source, num_target, density, edges, avg_deg, directed,
        reciprocity=-1)
    b_one_pop = _check_num_edges(
        source_ids, target_ids, num_edges, directed, multigraph)
    num_neurons = len(set(np.concatenate((source_ids, target_ids))))

    # for each node, check the neighbours that are in an area where
    # connections can be made: ± scale for lin, ± 10*scale for exp.
    # Get the sources and associated targets for each MPI process
    sources = []
    targets = []
    lim = scale if rule == 'lin' else 10*scale
    for s in source_ids[rank::size]:
        keep  = (np.abs(positions[0, target_ids] - positions[0, s]) < lim)
        keep *= (np.abs(positions[1, target_ids] - positions[1, s]) < lim)
        if b_one_pop:
            keep[s] = 0
        sources.append(s)
        targets.append(target_ids[keep])

    # the number of trials should be done depending on total number of
    # neighbours available, so we compute this number
    local_neighbours = 0

    for tgt_list in targets:
        local_neighbours += len(tgt_list)

    tot_neighbours = comm.gather(local_neighbours, root=0)
    if rank == 0:
        final_tot = np.sum(tot_neighbours)
        assert final_tot > num_edges, \
            "Scale is too small: there are not enough close neighbours to " +\
            "create the required number of connections. Increase `scale` " +\
            "or `neuron_density`."
    else:
        final_tot = None
    final_tot = comm.bcast(final_tot, root=0)

    neigh_norm = 1. / final_tot

    # try to create edges until num_edges is attained
    if rank == 0:
        ia_edges = np.zeros((num_edges, 2), dtype=int)
    else:
        ia_edges = None
    num_ecurrent = 0

    while num_ecurrent < num_edges:
        trials = []
        for tgt_list in targets:
            trials.append(max(
                int(len(tgt_list)*(num_edges - num_ecurrent)*neigh_norm), 1))
        # try to create edges
        edges_tmp = [[], []]
        dist_local = []
        total_trials = int(np.sum(trials))
        local_sources = np.repeat(sources, trials)
        local_targets = np.zeros(total_trials, dtype=int)
        current_pos = 0
        for tgts, num_try in zip(targets, trials):
            t = np.random.randint(0, len(tgts), num_try)
            local_targets[current_pos:current_pos + num_try] = tgts[t]
            current_pos += num_try
        test = dist_rule(rule, scale, positions[:, local_sources],
                         positions[:, local_targets], dist=dist_local)
        test = np.greater(test, np.random.uniform(size=total_trials))
        edges_tmp[0].extend(local_sources[test])
        edges_tmp[1].extend(local_targets[test])
        dist_local = np.array(dist_local)[test]

        comm.Barrier()

        # gather the result in root and assess the current number of edges
        edges_tmp  = comm.gather(edges_tmp, root=0)
        dist_local = comm.gather(dist_local, root=0)

        if rank == 0:
            edges_tmp = np.concatenate(edges_tmp, axis=1).T
            dist_local = np.concatenate(dist_local)

            # if we're at the end, we'll make too many edges, so we keep
            # only the necessary fraction that we pick randomly
            num_desired = num_edges - num_ecurrent
            num_tmp     = len(edges_tmp)

            if num_desired < num_tmp:
                chosen = np.random.choice(num_tmp, num_desired,
                                          replace=multigraph)
                edges_tmp = edges_tmp[chosen]
                dist_local = np.array(dist_local)[chosen]

            ia_edges, num_ecurrent = _filter(
                ia_edges, edges_tmp, num_ecurrent, edges_hash, b_one_pop,
                multigraph, distance=distance_tmp, dist_tmp=dist_local)

        num_ecurrent = comm.bcast(num_ecurrent, root=0)

        comm.Barrier()

    _finalize_random(rank)

    # the 'nngt' backend is made to be distributed, but the others are not
    if nngt.get_config("backend") == "nngt":
        local_edges = None
        if rank == 0:
            local_edges  = [ia_edges[i::size, :] for i in range(size)]
            distance_tmp = [distance_tmp[i::size] for i in range(size)]
        local_edges  = comm.scatter(local_edges, root=0)
        distance_tmp = comm.scatter(distance_tmp, root=0)
        distance.extend(distance_tmp)
        return local_edges
    else:
        # all the data is gather on the root processus
        if rank == 0:
            distance.extend(distance_tmp)
            return ia_edges
        else:
            return None


# --------------------- #
# Unavailable functions #
# --------------------- #

def _not_yet(*args, **kwargs):
    raise NotImplementedError("Not available with MPI yet.")

_circular = _not_yet
_erdos_renyi = _not_yet
_newman_watts = _not_yet
_price_scale_free = _not_yet
_random_scale_free = _not_yet
_unique_rows = _not_yet


# ----- #
# Tools #
# ----- #

def _mpi_and_random_init():
    '''
    Init MPI comm and information and seed the RNGs
    '''
    comm = MPI.COMM_WORLD.Clone()
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Random number generation seeding
    seeds = None

    if not nngt._seeded_local:
        # no local seeds were generated, set them from initial msd
        msd   = nngt.get_config("msd")
        seeds = [msd + i + 1 for i in range(size)]
        nngt._config['seeds'] = seeds
    elif not nngt._used_local:
        # local seeds were generated but not used, use them
        seeds = nngt.get_config('seeds')
    else:
        # local seeds were generated and used, generate new ones from new msd
        if rank == 0:
            msd = np.random.randint(0, 2**31 - size - 1)
        else:
            msd = None

        msd   = comm.bcast(msd, root=0)
        seeds = [msd + i + 1 for i in range(size)]

    seed  = seeds[rank]
    np.random.seed(seed)

    nngt._seeded_local = True
    nngt._used_local   = True

    return comm, size, rank


def _finalize_random(rank):
    '''
    Make sure everyone gets same seed back.
    '''
    comm = MPI.COMM_WORLD

    if rank == 0:
        new_seed = np.random.randint(0, 2**31 - 1)
    else:
        new_seed = None

    new_seed = comm.bcast(new_seed, root=0)

    np.random.seed(new_seed)
