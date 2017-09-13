#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
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
    "price_network",
]


def _distance_rule(source_ids, target_ids, density=-1, edges=-1, avg_deg=-1,
                   scale=-1, rule="exp", shape=None, positions=None,
                   directed=True, multigraph=False, distance=None, **kwargs):
    '''
    Returns a distance-rule graph
    '''
    distance = [] if distance is None else distance
    # mpi-related stuff
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

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

    # Random number generation seeding
    if rank == 0:
        msd = nngt.get_config('msd')
    else:
        msd = None
    msd   = comm.bcast(msd, root=0)
    seeds = nngt.get_config('seeds')
    seed  = seeds[rank] if seeds is not None else msd + rank + 1
    np.random.seed(seed)

    # for each node, check the neighbours that are in an area where
    # connections can be made: +/- scale for lin, +/- 10*scale for exp.
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

    # the number of trials should be done depending on the number of
    # neighbours that each node has, so compute this number
    tot_neighbours = 0

    for tgt_list in targets:
        tot_neighbours += len(tgt_list)

    comm.Barrier()

    tot_neighbours = comm.gather(tot_neighbours, root=0)
    final_tot = None
    if rank == 0:
        print(type(tot_neighbours))
        final_tot = np.sum(tot_neighbours)
        assert final_tot > num_edges, \
            "Scale is too small: there are not enough close neighbours to " +\
            "create the required number of connections. Increase `scale` " +\
            "or `neuron_density`."
    final_tot = comm.bcast(final_tot, root=0)
    norm = 1. / final_tot

    comm.Barrier()

    # try to create edges until num_edges is attained
    if rank == 0:
        ia_edges = np.zeros((num_edges, 2), dtype=int)
    else:
        ia_edges = None
    num_ecurrent = 0
    while num_ecurrent < num_edges:
        trials = []
        for tgt_list in targets:
            trials.append(
                int(len(tgt_list)*(num_edges - num_ecurrent)*norm) + 1)
        edges_tmp = [[], []]
        dist_local = []
        for s, tgts, num_try in zip(sources, targets, trials):
            # try to create edges
            dist_tmp = []
            t = np.random.randint(0, len(tgts), num_try)
            test = dist_rule(rule, positions[:, s], positions[:, tgts[t]],
                             scale, dist=dist_tmp)
            test = np.greater(test, np.random.uniform(size=num_try))
            edges_tmp[0].extend(np.full(num_try, s)[test])
            edges_tmp[1].extend(tgts[t][test])
            dist_local.extend(np.array(dist_tmp)[test])

        comm.Barrier()

        # gather the result in root and assess the current number of edges
        edges_tmp  = comm.gather(edges_tmp, root=0)
        dist_local = comm.gather(dist_local, root=0)

        if rank == 0:
            ia_edges_tmp = np.concatenate(edges_tmp, axis=1).T
            dist_local = np.concatenate(dist_local)
            # if we're at the end, we'll make too many edges, so we keep only
            # the necessary fraction that we pick randomly
            if num_edges - num_ecurrent < len(ia_edges_tmp):
                np.random.shuffle(ia_edges_tmp)
                ia_edges_tmp = ia_edges_tmp[:num_edges - num_ecurrent]
            ia_edges, num_ecurrent = _filter(
                ia_edges, ia_edges_tmp, num_ecurrent, b_one_pop, multigraph,
                distance=distance, dist_tmp=dist_local)
        num_ecurrent = comm.bcast(num_ecurrent, root=0)

    # make sure everyone gets same seed back
    if rank == 0:
        new_seed = np.random.randint(0, num_edges + 1)
    else:
        new_seed = None
    new_seed = comm.bcast(new_seed, root=0)
    np.random.seed(new_seed)

    return ia_edges
