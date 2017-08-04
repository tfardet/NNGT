#!/usr/bin/env python
#-*- coding:utf-8 -*-

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


MAXTESTS = 1000 # ensure that generation will finish
EPS = 1e-5


def _distance_rule(source_ids, target_ids, density, edges, avg_deg, scale,
                   rule, shape, positions, conversion_factor, directed,
                   multigraph, **kwargs):
    '''
    Returns a distance-rule graph
    '''
    # mpi-related stuff
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # computations
    def exp_rule(pos_src, pos_target):
        dist = np.linalg.norm(pos_src-pos_target,axis=0)
        return np.exp(np.divide(dist, -scale))
    def lin_rule(pos_src, pos_target):
        dist = np.linalg.norm(pos_src-pos_target,axis=0)
        return np.divide(scale-dist, scale).clip(min=0.)
    dist_test = exp_rule if rule == "exp" else lin_rule
    # compute the required values
    source_ids = np.array(source_ids).astype(int)
    target_ids = np.array(target_ids).astype(int)
    num_source, num_target = len(source_ids), len(target_ids)
    num_edges, _ = _compute_connections(num_source, num_target,
                             density, edges, avg_deg, directed, reciprocity=-1)
    b_one_pop = _check_num_edges(
        source_ids, target_ids, num_edges, directed, multigraph)
    num_neurons = len(set(np.concatenate((source_ids, target_ids))))

    # Random number generation seeding
    if rank == 0:
        msd = np.random.randint(0, num_edges + 1)
    else:
        msd = None
    msd = comm.bcast(msd, root=0)
    seed = msd + rank + 1
    np.random.seed(seed)

    # for each node, check the neighbours that are in an area where
    # connections can be made: +/- scale for lin, +/- 5*scale for exp.
    # Get the sources and associated targets for each MPI process
    sources = []
    targets = []
    lim = scale if rule == 'lin' else 5*scale
    for s in source_ids[rank::size]:
        keep  = (np.abs(positions[0, target_ids] - positions[0, s]) < lim)
        keep *= (np.abs(positions[1, target_ids] - positions[1, s]) < lim)
        sources.append(s)
        targets.append(target_ids[keep])

    # the number of trials should be done depending on the number of
    # neighbours that each node has, so compute this number
    tot_neighbours = 0

    for tgt_list in targets:
        tot_neighbours += len(tgt_list)

    tot_neighbours = comm.scatter(tot_neighbours, root=0)
    if rank == 0:
        tot_neighbours = np.sum(tot_neighbours)
        assert tot_neighbours > target_enum, \
            "Scale is too small: there are not enough close neighbours to " +\
            "create the required number of connections. Increase `scale` " +\
            "or `neuron_density`."
    tot_neighbours = comm.bcast(tot_neighbours, root=0)
    norm = 1. / tot_neighbours

    # try to create edges until num_edges is attained
    if rank == 0:
        ia_edges = np.zeros((max_create, 2), dtype=int)
    else:
        ia_edges = None
    num_ecurrent = 0
    while num_ecurrent < num_edges:
        trials = []
        for tgt_list in targets:
            trials.append(
                int(len(tgt_list)*(num_edges - current_edges)*norm) + 1)
        edges_tmp = [[], []]
        for s, tgts, num_try in zip(sources, targets, trials):
            # try to create edges
            t = np.random.randint(0, len(tgts), num_try)
            test = dist_test(positions[:, s], positions[:, tgts[t]])
            test = np.greater(test, np.random.uniform(size=num_create))
            edges_tmp[0].extend(np.full(num_try, s)[test])
            edges_tmp[1].extend(tgts[t][test])

        # gather the result in root and assess the current number of edges
        edges_tmp = comm.gather(edges_tmp, root=0)
        if rank == 0:
            ia_edges_tmp = np.concatenate(edges_tmp, axis=1).T
            # if we're at the end, we'll make too many edges, so we keep only
            # the necessary fraction that we pick randomly
            if num_edges - num_ecurrent < len(ia_edges_tmp):
                np.random.shuffle(ia_edges_tmp)
                ia_edges_tmp = ia_edges_tmp[:num_edges - num_ecurrent]
            ia_edges, num_ecurrent = _filter(
                ia_edges, ia_edges_tmp, num_ecurrent, b_one_pop, multigraph)
        num_ecurrent = comm.bcast(num_ecurrent, root=0)

    # make sure everyone gets same seed back
    if rank == 0:
        msd = np.random.randint(0, num_edges + 1)
    msd = comm.bcast(msd, root=0)

    return ia_edges
