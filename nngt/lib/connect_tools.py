#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

import warnings
import numpy as np
import scipy.sparse as ssp
from numpy.random import randint

import nngt
from nngt.lib import InvalidArgument


__all__ = [
    "_check_num_edges",
    "_compute_connections",
    "_filter",
    "_no_self_loops",
    "_set_options",
    "_unique_rows",
]


def _set_options(graph, population, shape, positions):
    if issubclass(graph.__class__, nngt.Network):
        Connections.delays(graph)
    elif population is not None:
        nngt.Network.make_network(graph, population)
    if shape is not None:
        nngt.SpatialGraph.make_spatial(graph, shape, positions)


def _compute_connections(num_source, num_target, density, edges, avg_deg,
                         directed, reciprocity=-1):
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


def _check_num_edges(source_ids, target_ids, num_edges, directed, multigraph):
    num_source, num_target = len(source_ids), len(target_ids)
    has_only_one_population = (False if num_source != num_target
                               else not np.all(source_ids - target_ids))
    if not has_only_one_population and not multigraph:
        b_d = (num_edges > num_source*num_target)
        b_nd = (num_edges > int(0.5*num_source*num_target))
        if (not directed and b_nd) or (directed and b_d):
            raise InvalidArgument("Required number of edges is too high")
    elif has_only_one_population and not multigraph:
        b_d = (num_edges > num_source*(num_target-1))
        b_nd = (num_edges > int((0.5*num_source-1)*num_target))
        if (not directed and b_nd) or (directed and b_d):
            raise InvalidArgument("Required number of edges is too high")
    return has_only_one_population


# ------------------------- #
# Edge checks and filtering #
# ------------------------- #

def _unique_rows(arr):
    '''
    Keep only unique edges
    '''
    b = np.ascontiguousarray(arr).view(np.dtype((np.void,
        arr.dtype.itemsize * arr.shape[1])))
    return np.unique(b).view(arr.dtype).reshape(-1,arr.shape[1]).astype(int)


def _no_self_loops(array):
    '''
    Remove self-loops
    '''
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
