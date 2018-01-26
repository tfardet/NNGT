#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

import numpy as np
import scipy.sparse as ssp
from scipy.spatial.distance import cdist
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
    "dist_rule",
    "max_proba_dist_rule"
]


def _set_options(graph, population, shape, positions):
    #~ if issubclass(graph.__class__, nngt.Network):
        #~ Connections.delays(graph)
    if population is not None:
        nngt.Graph.make_network(graph, population)
    if shape is not None:
        nngt.Graph.make_spatial(graph, shape, positions)


def _compute_connections(num_source, num_target, density, edges, avg_deg,
                         directed, reciprocity=-1):
    assert not np.allclose((density, edges, avg_deg), -1.), "At leat one " +\
        "of the following entries must be specified: 'density', 'edges', " +\
        "'avg_deg'."
    pre_recip_edges = 0
    if avg_deg > 0:
        pre_recip_edges = int(avg_deg * num_source)
    elif edges > 0:
        pre_recip_edges = int(edges)
    else:
        pre_recip_edges = int(density * num_source * num_target)
    dens = pre_recip_edges / float(num_source * num_target)
    edges = pre_recip_edges
    if edges:
        if not directed:
            pre_recip_edges = edges = int(edges/2)
        elif reciprocity > max(0,(2.-1./dens)):
            frac_recip = ((reciprocity - 1.
                           + np.sqrt(1. + dens*(reciprocity - 2.))) /
                             (2. - reciprocity))
            if frac_recip < 1.:
                pre_recip_edges = int(edges/(1+frac_recip))
            else:
                raise InvalidArgument(
                    "Such reciprocity cannot be obtained, request ignored.")
        elif reciprocity > 0.:
            raise InvalidArgument(
                "Reciprocity cannot be lower than 2-1/density.")
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

def _unique_rows(arr, return_index=False):
    '''
    Keep only unique edges
    '''
    b = np.ascontiguousarray(arr).view(
        np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    b, idx = np.unique(b, return_index=True)
    unique = b.view(arr.dtype).reshape(-1, arr.shape[1]).astype(int)
    if return_index:
        return unique, idx
    return unique


def _no_self_loops(array, return_test=False):
    '''
    Remove self-loops
    '''
    test = array[:, 0] != array[:, 1]
    if return_test:
        return array[test, :].astype(int), test
    return array[test, :].astype(int)


#~ def _filter(ia_edges, ia_edges_tmp, num_ecurrent, b_one_pop, multigraph,
            #~ distance=None, dist_tmp=None):
    #~ '''
    #~ Filter the edges: remove self loops and multiple connections if the graph
    #~ is not a multigraph.
    #~ '''
    #~ if b_one_pop:
        #~ ia_edges_tmp = _no_self_loops(ia_edges_tmp)
    #~ num_added = ia_edges_tmp.shape[0]
    #~ ia_edges[num_ecurrent:num_ecurrent+num_added,:] = ia_edges_tmp
    #~ old_ecurrent  = num_ecurrent
    #~ num_ecurrent += num_added
    #~ if not multigraph:
        #~ ia_edges_tmp = None
        #~ if distance is not None:
            #~ # get indices to keep only remaining distances
            #~ ia_edges_tmp, idx = _unique_rows(
                #~ ia_edges[:num_ecurrent,:], return_index=True)
            #~ valid_idx = np.array(idx[old_ecurrent:num_ecurrent] - old_ecurrent,
                                 #~ dtype=int)
            #~ distance.extend(np.array(dist_tmp)[valid_idx])
        #~ else:
            #~ ia_edges_tmp = _unique_rows(ia_edges[:num_ecurrent,:])
        #~ num_ecurrent = ia_edges_tmp.shape[0]
        #~ ia_edges[:num_ecurrent,:] = ia_edges_tmp
    #~ return ia_edges, num_ecurrent


def _filter(ia_edges, ia_edges_tmp, num_ecurrent, edges_hash, b_one_pop,
            multigraph, distance=None, dist_tmp=None):
    '''
    Filter the edges: remove self loops and multiple connections if the graph
    is not a multigraph.
    '''
    if b_one_pop:
        ia_edges_tmp, test = _no_self_loops(ia_edges_tmp, return_test=True)
        if dist_tmp is not None:
            dist_tmp = dist_tmp[test]

    if not multigraph:
        num_ecurrent = len(edges_hash)
        if distance is not None:
            for e, d in zip(ia_edges_tmp, dist_tmp):
                tpl_e = tuple(e)
                if tpl_e not in edges_hash:
                    ia_edges[num_ecurrent, :] = e
                    distance.append(d)
                    edges_hash[tpl_e] = None
                    num_ecurrent += 1
        else:
            for e in ia_edges_tmp:
                tpl_e = tuple(e)
                if tpl_e not in edges_hash:
                    ia_edges[num_ecurrent, :] = e
                    edges_hash[tpl_e] = None
                    num_ecurrent += 1
    else:
        num_added = len(ia_edges_tmp)
        ia_edges[num_ecurrent:num_ecurrent + num_added, :] = ia_edges_tmp
        num_ecurrent += num_added
        if distance is not None:
            distance.extend(dist_tmp)
    return ia_edges, num_ecurrent


# ------------- #
# Distance rule #
# ------------- #

def dist_rule(rule, scale, pos_src, pos_targets, dist=None):
    '''
    DR test from one source to several targets

    Parameters
    ----------
    rule : str
        Either 'exp', 'gaussian', or 'lin'.
    scale : float
        Characteristic scale.
    pos_src : array of shape (2, N)
        Positions of the sources.
    pos_targets : array of shape (2, N)
        Positions of the targets.
    dist : list, optional (default: None)
        List that will be filled with the distances of the edges.

    Returns
    -------
    Array of size N giving the probability of the edges according to the rule.
    '''
    vect = pos_targets - pos_src
    origin = np.array([(0., 0.)])
    # todo correct this
    dist_tmp = np.squeeze(cdist(vect.T, origin), axis=1)
    if dist is not None:
        dist.extend(dist_tmp)
    if rule == 'exp':
        return np.exp(np.divide(dist_tmp, -scale))
    elif rule == 'gaussian':
        return np.exp(-0.5*np.square(np.divide(dist_tmp, scale)))
    elif rule == 'lin':
        return np.divide(scale - dist_tmp, scale).clip(min=0.)
    else:
        raise InvalidArgument('Unknown rule "' + rule + '".')


def max_proba_dist_rule(rule, scale, max_proba, pos_src, pos_targets,
                        dist=None):
    '''
    DR test from one source to several targets

    Parameters
    ----------
    rule : str
        Either 'exp', 'gaussian', or 'lin'.
    scale : float
        Characteristic scale.
    norm : float
        Normalization factor giving proba at zero distance.
    pos_src : 2-tuple
        Positions of the sources.
    pos_targets : array of shape (2, N)
        Positions of the targets.
    dist : list, optional (default: None)
        List that will be filled with the distances of the edges.

    Returns
    -------
    Array of size N giving the probability of the edges according to the rule.
    '''
    x, y = pos_src
    s = np.repeat([[x], [y]], pos_targets.shape[1], axis=1)
    vect = pos_targets - np.repeat([[x], [y]], pos_targets.shape[1], axis=1)
    origin = np.array([(0., 0.)])
    # todo correct this
    dist_tmp = np.squeeze(cdist(vect.T, origin), axis=1)
    if dist is not None:
        dist.extend(dist_tmp)
    if rule == 'exp':
        return max_proba*np.exp(np.divide(dist_tmp, -scale))
    elif rule == 'gaussian':
        return max_proba*np.exp(-0.5*np.square(np.divide(dist_tmp, scale)))
    elif rule == 'lin':
        return max_proba*np.divide(scale - dist_tmp, scale).clip(min=0.)
    else:
        raise InvalidArgument('Unknown rule "' + rule + '".')
