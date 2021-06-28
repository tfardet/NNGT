#-*- coding:utf-8 -*-
#
# lib/graph_helpers.py
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

from copy import deepcopy

import numpy as np

from .errors import InvalidArgument
from .rng_tools import _eprop_distribution
from .test_functions import nonstring_container, is_integer


""" Helper functions for graph classes """

def _edge_prop(prop):
    ''' Return edge property `name` as a distribution dict '''
    if is_integer(prop) or isinstance(prop, float):
        return {"distribution": "constant", "value": prop}
    elif isinstance(prop, dict):
        return prop.copy()
    elif nonstring_container(prop):
        return {'distribution': 'custom', 'values': prop}
    elif prop is None:
        return {'distribution': 'constant'}
    else:
        raise InvalidArgument("Edge property must be either a dict, a list, or"
                              " a number; received {}".format(type(prop)))


def _get_edge_attr(graph, elist, attribute, prop=None, last_edges=False):
    '''
    Returns the values associated to a given edge attribute.

    Parameters
    ----------
    graph : the graph
    elist : the edges (N, 2)
    attributes : str
        The name of the attribute to set.
    prop : dict or array-like, optional (default: None)
        Properties associated to the `attribute`.

    Returns
    -------
    attr : array-like
        Values of the `attribute`
    '''
    # check the weights
    if "weight" == attribute:
        weights = np.ones(len(elist))

        if graph.is_weighted():
            prop = graph._w if prop is None else _edge_prop(prop)

            params = {
                k: v for (k, v) in prop.items() if k != "distribution"
            }

            weights = _eprop_distribution(
                graph, prop["distribution"], elist=elist,
                last_edges=last_edges, **params)

        # if dealing with network, check inhibitory weight factor
        if graph.is_network() and not np.isclose(graph._iwf, 1.):
            keep = graph.node_attributes['type'][elist[:, 0]] < 0
            weights[keep] *= graph._iwf

        return weights

    # also check delays
    if "delay" == attribute:
        delays = np.ones(len(elist))
        if prop is None and hasattr(graph, "_d"):
            prop = graph._d
        elif prop is not None:
            prop = _edge_prop(prop)
        params = {
            k: v for (k, v) in prop.items() if k != "distribution"
        }
        delays = _eprop_distribution(
            graph, prop["distribution"], elist=elist,
            last_edges=last_edges, **params)

        return delays

    # take care of others
    prop = _edge_prop(prop)
    params = {k: v for (k, v) in prop.items() if k != "distribution"}
    return _eprop_distribution(
        graph, prop["distribution"], elist=elist, last_edges=last_edges,
        **params)


def _get_syn_param(src_name, src_group, tgt_name, tgt_group, syn_spec,
                   key=None):
    '''
    Return the most specific synaptic properties in `syn_spec` with respect to
    connections between `src_group` and `tgt_group`.
    Priority is given to source (presynaptic properties): they come last.
    '''
    group_keys = []

    for k in syn_spec.keys():
        group_keys.extend(k)

    group_keys = set(group_keys)

    src_type = src_group.neuron_type
    tgt_type = tgt_group.neuron_type

    # entry for source type and target type
    dict_prop = syn_spec.get((src_type, tgt_type), {})
    key_prop = dict_prop.get(key, None)

    # entry for source type and target name
    if tgt_name in group_keys:
        dict_prop = syn_spec.get((src_type, tgt_name), dict_prop)
        key_prop = dict_prop.get(key, key_prop)
    # entry for source name and target type
    if src_name in group_keys:
        dict_prop = syn_spec.get((src_name, tgt_type), dict_prop)
        key_prop = dict_prop.get(key, key_prop)
    # entry for source name and target name
    if src_name in group_keys and tgt_name in group_keys:
        dict_prop = syn_spec.get((src_name, tgt_name), dict_prop)
        key_prop = dict_prop.get(key, key_prop)

    if key is not None:
        return deepcopy(key_prop)

    return deepcopy(dict_prop)


def _get_dtype(value):
    if is_integer(value):
        return "int"
    elif isinstance(value, float):
        return "double"
    if isinstance(value, (bytes, str)):
        return "string"

    return "object"


def _post_del_update(graph, nodes, remapping=None):
    '''
    Remove entries in positions and structure after node deletion.

    Parameters
    ----------
    graph : the graph
    nodes : set of nodes that have been deleted
    remapping : dict mapping old to new node ids
    '''
    num_old = graph.node_nb() + len(nodes)

    keep = np.ones(num_old, dtype=bool)
    keep[list(nodes)] = 0

    if graph.is_spatial():
        graph._pos = graph._pos[keep]

    if graph.structure is not None:
        struct = graph.structure

        # update structure
        struct._size = struct._max_id = struct._desired_size = graph.node_nb()

        struct._groups = struct._groups[keep]

        # map old and new node ids
        if remapping is None:
            idx = 0

            remapping = {}

            for n in range(num_old):
                if n not in nodes:
                    remapping[n] = idx
                    idx += 1

        # update groups
        for g in struct.values():
            g._ids = {remapping[n] for n in g._ids if n not in nodes}
            g._desired_size = len(g._ids)

        for m in struct._meta_groups.values():
            m._ids = {remapping[n] for n in m._ids if n not in nodes}
            m._desired_size = len(m._ids)


def _get_matrices(g, directed, weights, weighted, combine_weights,
                  normed=False, exponent=None, remove_self_loops=True):
    '''
    Return the relevant matrices:
    * W, Wu if weighted
    * A, Au otherwise
    '''
    if weighted:
        # weighted undirected
        W  = g.adjacency_matrix(weights=weights)

        if normed:
            W /= W.max()

        # remove potential self-loops
        if remove_self_loops:
            W.setdiag(0.)

        if exponent is not None:
            W = W.power(exponent)

        Wu = W

        if g.is_directed():
            if directed:
                Wu  = W + W.T
            elif not directed:
                if combine_weights == "sum":
                    Wu = W + W.T

                    if normed:
                        Wu /= Wu.max()
                elif combine_weights == "mean":
                    A = g.adjacency_matrix().astype(float)
                    A += A.T

                    Wu = (W + W.T).multiply(A.power(-1))
                elif combine_weights == "max":
                    Wu = W.maximum(W.T)
                elif combine_weights == "min":
                    A = g.adjacency_matrix()

                    s = A.multiply(A.T)
                    a = (A - s).maximum(0)

                    wa = W.multiply(a)

                    Wu = wa + wa.T + W.multiply(s).minimum(W.T.multiply(s))
                else:
                    raise ValueError("Unknown `combine_weights`: '{}'.".format(
                                     combine_weights))

        return W, Wu

    # binary undirected
    A = g.adjacency_matrix()

    # remove potential self-loops
    if remove_self_loops:
        A.setdiag(0.)

    Au = A

    if g.is_directed():
        Au = Au + Au.T

        if not directed:
            Au.data = np.ones(len(Au.data))

    return A, Au


# ------------------ #
# Graph-tool helpers #
# ------------------ #

def _get_gt_weights(g, weights):
    ''' Return the properly formatted weights '''
    if nonstring_container(weights):
        # user-provided array (test must come first since non hashable)
        return g.graph.new_edge_property("double", vals=weights)
    elif weights in g.edge_attributes:
        # existing edge attribute
        return g.graph.edge_properties[weights]
    elif weights is True:
        # "normal" weights
        return g.graph.edge_properties['weight']
    elif not weights:
        # unweighted
        return None

    raise ValueError("Unknown edge attribute '" + str(weights) + "'.")


def _get_gt_graph(g, directed, weights, combine_weights=None,
                  return_all=False):
    ''' Returns the correct graph(view) given the options '''
    import graph_tool as gt
    from graph_tool.stats import label_parallel_edges

    directed = g.is_directed() if directed is None else directed

    weights = "weight" if weights is True else weights
    weights = None if weights is False else weights

    if not directed and g.is_directed():
        if weights is not None:
            u, weights = _to_undirected(g, weights, combine_weights)

            if return_all:
                return u, u.graph, u.edge_attributes[weights]

            return u.graph

        graph = gt.GraphView(g.graph, directed=False)
        graph = gt.GraphView(graph, efilt=label_parallel_edges(graph).fa == 0)

        if return_all:
            return g, graph, None

        return graph

    if return_all:
        return g, g.graph, weights

    return g.graph


# -------------- #
# IGraph helpers #
# -------------- #

def _get_ig_weights(g, weights):
    if nonstring_container(weights):
        # user-provided array (test must come first since non hashable)
        return np.array(weights)
    elif weights in g.edge_attributes:
        # existing edge attribute
        return np.array(g.graph.es[weights])
    elif weights is True:
        # "normal" weights
        return np.array(g.graph.es["weight"])
    elif not weights:
        # unweighted
        return None

    raise ValueError("Unknown edge attribute '" + str(weights) + "'.")


def _get_ig_graph(g, directed, weights, combine_weights=None,
                  return_all=False):
    ''' Returns the correct graph(view) given the options '''
    directed = g.is_directed() if directed is None else directed

    weights = "weight" if weights is True else weights
    weights = None if weights is False else weights

    if not directed and g.is_directed():
        if weights is not None:
            u, weights = _to_undirected(g, weights, combine_weights)

            if return_all:
                return u, u.graph, u.edge_attributes[weights]

            return u.graph

        if return_all:
            return g, g.graph.as_undirected(), None

        return g.graph.as_undirected()

    if return_all:
        return g, g.graph, weights

    return g.graph


# ---------------- #
# Networkx helpers #
# ---------------- #

def _get_nx_weights(g, weights):
    ''' Return the properly formatted weights '''
    if nonstring_container(weights):
        # generate a function that returns the weights
        def f(source, target, attr):
            eid = g.edge_id((source, target))
            return weights[eid]

        return f
    elif weights in g.edge_attributes:
        # existing edge attribute
        return weights
    elif weights is True:
        # "normal" weights
        return 'weight'
    elif not weights:
        # unweighted
        return None

    raise ValueError("Unknown attribute '{}' for `weights`.".format(weights))


def _get_nx_graph(g, directed, weights, combine_weights=None,
                  return_all=False):
    ''' Returns the correct graph(view) given the options '''
    directed = g.is_directed() if directed is None else directed

    weights = "weight" if weights is True else weights
    weights = None if weights is False else weights

    if not directed and g.is_directed():
        if weights is not None:
            u, weights = _to_undirected(g, weights, combine_weights)

            if return_all:
                return u, u.graph, u.edge_attributes[weights]

            return u.graph

        if return_all:
            return g, g.graph.to_undirected(as_view=True), None

        return g.graph.to_undirected(as_view=True)

    if return_all:
        return g, g.graph, weights

    return g.graph


def _to_undirected(g, weights, combine_weights):
    if nonstring_container(weights):
        g = g.copy()
        g.new_edge_attribute("tmp", "double", values=weights)

        weights = "tmp"

    return g.to_undirected(combine_weights), weights
