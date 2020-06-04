#!/usr/bin/env python
#-*- coding:utf-8 -*-

from copy import deepcopy

import numpy as np

from .errors import InvalidArgument
from .rng_tools import _eprop_distribution
from .test_functions import nonstring_container, is_integer


""" Helper functions for graph classes """

def _edge_prop(prop):
    ''' Return edge property `name` as a distribution dict '''
    if is_integer(prop) or isinstance(prop, np.float):
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
            if prop is None:
                prop = graph._w
            else:
                prop = _edge_prop(prop)
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
    else:
        return deepcopy(dict_prop)


def _get_dtype(value):
    if is_integer(value):
        return "int"
    elif isinstance(value, float):
        return "double"
    if isinstance(value, (bytes, str)):
        return "string"

    return "object"


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


def _get_gt_graph(g, directed, weights):
    ''' Returns the correct graph(view) given the options '''
    import graph_tool as gt
    from graph_tool.stats import label_parallel_edges

    if not directed and g.is_directed():
        if weights is not None:
            raise ValueError(
                "Cannot make graph undirected if `weights` are used.")

        graph = gt.GraphView(g.graph, directed=False)

        return gt.GraphView(graph, efilt=label_parallel_edges(graph).fa == 0)

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


def _get_ig_graph(g, directed, weights):
    ''' Returns the correct graph(view) given the options '''
    if not directed and g.is_directed():
        if weights is not None:
            raise ValueError(
                "Cannot make graph undirected if `weights` are used.")

        return g.graph.as_undirected()

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


def _get_nx_graph(g, directed, weights):
    ''' Returns the correct graph(view) given the options '''
    if not directed and g.is_directed():
        if weights is not None:
            raise ValueError(
                "Cannot make graph undirected if `weights` are used.")

        return g.graph.to_undirected(as_view=True)

    return g.graph
