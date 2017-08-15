#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

from .errors import InvalidArgument
from .rng_tools import _eprop_distribution
from .test_functions import nonstring_container


""" Helper functions for graph classes """

def _edge_prop(prop):
    ''' Return edge property `name` as a distribution dict '''
    if isinstance(prop, int) or isinstance(prop, float):
        return {"distribution": "constant", "value": prop}
    elif isinstance(prop, dict):
        return prop.copy()
    elif nonstring_container(prop):
        return {'distribution': 'custom', 'values': prop}
    elif prop is None:
        return {'distribution': 'constant'}
    else:
        raise InvalidArgument("Edge property must be either a dict, a list, or"
                              " a number.")


def _set_edge_attr(graph, elist, attribute, prop=None, last_edges=False):
    '''
    Fill the `attributes` dictionnary by returning the values associated to a
    given edge attribute.

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
        if graph._weighted:
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
            keep = graph.nodes_attributes['type'][elist[:, 0]] < 0
            weights[keep] *= graph._iwf

        return weights

    # also check delays
    if "delay" == attribute:
        delays = np.ones(len(elist))
        if graph.is_network() and prop is None:
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
