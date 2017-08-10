#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

from .rng_tools import _eprop_distribution
from .test_functions import nonstring_container


""" Helper functions for graph classes """


def _edge_prop(name, arg_dict):
    ''' Return edge property `name` as a distribution dict '''
    final_prop = None
    if name in arg_dict:
        prop = arg_dict[name]
        if isinstance(prop, int) or isinstance(prop, float):
            final_prop = {"distribution": "constant", "value": prop}
        elif isinstance(prop, dict):
            final_prop = prop
        else:
            raise InvalidArgument("Edge property must be either a dict or"
                                  " a number.")
    else:
        final_prop = {"distribution": "constant"}
    return final_prop


def _set_edge_attr(graph, elist, attributes):
    ''' Fill the `attributes` dictionnary '''
    # check the weights
    if graph._weighted and "weight" not in attributes:
        distrib = graph._w["distribution"]
        params = {k: v for (k, v) in graph._w.items() if k != "distribution"}
        attributes["weight"] = _eprop_distribution(graph, distrib, elist=elist,
                                                   **params)
    if graph._weighted and not nonstring_container(attributes['weight']):
        prop = _edge_prop('weight', attributes)
        params = {k: v for (k, v) in prop.items() if k != "distribution"}
        attributes["weight"] = _eprop_distribution(
            graph, prop["distribution"], elist=elist, **params)

    # if dealing with network, check inhibitory weight factor
    if graph.is_network() and not np.isclose(graph._iwf, 1.):
            keep = graph.nodes_attributes['type'][elist[:, 0]] < 0
            attributes["weight"][keep] *= graph._iwf

    # also check delays
    if graph.is_network() and "delay" not in attributes:
        distrib = graph._d["distribution"]
        params = {k: v for (k, v) in graph._d.items() if k != "distribution"}
        attributes["delay"] = _eprop_distribution(graph, distrib, elist=elist,
                                                  **params)
    elif not nonstring_container(attributes.get('delay', [])):  # for 1 line
        prop = _edge_prop('delay', attributes)
        params = {k: v for (k, v) in prop.items() if k != "distribution"}
        attributes["delay"] = _eprop_distribution(
            graph, prop["distribution"], elist=elist, **params)
