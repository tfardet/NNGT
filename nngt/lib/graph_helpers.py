#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

from .rng_tools import _eprop_distribution

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
            raise InvalidArgument("`weights` must be either a dict or"
                                  " a number.")
    else:
        final_prop = {"distribution": "constant"}
    return final_prop


def _set_edge_attr(graph, elist, attributes):
    ''' Fill the `attributes` dictionnary '''
    if graph._weighted and "weight" not in attributes:
        distrib = graph._w["distribution"]
        params = {k: v for (k, v) in graph._w.items() if k != "distribution"}
        attributes["weight"] = _eprop_distribution(graph, distrib, elist=elist,
                                                   **params)
        if graph is not None and graph.is_network():
            if not np.isclose(graph._iwf, 1.):
                adj = graph.adjacency_matrix(types=True, weights=False)
                keep = (adj[elist[:, 0], elist[:, 1]] < 0).A1
                attributes["weight"][keep] *= graph._iwf
    if hasattr(graph, "_d") and "delay" not in attributes:
        distrib = graph._d["distribution"]
        params = {k: v for (k, v) in graph._d.items() if k != "distribution"}
        attributes["delay"] = _eprop_distribution(graph, distrib, elist=elist,
                                                  **params)
