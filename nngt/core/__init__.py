# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/core/__init__.py

"""
Core classes and functions. Most of them are not visible in the module as they
are directly loaded at :mod:`nngt` level.

.. class:: Graph
   :noindex:

.. class:: SpatialGraph
   :noindex:

.. class:: Network
   :noindex:

.. class:: SpatialNetwork
   :noindex:

"""

import nngt
from .gt_graph import _GtGraph
from .ig_graph import _IGraph
from .nx_graph import _NxGraph
from .nngt_graph import _NNGTGraph


_graphlib = {
    "graph-tool": _GtGraph,
    "igraph": _IGraph,
    "networkx": _NxGraph,
    "nngt": _NNGTGraph,
    #~ # "snap": _SnapGraph
}


#: Graph object (reference to one of the main libraries' wrapper
GraphObject = _graphlib[nngt._config["backend"]]


__all__ = [
    "GraphObject"
]
