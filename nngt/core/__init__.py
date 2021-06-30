#-*- coding:utf-8 -*-
#
# core/__init__.py
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
