#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Core classes and functions. Most of them are not visible in the module as they
are directly loaded at :mod:`nngt` level.

Content
=======
"""

import nngt
from .gt_graph import _GtGraph
from .ig_graph import _IGraph
from .nx_graph import _NxGraph
from .nngt_graph import _NNGTGraph
from .graph_datastruct import Connections


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
    "Connections",
    "GraphObject"
]
