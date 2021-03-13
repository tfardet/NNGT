#-*- coding:utf-8 -*-
#
# plot/__init__.py
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
Functions for plotting graphs and graph properties.

The following features are provided:

* basic graph plotting
* plotting the distribution of some attribute over the graph
* animation of some recorded activity
"""

import warnings as _warn
_warn.filterwarnings("ignore", module="matplotlib")

import matplotlib as _mpl

import nngt as _nngt

# module import

from .custom_plt import palette_continuous, palette_discrete, markers
from .animations import Animation2d, AnimationNetwork
from .plt_networks import chord_diagram, draw_network, hive_plot, library_draw
from .plt_properties import *
from . import plt_properties as _plt_prop


__all__ = [
    "Animation2d",
    "AnimationNetwork",
    "chord_diagram",
    "draw_network",
    "hive_plot",
    "library_draw",
    "palette_continuous",
    "palette_discrete",
    "markers",
]


if _nngt._config["mpl_backend"] is not None:
    _mpl.use(_nngt._config["mpl_backend"])


__all__.extend(_plt_prop.__all__)
