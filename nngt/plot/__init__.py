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
This modules provides the following features:

* plotting the distribution of some attribute over the graph
* basic graph plotting
* animation of some recorded activity
"""

import warnings as _warn
_warn.filterwarnings("ignore", module="matplotlib")

import matplotlib as _mpl

import nngt as _nngt

# module import

from .custom_plt import palette, markers
from .animations import Animation2d, AnimationNetwork
from .plt_networks import draw_network
from .plt_properties import *
from . import plt_properties as _plt_prop


__all__ = [
    "Animation2d",
    "AnimationNetwork",
    "draw_network",
    "palette",
    "markers",
]


if _nngt._config["mpl_backend"] is not None:
    _mpl.use(_nngt._config["mpl_backend"])


__all__.extend(_plt_prop.__all__)
