#-*- coding:utf-8 -*-
#
# lib/__init__.py
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
Various tools for random number generation, array searching and type testing.
"""

from .constants import *
from .errors import *
from .rng_tools import *
from .sorting import find_idx_nearest
from .test_functions import *


__all__ = [
    "delta_distrib",
    "find_idx_nearest",
    "gaussian_distrib",
    "InvalidArgument",
    "is_integer",
    "is_iterable",
    "lin_correlated_distrib",
    "log_correlated_distrib",
    "lognormal_distrib",
    "nonstring_container",
    "uniform_distrib",
]
