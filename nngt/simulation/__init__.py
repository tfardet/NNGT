#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
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
Main functions to send :class:`Network` instances to NEST, as well as helper
functions to excite or record the network activity.
"""

import sys as _sys
_sys.argv.append('--quiet')

import nngt as _nngt


# -------------- #
# Import modules #
# -------------- #

from . import nest_activity as _na
from . import nest_graph as _ng
from . import nest_utils as _nu
from .nest_activity import *
from .nest_graph import *
from .nest_utils import *


# ----------------- #
# Declare functions #
# ----------------- #

__all__ = []
__all__.extend(_na.__all__)
__all__.extend(_ng.__all__)
__all__.extend(_nu.__all__)

# test import of simulation plotting tools

if _nngt._config['with_plot']:
    from .nest_plot import plot_activity, raster_plot
    __all__.extend(("plot_activity", "raster_plot"))
