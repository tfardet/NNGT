#-*- coding:utf-8 -*-
#
# simulation/__init__.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2022 Tanguy Fardet
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
Module to interact easily with the NEST simulator. It allows to:

* build a NEST network from :class:`~nngt.Network` or
  :class:`~nngt.SpatialNetwork` objects,
* monitor the activity of the network (taking neural groups into account)
* plot the activity while separating the behaviours of predefined neural groups
"""

import logging as _logging
import sys as _sys
import types as _types

import nngt as _nngt
from nngt.lib.logger import _log_message


_logger = _logging.getLogger(__name__)


# --------- #
# Wrap nest #
# --------- #

import nest

warnlist = [
    "Connect", "Disconnect", "Create", "SetStatus", "ResetNetwork"
]

def _wrap_reset_kernel(func):
    '''
    Reset all NeuralPops and parent Networks before calling nest.ResetKernel.
    '''
    def wrapper(*args, **kwargs):
        _nngt.NeuralPop._nest_reset()

        return func(*args, **kwargs)

    return wrapper


def _wrap_warn(func):
    '''
    Warn when risky nest functions are called.
    '''
    def wrapper(*args, _warn=True, **kwargs):
        if _warn:
            _log_message(_logger, "WARNING", "This function could interfere "
                         "with NNGT, making your Network obsolete compared to "
                         "the one in NEST... make sure to check what is "
                         "modified! Pass the `_warn=False` keyword to the "
                         "function if you know what you are doing and want to "
                         "hide this message.")

        return func(*args, **kwargs)

    return wrapper


class NestMod(_types.ModuleType):

    '''
    Wrapped module to replace nest.
    '''

    def __getattribute__(self, attr):
        if attr in warnlist:
            return _wrap_warn(getattr(nest, attr))
        elif attr == "ResetKernel":
            return _wrap_reset_kernel(nest.ResetKernel)

        return getattr(nest, attr)


_sys.modules["nest"] = NestMod("nest")


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

try:
    import matplotlib
    _with_plot = True
except ImportError:
    _with_plot = False

if _with_plot:
    from .nest_plot import plot_activity, raster_plot
    __all__.extend(("plot_activity", "raster_plot"))
