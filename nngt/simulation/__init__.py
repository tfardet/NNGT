#-*- coding:utf-8 -*-
#
# simulation/__init__.py
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
Module to interact easily with the NEST simulator. It allows to:

* build a NEST network from :class:`~nngt.Network` or
  :class:`~nngt.SpatialNetwork` objects,
* monitor the activity of the network (taking neural groups into account)
* plot the activity while separating the behaviours of predefined neural groups
"""

import sys as _sys
import logging as _logging

import nngt as _nngt
from nngt.lib.logger import _log_message


_logger = _logging.getLogger(__name__)


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


# ---------------- #
# Wrap ResetKernel #
# ---------------- #

from nest import ResetKernel as _rk
from nest import Connect as _conn
from nest import Disconnect as _disc
from nest import Create as _cr
from nest import SetStatus as _setstat

try:
    from nest import ResetNetwork as _rn
except ImportError:
    pass

# store old functions
if not _nngt._old_nest_func:
    _nngt._old_nest_func["ResetKernel"] = _rk
    _nngt._old_nest_func["Connect"]     = _conn
    _nngt._old_nest_func["Disconnect"]  = _disc
    _nngt._old_nest_func["Create"]      = _cr
    _nngt._old_nest_func["SetStatus"]   = _setstat

    try:
        _nngt._old_nest_func["ResetNetwork"] = _rn
    except NameError:
        pass
else:
    _rk      = _nngt._old_nest_func["ResetKernel"]
    _conn    = _nngt._old_nest_func["Connect"]
    _disc    = _nngt._old_nest_func["Disconnect"]
    _cr      = _nngt._old_nest_func["Create"]
    _setstat = _nngt._old_nest_func["SetStatus"]

    try:
        _rn = _nngt._old_nest_func["ResetNetwork"]
    except KeyError:
        pass


def _new_reset_kernel():
    '''
    Call nest.ResetKernel, then reset all NeuralPops and parent Networks.
    '''
    _rk()
    _nngt.NeuralPop._nest_reset()


def _new_nest_func(old_nest_func):
    '''
    Print a warning to make sure user know what they are doing.
    '''
    def wrapper(*args, **kwargs):
        if kwargs.get("_warn", True):
            _log_message(_logger, "WARNING", "This function could interfere "
                         "with NNGT, making your Network obsolete compared to "
                         "the one in NEST... make sure to check what is "
                         "modified!")

        if "_warn" in kwargs:
            del kwargs["_warn"]

        return old_nest_func(*args, **kwargs)

    return wrapper


# nest is in sysmodules because it was imported in the main __init__.py
_sys.modules["nest"].ResetKernel = _new_reset_kernel
_sys.modules["nest"].Connect     = _new_nest_func(_conn)
_sys.modules["nest"].Disconnect  = _new_nest_func(_disc)
_sys.modules["nest"].Create      = _new_nest_func(_cr)
_sys.modules["nest"].SetStatus   = _new_nest_func(_setstat)

try:
    _sys.modules["nest"].ResetNetwork = _new_nest_func(_rn)
except NameError:
    pass
