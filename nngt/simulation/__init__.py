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
Main functions to send :class:`Network` instances to NEST, as well as helper
functions to excite or record the network activity.
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

if _nngt._config['with_plot']:
    from .nest_plot import plot_activity, raster_plot
    __all__.extend(("plot_activity", "raster_plot"))


# ---------------- #
# Wrap ResetKernel #
# ---------------- #

from nest import ResetKernel as _rk
from nest import ResetNetwork as _rn
from nest import Connect as _conn
from nest import Disconnect as _disc
from nest import Create as _cr
from nest import SetStatus as _setstat

# store old functions
if not _nngt._old_nest_func:
    _nngt._old_nest_func["ResetKernel"]  = _rk
    _nngt._old_nest_func["ResetNetwork"] = _rn
    _nngt._old_nest_func["Connect"]      = _conn
    _nngt._old_nest_func["Disconnect"]   = _disc
    _nngt._old_nest_func["Create"]       = _cr
    _nngt._old_nest_func["SetStatus"]    = _setstat
else:
    _rk      = _nngt._old_nest_func["ResetKernel"]
    _rn      = _nngt._old_nest_func["ResetNetwork"]
    _conn    = _nngt._old_nest_func["Connect"]
    _disc    = _nngt._old_nest_func["Disconnect"]
    _cr      = _nngt._old_nest_func["Create"]
    _setstat = _nngt._old_nest_func["SetStatus"]


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
_sys.modules["nest"].ResetKernel  = _new_reset_kernel
_sys.modules["nest"].ResetNetwork = _new_nest_func(_rn)
_sys.modules["nest"].Connect      = _new_nest_func(_conn)
_sys.modules["nest"].Disconnect   = _new_nest_func(_disc)
_sys.modules["nest"].Create       = _new_nest_func(_cr)
_sys.modules["nest"].SetStatus    = _new_nest_func(_setstat)
