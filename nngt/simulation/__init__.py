# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/simulation/__init__.py

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
    "Connect", "Disconnect", "Create", "SetStatus", "ResetNetwork", "set",
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
