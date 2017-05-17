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
Functions for plotting graphs and graph properties.


Content
=======
"""

import matplotlib as _mpl

import nngt as _nngt

if _nngt._config["backend"] is not None:
    _mpl.use(_nngt._config["backend"])
else:
    sav_backend = _mpl.get_backend()
    backends = [ 'GTK3Agg', 'Qt4Agg', 'Qt5Agg' ]
    keep_trying = True
    while backends and keep_trying:
        try:
            backend = backends.pop()
            _mpl.use(backend)
            keep_trying = False
        except:
            _mpl.use(sav_backend)


import warnings as _warn
_warn.filterwarnings("ignore", module="matplotlib")


# module import

from .custom_plt import palette
from .animations import Animation2d, AnimationNetwork
from .plt_networks import draw_network
from .plt_properties import *
from . import plt_properties as _plt_prop


__all__ = [
    "Animation2d",
    "AnimationNetwork",
    "draw_network",
    "palette",
]

__all__.extend(_plt_prop.__all__)
