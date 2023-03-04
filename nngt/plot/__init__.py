# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/plot/__init__.py

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
