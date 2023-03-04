# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/lib/__init__.py

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
