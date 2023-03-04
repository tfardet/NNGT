# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/analysis/__init__.py

"""
Details
=======
"""

from . import graph_analysis as _ga
from . import activity_analysis as _aa
from .graph_analysis import *
from .bayesian_blocks import bayesian_blocks
from .activity_analysis import *


__all__ = ['bayesian_blocks'] + _ga.__all__ + _aa.__all__
