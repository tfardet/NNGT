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
Content
=======
"""

import sys
sys.argv.append('--quiet')

import nngt


#
#---
# Dependencies
#---------------------

depends = ['nest', 'nngt.core']

from .nest_graph import *
from .nest_utils import *
from .nest_activity import *

nngt.__all__.append('simulation')


#
#---
# Declare functions
#---------------------

__all__ = [
    'activity_types',
	'get_nest_network',
	'make_nest_network',
    'monitor_groups',
    'monitor_nodes',
    'set_noise',
    'set_poisson_input',
    'set_step_currents',
]

# test import of simulation plotting tools

if nngt._config['with_plot']:
    from .nest_plot import plot_activity, raster_plot
    __all__.extend(("plot_activity", "raster_plot"))
