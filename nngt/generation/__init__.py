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
Details
=======
"""

from .graph_connectivity import *


__all__ = [
	'all_to_all',
    'connect_neural_groups',
    'connect_neural_types',
	'connect_nodes',
	'distance_rule',
	'erdos_renyi',
    'fixed_degree',
    'gaussian_degree',
	'random_scale_free',
	'price_scale_free',
	'newman_watts'
]
