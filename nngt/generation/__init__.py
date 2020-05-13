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
Functions that generates the underlying connectivity of graphs, as well
as the connection properties (weight/strength and delay).
"""

from . import connectors as _ct
from . import graph_connectivity as _gc
from . import rewiring as _rw

from .connectors import *
from .graph_connectivity import *
from .rewiring import *


__all__ = _gc.__all__ + _rw.__all__ + _ct.__all__
