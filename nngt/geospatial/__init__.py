#-*- coding:utf-8 -*-
#
# geospatial/__init__.py
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
The geospatial module contains functions and objects that enable
straightforward network plots together with geospatial data.

It relies on `geopandas <geopandas.org>`_ and
`cartopy <https://scitools.org.uk/cartopy/docs/latest>`_ in the background.

See ":ref:`sphx_glr_gallery_graph_structure_plot_map.py`" for an example.
"""

from .countries import (maps, country_names, country_codes, convertors, cities,
                        codes_to_names, natural_earth)

from .plot import draw_map
