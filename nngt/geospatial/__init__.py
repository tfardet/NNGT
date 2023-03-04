# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/geospatial/__init__.py

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
