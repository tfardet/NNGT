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

''' Spatial graphs generation and methods '''

import time

import numpy as np

import nngt
from nngt.geometry import Shape


# -------------------------- #
# Generate the spatial graph #
# -------------------------- #

ell = Shape.ellipse(radii=(3000., 5000.))

num_nodes = 1000
g = nngt.generation.gaussian_degree(100., 5., nodes=num_nodes, shape=ell)


# -------------- #
# Saving/loading #
# -------------- #

start = time.time()
g.to_file('sp_graph.el')
print('Saving in {} s.'.format(time.time() - start))

start = time.time()
g2 = nngt.Graph.from_file('sp_graph.el')
print('Loading in {} s.'.format(time.time() - start))

print('Both networks have same area: {}.'.format(
      np.isclose(g2.shape.area, ell.area)))
print('They also have the same boundaries: {}.'.format(
      np.all(np.isclose(g2.shape.bounds, ell.bounds))))


# ---- #
# Plot #
# ---- #

if nngt.get_config('with_plot'):
    nngt.plot.draw_network(g2, decimate=100, show=True)
