# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# doc/examples/spatial_graphs.py

''' Spatial graphs generation and methods '''

import os
import time

import numpy as np

import nngt
from nngt.geometry import Shape

# nngt.seed(0)


# ---------------------------- #
# Generate the spatial network #
# ---------------------------- #

ell = Shape.ellipse(radii=(3000., 5000.))

num_nodes  = 1000
population = nngt.NeuralPop.uniform(num_nodes)

g = nngt.generation.gaussian_degree(
    100., 5., nodes=num_nodes, shape=ell, population=population)


# -------------- #
# Saving/loading #
# -------------- #

start = time.time()
g.to_file('sp_graph.el')
print('Saving in {} s.'.format(time.time() - start))

start = time.time()
g2 = nngt.Graph.from_file('sp_graph.el')
print('Loading in {} s.'.format(time.time() - start))

# check equality of shapes and populations

print('Both networks have same area: {}.'.format(
      np.isclose(g2.shape.area, ell.area)))
print('They also have the same boundaries: {}.'.format(
      np.all(np.isclose(g2.shape.bounds, ell.bounds))))

same_groups = np.all(
    [g2.population[k] == g.population[k] for k in g.population])
same_ids = np.all(
    [g2.population[k].ids == g.population[k].ids for k in g.population])

print('They also have the same population: {}.'.format(same_groups * same_ids))


# remove file
os.remove('sp_graph.el')


# ---- #
# Plot #
# ---- #

if nngt.get_config('with_plot'):
    nngt.plot.draw_network(g2, decimate_connections=100, show=True)
