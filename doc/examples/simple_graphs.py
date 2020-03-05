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

''' Simple graph generation '''

import numpy as np

import nngt
import nngt.generation as ng


# np.random.seed(0)


# ------------------- #
# Generate the graphs #
# ------------------- #

num_nodes  = 1000
avg_deg_er = 25
avg_deg_sf = 100

# random graphs
g1 = ng.erdos_renyi(nodes=num_nodes, avg_deg=avg_deg_er)

# the same graph but undirected
g2 = ng.erdos_renyi(nodes=num_nodes, avg_deg=avg_deg_er, directed=False)

# 2-step generation of a scale-free with Gaussian weight distribution
w = {
    "distribution": "gaussian",
    "avg": 60.,
    "std": 5.
}

g3 = nngt.Graph(num_nodes, weights=w)
ng.random_scale_free(2.2, 2.9, avg_deg=avg_deg_sf, from_graph=g3)

# same in 1 step
g4 = ng.random_scale_free(
    2.2, 2.9, avg_deg=avg_deg_sf, nodes=num_nodes, weights=w)


# ----------------- #
# Check the results #
# ----------------- #

assert np.isclose(avg_deg_er, np.average(g1.get_degrees('in')), 1e-4)
assert np.isclose(avg_deg_sf, np.average(g3.get_degrees('in')), 1e-4)
assert np.isclose(avg_deg_sf, np.average(g4.get_degrees('in')), 1e-4)

print(
    "Erdos-Renyi: requested average degree of {}; got {} for directed graph "
    "and {} for undirected one.".format(
        avg_deg_er, g1.edge_nb() / float(num_nodes),
        g2.edge_nb() / float(num_nodes))
)

if nngt.get_config('with_plot'):
    from nngt.plot import degree_distribution

    degree_distribution(
        g4, deg_type=["in", "out"], logx=True, logy=True, show=True)

