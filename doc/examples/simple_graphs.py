# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# doc/examples/simple_graphs.py

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

