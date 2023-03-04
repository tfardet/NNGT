# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# doc/examples/multi_groups_network.py

""" Generation of multi-group networks """

import nngt
import nngt.generation as ng

import numpy as np


num_nodes = 1000


'''
Make the population
'''

# two groups
g1 = nngt.Group(500)  # nodes 0 to 499
g2 = nngt.Group(500)  # nodes 500 to 999

# make structure
struct = nngt.Structure.from_groups((g1, g2), ("left", "right"))

# create network from this population
net = nngt.Graph(structure=struct)


'''
Connect the groups
'''

# inter-groups (Erdos-Renyi)
prop_er1 = {"density": 0.005}
ng.connect_groups(net, "left", "right", "erdos_renyi", **prop_er1)

# intra-groups (Newman-Watts)
prop_nw = {
    "coord_nb": 20,
    "proba_shortcut": 0.1,
    "reciprocity_circular": 1.
}

ng.connect_groups(net, "left", "left", "newman_watts", **prop_nw)
ng.connect_groups(net, "right", "right", "newman_watts", **prop_nw)


'''
Plot the graph
'''

if nngt.get_config("with_plot"):
    nngt.plot.library_draw(net, show=False)

    pop_graph = net.get_structure_graph()

    nngt.plot.chord_diagram(pop_graph, names="name", use_gradient=True,
                            show=True)
