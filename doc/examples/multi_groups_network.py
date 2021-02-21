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
