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

# two groups of neurons
g1 = nngt.NeuralGroup(500, neuron_type=1)  # neurons 0 to 499
g2 = nngt.NeuralGroup(500, neuron_type=1)  # neurons 500 to 999

# make population (without NEST models)
pop = nngt.NeuralPop.from_groups(
    (g1, g2), ("left", "right"), with_models=False)

# create network from this population
net = nngt.Network(population=pop)


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
    import matplotlib.pyplot as plt

    colors = np.zeros(num_nodes)
    colors[500:] = 1

    if nngt.get_config("backend") == "graph-tool":
        from graph_tool.draw import graph_draw, prop_to_size, sfdp_layout
        pm = net.graph.new_vertex_property("int", colors)
        size = net.graph.new_vertex_property("double", val=5.)
        pos = sfdp_layout(net.graph, groups=pm, C=1., K=20, gamma=5, mu=20)
        graph_draw(net.graph, pos=pos, vertex_fill_color=pm, vertex_color=pm,
                   vertex_size=size, nodesfirst=True,
                   edge_color=[0.179, 0.203,0.210, 0.3])
    elif nngt.get_config("backend") == "networkx":
        import networkx as nx
        plt.figure()
        init_pos = {i: np.random.uniform(-1000., -900, 2) for i in range(500)}
        init_pos.update(
            {i: np.random.uniform(900., 1000, 2) for i in range(500, 1000)})
        layout = nx.spring_layout(net.graph, k=20, pos=init_pos)
        nx.draw(net, pos=layout, node_color=colors, node_size=20)
    elif nngt.get_config("backend") == "igraph":
        import igraph as ig
        colors = [(1, 0, 0) for _ in range(500)]
        colors.extend([(0, 0, 1) for _ in range(500)])
        ig.plot(net.graph, vertex_color=colors, vertex_size=5,
                edge_arrow_size=0.5)
    else:
        nngt.plot.draw_network(net)

    plt.show()
