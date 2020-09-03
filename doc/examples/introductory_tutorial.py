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

''' Node and edge attributes '''

import numpy as np

import nngt
import nngt.generation as ng


''' -------------- #
# Generate a graph #
# -------------- '''

# create an empty graph
g = nngt.Graph()

# add some nodes
g.new_node(10)            # create nodes 0, 1, ... to 9
print(g.node_nb(), '\n')  # returns 10

# add edges
g.new_edge(1, 4)          # create one connection going from 1 to 4
print(g.edge_nb())        # returns 1
g.new_edges([(0, 3), (5, 9), (9, 3)])
print(g.edge_nb(), '\n')  # returns 4


''' --------------- #
# Adding attributes #
# --------------- '''

g2 = nngt.Graph()

# add a new node with attributes
attributes = {
    'size': 2.,
    'color': 'blue',
    'a': 5,
    'blob': []
}

attribute_types = {
    'size': 'double',
    'color': 'string',
    'a': 'int',
    'blob': 'object'
}

g2.new_node(attributes=attributes, value_types=attribute_types)
print(g2.node_attributes, '\n')

# by default all nodes will have these properties with "empty" values
g2.new_node(2)
# for a double attribute like 'size', default value is NaN
print(g2.get_node_attributes(name="size"))
# for a string attribute like 'color', default value is ""
print(g2.get_node_attributes(name="color"))
# for an int attribute like 'a', default value is 0
print(g2.get_node_attributes(name='a'))
# for an object attribute like 'blob', default value is None
print(g2.get_node_attributes(name='blob'), '\n')

# attributes for multiple nodes can be set simultaneously
g2.new_node(3, attributes={'size': [4., 5., 1.], 'color': ['r', 'g', 'b']},
            value_types={'size': 'double', 'color': 'string'})
print(g2.node_attributes['size'])
print(g2.node_attributes['color'], '\n')

# creating attributes afterwards
import numpy as np
g3 = nngt.Graph(nodes=100)
g3.new_node_attribute('size', 'double',
                      values=np.random.uniform(0, 20, 100))
print(g3.node_attributes['size'][:5], '\n')

# edges attributes
edges = g3.new_edges(np.random.randint(0, 50, (10, 2)), ignore_invalid=True)
g3.new_edge_attribute('rank', 'int')
g3.set_edge_attribute('rank', val=2, edges=edges[:3, :])
print(g3.edge_attributes['rank'], '\n')

# check default values
e = g3.new_edge(99, 98)
g3.new_edges(np.random.randint(50, 100, (5, 2)), ignore_invalid=True)
print(g3.edge_attributes['rank'], '\n')


''' ---------------------------------------- #
# Generate and analyze more complex networks #
# ---------------------------------------- '''

from nngt import generation as ng
from nngt import analysis as na
from nngt import plot as nplt

# make an ER network
g = ng.erdos_renyi(nodes=1000, avg_deg=100)

if nngt.get_config("with_plot"):
    nplt.degree_distribution(g, ('in', 'total'), show=False)

print("Clustering ER: {}".format(na.global_clustering(g)))

# then a scale-free network
g = ng.random_scale_free(1.8, 3.2, nodes=1000, avg_deg=100)

if nngt.get_config("with_plot"):
    nplt.degree_distribution(g, ('in', 'out'), num_bins=30, logx=True,
                             logy=True, show=True)

print("Clustering SF: {}".format(na.global_clustering(g)))
