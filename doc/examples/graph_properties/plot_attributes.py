#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2020  Tanguy Fardet
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

"""
Plot various graph properties
=============================
"""

import nngt
import nngt.plot as nplt
from nngt.geometry import Shape

import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.edgecolor': 'grey', 'xtick.color': 'grey', 'ytick.color': 'grey',
    "figure.facecolor": (0, 0, 0, 0), "axes.facecolor": (0, 0, 0, 0),
    "axes.labelcolor": "grey", "text.color": "grey"
})


nngt.seed(0)


# %%
# Let's start by making a random exponential graph

shape = Shape.disk(100)

g = nngt.generation.distance_rule(5, shape=shape, nodes=1000, avg_deg=20)


# %%
# Let's plot the distances

nplt.edge_attributes_distribution(g, "distance", show=True)

# %%
# We then compute the betweenness and see how it correlates with the distance

nbetw, ebetw = nngt.analysis.betweenness(g)

g.new_edge_attribute("betweenness", "float", values=ebetw)

nplt.correlation_to_attribute(g, "distance", "betweenness",
                              attribute_type="edge", show=True)


# %%
# Let's check the correlations between various node properties and their degree

g.new_node_attribute("betweenness", "float", values=nbetw)

attr = ["betweenness", "clustering", "in-degree", "subgraph_centrality"]

nplt.correlation_to_attribute(g, "out-degree", attr, show=True)
