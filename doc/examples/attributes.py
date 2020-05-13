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

num_nodes = 1000
avg_deg   = 25

graph = ng.erdos_renyi(nodes=num_nodes, avg_deg=avg_deg)


''' ----------------- #
# Add node attributes #
# ----------------- '''

# Let's make a network of animals where nodes represent either cats or dogs.
# (no discrimination against cats or dogs was intended, no animals were harmed
# while writing or running this code)
animals  = ["cat" for _ in range(600)]  # 600 cats
animals += ["dog" for _ in range(400)]  # and 400 dogs
np.random.shuffle(animals)              # which we assign randomly to the nodes

graph.new_node_attribute("animal", value_type="string", values=animals)

# Let's check the type of the first six animals
print(graph.get_node_attributes([0, 1, 2, 3, 4, 5], "animal"))

# Nodes can have attributes of multiple types, let's add a size to our animals
catsizes = np.random.normal(50, 5, 600)   # cats around 50 cm
dogsizes = np.random.normal(80, 10, 400)  # dogs around 80 cm

# We first create the attribute without values (for "double", default to NaN)
graph.new_node_attribute("size", value_type="double")

# We now have to attributes: one containing strings, the other numbers (double)
print(graph.node_attributes)

# get the cats and set their sizes
cats = graph.get_nodes(attribute="animal", value="cat")
graph.set_node_attribute("size", values=catsizes, nodes=cats)

# We set 600 values so there are 400 NaNs left
assert np.sum(np.isnan(graph.get_node_attributes(name="size"))) == 400, \
    "There were not 400 NaNs as predicted."

# None of the NaN values belongs to a cat
assert not np.any(np.isnan(graph.get_node_attributes(cats, name="size"))), \
    "Got some cats with NaN size! :'("

# get the dogs and set their sizes
dogs = graph.get_nodes(attribute="animal", value="dog")
graph.set_node_attribute("size", values=dogsizes, nodes=dogs)

# Some of the animals are part of human househols, they have therefore "owners"
# which will be represented here through a Human class.
# Animals without an owner will have an empty list as attribute.

class Human:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return "Human<{}>".format(self.name)

# John owns all animals between 8 and 48
John    = Human("John")
animals = [i for i in range(8, 49)]

graph.new_node_attribute("owners", value_type="object", val=[])
graph.set_node_attribute("owners", val=[John], nodes=animals)

# Now suppose another human, Julie, owns all animals between 0 and 40
Julie   = Human("Julie")
animals = [i for i in range(0, 41)]

# to update the values, we need to get them to add Bob to the list
owners = graph.get_node_attributes(name="owners", nodes=animals)

for interactions in owners:
    interactions.append(Julie)

graph.set_node_attribute("owners", values=owners, nodes=animals)

# now some of the initial owners should have had their attributes updated
new_owners = graph.get_node_attributes(name="owners")
print("There are animals owned only by", new_owners[0], "others owned only by",
      new_owners[48], "and some more owned by both", new_owners[40])


''' ---------- #
# Edge weights #
# ---------- '''

# Same as for node attributes, one can give attributes to the edges
# Let's give weights to the edges depending on how often the animals interact!
# cat's interact a lot among themselves, so we'll give them high weights
cat_edges = graph.get_edges(source_node=cats, target_node=cats)

# check that these are indeed only between cats
cat_set  = set(cats)
node_set = set(np.unique(cat_edges))

assert cat_set == node_set, "Damned, something wrong happened to the cats!"

# uniform distribution of weights between 30 and 50
graph.set_weights(elist=cat_edges, distribution="uniform",
                  parameters={"lower": 30, "upper": 50})

# dogs have less occasions to interact except some which spend a lot of time
# together, so we use a lognormal distribution
dog_edges = graph.get_edges(source_node=dogs, target_node=dogs)
graph.set_weights(elist=dog_edges, distribution="lognormal",
                  parameters={"position": 2.2, "scale": 0.5})

# Cats do not like dogs, so we set their weights to -5
# Dogs like chasing cats but do not like them much either so we let the default
# value of 1
cd_edges = graph.get_edges(source_node=cats, target_node=dogs)
graph.set_weights(elist=cd_edges, distribution="constant",
                  parameters={"value": -5})

# Let's check the distribution (you should clearly see 4 separate shapes)
if nngt.get_config("with_plot"):
    nngt.plot.edge_attributes_distribution(graph, "weight")


''' ------------------- #
# Other edge attributes #
# ------------------- '''

# non-default edge attributes can be created as the node attributes
# let's create a class for humans and store it when two animals have interacted
# with the same human (the default will be an empty list if they did not)

# Alice interacted with all animals between 8 and 48
Alice   = Human("Alice")
animals = [i for i in range(8, 49)]
edges   = graph.get_edges(source_node=animals, target_node=animals)

graph.new_edge_attribute("common_interaction", value_type="object", val=[])
graph.set_edge_attribute("common_interaction", val=[Alice], edges=edges)

# Now suppose another human, Bob, interacted with all animals between 0 and 40
Bob     = Human("Bob")
animals = [i for i in range(0, 41)]
edges2  = graph.get_edges(source_node=animals, target_node=animals)

# to update the values, we need to get them to add Bob to the list
ci = graph.get_edge_attributes(name="common_interaction", edges=edges2)

for interactions in ci:
    interactions.append(Bob)

graph.set_edge_attribute("common_interaction", values=ci, edges=edges2)

# now some of the initial `edges` should have had their attributes updated
new_ci = graph.get_edge_attributes(name="common_interaction", edges=edges)
print(np.sum([0 if len(interaction) < 2 else 1 for interaction in new_ci]),
      "interactions have been updated among the", len(edges), "from Alice.")

