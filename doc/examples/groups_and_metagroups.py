# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# doc/examples/groups_and_metagroups.py

""" Generation of multi-group networks containing metagroups """

import nngt
import nngt.generation as ng

import numpy as np


'''
Make a mixed excitatory and inhibitory population, then subdived it in subgroups
'''

num_neurons = 1000

pop = nngt.NeuralPop.exc_and_inhib(num_neurons)

# create two separated subgroups associated to two shapes where the neurons
# will be seeded

# we select 500 random nodes for the left group
left_nodes = np.random.choice([i for i in range(num_neurons)],
                              500, replace=False)
left = nngt.NeuralGroup(left_nodes, neuron_type=None)  # here we first create...
pop.add_meta_group(left, "left")  # ... then add

# right group is the complement
right_nodes = list(set(pop.ids).difference(left_nodes))
right = pop.create_meta_group(right_nodes, "right")  # here both in one call

# create another pair of random metagroups

# we select 500 random nodes for the left group
group1 = pop.create_meta_group([i for i in range(500)], "g1")
group2 = pop.create_meta_group([i for i in range(500, num_neurons)], "g2")


'''
We then create the shapes associated to the left and right groups and seed
the neurons accordingly in the network
'''

left_shape  = nngt.geometry.Shape.disk(300, (-300, 0))
right_shape = nngt.geometry.Shape.rectangle(800, 200, (300, 0))

left_pos  = left_shape.seed_neurons(left.size)
right_pos = right_shape.seed_neurons(right.size)

# we order the positions according to the neuron ids
positions = np.empty((num_neurons, 2))

for i, p in zip(left_nodes, left_pos):
    positions[i] = p

for i, p in zip(right_nodes, right_pos):
    positions[i] = p

# create network from this population
net = nngt.Network(population=pop, positions=positions)


'''
Access metagroups
'''

print(pop.meta_groups)
print(pop["left"])


'''
Plot the graph
'''

plt = None

# we plot the graph, setting the node shape from the left and right groups
# and the color from the neuronal type (exc. and inhib.)

nngt.plot.draw_network(net, nshape=[left, right], nsize=20,
                       show_environment=False)

if nngt.get_config("with_plot"):
    import matplotlib.pyplot as plt
    plt.show()

# further tests to make sure every configuration works

nngt.plot.draw_network(net, nshape=[left, right], show_environment=False,
                       max_nsize=20, simple_nodes=True)

nngt.plot.draw_network(net, nshape=["o" for _ in range(net.node_nb())],
                       show_environment=False, simple_nodes=True)

nngt.plot.draw_network(net, nshape=["o" for _ in range(net.node_nb())],
                       show_environment=False)

nngt.plot.draw_network(net, nshape="s", show_environment=False,
                       simple_nodes=True)

nngt.plot.draw_network(net, nshape="s", show_environment=False)

if nngt.get_config("with_plot"):
    plt.show()
