#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
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

''' Using complex NEST models '''

import nngt
import nngt.generation as ng


'''
Build a network with two populations:
* excitatory (80%)
* inhibitory (20%)
'''
num_neurons = 50   # number of neurons
avg_degree  = 20   # average number of neighbours
std_degree  = 3    # deviation for the Gaussian graph

# parameters
neuron_model = "ht_neuron"      # hill-tononi model
exc_syn = {'receptor_type': 1}  # 1 is 'AMPA' in this model
inh_syn = {'receptor_type': 3}  # 3 is 'GABA_A' in this model

pop = nngt.NeuralPop.exc_and_inhib(
    num_neurons, en_model=neuron_model, in_model=neuron_model,
    es_param=exc_syn, is_param=inh_syn)

# create the network and send it to NEST
w_prop = {"distribution": "gaussian", "avg": 1., "std": .2}
net = nngt.generation.gaussian_degree(
    avg_degree, std_degree, population=pop, weights=w_prop)

'''
Send to NEST and set excitation and recorders
'''
import nest
from nngt.simulation import monitor_groups, plot_activity, set_noise

gids = net.to_nest()

# add noise to the excitatory neurons
excs = list(pop["excitatory"].nest_gids)
set_noise(excs, 10., 2.)

# record
groups = [key for key in net.population]
recorder, record = monitor_groups(groups, net)

'''
Simulate and plot.
'''
simtime = 2000.
nest.Simulate(simtime)

plot_activity(
    recorder, record, network=net, show=True, hist=False,
    limits=(0, simtime))
