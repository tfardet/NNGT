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

''' Network generation for NEST '''

import numpy as np

import nngt
import nngt.generation as ng


# nngt.seed(0)


# -------------------- #
# Generate the network #
# -------------------- #

'''
Build a network with two populations:
* excitatory (80%)
* inhibitory (20%)
'''
num_nodes = 1000

# 800 excitatory neurons, 200 inhibitory
net = nngt.Network.ei_network(num_nodes, ei_ratio=0.2)

'''
Connect the populations.
'''
# exc -> inhib (Erdos-Renyi)
ng.connect_neural_types(net, 1, -1, "erdos_renyi", density=0.035)

# exc -> exc (Newmann-Watts)
prop_nw = {
    "coord_nb": 10,
    "proba_shortcut": 0.1
}
ng.connect_neural_types(net, 1, 1, "newman_watts", **prop_nw)

# inhib -> exc (Random scale-free)
prop_rsf = {
    "in_exp": 2.1,
    "out_exp": 2.6,
    "density": 0.2
}
ng.connect_neural_types(net, -1, 1, "random_scale_free", **prop_rsf)

# inhib -> inhib (Erdos-Renyi)
ng.connect_neural_types(net, -1, -1, "erdos_renyi", density=0.04)


# ------------------ #
# Simulate with NEST #
# ------------------ #

if nngt.get_config('with_nest'):
    import nest
    import nngt.simulation as ns

    '''
    Prepare the network and devices.
    '''
    # send to NEST,
    gids = net.to_nest()
    # excite
    ns.set_poisson_input(gids, rate=100000.)
    # record
    groups = [key for key in net.population]
    recorder, record = ns.monitor_groups(groups, net)

    '''
    Simulate and plot.
    '''
    simtime = 100.
    nest.Simulate(simtime)

    if nngt.get_config('with_plot'):
        ns.plot_activity(
            recorder, record, network=net, show=True, hist=False,
            limits=(0,simtime))
