# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# doc/examples/basic_nest_network.py

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
net = nngt.Network.exc_and_inhib(num_nodes, ei_ratio=0.2)

'''
Connect the populations.
'''
# exc -> inhib (Erdos-Renyi)
ng.connect_neural_types(net, 1, -1, "erdos_renyi", density=0.035)

# exc -> exc (Newmann-Watts)
prop_nw = {
    "coord_nb": 10,
    "proba_shortcut": 0.1,
    "reciprocity_circular": 1.
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
    import nngt.simulation as ns
    import nest

    '''
    Prepare the network and devices.
    '''
    # send to NEST
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
            recorder, record, network=net, show=True, limits=(0,simtime))

    '''
    A reminder about weights of inhibitory connections
    '''

    # sign of NNGT versus NEST inhibitory connections
    igroup = net.population["inhibitory"]

    # in NNGT
    iedges = net.get_edges(source_node=igroup.ids)
    w_nngt = set(net.get_weights(edges=iedges))

    # in NEST
    try:
        # nest 2
        iconn  = nest.GetConnections(
            source=list(net.population["inhibitory"].nest_gids),
            target=list(net.population.nest_gids))
    except:
        # nest 3
        import nest
        s = nest.NodeCollection(net.population["inhibitory"].nest_gids)
        t = nest.NodeCollection(net.population.nest_gids)

        iconn  = nest.GetConnections(source=s, target=t)

    w_nest = set(nest.GetStatus(iconn, "weight"))

    # In NNGT, inhibitory weights are positive to work with graph analysis
    # methods; they are automatically converted to negative weights in NEST
    print("NNGT weights:", w_nngt, "versus NEST weights", w_nest)

    assert w_nngt == {1} and w_nest == {-1}
