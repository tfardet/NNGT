# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# doc/examples/nest_receptor_ports.py

''' Using complex NEST models '''

import numpy as np
import nngt
import nngt.generation as ng


# np.random.seed(0)


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

synapses = {
    (1, 1):   exc_syn,
    (1, -1):  exc_syn,
    (-1, 1):  inh_syn,
    (-1, -1): inh_syn,
}

pop = nngt.NeuralPop.exc_and_inhib(
    num_neurons, en_model=neuron_model, in_model=neuron_model,
    syn_spec=synapses)

# create the network and send it to NEST
w_prop = {"distribution": "gaussian", "avg": 0.2, "std": .05}
net = nngt.generation.gaussian_degree(
    avg_degree, std_degree, population=pop, weights=w_prop)

'''
Send to NEST and set excitation and recorders
'''
if nngt.get_config('with_nest'):
    import nest
    import nngt.simulation as ns

    nest.ResetKernel()

    gids = net.to_nest()

    # add noise to the excitatory neurons
    excs = list(pop["excitatory"].nest_gids)
    inhs = list(pop["inhibitory"].nest_gids)
    ns.set_noise(excs, 10., 2.)
    ns.set_noise(inhs, 5., 1.)

    # record
    groups = [key for key in net.population]
    recorder, record = ns.monitor_groups(groups, net)

    '''
    Simulate and plot.
    '''
    simtime = 2000.
    nest.Simulate(simtime)

    if nngt.get_config('with_plot'):
        ns.plot_activity(
            recorder, record, network=net, show=True, histogram=False,
            limits=(0, simtime))
