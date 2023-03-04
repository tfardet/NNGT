# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# doc/examples/nest_network.py

""" Network generation for NEST """

import numpy as np

import nngt
import nngt.generation as ng


''' Create groups with different parameters '''
# adaptive spiking neurons
base_params = {
    'E_L': -60., 'V_th': -58., 'b': 20., 'tau_w': 100.,
    'V_reset': -65., 't_ref': 2., 'g_L': 10., 'C_m': 250.
}
# oscillators
params1, params2 = base_params.copy(), base_params.copy()
params1.update(
    {'E_L': -65., 'b': 40., 'I_e': 200., 'tau_w': 400., "V_th": -57.})
# bursters
params2.update({'b': 25., 'V_reset': -55., 'tau_w': 300.})

oscill = nngt.NeuralGroup(
    nodes=400, neuron_model='aeif_psc_alpha', neuron_type=1,
    neuron_param=params1)

burst = nngt.NeuralGroup(
    nodes=200, neuron_model='aeif_psc_alpha', neuron_type=1,
    neuron_param=params2)

adapt = nngt.NeuralGroup(
    nodes=200, neuron_model='aeif_psc_alpha', neuron_type=1,
    neuron_param=base_params)

model = 'model'

try:
    import nest
    nest.NodeCollection()
    model = 'synapse_model'
except:
    pass

synapses = {
    'default': {model: 'tsodyks2_synapse'},
    ('oscillators', 'bursters'): {model: 'tsodyks2_synapse', 'U': 0.6},
    ('oscillators', 'oscillators'): {model: 'tsodyks2_synapse', 'U': 0.7},
    ('oscillators', 'adaptive'): {model: 'tsodyks2_synapse', 'U': 0.5}
}

'''
Create the population that will represent the neuronal
network from these groups
'''
pop = nngt.NeuralPop.from_groups(
    [oscill, burst, adapt],
    names=['oscillators', 'bursters', 'adaptive'], syn_spec=synapses)

'''
Create the network from this population,
using a Gaussian in-degree
'''
net = ng.gaussian_degree(
    100., 15., population=pop, weights=155., delays=5.)


'''
Send the network to NEST, monitor and simulate
'''
if nngt.get_config('with_nest'):
    import nngt.simulation as ns
    import nest

    nest.ResetKernel()

    nest.SetKernelStatus({'local_num_threads': 4})

    gids = net.to_nest()

    nngt.simulation.randomize_neural_states(net, {"w": ("uniform", 0, 200)})

    recorders, records = ns.monitor_groups(pop.keys(), net)

    nest.Simulate(1600.)

    if nngt.get_config('with_plot'):
        ns.plot_activity(recorders, records, network=net, show=True)
