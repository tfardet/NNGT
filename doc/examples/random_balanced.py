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

"""
Use NNGT to analyze NEST-simulated activity of a random balanced network.
"""

import os

import numpy as np
from scipy.special import lambertw

import nngt
import nngt.generation as ng


'''
Simulation parameters
'''

num_omp = int(os.environ.get("OMP", 8))
nngt.set_config("omp", num_omp)
nngt.set_config("seeds", [10 + i for i in range(num_omp)])

dt = 0.1         # the resolution in ms
simtime = 1000.  # Simulation time in ms
delay = 1.5      # synaptic delay in ms

g = 4.0          # ratio inhibitory weight/excitatory weight
eta = 2.0        # external rate relative to threshold rate
epsilon = 0.1    # connection probability


'''
Tools
'''

def ComputePSPnorm(tauMem, CMem, tauSyn):
    a = (tauMem / tauSyn)
    b = (1.0 / tauSyn - 1.0 / tauMem)

    # time of maximum
    t_max = 1.0 / b * (-lambertw(-np.exp(-1.0 / a) / a, -1) - 1.0 / a)
    t_max = np.real(t_max)

    # maximum of PSP for current of unit amplitude
    return (np.exp(1.0) / (tauSyn * CMem * b) *
            ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b -
             t_max * np.exp(-t_max / tauSyn)))


'''
Network parameters
'''

order = 1000
NE = 4 * order          # number of excitatory neurons
NI = 1 * order          # number of inhibitory neurons
N_neurons = NE + NI     # number of neurons in total
N_rec = 50              # record from 50 neurons

CE = int(epsilon * NE)  # number of excitatory synapses per neuron
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
C_tot = int(CI + CE)    # total number of synapses per neuron

tauSyn = 0.5  # synaptic time constant in ms
tauMem = 20.  # time constant of membrane potential in ms
CMem = 250.   # capacitance of membrane in in pF
theta = 20.   # membrane threshold potential in mV

neuron_params = {"C_m": CMem,
                 "tau_m": tauMem,
                 "tau_syn_ex": tauSyn,
                 "tau_syn_in": tauSyn,
                 "t_ref": 2.0,
                 "E_L": 0.0,
                 "V_reset": 0.0,
                 "V_m": 0.0,
                 "V_th": theta}

J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
J = 0.8            # postsynaptic amplitude in mV
J_ex = J / J_unit  # amplitude of excitatory postsynaptic current
J_in = g * J_ex    # amplitude of inhibitory postsynaptic current

nu_th = (theta * CMem) / (J_ex * CE * np.e * tauMem * tauSyn)
nu_ex = eta * nu_th
p_rate = 400. * nu_ex * CE


'''
Create the population and network
'''

pop = nngt.NeuralPop.exc_and_inhib(
    N_neurons, en_model="iaf_psc_alpha", en_param=neuron_params,
    in_model="iaf_psc_alpha", in_param=neuron_params)

net = nngt.Network(population=pop)

ng.connect_groups(net, pop, "excitatory", "gaussian_degree", degree_type="in",
                  avg=CE, std=0.1*CE, weights=J_ex, delays=delay)

ng.connect_groups(net, pop, "inhibitory", "gaussian_degree", degree_type="in",
                  avg=CI, std=0.1*CI, weights=J_in, delays=delay)


'''
Tools to analyze the activity
'''

def coefficient_of_variation(spikes):
    from scipy.stats import variation
    times = np.array(spikes["times"])
    nrns  = np.array(spikes["senders"])

    return np.nanmean([variation(np.diff(times[nrns == i]))
                       for i in np.unique(nrns)])


def cross_correlation(spikes, time, numbins=1000):
    times = np.array(spikes["times"])
    nrns  = np.array(spikes["senders"])

    a = [np.histogram(times[nrns == x], numbins, range=[0, time])[0]
         for x in np.unique(nrns)]

    c = np.cov(a)
    std = np.sqrt(np.diag(c))
    c = np.divide(c, std[:,np.newaxis])
    c = np.divide(c, std[:,np.newaxis].T)

    return np.nanmean(c)


'''
Send the network to NEST, set noise and randomize parameters
'''

if nngt.get_config('with_nest'):
    import nest
    import nngt.simulation as ns
    from nngt.analysis import get_spikes

    nest.ResetKernel()

    nest.SetKernelStatus({"resolution": dt, "print_time": True,
                          "overwrite_files": True, 'local_num_threads': 4})

    gids = net.to_nest()

    pg = ns.set_poisson_input(gids, rate=p_rate,
                              syn_spec={"weight": J_ex, "delay": delay})

    recorders, records = ns.monitor_groups(
        ["excitatory", "inhibitory"], network=net)

    nest.Simulate(simtime)

    if nngt.get_config('with_plot'):
        ideg = net.get_degrees("in", edge_type="inhibitory")
        edeg = net.get_degrees("in", edge_type="excitatory")

        # plot the basic activity
        ns.plot_activity(
            recorders, records, network=net, show=False)
        # sort by firing rate
        ns.plot_activity(
            recorders, records, network=net, sort="firing_rate", show=False)
        # by in-degree (not working)
        ns.plot_activity(
            recorders, records, network=net, sort="in-degree", show=False)
        # k_e - k_i (good)
        ns.plot_activity(
            recorders, records, network=net, sort=edeg-ideg, show=True)

        # then show correlations
        exc_nodes = pop["excitatory"].ids
        inh_nodes = pop["inhibitory"].ids
        nngt.plot.correlation_to_attribute(
            net, (edeg-ideg)[exc_nodes], "firing_rate", nodes=exc_nodes,
            show=False)
        nngt.plot.correlation_to_attribute(
            net, (edeg-ideg)[inh_nodes], "firing_rate", nodes=inh_nodes,
            show=True)

        # print the CV and CC
        data_exc = get_spikes(recorders[0], astype="np")
        data_inh = get_spikes(recorders[1], astype="np")

        spikes = {
            "senders": list(data_exc[:, 0]) + list(data_inh[:, 0]),
            "times": list(data_exc[:, 1]) + list(data_inh[:, 1]),
        }

        print(coefficient_of_variation(spikes),
              cross_correlation(spikes, simtime))
