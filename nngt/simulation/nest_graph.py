#!/usr/bin/env python
#-*- coding:utf-8 -*-

import nest
import numpy as np
import scipy.sparse as ssp
from scipy.optimize import root
from scipy.signal import argrelmax, argrelmin

import matplotlib.pyplot as plt

from nngt.globals import WEIGHT, DELAY
from nngt.lib import InvalidArgument
from nngt.lib.sorting import _sort_groups

__all__ = [
    'make_nest_network',
    'get_nest_network',
    'reproducible_weights'
]


#-----------------------------------------------------------------------------#
# Topology
#------------------------
#

def make_nest_network(network, use_weights=True):
    '''
    Create a new network which will be filled with neurons and
    connector objects to reproduce the topology from the initial network.

    Parameters
    ----------
    network: :class:`nngt.Network` or :class:`nngt.SpatialNetwork`
        the network we want to reproduce in NEST.
    use_weights : bool, optional (default: True)
        Whether to use the network weights or default ones (value: 10.).

    Returns
    -------
    gids : tuple (nodes in NEST)
        GIDs of the neurons in the network.
    '''
    gids = []

    # link NEST Gids to nngt.Network ids as neurons are created
    num_neurons = network.node_nb()
    ia_nngt_ids = np.zeros(num_neurons, dtype=int)
    ia_nest_gids = np.zeros(num_neurons, dtype=int)
    ia_nngt_nest = np.zeros(num_neurons, dtype=int)
    current_size = 0

    # sort groups by size
    _, groups = _sort_groups(network.population)

    for group in groups:
        group_size = len(group.id_list)
        if group_size:
            ia_nngt_ids[current_size:current_size + group_size] = group.id_list
            # clean up neuron_param dict
            defaults = nest.GetDefaults(group.neuron_model)
            n_param = {key: val for key, val in group.neuron_param.items()
                       if key in defaults and key != "model"}
            # create neurons
            gids_tmp = nest.Create(group.neuron_model, group_size, n_param)
            idx_nest = ia_nngt_ids[np.arange(
                current_size, current_size + group_size)]
            ia_nest_gids[current_size:current_size + group_size] = gids_tmp
            ia_nngt_nest[idx_nest] = gids_tmp
            current_size += group_size
            gids.extend(gids_tmp)
        
    # conversions ids/gids
    network.nest_gid = ia_nngt_nest
    network._id_from_nest_gid = {
        gid: idx for (idx, gid) in zip(ia_nngt_ids, ia_nest_gids)
    }

    # get all properties as scipy.sparse.csr matrices
    csr_weights = network.adjacency_matrix(False, True)
    csr_delays = network.adjacency_matrix(False, DELAY)

    cspec = 'one_to_one'
    for group in network.population.values():
        # get the nodes ids and switch the sources back to their real values
        if len(group.id_list) > 0:
            min_idx = np.min(group.id_list)
            src_ids = csr_weights[group.id_list, :].nonzero()[0] + min_idx
            trgt_ids = csr_weights[group.id_list, :].nonzero()[1]
            # switch to nest gids
            src_ids = network.nest_gid[src_ids]
            trgt_ids = network.nest_gid[trgt_ids]
            # prepare the synaptic parameters
            syn_param = {"model": group.syn_model}
            for key, val in group.syn_param.items():
                if key != "receptor_port":
                    syn_param[key] = np.repeat(val, len(src_ids))
                else:
                    syn_param[key] = val
            # prepare weights
            syn_sign = group.neuron_type
            if use_weights:
                syn_param[WEIGHT] = syn_sign*csr_weights[group.id_list, :].data
            else:
                syn_param[WEIGHT] = np.repeat(syn_sign, len(src_ids))
            syn_param[DELAY] = csr_delays[group.id_list, :].data
            nest.Connect(
                src_ids, trgt_ids, syn_spec=syn_param, conn_spec=cspec)

    return tuple(ia_nest_gids)


def get_nest_network(id_converter=None):
    '''
    Get the adjacency matrix describing a NEST network.

    Parameters
    ----------
    id_converter : dict, optional (default: None)
        A dictionary which maps NEST gids to the desired neurons ids.

    Returns
    -------
    mat_adj : :class:`~scipy.sparse.lil_matrix`
        Adjacency matrix of the network.
    '''
    gids = nest.GetNodes()[0]
    n = len(gids)
    mat_adj = ssp.lil_matrix((n,n))
    if id_converter is None:
        id_converter = { idx:i for i,idx in enumerate(gids) }

    for i in range(n):
        src = id_converter[gids[i]]
        connections = nest.GetConnections(source=(gids[i],))
        info = nest.GetStatus(connections)
        for dic in info:
            mat_adj.rows[src].append(id_converter[dic['target']])
            mat_adj.data[src].append(dic[WEIGHT])

    return mat_adj
    

#-----------------------------------------------------------------------------#
# Weights
#------------------------
#

def _value_psp(weight, neuron_model, di_param, timestep, simtime):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution":timestep})
    # create neuron and recorder
    neuron = nest.Create(neuron_model, params=di_param)
    V_rest = nest.GetStatus(neuron)[0]["E_L"]
    nest.SetStatus(neuron, {"V_m": V_rest})
    vm = nest.Create("voltmeter", params={"interval": timestep})
    nest.Connect(vm, neuron)
    # send the initial spike
    sg = nest.Create("spike_generator", params={'spike_times':[timestep],
                                                'spike_weights':weight})
    nest.Connect(sg, neuron)
    nest.Simulate(simtime)
    # get the max and its time
    dvm = nest.GetStatus(vm)[0]
    da_voltage = dvm["events"]["V_m"]
    idx = np.argmax(da_voltage)
    if idx == len(da_voltage - 1):
        raise InvalidArgument("simtime too short: PSP maximum is out of range")
    else:
        val = da_voltage[idx] - V_rest
        return val
        
def _find_extremal_weights(min_weight, max_weight, neuron_model, di_param={},
                           precision=0.1, timestep=0.01, simtime=10.):
    '''
    Find the values of the connection weights that will give PSP responses of
    `min_weight` and `max_weight` in mV.
    
    Parameters
    ----------
    min_weight : float
        Minimal weight.
    max_weight : float
        Maximal weight.
    neuron_model : string
        Name of the model used.
    di_param : dict, optional (default: {})
        Parameters of the model, default parameters if not supplied.
    precision : float, optional (default : -1.)
        Precision with which the result should be obtained. If the value is
        equal to or smaller than zero, it will default to 0.1% of the value.
    timestep : float, optional (default: 0.01)
        Timestep of the simulation in ms.
    simtime : float, optional (default: 10.)
        Simulation time in ms (default: 10).
    
    .. note :
        If the parameters used are not the default ones, they MUST be provided,
        otherwise the resulting weights will likely be WRONG.
    '''
    # define the function for root finding
    def _func_min(weight):
        val = _value_psp(weight, neuron_model, di_param, timestep, simtime)
        return val - min_weight
    def _func_max(weight):
        val = _value_psp(weight, neuron_model, di_param, timestep, simtime)
        return val - max_weight
    # @todo: find highest and lowest value that result in spike emission
    # get root
    min_w = root(_func_min, min_weight, tol=0.1*min_weight/100.).x[0]
    max_w = root(_func_max, max_weight, tol=0.1*max_weight/100.).x[0]
    return min_w, max_w

def _get_psp_list(bins, neuron_model, di_param, timestep, simtime):
    '''
    Return the list of effective weights from a list of NEST connection
    weights.
    '''
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution":timestep})
    # create neuron and recorder
    neuron = nest.Create(neuron_model, params=di_param)
    vm = nest.Create("voltmeter", params={"interval": timestep})
    nest.Connect(vm, neuron)
    # send the spikes
    times = [ timestep+n*simtime for n in range(len(bins)) ]
    sg = nest.Create("spike_generator", params={'spike_times':times,
                                                'spike_weights':bins})
    nest.Connect(sg, neuron)
    nest.Simulate((len(bins)+1)*simtime)
    # get the max and its time
    dvm = nest.GetStatus(vm)[0]
    da_voltage = dvm["events"]["V_m"]
    da_times = dvm["events"]["times"]
    da_max_psp = da_voltage[ argrelmax(da_voltage) ]
    da_min_psp = da_voltage[ argrelmin(da_voltage) ]
    da_max_psp -= da_min_psp
    if len(bins) != len(da_max_psp):
        raise InvalidArgument("simtime too short: all PSP maxima are not in \
range")
    else:
        plt.plot(da_times, da_voltage)
        plt.show()
        return da_max_psp

def reproducible_weights(weights, neuron_model, di_param={}, timestep=0.05,
                         simtime=50., num_bins=1000, log=False):
    '''
    Find the values of the connection weights that will give PSP responses of
    `min_weight` and `max_weight` in mV.
    
    Parameters
    ----------
    weights : list of floats
        Exact desired synaptic weights.
    neuron_model : string
        Name of the model used.
    di_param : dict, optional (default: {})
        Parameters of the model, default parameters if not supplied.
    timestep : float, optional (default: 0.01)
        Timestep of the simulation in ms.
    simtime : float, optional (default: 10.)
        Simulation time in ms (default: 10).
    num_bins : int, optional (default: 10000)
        Number of bins used to discretize the exact synaptic weights.
    log : bool, optional (default: False)
        Whether bins should use a logarithmic scale.
    
    .. note :
        If the parameters used are not the default ones, they MUST be provided,
        otherwise the resulting weights will likely be WRONG.
    '''
    min_weight = np.min(weights)
    max_weight = np.max(weights)
    # get corrected weights
    min_corr, max_corr = _find_extremal_weights(min_weight, max_weight,
                    neuron_model, di_param, timestep=timestep, simtime=simtime)
    #~ # bin them
    bins = None
    if log:
        log_min = np.log10(min_corr)
        log_max = np.log10(max_corr)
        bins = np.logspace(log_min, log_max, num_bins)
    else:
        bins = np.linspace(min_corr, max_corr, num_bins)
    binned_weights = _get_psp_list(bins,neuron_model,di_param,timestep,simtime)
    idx_binning = np.digitize(weights, binned_weights)
    return bins[ idx_binning ]

    
