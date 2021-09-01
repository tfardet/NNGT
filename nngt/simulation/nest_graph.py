#-*- coding:utf-8 -*-
#
# simulation/nest_graph.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import nest
import numpy as np
import scipy.sparse as ssp
from scipy.optimize import root
from scipy.signal import argrelmax, argrelmin

import nngt
from nngt.lib import InvalidArgument, nonstring_container, WEIGHT, DELAY
from nngt.lib.sorting import _sort_groups
from nngt.lib.test_functions import mpi_checker
from nngt.lib.graph_helpers import _get_syn_param
from .nest_utils import nest_version, _get_nest_gids


__all__ = [
    'make_nest_network',
    'get_nest_adjacency',
    'reproducible_weights'
]


# -------- #
# Topology #
# -------- #

@mpi_checker()
def make_nest_network(network, send_only=None, weights=True):
    '''
    Create a new network which will be filled with neurons and
    connector objects to reproduce the topology from the initial network.

    .. versionchanged:: 0.8
        Added `send_only` parameter.

    Parameters
    ----------
    network: :class:`nngt.Network` or :class:`nngt.SpatialNetwork`
        the network we want to reproduce in NEST.
    send_only : int, str, or list of str, optional (default: None)
        Restrict the nodes that are created in NEST to either inhibitory or
        excitatory neurons `send_only` :math:`\in \{ 1, -1\}` to a group or a
        list of groups.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.

    Returns
    -------
    gids : tuple or NodeCollection (nodes in NEST)
        GIDs of the neurons in the NEST network.
    '''
    gids = _get_nest_gids([])
    pop  = network.population

    send = list(network.population.keys())

    if send_only in (-1, 1):
        send = [g for g in send if pop[g].neuron_type == send_only]
    elif isinstance(send_only, str):
        send = [pop[send_only]]
    elif nonstring_container(send_only):
        send = [g for g in send_only]

    send = [g for g in send if pop[g].ids]

    # link NEST Gids to nngt.Network ids as neurons are created
    num_neurons  = network.node_nb()
    ia_nngt_ids  = np.full(num_neurons, -1, dtype=int)
    ia_nest_gids = np.full(num_neurons, -1, dtype=int)
    ia_nngt_nest = np.full(num_neurons, -1, dtype=int)
    current_size = 0

    for name in send:
        group = pop[name]
        group_size = len(group.ids)

        if group_size:
            ia_nngt_ids[current_size:current_size + group_size] = group.ids

            # clean up neuron_param dict, separate scalar and non-scalar
            defaults     = nest.GetDefaults(group.neuron_model)
            scalar_param = {}
            ns_param     = {}

            for key, val in group.neuron_param.items():
                if key in defaults and key != "model":
                    if nonstring_container(val) and len(val) == group_size:
                        ns_param[key] = val
                    else:
                        scalar_param[key] = val

            # create neurons:
            gids_tmp = nest.Create(group.neuron_model, group_size,
                                   scalar_param, _warn=False)

            # set non-scalar properties
            for k, v in ns_param.items():
                nest.SetStatus(gids_tmp, k, v, _warn=False)

            # create ids
            idx_nest = ia_nngt_ids[
                np.arange(current_size, current_size + group_size)]

            ia_nest_gids[current_size:current_size + group_size] = gids_tmp
            ia_nngt_nest[idx_nest] = gids_tmp
            current_size += group_size
            gids += gids_tmp

    # conversions ids/gids
    network.nest_gids = ia_nngt_nest
    network._id_from_nest_gid = {
        gid: idx for (idx, gid) in zip(ia_nngt_ids, ia_nest_gids)
    }

    # get all properties as scipy.sparse.csr matrices
    csr_weights = network.adjacency_matrix(types=False, weights=weights)
    csr_delays  = network.adjacency_matrix(types=False, weights=DELAY)

    cspec = 'one_to_one'

    num_conn = 0

    for src_name in send:
        src_group = pop[src_name]
        syn_sign = src_group.neuron_type

        # local connectivity matrix and offset to correct neuron id
        arr_idx = np.sort(src_group.ids).astype(int)

        local_csr = csr_weights[arr_idx, :]

        assert local_csr.shape[1] == network.node_nb()

        if len(src_group.ids) > 0 and pop.syn_spec is not None:
            # check whether custom synapses should be used
            local_tgt_names = [name for name in send if pop[name].ids]

            for tgt_name in send:
                tgt_group = pop[tgt_name]
                # get list of targets for each
                arr_tgt_idx = np.sort(tgt_group.ids).astype(int)

                # get non-wero entries (global NNGT ids)
                keep = local_csr[:, arr_tgt_idx].nonzero()

                local_tgt_ids = arr_tgt_idx[keep[1]]
                local_src_ids = arr_idx[keep[0]]

                if len(local_tgt_ids) and len(local_src_ids):
                    # get the synaptic parameters
                    syn_spec = _get_syn_param(
                        src_name, src_group, tgt_name, tgt_group, pop.syn_spec)

                    # check whether sign must be given or not
                    local_sign = syn_sign
                    if "receptor_type" in syn_spec:
                        if "cond" in tgt_group.neuron_model:
                            # do not specify the sign for conductance-based
                            # multisynapse model
                            local_sign = 1

                    # using A1 to get data from matrix
                    if weights:
                        syn_spec[WEIGHT] = local_sign *\
                            csr_weights[local_src_ids, local_tgt_ids].A1
                    else:
                        syn_spec[WEIGHT] = np.repeat(local_sign, len(tgt_ids))
    
                    syn_spec[DELAY] = \
                        csr_delays[local_src_ids, local_tgt_ids].A1

                    num_conn += len(local_src_ids)

                    # check backend
                    with_mpi = nngt.get_config("mpi")
                    if nngt.get_config("backend") == "nngt" and with_mpi:
                        comm = nngt.get_config("mpi_comm")
                        for i in range(comm.Get_size()):
                            sources = \
                                comm.bcast(network.nest_gids[local_src_ids], i)
                            targets = \
                                comm.bcast(network.nest_gids[local_tgt_ids], i)
                            sspec   = comm.bcast(syn_spec, i)
                            nest.Connect(sources, targets, syn_spec=sspec,
                                         conn_spec=cspec, _warn=False)
                            comm.Barrier()
                    else:
                        nest.Connect(
                            network.nest_gids[local_src_ids],
                            network.nest_gids[local_tgt_ids],
                            syn_spec=syn_spec, conn_spec=cspec, _warn=False)

        elif len(src_group.ids) > 0:
            # get NEST gids of sources and targets for each edge
            src_ids = network.nest_gids[local_csr.nonzero()[0] + min_sidx]
            tgt_ids = network.nest_gids[local_csr.nonzero()[1]]

            # prepare weights
            syn_spec = {
                WEIGHT: np.repeat(syn_sign, len(src_ids)).astype(float)
            }

            if weights not in {None, False}:
                syn_spec[WEIGHT] *= csr_weights[src_group.ids, :].data

            syn_spec[DELAY] = csr_delays[src_group.ids, :].data

            nest.Connect(src_ids, tgt_ids, syn_spec=syn_spec, conn_spec=cspec,
                         _warn=False)

    # tell the populaton that the network it describes was sent to NEST
    network.population._sent_to_nest()

    return gids


def get_nest_adjacency(id_converter=None):
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
    gids = (nest.GetNodes()[0] if nest_version == 2
            else np.asarray(nest.GetNodes()))

    n = len(gids)

    mat_adj = ssp.lil_matrix((n,n))

    if id_converter is None:
        id_converter = {idx: i for i, idx in enumerate(gids)}

    for i in range(n):
        src = id_converter[gids[i]]
        connections = nest.GetConnections(source=(gids[i],))
        info = nest.GetStatus(connections)
        for dic in info:
            mat_adj.rows[src].append(id_converter[dic['target']])
            mat_adj.data[src].append(dic[WEIGHT])

    return mat_adj


# ------- #
# Weights #
# ------- #

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

    Note
    ----
    If the parameters used are not the default ones, they MUST be provided,
    otherwise the resulting weights will likely be WRONG.
    '''
    min_weight = np.min(weights)
    max_weight = np.max(weights)

    # get corrected weights
    min_corr, max_corr = _find_extremal_weights(
        min_weight, max_weight, neuron_model, di_param, timestep=timestep,
        simtime=simtime)

    # bin them
    bins = None

    if log:
        log_min = np.log10(min_corr)
        log_max = np.log10(max_corr)
        bins = np.logspace(log_min, log_max, num_bins)
    else:
        bins = np.linspace(min_corr, max_corr, num_bins)

    binned_weights = _get_psp_list(
        bins, neuron_model, di_param, timestep, simtime)

    idx_binning = np.digitize(weights, binned_weights)

    return bins[ idx_binning ]


# ----- #
# Tools #
# ----- #

def _value_psp(weight, neuron_model, di_param, timestep, simtime):
    nest.ResetKernel(_warn=False)
    nest.SetKernelStatus({"resolution": timestep})

    # create neuron and recorder
    neuron = nest.Create(neuron_model, params=di_param, _warn=False)
    V_rest = nest.GetStatus(neuron)[0]["E_L"]
    nest.SetStatus(neuron, {"V_m": V_rest}, _warn=False)
    vm = nest.Create("voltmeter", params={"interval": timestep}, _warn=False)
    nest.Connect(vm, neuron, _warn=False)
    # send the initial spike
    sg = nest.Create(
        "spike_generator", params={'spike_times': [timestep],
                                   'spike_weights': weight},
        _warn=False)

    nest.Connect(sg, neuron, _warn=False)
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

    Note
    ----
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
    nest.ResetKernel(_warn=False)
    nest.SetKernelStatus({"resolution": timestep})
    # create neuron and recorder
    neuron = nest.Create(neuron_model, params=di_param, _warn=False)
    vm = nest.Create("voltmeter", params={"interval": timestep}, _warn=False)
    nest.Connect(vm, neuron, _warn=False)
    # send the spikes
    times = [timestep + n*simtime for n in range(len(bins))]
    sg = nest.Create(
        "spike_generator", params={'spike_times': times,
                                   'spike_weights': bins},
        _warn=False)
    nest.Connect(sg, neuron, _warn=False)
    nest.Simulate((len(bins)+1)*simtime)
    # get the max and its time
    dvm = nest.GetStatus(vm)[0]
    da_voltage = dvm["events"]["V_m"]
    da_times = dvm["events"]["times"]
    da_max_psp = da_voltage[argrelmax(da_voltage)]
    da_min_psp = da_voltage[argrelmin(da_voltage)]
    da_max_psp -= da_min_psp
    if len(bins) != len(da_max_psp):
        raise InvalidArgument("simtime too short: all PSP maxima are not in "
                              "range.")
    else:
        return da_max_psp
