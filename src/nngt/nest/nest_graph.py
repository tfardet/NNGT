#!/usr/bin/env python
#-*- coding:utf-8 -*-

import nest
import numpy as np
import scipy.sparse as ssp

from ..globals import WEIGHT, DELAY

__all__ = [ 'make_nest_network', 'get_nest_network' ]



def make_nest_network(network, use_weights=True):
    '''
    Create a new subnetwork which will be filled with neurons and
    connector objects to reproduce the topology from the initial network.

    Parameters
    ----------
    network: :class:`nngt.Network` or :class:`nngt.SpatialNetwork`
        the network we want to reproduce in NEST.
    use_weights : bool, optional (default: True)
        Whether to use the network weights or default ones (value: 10.).

    Returns
    -------
    subnet : tuple (node in NEST)
        GID of the new NEST subnetwork
    gids : tuple (nodes in NEST)
        GIDs of the neurons in `subnet`
    '''

    # create the node subnetwork
    subnet = nest.Create('subnet')
    nest.ChangeSubnet(subnet)

    # link NEST Gids to nngt.Network ids as neurons are created
    num_neurons = network.node_nb()
    ia_nngt_ids = np.zeros(num_neurons, dtype=int)
    ia_nest_gids = np.zeros(num_neurons, dtype=int)
    ia_nngt_nest = np.zeros(num_neurons, dtype=int)
    current_size = 0

    for group in network.population.itervalues():
        group_size = len(group.id_list)
        ia_nngt_ids[current_size:current_size+group_size] = group.id_list
        gids = nest.Create(group.neuron_model, group_size, group.neuron_param)
        idx_nest = ia_nngt_ids[np.arange(current_size,current_size+group_size)]
        ia_nest_gids[current_size:current_size+group_size] = gids
        ia_nngt_nest[idx_nest] = gids
        current_size += group_size
        
    # get all properties as scipy.sparse.lil matrices
    lil_adjacency = network.adjacency_matrix().tolil()
    lil_weights = network[WEIGHT]
    lil_delays = network[DELAY]
    
    # connect neurons
    dic = { "target": None, WEIGHT: None, DELAY: None }
    for i in range(num_neurons):
        dic_prop = network.neuron_properties(ia_nngt_ids[i])
        syn_model = dic_prop["syn_model"]
        for key, val in dic_prop["syn_param"].iteritems():
            dic[key] = np.repeat(val, num_connect)
        syn_sign = dic_prop["neuron_type"]
        ia_targets = np.array(lil_adjacency.rows[ia_nngt_ids[i]], dtype=int)
        dic["target"] = ia_nngt_nest[ia_targets]
        num_connect = len(ia_targets)
        if use_weights:
            dic[WEIGHT] = syn_sign*np.array(lil_weights.data[ia_nngt_ids[i]])
        else:
            dic[WEIGHT] = syn_sign*np.repeat(10., num_connect)
        dic[DELAY] = np.array(lil_delays.data[ia_nngt_ids[i]])
        nest.DataConnect((ia_nngt_nest[ia_nngt_ids[i]],), (dic,), syn_model)
        
    # conversions ids/gids
    network.nest_id = ia_nngt_nest
    network.id_from_nest_id = { gid:idx for (idx,gid) in zip(ia_nngt_ids,
                                                             ia_nest_gids) }
    return subnet, tuple(ia_nest_gids)


def get_nest_network(nest_subnet, id_converter=None):
    '''
    Get the adjacency matrix describing a NEST subnetwork.

    Parameters
    ----------
    nest_subnet : tuple
        Subnetwork node in NEST.
    id_converter : dict, optional (default: None)
        A dictionary which maps NEST gids to the desired neurons ids.

    Returns
    -------
    mat_adj : :class:`~scipy.sparse.lil_matrix`
        Adjacency matrix of the network.
    '''
    gids = nest.GetNodes(nest_subnet)[0]
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
            mat_adj.data[src].append(dic['weight'])

    return mat_adj
    
