#!/usr/bin/env python
#-*- coding:utf-8 -*-

import nest
import numpy as np


def make_nest_network(network):
    '''
    Create a new subnetwork which will be filled with neurons and
    connector objects to reproduce the topology from the initial network.

    Parameters
    ----------
    network: :class:`nngt.Network` or :class:`nngt.SpatialNetwork`
        the network we want to reproduce in NEST.

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
    print("neurons:", num_neurons)
    ia_nngt_ids = np.zeros(num_neurons, dtype=int)
    ia_nest_gids = np.zeros(num_neurons, dtype=int)
    current_size = 0

    for group in network.population.itervalues():
        group_size = len(group.id_list)
        ia_nngt_ids[current_size:current_size+group_size] = group.id_list
        gids = nest.Create(group.neuron_model, group_size, group.neuron_param)
        print("gids",gids)
        ia_nest_gids[current_size:current_size+group_size] = gids
        current_size += group_size

    # get all properties as scipy.sparse.lil matrices
    #~ neural_model = network.get_neural_model()
    lil_adjacency = network.adjacency_matrix().tolil()
    #~ lil_delays = network.connect_properties()

    # connect neurons
    dic = { "target": None, "weight": None, "delay": None }
    for i in range(num_neurons):
        da_targets = np.array(lil_adjacency.rows[ia_nngt_ids[i]], dtype=int)
        dic["target"] = ia_nest_gids[da_targets]
        num_connect = len(da_targets)
        dic["weight"] = np.repeat(1., num_connect)
        dic["delay"] = np.repeat(1., num_connect)
        dic_prop = network.neuron_properties(i)
        for key, val in dic_prop["syn_param"].items():
            dic[key] = np.repeat(val, num_connect)
            #@todo: adapt to allow for variance in the parameters
        nest.DataConnect((ia_nest_gids[i],), (dic,), "static_synapse")
    return subnet, tuple(ia_nest_gids)
	
