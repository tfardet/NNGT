#!/usr/bin/env python
#-*- coding:utf-8 -*-


def make_nest_network(network, node_model='iaf_neuron', syn_model='static_synapse'):
	'''
	Create a new subnetwork which will be filled with neurons and
	connector objects to reproduce the topology from the initial network.
    
    Parameters
    ----------
    network: AGNet.core.NeuralNetwork
		the network we want to reproduce in NEST.
	node_model: string
		Neural model to use when creating nodes if 'network' has no
		specific model property
	syn_model: string
		Synaptic model to use when connecting nodes if 'network' has no
		specific model property
    
    Returns
    -------
    subnet : tuple (node in NEST)
        GID of the new NEST subnetwork
	'''
	
	subnet = nest.Create('subnet')
	nest.ChangeSubnet(subnet)
	num_neurons = network.num_vertices()
	
	# get all properties as scipy.sparse.lil matrices
	b_uniform = network.is_uniform()
	neural_model = network.get_neural_model()
	lil_adjacency = network.get_adjacency_matrix().tolil()
	lil_weights, lil_delays = network.connect_properties()
	
	# create the nodes, then connect them
	if b_uniform:
		nest.Create(neural_model,num_neurons)
	else:
		for tpl_model in neural_model:
			nest.Create(tpl_model[0],params=tpl_model[1])
			
	
	
	
