#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Main object for database management """

import platform
from datetime import datetime
from itertools import permutations

import nest

from nngt.lib.db_tools import psutil
from .db_generation import *


__all__ = [ 'nngt_db' ]



class NNGTdb:
    '''
    Class containing the database object and methods to store the simulation
    properties.
    '''

    def __init__(self):
        self.db = db
        self.db.connect()
        self.current_simulation = None
        self.simulator_lib = None
        self.computer = None
        self.neuralnet = None
        self.entries = {}

    def make_computer_entry(self):
        '''
        Get the computer properties.

        Returns
        -------
        computer : :class:`~nngt.database.Computer`
            New computer entry.
        '''
        name = platform.node()
        comp_prop = {
            'name': name if name else "unknown",
            'platform': platform.platform(),
            'python': platform.python_version(),
            'cores': psutil.cpu_count(),
            'ram': psutil.virtual_memory().total,
        }
        computer = Computer(**comp_prop)
        return computer

    def make_network_entry(self, network):
        '''
        Get the network properties.

        Parameters
        ----------
        network : :class:`~nngt.Network` or subclass
            Network used in the current simulation.

        Returns
        -------
        neuralnet : :class:`~nngt.database.NeuralNetwork`
            New NeuralNetwork entry.
        '''
        weighted = network.is_weighted()
        net_prop = {
            'network_type': network.type,
            'directed': network.is_directed(),
            'nodes': network.node_nb(),
            'edges': network.edge_nb(),
            'weighted': weighted,
            'compressed_file': network.to_string()
        }
        if weighted:
            net_prop['weight_distribution'] = network._w['distrib']
        neuralnet = NeuralNetwork(**net_prop)
        return neuralnet

    def make_neuron_entry(self, network, group):
        '''
        Get the neuronal properties.

        Parameters
        ----------
        network : :class:`~nngt.Network` or subclass
            Network used in the current simulation.
        group : :class:`~nngt.core.NeuralGroup`
            Group which properties will be fetched.

        Returns
        -------
        neuron : :class:`~nngt.database.Neuron`
            New neuron entry.
        '''
        nngt_id = group.id_list[0]
        gid = network.nest_gid[nngt_id]
        # get the dictionary
        neuron_prop = nest.GetStatus((gid,))[0]
        # update Neuron class accordingly
        update_node_class("neuron", **neuron_prop)
        neuron = Neuron(**neuron_prop)
        return neuron

    def make_synapse_entry(self, network, group_pre, group_post):
        '''
        Get the synaptic properties.

        Parameters
        ----------
        network : :class:`~nngt.Network` or subclass
            Network used in the current simulation.
        group_pre : :class:`~nngt.core.NeuralGroup`
            Pre-synaptic group.
        group_post : :class:`~nngt.core.NeuralGroup`
            Post-synaptic group.

        Returns
        -------
        synapse : :class:`~nngt.database.Synapse`
            New synapse entry.
        '''
        syn_model = group_pre.syn_model
        source_gids = network.nest_gid[group_pre.id_list]
        target_gids = network.nest_gid[group_post.id_list]
        connections = nest.GetConnections(synapse_model=syn_model,
                                        source=source_gids, target=target_gids)
        # get the dictionary
        syn_prop={}
        if connections:
            syn_prop = nest.GetStatus((connections[0],))[0]
            # update Synapse class accordingly
            update_node_class("synapse", **syn_prop)
        synapse = Synapse(**syn_prop)
        return synapse

    def make_connection_entry(self, neuron_pre, neuron_post, synapse):
        '''
        Create the entries for the Connections table from a list of
        (pre, post, syn) triples.

        Parameters
        ----------
        neuron_pre : :class:`~nngt.database.Neuron`
            Pre-synaptic neuron entry.
        neuron_post : :class:`~nngt.database.Neuron`
            Post-synaptic neuron entry.
        synapse : :class:`~nngt.database.Synapse`
            Synapse entry.
        
        Returns
        -------
        :class:`~nngt.database.Connection`
        '''
        return Connection(pre=triple.pre, post=triple.post, synapse=triple.syn)

    def get_simulation_prop(self, simulator, c_entry, net_entry, network):
        '''
        Get the simulation properties.
        
        Parameters
        ----------
        simulator : str
            Name of the simulator use (NEST, BRIAN...).
        c_entry : :class:`~nngt.database.Computer`
            Current Computer entry in the database.
        net_entry : :class:`~nngt.database.NeuralNetwork`
            Current NeuralNetwork entry in the database.
        network : :class:`~nngt.Network`
            Network used for the simulation.

        Returns
        -------
        sim_prop : dict
            Dictionary containing the relevant key/value pairs to fill the
            :class:`~nngt.database.Simulation` class.
        '''
        pop, size = [], []
        for name, group in iter(network.population.items()):
            pop.append(name)
            size.append(len(group.id_list))
        self.current_simulation = {
            'start_time': datetime(),
            'simulated_time': nest.GetKernelStatus('time'),
            'resolution': nest.GetKernelStatus('resolution'),
            'simulator': simulator.lower(),
            'grnd_seed': nest.GetKernelStatus('grng_seed'),
            'local_seeds': nest.GetKernelStatus('rng_seeds'),
            'computer': c_entry,
            'network': net_entry,
            'population': pop,
            'pop_sizes': size
        }
        
    def log_simulation_start(self, network, simulator):
        '''
        Record the simulation start time, all nodes, connections, network, and
        computer properties, as well as some of simulation's.

        Parameters
        ----------
        network : :class:`~nngt.Network` or subclass
            Network used for the current simulation.
        simulator : str
            Name of the simulator.
        '''
        if not self.is_clear():
            raise RuntimeError("Database log started without clearing the \
previous one.")
        # computer and network data
        self.computer = self.make_computer_entry()
        self.neuralnet = self.make_network_entry(network)
        # neurons, synapses and connections
        perm_names = tuple(permutations(network.population.keys(), 2))
        perm_groups = tuple(permutations(network.population.values(), 2))
        self.entries = {}
        for (name_pre, name_post), (pre, post) in zip(perm_names, perm_groups):
            if name_pre not in entries:
                self.entries[name_pre] = self.make_neuron_entry(network, pre)
            if name_post not in entries:
                self.entries[name_post] = self.make_neuron_entry(network, post)
            synapse = self.make_synapse_entry(network, entries[name_pre],
                                              entries[name_post])
            self.entries["syn_{}->{}".format(name_pre, name_post)] = synapse
            conn = self.make_connection_entry(neuron_pre, neuron_post, synapse)
            self.entries["conn_{}->{}".format(name_pre, name_post)]
        

    def log_simulation_end(self, activity=None):
        '''
        Record the simulation completion and simulated times, save the data,
        then reset.
        '''
        if self.is_clear():
            raise RuntimeError("Database log ended with empy log.")
        # get completion time and simulated time
        self.current_simulation['completion_time'] = datetime()
        start_time = self.current_simulation['simulated_time']
        new_time = nest.GetKernelStatus('time')
        self.current_simulation['simulated_time'] = new_time - start_time
        # save data and reset
        self.computer.save()
        self.neuralnet.save()
        for entry in iter(self.entries.values()):
            entry.save()
        simul_data = Simulation(**self.current_simulation)
        simul_data.save()
        self.reset()

    def is_clear(self):
        ''' Check that the logs are clear. '''
        clear = True
        clear *= self.current_simulation is None
        clear *= self.simulator_lib is None
        clear *= self.computer is None
        clear *= self.neuralnet is None
        clear *= not self.entries
        return clear

    def reset(self):
        ''' Reset log status. '''
        self.current_simulation = None
        self.simulator_lib = None
        self.computer = None
        self.neuralnet = None
        self.entries = {}


#-----------------------------------------------------------------------------#
# Main database object
#------------------------
#
        
nngt_db = NNGTdb() #: main database object

