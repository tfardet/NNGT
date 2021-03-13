#-*- coding:utf-8 -*-
#
# database/db_main.py
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

""" Main object for database management """

import platform
from datetime import datetime
from itertools import permutations

try:
    import nest
except ImportError:
    raise ImportError("Database module requires NEST to work.")

import peewee

import nngt
from nngt.lib.db_tools import psutil
from .db_generation import *


__all__ = ['NNGTdb']


class NNGTdb:
    '''
    Class containing the database object and methods to store the simulation
    properties.
    '''
    
    tables = {
        'activity': Activity,
        'computer': Computer,
        'connection': Connection,
        'neuralnetwork': NeuralNetwork,
        'neuron': Neuron,
        'simulation': Simulation,
        'synapse': Synapse
    }

    def __init__(self):
        self.db = nngt._main_db
        self.db.connect()
        self.db.create_tables(self.tables.values(), safe=True)
        self._update_models()
        self.activity = None
        self.current_simulation = None
        self.simulator_lib = None
        self.computer = None
        self.neuralnet = None
        self.connections = {}
        self.nodes = {}
    
    def _update_class(self, table, **kwargs):
        ''' Add a field for each property of the considered node. '''
        klass = self.tables[table]
        columns = [ x.name for x in self.db.get_columns(table) ]
        migrator = db_migrator[self.db.__class__.__name__](self.db)
        type_names = False
        if "dtype" in kwargs:
            type_names = kwargs["dtype"]
            del kwargs["dtype"]
        for attr, value in kwargs.items():
            if attr not in ignore:
                # generate field instance
                dtype = value if type_names else value.__class__.__name__
                val_field = val_to_field[dtype](null=True)
                if len(attr) == 1 and attr.isupper():
                    attr = 2*attr.lower()
                klass._meta.add_field(attr, val_field)
                # check whether the column exists on the table, if not create
                if not attr in columns:
                    try:
                        migrate(migrator.add_column(table, attr, val_field))
                    except peewee.OperationalError:
                        pass
        return klass
    
    def _update_models(self):
        ''' Update the models so that we can query the database with them '''
        tables = self.db.get_tables()
        for table in tables:
            delete = []
            col_names = self.db.get_columns(table)
            col_dict = { x.name: x.data_type for x in col_names}
            for attr in iter(col_dict.keys()):
                if attr == "compressed_file":
                    col_dict["compressed_file"] = "compressed"
                elif "_id" in attr:
                    delete.append(attr)
            col_dict["dtype"] = True
            for key in delete:
                del col_dict[key]
            self.tables[table] = self._update_class(table, **col_dict)

    def _make_computer_entry(self):
        ''' Get the computer properties.

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

    def _make_network_entry(self, network):
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
        if network is not None:
            weighted = network.is_weighted()
            net_prop = {
                'network_type': network.type,
                'directed': network.is_directed(),
                'nodes': network.node_nb(),
                'edges': network.edge_nb(),
                'weighted': weighted,
                'compressed_file': str(network).encode('utf-8')
            }
            if weighted:
                net_prop['weight_distribution'] = network._w
            neuralnet = NeuralNetwork(**net_prop)
        else:
            neuralnet = NeuralNetwork()
        return neuralnet

    def _make_neuron_entry(self, network, group):
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
        nngt_id = group.ids[0]
        gid = network.nest_gids[nngt_id]
        # get the dictionary
        neuron_prop = nest.GetStatus((gid,))[0]
        # update Neuron class accordingly
        Neuron = self._update_class("neuron", **neuron_prop)
        neuron = Neuron(**neuron_prop)
        return neuron

    def _make_synapse_entry(self, network, group_pre, group_post):
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
        syn_model = "static_synapse"
        if (group_pre.name, group_post.name) in network.population.syn_spec:
            pre, post = group_pre.name, group_post.name
            syn_model = network.population.syn_spec[(pre, post)]
            if isinstance(syn_model, dict):
                syn_model = syn_model.get("model", "static_synapse")
        source_gids = tuple(network.nest_gids[group_pre.ids])
        target_gids = tuple(network.nest_gids[group_post.ids])
        connections = nest.GetConnections(
            synapse_model=syn_model, source=source_gids, target=target_gids)
        # get the dictionary
        syn_prop={}
        if connections:
            syn_prop = nest.GetStatus((connections[0],))[0]
            # check single uppercase letters
            sngl_upper = []
            for k in syn_prop:
                if len(k) == 1 and k.isupper():
                    sngl_upper.append(k)
            for k in sngl_upper:
                syn_prop[2*k.lower()] = syn_prop[k]
                del syn_prop[k]
            # update Synapse class accordingly
            Synapse = self._update_class("synapse", **syn_prop)
        synapse = Synapse(**syn_prop)
        return synapse

    def _make_connection_entry(self, neuron_pre, neuron_post, synapse):
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
        return Connection(pre=neuron_pre, post=neuron_post, synapse=synapse)

    def _get_simulation_prop(self, network, simulator):
        '''
        Get the simulation properties.
        
        Parameters
        ----------
        network : :class:`~nngt.Network`
            Network used for the simulation.
        simulator : str
            Name of the simulator use (NEST, BRIAN...).

        Returns
        -------
        sim_prop : dict
            Dictionary containing the relevant key/value pairs to fill the
            :class:`~nngt.database.Simulation` class.
        '''
        pop, size = [], []
        for name, group in iter(network.population.items()):
            pop.append(name)
            size.append(len(group.ids))
        self.current_simulation = {
            'start_time': datetime.now(),
            'simulated_time': nest.GetKernelStatus('time'),
            'resolution': nest.GetKernelStatus('resolution'),
            'simulator': simulator.lower(),
            'grnd_seed': nest.GetKernelStatus('grng_seed'),
            'local_seeds': nest.GetKernelStatus('rng_seeds'),
            'population': pop,
            'pop_sizes': size
        }

    def _make_activity_entry(self, network=None):
        '''
        Create an activity entry from an
        :class:`~nngt.simulation.ActivityRecord` object.
        '''
        raster      = nngt.analysis.get_spikes(astype="np")
        activity    = nngt.simulation.analyze_raster(raster, network=network)
        di_activity = activity.properties
        di_activity["raster"] = raster
        act_attr = { k: v.__class__.__name__ for k, v in  di_activity.items() }
        if "spike_files" in act_attr:
            act_attr["spike_files"] = "compressed"
        act_attr["dtypes"] = True
        ''' ..todo ::
            compress the spike files '''
        Activity = self._update_class("activity", **act_attr)
        activity_entry = Activity(**di_activity)
        self.current_simulation['activity'] = activity_entry
        return activity_entry
        
    def log_simulation_start(self, network, simulator, save_network=True):
        '''
        Record the simulation start time, all nodes, connections, network, and
        computer properties, as well as some of simulation's.

        Parameters
        ----------
        network : :class:`~nngt.Network` or subclass
            Network used for the current simulation.
        simulator : str
            Name of the simulator.
        save_network : bool, optional (default: True)
            Whether to save the network or not.
        '''
        if not self.is_clear():
            raise RuntimeError("Database log started without clearing the "
                               "previous one.")
        self._get_simulation_prop(network, simulator)
        # computer and network data
        self.computer  = self._make_computer_entry()
        self.neuralnet = (self._make_network_entry(network)
                          if save_network else None)
        self.current_simulation['computer'] = self.computer
        self.current_simulation['network']  = self.neuralnet
        # neurons, synapses and connections
        perm_names = tuple(permutations(network.population.keys(), 2))
        perm_groups = tuple(permutations(network.population.values(), 2))
        if not perm_names:
            group_name = list(network.population.keys())[0]
            group = network.population[group_name]
            perm_names = ((group_name, group_name),)
            perm_groups = ((group, group),)
        self.nodes = {}
        for (name_pre, name_post), (pre, post) in zip(perm_names, perm_groups):
            if name_pre not in self.nodes:
                self.nodes[name_pre] = self._make_neuron_entry(network, pre)
            if name_post not in self.nodes:
                self.nodes[name_post] = self._make_neuron_entry(network, post)
            synapse = self._make_synapse_entry(network, pre, post)
            self.nodes["syn_{}->{}".format(name_pre, name_post)] = synapse
            conn = self._make_connection_entry(self.nodes[name_pre],
                                               self.nodes[name_post], synapse)
            self.connections["{}->{}".format(name_pre, name_post)] = conn
        

    def log_simulation_end(self, network=None, log_activity=True):
        '''
        Record the simulation completion and simulated times, save the data,
        then reset.
        '''
        if self.is_clear():
            raise RuntimeError("Database log ended with empy log.")
        # get completion time and simulated time
        self.current_simulation['completion_time'] = datetime.now()
        start_time = self.current_simulation['simulated_time']
        new_time   = nest.GetKernelStatus('time')
        self.current_simulation['simulated_time'] = new_time - start_time
        # save activity if provided
        if log_activity:
            self.activity = self._make_activity_entry(network)
        else:
            self.activity = Activity()
            self.current_simulation['activity'] = self.activity
        # save data and reset
        self.activity.save()
        self.computer.save()
        if self.neuralnet is not None:
            self.neuralnet.save()
        for entry in iter(self.nodes.values()):
            entry.save()
        for entry in iter(self.connections.values()):
            entry.save()
        simul_data = Simulation(**self.current_simulation)
        simul_data.save()
        # ~ if nngt.get_config("db_to_file"):
            # ~ from .csv_utils import dump_csv
            # ~ db_cls = list(self.tables.values())
            # ~ q = (Simulation.select(*db_cls).join(Computer).switch(Simulation)
                  # ~ .join(NeuralNetwork).switch(Simulation).join(Activity)
                  # ~ .switch(Simulation).join(Connection).join(Neuron, on=Connection.pre)
                  # ~ .switch(Connection).join(Neuron, on=Connection.post)
                  # ~ .switch(Connection).join(Synapse)).select(*db_cls)
            # ~ dump_csv(q, "{}_{}.csv".format(self.computer.name,
                                           # ~ simul_data.completion_time))
        self.reset()
    
    def get_results(self, table, column=None, value=None):
        '''
        Return the entries where the attribute `column` satisfies the required
        equality.
        
        Parameters
        ----------
        table : str
            Name of the table where the search should be performed (among
            ``'simulation'``, ``'computer'``, ``'neuralnetwork'``,
            ``'activity'``, ``'synapse'``, ``'neuron'``, or ``'connection'``).
        column : str, optional (default: None)
            Name of the variable of interest (a column on the table). If None,
            the whole table is returned.
        value : `column` corresponding type, optional (default: None)
            Specific value for the variable of interest. If None, the whole
            column is returned.
        
        Returns
        -------
        :class:`peewee.SelectQuery` with entries matching the request.
        '''
        TableModel = self.tables[table]
        if column is None:
            return TableModel.select()
        elif value is None:
            return TableModel.select(getattr(TableModel, column))
        else:
            return TableModel.select().where(
                getattr(TableModel, column) == value)

    def is_clear(self):
        ''' Check that the logs are clear. '''
        clear = True
        clear *= self.current_simulation is None
        clear *= self.simulator_lib is None
        clear *= self.computer is None
        clear *= self.neuralnet is None
        clear *= not self.nodes
        clear *= not self.connections
        return clear

    def reset(self):
        ''' Reset log status. '''
        self.current_simulation = None
        self.simulator_lib = None
        self.computer = None
        self.neuralnet = None
        self.connections = {}
        self.nodes = {}

