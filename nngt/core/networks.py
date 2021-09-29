#-*- coding:utf-8 -*-
#
# core/networks.py
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

""" Network and SpatialNetwork classes for neuroscience integration """

import numpy as np

import nngt
from nngt.lib import (InvalidArgument, nonstring_container, default_neuron,
                      default_synapse)
from .graph import Graph
from .spatial_graph import SpatialGraph


# ------- #
# Network #
# ------- #

class Network(Graph):

    """
    The detailed class that inherits from :class:`~nngt.Graph` and implements
    additional properties to describe various biological functions
    and interact with the NEST simulator.
    """

    #-------------------------------------------------------------------------#
    # Class attributes and methods

    __num_networks = 0
    __max_id       = 0

    @classmethod
    def num_networks(cls):
        ''' Returns the number of alive instances. '''
        return cls.__num_networks

    @classmethod
    def from_gids(cls, gids, get_connections=True, get_params=False,
                  neuron_model=default_neuron, neuron_param=None,
                  syn_model=default_synapse, syn_param=None, **kwargs):
        '''
        Generate a network from gids.

        Warning
        -------
        Unless `get_connections` and `get_params` is True, or if your
        population is homogeneous and you provide the required information, the
        information contained by the network and its `population` attribute
        will be erroneous!
        To prevent conflicts the :func:`~nngt.Network.to_nest` function is not
        available. If you know what you are doing, you should be able to find a
        workaround...

        Parameters
        ----------
        gids : array-like
            Ids of the neurons in NEST or simply user specified ids.
        get_params : bool, optional (default: True)
            Whether the parameters should be obtained from NEST (can be very
            slow).
        neuron_model : string, optional (default: None)
            Name of the NEST neural model to use when simulating the activity.
        neuron_param : dict, optional (default: {})
            Dictionary containing the neural parameters; the default value will
            make NEST use the default parameters of the model.
        syn_model : string, optional (default: 'static_synapse')
            NEST synaptic model to use when simulating the activity.
        syn_param : dict, optional (default: {})
            Dictionary containing the synaptic parameters; the default value
            will make NEST use the default parameters of the model.

        Returns
        -------
        net : :class:`~nngt.Network` or subclass
            Uniform network of disconnected neurons.
        '''
        from nngt.lib.errors import not_implemented

        if neuron_param is None:
            neuron_param = {}
        if syn_param is None:
            syn_param = {}

        # create the population
        size  = len(gids)
        nodes = [i for i in range(size)]

        group = nngt.NeuralGroup(
            nodes, neuron_type=1, neuron_model=neuron_model,
            neuron_param=neuron_param)

        pop = nngt.NeuralPop.from_groups([group])

        # create the network
        net = cls(population=pop, **kwargs)
        net.nest_gids = np.array(gids)
        net._id_from_nest_gid = {gid: i for i, gid in enumerate(gids)}
        net.to_nest = not_implemented

        if get_connections:
            from nngt.simulation import get_nest_adjacency
            converter = {gid: i for i, gid in enumerate(gids)}
            mat = get_nest_adjacency(converter)
            edges = np.array(mat.nonzero()).T
            w = mat.data
            net.new_edges(edges, {'weight': w}, check_duplicates=False,
                          check_self_loops=False, check_existing=False)

        if get_params:
            raise NotImplementedError('`get_params` not implemented yet.')

        return net

    @classmethod
    def uniform(cls, size, neuron_model=default_neuron,
                        neuron_param=None, syn_model=default_synapse,
                        syn_param=None, **kwargs):
        '''
        Generate a network containing only one type of neurons.

        Parameters
        ----------
        size : int
            Number of neurons in the network.
        neuron_model : string, optional (default: 'aief_cond_alpha')
            Name of the NEST neural model to use when simulating the activity.
        neuron_param : dict, optional (default: {})
            Dictionary containing the neural parameters; the default value will
            make NEST use the default parameters of the model.
        syn_model : string, optional (default: 'static_synapse')
            NEST synaptic model to use when simulating the activity.
        syn_param : dict, optional (default: {})
            Dictionary containing the synaptic parameters; the default value
            will make NEST use the default parameters of the model.

        Returns
        -------
        net : :class:`~nngt.Network` or subclass
            Uniform network of disconnected neurons.
        '''
        if neuron_param is None:
            neuron_param = {}

        if syn_param is None:
            syn_param = {}

        pop = nngt.NeuralPop.uniform(
            size, neuron_model=neuron_model, neuron_param=neuron_param,
            syn_model=syn_model, syn_param=syn_param, parent=None)

        net = cls(population=pop, **kwargs)

        return net

    @classmethod
    def exc_and_inhib(cls, size, iratio=0.2, en_model=default_neuron,
            en_param=None, in_model=default_neuron, in_param=None,
            syn_spec=None, **kwargs):
        '''
        Generate a network containing a population of two neural groups:
        inhibitory and excitatory neurons.

        Parameters
        ----------
        size : int
            Number of neurons in the network.
        i_ratio : double, optional (default: 0.2)
            Ratio of inhibitory neurons: :math:`\\frac{N_i}{N_e+N_i}`.
        en_model : string, optional (default: 'aeif_cond_alpha')
           Nest model for the excitatory neuron.
        en_param : dict, optional (default: {})
            Dictionary of parameters for the the excitatory neuron.
        in_model : string, optional (default: 'aeif_cond_alpha')
           Nest model for the inhibitory neuron.
        in_param : dict, optional (default: {})
            Dictionary of parameters for the the inhibitory neuron.
        syn_spec : dict, optional (default: static synapse)
            Dictionary containg a directed edge between groups as key and the
            associated synaptic parameters for the post-synaptic neurons (i.e.
            those of the second group) as value. If provided, all connections
            between groups will be set according to the values contained in
            `syn_spec`. Valid keys are:

            - `('excitatory', 'excitatory')`
            - `('excitatory', 'inhibitory')`
            - `('inhibitory', 'excitatory')`
            - `('inhibitory', 'inhibitory')`

        Returns
        -------
        net : :class:`~nngt.Network` or subclass
            Network of disconnected excitatory and inhibitory neurons.

        See also
        --------
        :func:`~nngt.NeuralPop.exc_and_inhib`
        '''
        pop = nngt.NeuralPop.exc_and_inhib(
            size, iratio, en_model, en_param, in_model, in_param,
            syn_spec=syn_spec)

        net = cls(population=pop, **kwargs)

        return net

    #-------------------------------------------------------------------------#
    # Constructor, destructor and attributes

    def __init__(self, name="Network", weighted=True, directed=True,
                 copy_graph=None, population=None, inh_weight_factor=1.,
                 **kwargs):
        '''
        Initializes :class:`~nngt.Network` instance.

        .. versionchanged: 2.4
            Move `from_graph` to `copy_graph` to reflect changes in Graph.

        Parameters
        ----------
        nodes : int, optional (default: 0)
            Number of nodes in the graph.
        name : string, optional (default: "Graph")
            The name of this :class:`Graph` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        copy_graph : :class:`~nngt.core.GraphObject`, optional (default: None)
            An optional :class:`~nngt.core.GraphObject` to serve as base.
        population : :class:`nngt.NeuralPop`, (default: None)
            An object containing the neural groups and their properties:
            model(s) to use in NEST to simulate the neurons as well as their
            parameters.
        inh_weight_factor : float, optional (default: 1.)
            Factor to apply to inhibitory synapses, to compensate for example
            the strength difference due to timescales between excitatory and
            inhibitory synapses.

        Returns
        -------
        self : :class:`~nggt.Network`
        '''
        self.__id = self.__class__.__max_id

        self.__class__.__num_networks += 1
        self.__class__.__max_id += 1

        assert directed, "Network class cannot be undirected."

        if population is None:
            raise InvalidArgument("Network needs a NeuralPop to be created")

        nodes = population.size

        if "nodes" in kwargs.keys():
            assert kwargs["nodes"] == nodes, "Incompatible values for " +\
                "`nodes` = {} with a `population` of size {}.".format(
                    kwargs["nodes"], nodes)
            del kwargs["nodes"]

        if "delays" not in kwargs:  # set default delay to 1.
            kwargs["delays"] = 1.

        super().__init__(nodes=nodes, name=name, weighted=weighted,
                         directed=directed, copy_graph=copy_graph,
                         inh_weight_factor=inh_weight_factor, **kwargs)

        self._init_bioproperties(population)

    def __del__(self):
        super().__del__()
        self.__class__.__num_networks -= 1

    @property
    def population(self):
        '''
        :class:`~nngt.NeuralPop` that divides the neurons into groups with
        specific properties.
        '''
        return self._population

    @population.setter
    def population(self, population):
        if issubclass(population.__class__, nngt.NeuralPop):
            if self.node_nb() == population.size:
                if population.is_valid:
                    self._population = population
                else:
                    raise AttributeError("NeuralPop is not valid (not all "
                                         "neurons are associated to a group).")
            else:
                raise AttributeError("Network and NeuralPop must have same "
                                     "number of neurons.")
        else:
            raise AttributeError("Expecting NeuralPop but received "
                                 "'{}'".format(population.__class__.__name__))

    @property
    def nest_gids(self):
        return self._nest_gids

    @nest_gids.setter
    def nest_gids(self, gids):
        self._nest_gids = gids
        for group in self.population.values():
            group._nest_gids = gids[group.ids]

    def get_edge_types(self, edges=None):
        '''
        Return the type of all or a subset of the edges.
        For all edges, the types are ordered according to the edges ids, i.e.
        in the same order as :property:`~nngt.Graph.edges_array`.

        .. versionchanged:: 2.4
            Updated it to make it compatible with the default
            :class:`~nngt.Graph` function, including the `edges` argument.

        Parameters
        ----------
        edges : (E, 2) array, optional (default: all edges)
            Edges for which the type should be returned.

        Returns
        -------
        the list of types (1 for excitatory, -1 for inhibitory)
        '''
        edges = self.edges_array if edges is None else edges

        types = np.ones(len(edges))

        inhib_neurons = set()

        for g in self._population.values():
            if g.neuron_type == -1:
                inhib_neurons.update(g.ids)

        for i, e in enumerate(edges):
            if e[0] in inhib_neurons:
                types[i] = -1

        return types

    def id_from_nest_gid(self, gids):
        '''
        Return the ids of the nodes in the :class:`nngt.Network` instance from
        the corresponding NEST gids.

        Parameters
        ----------
        gids : int or tuple
            NEST gids.

        Returns
        -------
        ids : int or tuple
            Ids in the network. Same type as the requested `gids` type.
        '''
        if nonstring_container(gids):
            return np.array([self._id_from_nest_gid[gid] for gid in gids],
                            dtype=int)
        else:
            return self._id_from_nest_gid[gids]

    def to_nest(self, send_only=None, weights=True):
        '''
        Send the network to NEST.

        .. seealso::
            :func:`~nngt.simulation.make_nest_network` for parameters
        '''
        from nngt.simulation import make_nest_network
        if nngt._config['with_nest']:
            return make_nest_network(
                self, send_only=send_only, weights=weights)
        else:
            raise RuntimeError("NEST is not present.")

    #-------------------------------------------------------------------------#
    # Init tool

    def _init_bioproperties(self, population):
        ''' Set the population attribute and link each neuron to its group. '''
        self._population = None
        self._nest_gids = None
        self._id_from_nest_gid = None
        if not hasattr(self, '_iwf'):
            self._iwf = 1.
        if issubclass(population.__class__, nngt.NeuralPop):
            if population.is_valid or not self.node_nb():
                self._population = population
                nodes = population.size
                # create the delay attribute if necessary
                if "delay" not in self.edge_attributes:
                    self.set_delays()
            else:
                raise AttributeError("NeuralPop is not valid (not all neurons "
                                     "are associated to a group).")
        else:
            raise AttributeError("Expected NeuralPop but received "
                                 "{}".format(pop.__class__.__name__))

    #-------------------------------------------------------------------------#
    # Setter

    def set_types(self, edge_type, nodes=None, fraction=None):
        '''
        .. warning::
            This function is not available for :class:`~nngt.Network`
            subclasses.
        '''
        raise NotImplementedError("Cannot be used on :class:`~nngt.Network`.")

    def get_neuron_type(self, neuron_ids):
        '''
        Return the type of the neurons (+1 for excitatory, -1 for inhibitory).

        Parameters
        ----------
        neuron_ids : int or tuple
            NEST gids.

        Returns
        -------
        ids : int or tuple
            Ids in the network. Same type as the requested `gids` type.
        '''
        if is_integer(neuron_ids):
            group_name = self._population._neuron_group[neuron_ids]
            neuron_type = self._population[group_name].neuron_type
            return neuron_type
        else:
            groups = (self._population._neuron_group[i] for i in neuron_ids)
            types = tuple(self._population[gn].neuron_type for gn in groups)
            return types

    #-------------------------------------------------------------------------#
    # Getter

    def neuron_properties(self, idx_neuron):
        '''
        Properties of a neuron in the graph.

        Parameters
        ----------
        idx_neuron : int
            Index of a neuron in the graph.

        Returns
        -------
        dict of the neuron's properties.
        '''
        group_name = self._population._neuron_group[idx_neuron]
        return self._population[group_name].properties()


# -------------- #
# SpatialNetwork #
# -------------- #

class SpatialNetwork(Network, SpatialGraph):

    """
    Class that inherits from :class:`~nngt.Network` and
    :class:`~nngt.SpatialGraph` to provide a detailed description of a real
    neural network in space, i.e. with positions and biological properties to
    interact with NEST.
    """

    #-------------------------------------------------------------------------#
    # Class attributes

    __num_networks = 0
    __max_id = 0

    #-------------------------------------------------------------------------#
    # Constructor, destructor, and attributes

    def __init__(self, population=None, name="SpatialNetwork", weighted=True,
                 directed=True, shape=None, copy_graph=None, positions=None,
                 **kwargs):
        '''
        Initialize SpatialNetwork instance.

        .. versionchanged: 2.4
            Move `from_graph` to `copy_graph` to reflect changes in Graph.

        Parameters
        ----------
        name : string, optional (default: "Graph")
            The name of this :class:`Graph` instance.
        weighted : bool, optional (default: True)
            Whether the graph edges have weight properties.
        directed : bool, optional (default: True)
            Whether the graph is directed or undirected.
        shape : :class:`~nngt.geometry.Shape`, optional (default: None)
            Shape of the neurons' environment (None leads to a square of side
            1 cm)
        positions : :class:`numpy.array`, optional (default: None)
            Positions of the neurons; if not specified and `nodes` != 0, then
            neurons will be reparted at random inside the
            :class:`~nngt.geometry.Shape` object of the instance.
        population : class:`~nngt.NeuralPop`, optional (default: None)
            Population from which the network will be built.

        Returns
        -------
        self : :class:`~nngt.SpatialNetwork`
        '''
        self.__id = self.__class__.__max_id
        self.__class__.__num_networks += 1
        self.__class__.__max_id += 1

        super().__init__(
            name=name, weighted=weighted, directed=directed,
            shape=shape, positions=positions, population=population,
            copy_graph=copy_graph, **kwargs)

    def __del__ (self):
        super().__del__()
        self.__class__.__num_networks -= 1

    #-------------------------------------------------------------------------#
    # Setter

    def set_types(self, syn_type, nodes=None, fraction=None):
        '''
        .. warning::
            This function is not available for :class:`~nngt.Network`
            subclasses.
        '''
        raise NotImplementedError("Cannot be used on "
                                  ":class:`~nngt.SpatialNetwork`.")
