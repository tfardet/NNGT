#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Graph data strctures in NNGT """

from collections import OrderedDict
import weakref

import numpy as np
from numpy.random import randint, uniform
import scipy.sparse as ssp
import scipy.spatial as sptl

from nngt.globals import (default_neuron, default_synapse, POS, WEIGHT, DELAY,
                          DIST, TYPE, BWEIGHT)
from nngt.lib import InvalidArgument
from nngt.lib.rng_tools import _eprop_distribution


__all__ = [
    'GroupProperty',
    'NeuralPop',
    'Shape',
]



#-----------------------------------------------------------------------------#
# NeuralPop
#------------------------
#

class NeuralPop(OrderedDict):

    """
    The basic class that contains groups of neurons and their properties.

    :ivar has_models: :class:`bool`,
        ``True`` if every group has a ``model`` attribute.
    """

    #-------------------------------------------------------------------------#
    # Class attributes and methods

    @classmethod
    def from_network(cls, graph, *args):
        '''
        Make a NeuralPop object from a network. The groups of neurons are
        determined using instructions from an arbitrary number of
        :class:`~nngt.properties.GroupProperties`.
        '''
        return cls.__init__(parent=graph,graph=graph,group_prop=args)

    @classmethod
    def uniform(cls, size, parent=None, neuron_model=default_neuron,
                neuron_param={}, syn_model=default_synapse, syn_param={}):
        ''' Make a NeuralPop of identical neurons '''
        pop = cls(size, parent)
        pop.new_group("default", range(size), 1, neuron_model, neuron_param,
           syn_model, syn_param)
        return pop

    @classmethod
    def exc_and_inhib(cls, size, iratio=0.2, parent=None,
            en_model=default_neuron, en_param={}, es_model=default_synapse,
            es_param={}, in_model=default_neuron, in_param={},
            is_model=default_synapse, is_param={}):
        '''
        Make a NeuralPop with a given ratio of inhibitory and excitatory
        neurons.
        '''
        num_exc_neurons = int(size*(1-iratio))
        pop = cls(size, parent)
        pop.new_group("excitatory", range(num_exc_neurons), 1, en_model,
           en_param, es_model, es_param)
        pop.new_group("inhibitory", range(num_exc_neurons, size), -1, in_model,
           in_param, is_model, es_param)
        return pop

    @classmethod
    def copy(cls, pop):
        ''' Copy an existing NeuralPop '''
        new_pop = cls.__init__(pop.has_models)
        for name, group in pop.items():
             new_pop.new_group(name, group.id_list, group.model,
                               group.neuron_param)
        return new_pop

    #-------------------------------------------------------------------------#
    # Contructor and instance attributes

    def __init__(self, size=None, parent=None, with_models=True, **kwargs):
        '''
        Initialize NeuralPop instance

        Parameters
        ----------
        with_models : :class:`bool`
            whether the population's groups contain models to use in NEST
        **kwargs : :class:`dict`

        Returns
        -------
        pop : :class:`~nngt.properties.NeuralPop` instance
        '''
        self._is_valid = False
        self._size = size if parent is None else parent.node_nb()
        self._neuron_group = np.empty(self._size, dtype=object)
        super(NeuralPop, self).__init__()
        if "graph" in kwargs.keys():
            dic = _make_groups(kwargs["graph"], kwargs["group_prop"])
            self._is_valid = True
            self.update(dic)
        self._has_models = with_models
    
    def __getitem__(self, key):
        if isinstance(key, int):
            new_key = tuple(self.keys())[key]
            return OrderedDict.__getitem__(self, new_key)
        else:
            return OrderedDict.__getitem__(self, key)
    
    def __setitem__(self, key, value):
        if isinstance(key, int):
            new_key = tuple(self.keys())[key]
            OrderedDict.__setitem__(self, new_key, value)
        else:
            OrderedDict.__setitem__(self, key, value)

    @property
    def size(self):
        return self._size

    @property
    def has_models(self):
        return self._has_models

    @property
    def is_valid(self):
        return self._is_valid

    #-------------------------------------------------------------------------#
    # Methods

    def set_model(self, model, group=None):
        '''
        Set the groups' models.

        Parameters
        ----------
        model : dict
            Dictionary containing the model type as key ("neuron" or "synapse")
            and the model name as value (e.g. {"neuron": "iaf_neuron"}).
        group : list of strings, optional (default: None)
            List of strings containing the names of the groups which models
            should be updated.

        .. warning::
            No check is performed on the validity of the models, which means
            that errors will only be detected when building the graph in NEST.

        .. note::
            By default, synapses are registered as "static_synapse"s in NEST;
            because of this, only the ``neuron_model`` attribute is checked by
            the ``has_models`` function: it will answer ``True`` if all groups
            have a 'non-None' ``neuron_model`` attribute.
        '''
        if group is None:
            group = self.keys()
        try:
            for key,val in iter(model.items()):
                for name in group:
                    if key == "neuron":
                        self[name].neuron_model = val
                    elif key == "synapse":
                        self[name].syn_model = val
                    else:
                        raise ValueError("Model type {} is not valid; choose \
among 'neuron' or 'synapse'.".format(key))
            else:
                raise
        except:
            raise InvalidArgument("Invalid model dict or group; see docstring.")
        b_has_models = True
        for group in iter(self.values()):
            b_has_model *= group.has_model
        self._has_models = b_has_models

    def set_param(self, param, group=None):
        '''
        Set the groups' parameters.

        Parameters
        ----------
        param : dict
            Dictionary containing the model type as key ("neuron" or "synapse")
            and the model parameter as value (e.g. {"neuron": {"C_m": 125.}}).
        group : list of strings, optional (default: None)
            List of strings containing the names of the groups which models
            should be updated.

        .. warning::
            No check is performed on the validity of the parameters, which
            means that errors will only be detected when building the graph in
            NEST.
        '''
        if group is None:
            group = self.keys()
        try:
            for key,val in iter(param.items()):
                for name in group:
                    if key == "neuron":
                        self[name].neuron_param = val
                    elif key == "synapse":
                        self[name].syn_param = val
                    else:
                        raise ValueError("Model type {} is not valid; choose \
among 'neuron' or 'synapse'.".format(key))
        except:
            raise InvalidArgument("Invalid param dict or group; see docstring.")

    def new_group(self, name, id_list, ntype=1, neuron_model=None, neuron_param={},
                  syn_model=default_synapse, syn_param={}):
        # create a group
        group = NeuralGroup(id_list, ntype, neuron_model, neuron_param, syn_model, syn_param)
        if self._has_models and not group.has_model:
            raise AttributeError("This NeuralPop requires group to have a \
model attribute that is not `None`; to disable this, use `set_models(None)` \
method on this NeuralPop instance.")
        elif group.has_model and not self._has_models:
            warnings.warn("This NeuralPop is not set to take models into \
account; use the `set_models` method to change its behaviour.")
        self[name] = group
        # update the group node property
        self._neuron_group[id_list] = name
        if None in list(self._neuron_group):
            self._is_valid = False
        else:
            self._is_valid = True

    def get_param(self, groups=None, neurons=None, element="neuron"):
        '''
        Return the `element` (neuron or synapse) parameters for neurons or
        groups of neurons in the population.

        Parameters
        ----------
        groups : ``str``, ``int`` or array-like, optional (default: ``None``)
            Names or numbers of the groups for which the neural properties
            should be returned.
        neurons : int or array-like, optional (default: ``None``)
            IDs of the neurons for which parameters should be returned.
        element : ``list`` of ``str``, optional (default: ``"neuron"``)
            Element for which the parameters should be returned (either
            ``"neuron"`` or ``"synapse"``).

        Returns
        -------
        param : ``list``
            List of all dictionaries with the elements' parameters.
        '''
        if neurons is not None:
            groups = self._neuron_group[neurons]
        elif groups is None:
            groups = tuple(self.keys())
        key = "neuron_param" if element == "neuron" else "syn_param"
        if isinstance(groups, (str, int)):
            return self[groups].properties[key]
        else:
            param = []
            for group in groups:
                param.append(self[group].properties[key])
            return param

    def get_group(neurons, numbers=False):
        '''
        Return the group of the neurons.
        
        Parameters
        ----------
        neurons : int or array-like
            IDs of the neurons for which the group should be returned.
        numbers : bool, optional (default: False)
            Whether the group identifier should be returned as a number; if
            ``False``, the group names are returned.
        '''
        if not numbers:
            return self._neuron_group[neurons]
        elif isinstance(neurons, int):
            keys.index(self._neuron_group[neurons])
        else:
            keys = tuple(self.keys())
            return [keys.index(self._neuron_group[n]) for n in neurons]

    def add_to_group(self, group_name, id_list):
        self[group_name].id_list.extend(id_list)
        self._neuron_group[id_list] = group_name
        if None in list(self._neuron_group):
            self._is_valid = False
        else:
            self._is_valid = True


#-----------------------------------------------------------------------------#
# NeuralGroup and GroupProperty
#------------------------
#


class NeuralGroup:

    """
    Class defining groups of neurons.

    :ivar id_list: :class:`list` of :class:`int`s
        the ids of the neurons in this group.
    :ivar neuron_type: :class:`int`
        the default is ``1`` for excitatory neurons; ``-1`` is for interneurons
    :ivar model: :class:`string`, optional (default: None)
        the name of the model to use when simulating the activity of this group
    :ivar neuron_param: :class:`dict`, optional (default: {})
        the parameters to use (if they differ from the model's defaults)

    .. warning::
        Equality between :class:`~nngt.properties.NeuralGroup`s only compares
        the neuronal and synaptic ``model`` and ``param`` attributes, i.e.
        groups differing only by their ``id_list`` will register as equal.
    .. note::
        By default, synapses are registered as ``"static_synapse"`` in NEST;
        because of this, only the ``neuron_model`` attribute is checked by the
        ``has_model`` function.
    """

    def __init__ (self, id_list=[], ntype=1, model=None, neuron_param={},
                  syn_model=None, syn_param={}):
        self._has_model = False if model is None else True
        self._neuron_model = model
        self._id_list = list(id_list)
        self._nest_gids = None
        if self._has_model:
            self.neuron_param = neuron_param
            self.neuron_type = ntype
            self.syn_model = (syn_model if syn_model is not None
                              else "static_synapse")
            self.syn_param = syn_param

    def __eq__ (self, other):
        if isinstance(other, NeuralGroup):
            same_nmodel = ( self.neuron_model == other.neuron_model *
                            self.neuron_param == other.neuron_param )
            same_smodel = ( self.syn_model == other.syn_model *
                            self.syn_param == other.syn_param )
            return same_nmodel * same_smodel
        else:
            return False

    @property
    def neuron_model(self):
        return self._neuron_model

    @neuron_model.setter
    def neuron_model(self, value):
        self._neuron_model = value
        self._has_model = False if value is None else True

    @property
    def id_list(self):
        return self._id_list

    @property
    def nest_gids(self):
        return self._nest_gids

    @property
    def has_model(self):
        return self._has_model

    @property
    def properties(self):
        dic = { "neuron_type": self.neuron_type,
                "neuron_model": self._neuron_model,
                "neuron_param": self.neuron_param,
                "syn_model": self.syn_model,
                "syn_param": self.syn_param }
        return dic

class GroupProperty:

    """
    Class defining the properties needed to create groups of neurons from an
    existing :class:`~nngt.GraphClass` or one of its subclasses.

    :ivar size: :class:`int`
        Size of the group.
    :ivar constraints: :class:`dict`, optional (default: {})
        Constraints to respect when building the
        :class:`~nngt.properties.NeuralGroup` .
    :ivar neuron_model: :class:`string`, optional (default: None)
        name of the model to use when simulating the activity of this group.
    :ivar neuron_param: :class:`dict`, optional (default: {})
        the parameters to use (if they differ from the model's defaults)
    """

    def __init__ (self, size, constraints={}, neuron_model=None,
                  neuron_param={}, syn_model=None, syn_param={}):
        '''
        Create a new instance of GroupProperties.

        Notes
        -----
        The constraints can be chosen among:
            - "avg_deg", "min_deg", "max_deg" (:class:`int`) to constrain the
            total degree of the nodes
            - "avg/min/max_in_deg", "avg/min/max_out_deg", to work with the
            in/out-degrees
            - "avg/min/max_betw" (:class:`double`) to constrain the betweenness
            centrality
            - "in_shape" (:class:`nngt.Shape`) to chose neurons inside a given
            spatial region

        Examples
        --------
        >>> di_constrain = { "avg_deg": 10, "min_betw": 0.001 }
        >>> group_prop = GroupProperties(200, constraints=di_constrain)
        '''
        self.size = size
        self.constraints = constraints
        self.neuron_model = neuron_model
        self.neuron_param = neuron_param
        self.syn_model = syn_model
        self.syn_param = syn_param

def _make_groups(graph, group_prop):
    '''
    Divide `graph` into groups using `group_prop`, a list of group properties
    @todo
    '''
    pass


#-----------------------------------------------------------------------------#
# Connections
#------------------------
#

class Connections:

    """
    The basic class that computes the properties of the connections between
    neurons for graphs.
    """

    #-------------------------------------------------------------------------#
    # Class methods

    @staticmethod
    def distances(graph, elist=None, pos=None, dlist=None, overwrite=False):
        '''
        Compute the distances between connected nodes in the graph. Try to add
        only the new distances to the graph. If they overlap with previously
        computed distances, recomputes everything.

        Parameters
        ----------
        graph : class:`~nngt.Graph` or subclass
            Graph the nodes belong to.
        elist : class:`numpy.array`, optional (default: None)
            List of the edges.
        pos : class:`numpy.array`, optional (default: None)
            Positions of the nodes; note that if `graph` has a "position"
            attribute, `pos` will not be taken into account.
        dlist : class:`numpy.array`, optional (default: None)
            List of distances (for user-defined distances)

        Returns
        -------
        new_dist : class:`numpy.array`
            Array containing *ONLY* the newly-computed distances.
        '''
        n = graph.node_nb()
        elist = graph.edges_array if elist is None else elist
        if dlist is not None:
            graph.set_edge_attribute(DIST, value_type="double", values=dlist)
            return dlist
        else:
            pos = graph._pos if hasattr(graph, "_pos") else pos
            # compute the new distances
            if graph.edge_nb():
                ra_x = pos[0, elist[:,0]] - pos[0, elist[:,1]]
                ra_y = pos[1, elist[:,0]] - pos[1, elist[:,1]]
                ra_dist = np.sqrt( np.square(ra_x) + np.square(ra_y) )
                #~ ra_dist = np.tile( , 2)
                # update graph distances
                graph.set_edge_attribute(DIST, value_type="double",
                                         values=ra_dist, edges=elist)
                return ra_dist
            else:
                return []

    @staticmethod
    def delays(graph, dlist=None, elist=None, distribution="constant",
               parameters={}, noise_scale=None):
        '''
        Compute the delays of the neuronal connections.

        Parameters
        ----------
        graph : class:`~nngt.Graph` or subclass
            Graph the nodes belong to.
        dlist : class:`numpy.array`, optional (default: None)
            List of user-defined delays).
        elist : class:`numpy.array`, optional (default: None)
            List of the edges which value should be updated.
        distribution : class:`string`, optional (default: "constant")
            Type of distribution (choose among "constant", "uniform",
            "lognormal", "gaussian", "user_def", "lin_corr", "log_corr").
        parameters : class:`dict`, optional (default: {})
            Dictionary containing the distribution parameters.
        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.

        Returns
        -------
        new_delays : class:`scipy.sparse.lil_matrix`
            A sparse matrix containing *ONLY* the newly-computed weights.
        '''
        parameters["btype"] = parameters.get("btype", "edge")
        parameters["use_weights"] = parameters.get("use_weights", False)
        elist = np.array(elist) if elist is not None else elist
        if dlist is not None:
            num_edges = graph.edge_nb() if elist is None else elist.shape[0]
            if len(dlist) != num_edges:
                raise InvalidArgument("`dlist` must have one entry per edge.")
        else:
            dlist = _eprop_distribution(graph, distribution, elist=elist,
                                        **parameters)
        # add to the graph container
        graph.set_edge_attribute(
            DELAY, value_type="double", values=dlist, edges=elist)
        return dlist

    @staticmethod
    def weights(graph, elist=None, wlist=None, distribution="constant",
                parameters={}, noise_scale=None):
        '''
        Compute the weights of the graph's edges.
        @todo: take elist into account

        Parameters
        ----------
        graph : class:`~nngt.Graph` or subclass
            Graph the nodes belong to.
        elist : class:`numpy.array`, optional (default: None)
            List of the edges (for user defined weights).
        wlist : class:`numpy.array`, optional (default: None)
            List of the weights (for user defined weights).
        distribution : class:`string`, optional (default: "constant")
            Type of distribution (choose among "constant", "uniform",
            "lognormal", "gaussian", "user_def", "lin_corr", "log_corr").
        parameters : class:`dict`, optional (default: {})
            Dictionary containing the distribution parameters.
        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.

        Returns
        -------
        new_weights : class:`scipy.sparse.lil_matrix`
            A sparse matrix containing *ONLY* the newly-computed weights.
        '''
        parameters["btype"] = parameters.get("btype", "edge")
        parameters["use_weights"] = parameters.get("use_weights", False)
        #~ elist = np.array(elist) if elist is not None else elist
        elist = None
        if wlist is not None:
            num_edges = graph.edge_nb() if elist is None else elist.shape[0]
            if len(wlist) != num_edges:
                raise InvalidArgument(
                    '''`wlist` must have one entry per edge. For graph {},
there are {} edges while {} values where provided'''.format(
                    graph.name, num_edges, len(wlist)))
        else:
            wlist = _eprop_distribution(graph, distribution, elist=elist,
                                        **parameters)
        # add to the graph container
        bwlist = (np.max(wlist) - wlist if np.any(wlist)
                  else np.repeat(0, len(wlist)))
        graph.set_edge_attribute(
            WEIGHT, value_type="double", values=wlist, edges=elist)
        graph.set_edge_attribute(
            BWEIGHT, value_type="double", values=bwlist, edges=elist)
        return wlist

    @staticmethod
    def types(graph, inhib_nodes=None, inhib_frac=None):
        '''
        @todo

        Define the type of a set of neurons.
        If no arguments are given, all edges will be set as excitatory.

        Parameters
        ----------
        graph : :class:`~nngt.Graph` or subclass
            Graph on which edge types will be created.
        inhib_nodes : int, float or list, optional (default: `None`)
            If `inhib_nodes` is an int, number of inhibitory nodes in the graph
            (all connections from inhibitory nodes are inhibitory); if it is a
            float, ratio of inhibitory nodes in the graph; if it is a list, ids
            of the inhibitory nodes.
        inhib_frac : float, optional (default: `None`)
            Fraction of the selected edges that will be set as refractory (if
            `inhib_nodes` is not `None`, it is the fraction of the nodes' edges
            that will become inhibitory, otherwise it is the fraction of all
            the edges in the graph).

        Returns
        -------
        t_list : :class:`~numpy.ndarray`
            List of the edges' types.
        '''
        t_list = np.repeat(1.,graph.edge_nb())
        edges = graph.edges_array
        num_inhib = 0
        idx_inhib = []
        if inhib_nodes is None and inhib_frac is None:
            graph.new_edge_attribute("type", "double", val=1.)
            return t_list
        else:
            n = graph.node_nb()
            if inhib_nodes is None:
                # set inhib_frac*num_edges random inhibitory connections
                num_edges = graph.edge_nb()
                num_inhib = int(num_edges*inhib_frac)
                num_current = 0
                while num_current < num_inhib:
                    new = randint(0,num_edges,num_inhib-num_current)
                    idx_inhib = np.unique( np.concatenate( (idx_inhib, new) ) )
                    num_current = len(idx_inhib)
                t_list[idx_inhib.astype(int)] *= -1.
            else:
                # get the dict of inhibitory nodes
                num_inhib_nodes = 0
                idx_nodes = {}
                if hasattr(inhib_nodes, '__iter__'):
                    idx_nodes = { i:-1 for i in inhib_nodes }
                    num_inhib_nodes = len(idx_nodes)
                if issubclass(inhib_nodes.__class__, float):
                    if inhib_nodes > 1:
                        raise InvalidArgument("Inhibitory ratio (float value \
for `inhib_nodes`) must be smaller than 1")
                        num_inhib_nodes = int(inhib_nodes*n)
                if issubclass(inhib_nodes.__class__, int):
                    num_inhib_nodes = int(inhib_nodes)
                while len(idx_nodes) != num_inhib_nodes:
                    indices = randint(0,n,num_inhib_nodes-len(idx_nodes))
                    di_tmp = { i:-1 for i in indices }
                    idx_nodes.update(di_tmp)
                for v in edges[:,0]:
                    if v in idx_nodes:
                        idx_inhib.append(v)
                idx_inhib = np.unique(idx_inhib)
                # set the inhibitory edge indices
                for v in idx_inhib:
                    idx_edges = np.argwhere(edges[:,0]==v)
                    n = len(idx_edges)
                    if inhib_frac is not None:
                        idx_inh = []
                        num_inh = n*inhib_frac
                        i = 0
                        while i != num_inh:
                            ids = randint(0,n,num_inh-i)
                            idx_inh = np.unique(np.concatenate((idx_inh,ids)))
                            i = len(idx_inh)
                        t_list[idx_inh] *= -1.
                    else:
                        t_list[idx_edges] *= -1.
            graph.set_edge_attribute("type", value_type="double", values=t_list)
            return t_list




#-----------------------------------------------------------------------------#
# Shape
#------------------------
#

class Shape:
    """
    Class containing the shape of the area where neurons will be distributed to
    form a network.

    ..warning :
        so far, only a rectangle can be created.

    Attributes
    ----------
    area: double
        Area of the shape in mm^2.
    com: tuple of doubles
        Position of the center of mass of the current shape.

    Methods
    -------
    add_subshape: void
        @todo
        Add a :class:`~nngt.Shape` to a preexisting one.
    """

    @classmethod
    def rectangle(cls, parent, height, width, pos_com=(0.,0.)):
        '''
        Generate a rectangle of given height, width and center of mass.

        Parameters
        ----------
        parent : :class:`~nngt.SpatialGraph` or subclass
            The parent container.
        height : float
            Height of the rectangle.
        width : float
            Width of the rectangle.
        pos_com : tuple of floats, optional (default: (0., 0.))
            Position of the rectangle's center of mass

        Returns
        -------
        shape : :class:`~nngt.Shape`
            Rectangle shape.
        '''
        shape = cls(parent)
        half_diag = np.sqrt(np.square(height/2.)+np.square(width/2.))/2.
        pos_com = np.array(pos_com)
        points = [  pos_com + [half_diag,half_diag],
                    pos_com + [half_diag,-half_diag],
                    pos_com - [half_diag,half_diag],
                    pos_com - [half_diag,-half_diag] ]
        shape._convex_hull = sptl.Delaunay(points)
        shape._com = pos_com
        return shape

    def __init__(self, parent=None, ):
        self.parent = weakref.proxy(parent) if parent is not None else None
        self._area = 0.
        self._com = (0.,0.)
        self._convex_hull = None

    @property
    def area(self):
        ''' Area of the shape. '''
        return self._area

    @property
    def com(self):
        ''' Center of mass of the shape. '''
        return self._com

    @property
    def vertices(self):
        return self._convex_hull.points

    def set_parent(self, parent):
        self.parent = weakref.proxy(parent) if parent is not None else None

    def add_subshape(self, subshape, position, unit='mm'):
        """
        Add a :class:`~nngt.core.Shape` to the current one.

        Parameters
        ----------
        subshape: :class:`~nngt.Shape`
            Subshape to add.
        position: tuple of doubles
            Position of the subshape's center of gravity in space.
        unit: string (default 'mm')
            Unit in the metric system among 'um', 'mm', 'cm', 'dm', 'm'

        Returns
        -------
        None
        """
        pass

    def rnd_distrib(self, nodes=None):
        #@todo: make it general
        if self.parent is not None:
            nodes = self.parent.node_nb()
        points = self._convex_hull.points
        min_x, max_x = points[:,0].min(), points[:,0].max()
        min_y, max_y = points[:,1].min(), points[:,1].max()
        ra_x = uniform(min_x, max_x, size=nodes)
        ra_y = uniform(min_y, max_y, size=nodes)
        return np.vstack((ra_x,ra_y))

