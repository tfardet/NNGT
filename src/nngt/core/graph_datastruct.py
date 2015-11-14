#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Graph data strctures in NNGT """

import numpy as np

from ..properties.populations import NeuralGroup, _make_groups




#
#---
# NeuralPop
#------------------------

class NeuralPop(dict,object):
    
    """    
    The basic class that contains groups of neurons and their properties.

    :ivar has_models: :class:`bool`,
        ``True`` if every group has a ``model`` attribute.
    """

    #-------------------------------------------------------------------------#
    # Class attributes and methods
    
    @classmethod
    def pop_from_network(cls, graph, *args):
        '''
        Make a NeuralPop object from a network. The groups of neurons are 
        determined using instructions from an arbitrary number of
        :class:`~nngt.properties.GroupProperties`.
        '''
        return cls.__init__(parent=graph,graph=graph,group_prop=args)

    @classmethod
    def uniform_population(cls, size, parent=None, neuron_model="aeif_neuron",
            neuron_param={}, syn_model="static_synapse", syn_param={}):
        ''' Make a NeuralPop of identical neurons '''
        pop = cls(size, parent)
        pop.new_group("default", range(size), 1, neuron_model, neuron_param,
           syn_model, syn_param)
        return pop

    @classmethod
    def ei_population(cls, size, ei_ratio=0.2, parent=None,
            en_model="aeif_neuron", en_param={}, es_model="static_synapse",
            es_param={}, in_model="aeif_neuron", in_param={},
            is_model="static_synapse", is_param={}):
        '''
        Make a NeuralPop with a given ratio of inhibitory and excitatory
        neurons
        '''
        num_inhib_neuron = int(ei_ratio*size)
        pop = cls(size, parent)
        pop.new_group("excitatory", range(num_inhib_neuron,size), 1, en_model,
           en_param, es_model, es_param)
        pop.new_group("inhibitory", range(num_inhib_neuron), -1, in_model,
           in_param, is_model, es_param)
        return pop
    
    @classmethod
    def copy(cls, pop):
        ''' Copy an existing NeuralPop '''
        new_pop = cls.__init__(pop.has_models)
        for name, group in pop.items():
             new_pop.new_group(name, group.id_list, group.model, group.neuron_param)
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
        self.parent = parent
        self._is_valid = False
        self._size = size if parent is None else parent.node_nb()
        self._neuron_group = np.empty(self._size, dtype=object)
        if "graph" in kwargs.keys():
            dic = _make_groups(kwargs["graph"], kwargs["group_prop"])
            self._is_valid = True
        else:
            super(NeuralPop, self).__init__()
        self._has_models = with_models
        
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
    
    def set_models(self, models=None):
        if isinstance(models, str) or models is None:
            for group in self.itervalues():
                group.model = models
        else:
            try:
                if list(model.keys()) == list(self.keys()):
                    for name, model in models.items():
                        self[name].model = model
                else:
                    raise
            except:
                raise ArgumentError("set_models argument should be either a \
                            string or a dict with the same keys as NeuralPop")
        b_has_models = True
        for group in self.itervalues():
            b_has_model *= group.has_model
        self._has_models = b_has_models
    
    def new_group(self, name, id_list, ntype=1, neuron_model=None, neuron_param={},
                  syn_model="static_synapse", syn_param={}):
        # create a group
        group = NeuralGroup(id_list, ntype, neuron_model, neuron_param, syn_model, syn_param)
        if self._has_models and not group.has_model:
            raise AttributeError("This NeuralPop requires group to have a \
                                 model attribute that is not `None`; to \
                                 disable this, use `set_models(None)` method \
                                 on this NeuralPop instance.")
        elif group.has_model and not self._has_models:
            warnings.warn("This NeuralPop is not set to take models into \
                          account; use the `set_models` method to change its \
                          behaviour.")
        self[name] = group
        # update the group node property
        self._neuron_group[id_list] = name
        if None in list(self._neuron_group):
            self._is_valid = False
        else:
            self._is_valid = True

    def add_to_group(self, group_name, id_list):
        self[group_name].id_list.extend(id_list)
        self._neuron_group[id_list] = group_name
        if None in list(self._neuron_group):
            self._is_valid = False
        else:
            self._is_valid = True


#
#---
# GroupProperties
#------------------------

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


#
#---
# Connections
#------------------------

class Connections:
    
    """    
    The basic class that contains the properties of the connections between
    neurons for sparse graphs.

    :ivar delays: :class:`~scipy.sparse.lil_matrix` of :class:`double`
        a *lil* sparse matrix containing the delay of spike propagation between
        pairs of neurons.
    :ivar distances: :class:`~scipy.sparse.lil_matrix` of :class:`double`,
        optional (default: None)
        sparse matrix containing the distances between connected neurons
    """
    
    def __init__(self, parent, dense=False):
        '''
        Initialize SparseConnections instance.

        Parameters
        ----------
        @todo: init from existing graph (duplicate the adjacency matrix), set
        
        init empty
            
        Returns
        -------
        pop : :class:`~nngt.properties.SparseConnections` instance
        '''
        self._parent = parent
        n = parent.node_nb()
        self._distance = None
        self._delay = None
        if hasattr(parent,"_init_spatial_properties"):
            self._distance = (sp.zeros((n,n)) if dense
                              else ssp.lil_matrix((n,n)))
        if hasattr(parent,"_init_bioproperties"):
            self._delay = sp.zeros((n,n)) if dense else ssp.lil_matrix((n,n))
        
    
    @property
    def parent(self):
        return self._parent


#
#---
# Shape class
#--------------------

class Shape:
    """
    Class containing the shape of the area where neurons will be
    distributed to form a network.

    Attributes
    ----------
    area: double
        Area of the shape in mm^2.
    gravity_center: tuple of doubles
        Position of the center of gravity of the current shape.

    Methods
    -------
    add_subshape: void
        Add a AGNet.generation.Shape to a preexisting one.
    """

    def __init__(self):
        self.__area = 0.
        self.__gravity_center = (0.,0.)

    def add_subshape(self,subshape,position,unit='mm'):
        """
        Add a AGNet.generation.Shape to the current one.
        
        Parameters
        ----------
        subshape: AGNet.generation.Shape
            Length of the rectangle (by default in mm).
        position: tuple of doubles
            Position of the subshape's center of gravity in space.
        unit: string (default 'mm')
            Unit in the metric system among 'um', 'mm', 'cm', 'dm', 'm'
        
        Returns
        -------
        None
        """

    def rnd_distrib(self):
        pass
