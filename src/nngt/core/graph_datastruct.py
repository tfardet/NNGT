#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Graph data strctures in NNGT """

import numpy as np
import scipy.sparse as ssp

from ..properties.populations import NeuralGroup, _make_groups
from ..lib.weights import *



#-----------------------------------------------------------------------------#
# NeuralPop
#------------------------
#

class NeuralPop(dict):
    
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
    def ei_population(cls, size, iratio=0.2, parent=None,
            en_model="aeif_neuron", en_param={}, es_model="static_synapse",
            es_param={}, in_model="aeif_neuron", in_param={},
            is_model="static_synapse", is_param={}):
        '''
        Make a NeuralPop with a given ratio of inhibitory and excitatory
        neurons.
        '''
        num_inhib_neuron = int(iratio*size)
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

    def add_to_group(self, group_name, id_list):
        self[group_name].id_list.extend(id_list)
        self._neuron_group[id_list] = group_name
        if None in list(self._neuron_group):
            self._is_valid = False
        else:
            self._is_valid = True


#-----------------------------------------------------------------------------#
# GroupProperty
#------------------------
#

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


#-----------------------------------------------------------------------------#
# Connections
#------------------------
#

class Connections:
    
    """    
    The basic class that computes the properties of the connections between
    neurons for graphs.
    """
    
    di_wfunc = {
        "uniform": uniform_weights,
        "lognormal": lognormal_weights,
        "gaussian": gaussian_weights,
        "lin_corr": lin_correlated_weights,
        "log_corr": log_correlated_weights
    }

    #-------------------------------------------------------------------------#
    # Class methods
    
    @classmethod
    def distances(cls, graph, elist=None, pos=None):
        '''
        Compute the distances between connected nodes in the graph. Try to add 
        only the new distances to the graph. If they overlap with previously 
        computed distances, recomputes everything.
        
        Parameters
        ----------
        graph : class:`~nngt.Graph` or subclass
            Graph the nodes belong to.
        elist : class:`numpy.array`, optional (default: None)
            List of the edges
        pos : class:`numpy.array`, optional (default: None)
            Positions of the nodes; if None, the graph must have a "position" 
            item.
        
        Returns
        -------
        new_dist : class:`scipy.sparse.lil_matrix`
            Sparse matrix containing *ONLY* the newly-computed distances.
        '''
        n = graph.node_nb()
        if elist is None:
            mat_adj = graph.adjacency_matrix().tocoo()
            elist = np.array( [mat_adj.row, mat_adj.col] ).T
        if pos is None:
            pos = graph["position"]
        # compute the new distances
        ra_x = pos[0,elist[:,0]] - pos[0,elist[:,1]]
        ra_y = pos[1,elist[:,0]] - pos[1,elist[:,1]]
        ra_dist = np.tile( np.sqrt( np.square(ra_x) + np.square(ra_y) ), 2)
        ia_sources = np.concatenate((elist[:,0], elist[:,1]))
        ia_targets = np.concatenate((elist[:,1], elist[:,0]))
        new_dist = ssp.coo_matrix( (ra_dist, (ia_sources,ia_targets)), (n,n) )
        # update graph distances
        current_dist = ssp.csr_matrix((n,n))
        if "distance" in graph.attributes():
            current_dist = graph._data["distance"].tocsr()
        total_dist = current_dist.nnz + new_dist.nnz
        new_dist = current_dist + new_dist.tocsr()
        if new_dist.nnz == total_dist:
            new_dist = new_dist.tolil()
            graph._data["distance"] = new_dist
        else:
            graph._data["distance"] = Connections.distances(network).tolil()
        return new_dist
    
    @classmethod
    def delays(cls, graph, elist=None, pos=None, distrib=None,
                   correl=None):
        pass
    
    @classmethod
    def weights(cls, graph, elist=None, wlist=None, distrib=None,
                distrib_prop={}, correl=None, noise_scale=None):
        '''
        Compute the weights of the graph's edges.
        
        Parameters
        ----------
        graph : class:`~nngt.Graph` or subclass
            Graph the nodes belong to.
        elist : class:`numpy.array`, optional (default: None)
            List of the edges (for user defined weights).
        wlist : class:`numpy.array`, optional (default: None)
            List of the weights (for user defined weights).
        distrib : class:`string`, optional (default: None)
            Type of distribution (choose among "uniform", "lognormal",
            "gaussian", "user_def", "lin_corr", "log_corr").
        distrib_prop : class:`dict`, optional (default: {})
            Dictionary containing the distribution parameters.
        correl : class:`string`, optional (default: None)
            Property to which the weights should be correlated.
        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.
        
        Returns
        -------
        class:`scipy.sparse.coo_matrix`
        '''
        n = graph.node_nb()
        corr = correl
        if corr is not None:
            corr = ( graph.get_betweenness(False)[1] if correl == "betweenness"
                                                     else graph[correl] )
        new_weights = None
        if elist is None:
            new_weights = cls.di_wfunc[distrib](graph=graph,
                            correl_attribute=corr, **distrib_prop)
        else:
            new_weights = ssp.coo_matrix((wlist,(elist[:,0],elist[:,1])),(n,n))
        new_weights = new_weights.tolil()
        # add to the graph container
        if graph.is_weighted():
            mat_weights = graph._data["weight"]
            graph._data["weight"] = (mat_weights+new_weights).tolil()
        else:
            graph._data["weight"] = new_weights
        # add to the graph-object attribute
        sources, targets = graph["edges"][:,1], graph["edges"][:,0]
        lst_w = graph._data["weight"][sources,targets].data[0]
        graph.graph.new_edge_attribute("weight", "double", values=lst_w)
    
    @classmethod
    def types(cls, graph, elist=None):
        pass


#-----------------------------------------------------------------------------#
# Shape
#------------------------
#

class Shape:
    """
    Class containing the shape of the area where neurons will be distributed to
    form a network.

    Attributes
    ----------
    area: double
        Area of the shape in mm^2.
    com: tuple of doubles
        Position of the center of mass of the current shape.

    Methods
    -------
    add_subshape: void
        Add a AGNet.generation.Shape to a preexisting one.
    """

    def __init__(self, parent=None):
        self._parent = parent
        self._area = 0.
        self._com = (0.,0.)
    
    @property
    def area(self):
        return self._area
    
    @property
    def com(self):
        return self._com

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

    def rnd_distrib(self, nodes=None):
        #@todo: make it general
        if self._parent is not None:
            nodes = self._parent.node_nb()
        ra_x = np.random.uniform(size=nodes)
        ra_y = np.random.uniform(size=nodes)
        return np.array([ra_x,ra_y])
        
