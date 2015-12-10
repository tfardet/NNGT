#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Graph data strctures in NNGT """

import weakref

import numpy as np
import scipy.sparse as ssp

from nngt.globals import ( default_neuron, default_synapse, POS, WEIGHT, DELAY,
                       DIST, TYPE )
from nngt.properties.populations import NeuralGroup, _make_groups
from nngt.lib import ( InvalidArgument, delta_distrib, uniform_distrib,
        lognormal_distrib, gaussian_distrib, lin_correlated_distrib,
        log_correlated_distrib )



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
    def uniform_population(cls, size, parent=None, neuron_model=default_neuron,
            neuron_param={}, syn_model=default_synapse, syn_param={}):
        ''' Make a NeuralPop of identical neurons '''
        pop = cls(size, parent)
        pop.new_group("default", range(size), 1, neuron_model, neuron_param,
           syn_model, syn_param)
        return pop

    @classmethod
    def ei_population(cls, size, iratio=0.2, parent=None,
            en_model=default_neuron, en_param={}, es_model=default_synapse,
            es_param={}, in_model=default_neuron, in_param={},
            is_model=default_synapse, is_param={}):
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
            for key,val in model.iteritems():
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
        for group in self.itervalues():
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
            for key,val in param.iteritems():
                for name in group:
                    if key == "neuron":
                        self[name].neuron_param = val
                    elif key == "synapse":
                        self[name].syn_param = val
                    else:
                        raise ValueError("Model type {} is not valid; choose \
among 'neuron' or 'synapse'.".format(key))
            else:
                raise
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
    
    di_dfunc = {
        "constant": delta_distrib,
        "uniform": uniform_distrib,
        "lognormal": lognormal_distrib,
        "gaussian": gaussian_distrib,
        "lin_corr": lin_correlated_distrib,
        "log_corr": log_correlated_distrib
    }

    #-------------------------------------------------------------------------#
    # Class methods
    
    @classmethod
    def distances(cls, graph, elist=None, pos=None, overwrite=False):
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
            pos = graph[POS]
        # compute the new distances
        ra_x = pos[0,elist[:,0]] - pos[0,elist[:,1]]
        ra_y = pos[1,elist[:,0]] - pos[1,elist[:,1]]
        ra_dist = np.tile( np.sqrt( np.square(ra_x) + np.square(ra_y) ), 2)
        ia_sources = np.concatenate((elist[:,0], elist[:,1]))
        ia_targets = np.concatenate((elist[:,1], elist[:,0]))
        new_dist = ssp.coo_matrix( (ra_dist, (ia_sources,ia_targets)), (n,n) )
        # update graph distances
        current_dist = ssp.csr_matrix((n,n))
        if DIST in graph.attributes() and not overwrite:
            current_dist = graph._data[DIST].tocsr()
        total_dist = current_dist.nnz + new_dist.nnz
        new_dist = (current_dist + new_dist.tocsr()).tolil()
        if current_dist.nnz:
            graph._data[DIST] = new_dist
        else:
            new_dist = ssp.coo_matrix((ra_dist,(ia_sources,ia_targets)),(n,n))
            new_dist = new_dist.tolil()
            graph._data[DIST] = new_dist
        return new_dist
    
    @classmethod
    def delays(cls, graph, elist=None, dlist=None, pos=None,
               distrib="constant", distrib_prop={}, correl=None,
               noise_scale=None):
        '''
        Compute the delays of the neuronal connections.
        
        Parameters
        ----------
        graph : class:`~nngt.Graph` or subclass
            Graph the nodes belong to.
        elist : class:`numpy.array`, optional (default: None)
            List of the edges (for user defined weights).
        dlist : class:`numpy.array`, optional (default: None)
            List of the weights (for user defined weights).
        distrib : class:`string`, optional (default: "constant")
            Type of distribution (choose among "constant", "uniform", 
            "lognormal", "gaussian", "user_def", "lin_corr", "log_corr").
        distrib_prop : class:`dict`, optional (default: {})
            Dictionary containing the distribution parameters.
        correl : class:`string`, optional (default: None)
            Property to which the weights should be correlated.
        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.
        
        Returns
        -------
        new_delays : class:`scipy.sparse.lil_matrix`
            A sparse matrix containing *ONLY* the newly-computed weights.
        '''
        n = graph.node_nb()
        corr = correl
        if corr is not None:
            corr = ( graph.get_betweenness(False)[1] if correl == "betweenness"
                                                     else graph[correl] )
        new_delays = None
        if elist is None:
            new_delays = cls.di_dfunc[distrib](graph, correl_attribute=corr,
                                               **distrib_prop)
        elif dlist is not None:
            new_delays = ssp.coo_matrix((dlist,(elist[:,0],elist[:,1])),(n,n))
        else:
            new_delays = cls.di_dfunc[distrib](graph, elist=elist,
                            correl_attribute=corr, **distrib_prop)
        new_delays = new_delays.tolil()
        # add to the graph container
        di_data = graph._data
        mat_delays = di_data[DELAY] if DELAY in di_data.keys() else None
        if mat_delays is not None:
            if not mat_delays.nnz:
                di_data[DELAY] = new_delays
            elif elist is not None:
                eslice = elist[:,0],elist[:,1]
                di_data[DELAY][eslice] = new_delays[eslice]
        else:
            di_data[DELAY] = new_delays
        return new_delays
    
    @classmethod
    def weights(cls, graph, elist=None, wlist=None, distrib="constant",
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
        distrib : class:`string`, optional (default: "constant")
            Type of distribution (choose among "constant", "uniform", 
            "lognormal", "gaussian", "user_def", "lin_corr", "log_corr").
        distrib_prop : class:`dict`, optional (default: {})
            Dictionary containing the distribution parameters.
        correl : class:`string`, optional (default: None)
            Property to which the weights should be correlated.
        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.
        
        Returns
        -------
        new_weights : class:`scipy.sparse.lil_matrix`
            A sparse matrix containing *ONLY* the newly-computed weights.
        '''
        n = graph.node_nb()
        corr = correl
        if corr is not None:
            corr = ( graph.get_betweenness(False)[1] if correl == "betweenness"
                                                     else graph[correl] )
        new_weights = None
        if elist is None:
            new_weights = cls.di_dfunc[distrib](graph, correl_attribute=corr,
                                                **distrib_prop)
        elif wlist is not None:
            new_weights = ssp.coo_matrix((dlist,(elist[:,0],elist[:,1])),(n,n))
        else:
            new_weights = cls.di_dfunc[distrib](graph, elist=elist,
                            correl_attribute=corr, **distrib_prop)
        new_weights = new_weights.tolil()
        # add to the graph container
        mat_weights = (graph._data[WEIGHT] if WEIGHT in graph._data.keys() 
                                           else None)
        if mat_weights is not None and elist is not None:
            eslice = elist[:,0],elist[:,1]
            graph._data[WEIGHT][eslice] = new_weights[eslice]
        else:
            graph._data[WEIGHT] = new_weights
        # add to the graph object
        if "edges" in graph.attributes():
            sources, targets = graph["edges"][:,0], graph["edges"][:,1]
            lst_w = graph._data[WEIGHT][sources,targets].data[0]
            if "weight" in graph.graph.eproperties.keys():
                graph.graph.eproperties["weight"] = lst_w
            else:
                graph.graph.new_edge_attribute(WEIGHT, "double", values=lst_w)
        else:
            graph.graph.new_edge_attribute(WEIGHT, "double")
        return new_weights
    
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
        self.parent = weakref.proxy(parent) if parent is not None else None
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
        if self.parent is not None:
            nodes = self.parent.node_nb()
        ra_x = np.random.uniform(size=nodes)
        ra_y = np.random.uniform(size=nodes)
        return np.array([ra_x,ra_y])
        
