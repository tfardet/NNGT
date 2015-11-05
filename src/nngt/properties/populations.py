#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Populations of neurons in NNGT """

import warnings



#
#---
# NeuralPop
#------------------------

class NeuralPop(dict):
    
    """    
    The basic class that contains groups of neurons and their properties.

    :ivar has_models: :class:`bool`,
        ``True`` if every group has a ``model`` attribute.
    """
    
    @classmethod
    def pop_from_network(cls, graph, *args):
        '''
        Make a NeuralPop object from a network. The groups of neurons are 
        determined using instructions from an arbitrary number of
        :class:`~nngt.properties.GroupProperties`.
        '''
        return cls.__init__(graph=graph,group_prop=args)
    
    @classmethod
    def copy(cls, pop):
        ''' Copy an existing NeuralPop '''
        new_pop = cls.__init__(pop.has_models)
        for name, group in pop.items():
             new_pop.new_group(name, group.id_list, group.model, group.param)
        return new_pop
        
    def __init__(self, with_models=True, **kwargs):
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
        if "graph" in kwargs:
            dic = _make_groups(kwargs["graph"], kwargs["group_prop"])
        else:
            super(NeuralPop, self).__init__()
        self._has_models = with_models
        
    @property
    def has_models(self):
        return self._has_models
    
    def set_models(self, models=None):
        if isinstance(models, str) or models is None:
            for group in self.itervalues():
                group.model = models
        elif list(model.keys()) == list(self.keys()):
            for name, model in models.items():
                self[name].model = model
        else:
            raise ArgumentError("set_models argument should be either a string\
                                 or a dict with the same keys as NeuralPop")
        b_has_models = True
        for group in self.itervalues():
            b_has_model *= group.has_model
        self._has_models = b_has_models
    
    def new_group(self, name, id_list, model=None, param={}):
        group = NeuralGroup(id_list, model, param)
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
    
    def neuron_property(self, idx):
        pass
        


#
#---
# NeuralGroup
#------------------------

class NeuralGroup(_Group):
    
    """
    Class defining groups of neurons.
    
    :ivar id_list: :class:`list` of :class:`int`s
        the ids of the neurons in this group.
    :ivar model: :class:`string`, optional (default: None)
        the name of the model to use when simulating the activity of this group
    :ivar param: :class:`dict`, optional (default: {})
        the parameters to use (if they differ from the model's defaults)
        
    .. warning::
        Equality between :class:`~nngt.properties.NeuralGroup`s only compares 
        the ``model`` and ``param`` attributes, i.e. groups differing only by 
        their ``id_list``s will register as equal.
    """
    
    def __init__ (self, id_list=[], model=None, param={}):
        self._model = model
        self.param = param
        self._has_model = False if model is None else True
        self._id_list = list(id_list)
    
    def __eq__ (self, other):
        if isinstance(other, NeuralGroup):
            return self.model == other.model * self.param == other.param
        else:
            return False
        
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
        self._has_model = False if model is None else True
    
    @property
    def id_list(self):
        return self._id_list
    
    @property
    def has_model(self):
        return self._has_model
            

#
#---
# GroupProperties
#------------------------

class GroupProperties(_Group):
    
    """
    Class defining the properties needed to create groups of neurons from an
    existing :class:`~nngt.GraphClass` or one of its subclasses.
    
    :ivar size: :class:`int`
        Size of the group.
    :ivar constraints: :class:`dict`, optional (default: {})
        Constraints to respect when building the 
        :class:`~nngt.properties.NeuralGroup` .
    :ivar model: :class:`string`, optional (default: None)
        name of the model to use when simulating the activity of this group.
    :ivar param: :class:`dict`, optional (default: {})
        the parameters to use (if they differ from the model's defaults)
    """
    
    def __init__ (self, size, constraints={}, model=None, param={}):
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
        self.model = model
        self.param = param

    
#
#---
# Making groups
#------------------------

def _make_groups(graph, group_prop):
    '''
    Divide `graph` into groups using `group_prop`, a list of group properties
    @todo
    '''
    pass
