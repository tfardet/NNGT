#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Populations of neurons in NNGT """

import warnings


__all__ = ["NeuralGroup", "_make_groups"]



#
#---
# NeuralGroup
#------------------------

class NeuralGroup:
    
    """
    Class defining groups of neurons.
    
    :ivar id_list: :class:`list` of :class:`int`s
        the ids of the neurons in this group.
    :ivar neuron_type: :class:`int`
        the default is ``1`` for excitatory neurons; ``-1`` is for interneurons 
    :ivar model: :class:`string`, optional (default: None)
        the name of the model to use when simulating the activity of this group
    :ivar param: :class:`dict`, optional (default: {})
        the parameters to use (if they differ from the model's defaults)
        
    .. warning::
        Equality between :class:`~nngt.properties.NeuralGroup`s only compares 
        the neuronal and synaptic ``model`` and ``param`` attributes, i.e. 
        groups differing only by their ``id_list``s will register as equal.
    .. note::
        By default, synapses are registered as "static_synapse"s in NEST; 
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
            self.syn_model = syn_model
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

    def properties(self):
        dic = { "neuron_type": self.neuron_type,
                "neuron_model": self._neuron_model,
                "neuron_param": self.neuron_param,
                "syn_model": self.syn_model,
                "syn_param": self.syn_param }
        return dic

    
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
