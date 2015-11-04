#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Populations of neurons in NNGT """


#
#---
# NeuralPop
#------------------------

class NeuralPop(dict):
    
    """
    .. py:currentmodule:: nggt.properties
    
    The basic class that contains populations of neurons and their properties.

    :ivar id: :class:`int`
        unique id that identifies the instance.
    :ivar graph: :class:`~nngt.core.GraphObject`
        main attribute of the class instance.
    """

    __num_graphs = 0
    __max_id = 0
    __di_property_func = {
            "reciprocity": reciprocity, "clustering": clustering,
            "assortativity": assortativity, "diameter": diameter,
            "scc": num_scc, "wcc": num_wcc, "radius": spectral_radius, 
            "num_iedges": num_iedges }
    __properties = __di_property_func.keys()
    
    @classmethod
    def graph_into_pop(cls, graph, *args):
        '''
        Divide a graph into populations of neurons using instruction from
        an arbitrary number of :class:`PopIntructions`.
        @todo
        '''
        return cls.__num_graphs


    def __init__(self, *args, **kwargs):
        '''
        Initialize NeuralPop instance

        Parameters
        ----------
        
        
        Returns
        -------
        
        '''
        pass
