#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Connections properties in NNGT """

import scipy.sparse as ssp



#
#---
# SparseConnections
#------------------------

class SparseConnections:
    
    """    
    The basic class that contains the properties of the connections between
    neurons for sparse graphs.

    :ivar delays: :class:`~scipy.sparse.lil_matrix` of :class:`double`
        a *lil* sparse matrix containing the delay of spike propagation between
        pairs of neurons.
    :ivar syn_model: :class:`~scipy.sparse.lil_matrix` of :class:`str`
        name of the synaptic model to be used when simulating the network's 
        activity.
    :ivar param: :class:`~scipy.sparse.lil_matrix` of :class:`dict`
        NEST parameters for each of the connections.
    :ivar distances: :class:`~scipy.sparse.lil_matrix` of :class:`double`,
        optional (default: None)
        sparse matrix containing the distances between connected neurons
    """
    
    def __init__(self):
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
        pass    


#
#---
# DenseConnections
#------------------------

class DenseConnections:

    """    
    The basic class that contains the properties of the connections between
    neurons for sparse graphs.

    :ivar delays: :class:`numpy.array` of :class:`double`
        matrix containing the delay of spike propagation between pairs of
        neurons.
    :ivar syn_model: :class:`numpy.array` of :class:`str`
        name of the synaptic model to be used when simulating the network's 
        activity.
    :ivar param: :class:`numpy.array` of :class:`dict`
        NEST parameters for each of the connections.
    :ivar distances: :class:`numpy.array` of :class:`double`,
        optional (default: None)
        matrix containing the distances between connected neurons
    """
    
    def __init__(self):
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
