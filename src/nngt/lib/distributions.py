#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generating the weights of the graph object's connections """

import numpy as np
import scipy.sparse as ssp



#-----------------------------------------------------------------------------#
# Return the right distribution
#------------------------
#

def eprop_distribution(graph, distrib_type, matrix=False, elist=None, **kw):
    ra_values = di_dfunc[distrib_type](graph, elist=elist, **kw)
    num_edges = graph.edge_nb()
    if matrix:
        return _make_matrix(graph, num_edges, ra_values, elist)
    else:
        return ra_values


#-----------------------------------------------------------------------------#
# Generating the matrix
#------------------------
#

def _make_matrix(graph, ecount, values, elist=None):
    mat_distrib = None
    n = graph.node_nb()
    if elist is not None and graph.edge_nb():
        mat_distrib = ssp.coo_matrix((values,(elist[:,0],elist[:,1])),(n,n))
    else:
        mat_distrib = graph.adjacency_matrix()
        mat_distrib.data = values
    mat_distrib = mat_distrib.tolil()
    mat_distrib.setdiag(np.zeros(n))
    return mat_distrib


#-----------------------------------------------------------------------------#
# Distribution generators
#------------------------
#

def delta_distrib(graph, elist=None, value=1., **kwargs):
    '''
    Delta distribution for edge attributes.
    
    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph for which an edge attribute will be generated.
    elist : @todo
    value : float, optional (default: 1.)
        Value of the delta distribution.
    
    Returns : :class:`numpy.ndarray`
        Attribute value for each edge in `graph`.
    '''
    ecount = elist.shape[0] if elist is not None else graph.edge_nb()
    return np.repeat(value, ecount)

def uniform_distrib(graph, elist=None, min=0., max=1.5,
                    **kwargs):
    '''
    Uniform distribution for edge attributes.
    
    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph for which an edge attribute will be generated.
    elist : @todo
    min : float, optional (default: 0.)
        Min value of the uniform distribution.
    min : float, optional (default: 1.5)
        Max value of the uniform distribution.
    
    Returns : :class:`numpy.ndarray`
        Attribute value for each edge in `graph`.
    '''
    ecount = elist.shape[0] if elist is not None else graph.edge_nb()
    return np.random.uniform(min, max, ecount)

def gaussian_distrib(graph, elist=None, avg=1., std=0.2, **kwargs):
    '''
    Gaussian distribution for edge attributes.
    
    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph for which an edge attribute will be generated.
    elist : @todo
    avg : float, optional (default: 0.)
        Average of the Gaussian distribution.
    std : float, optional (default: 1.5)
        Standard deviation of the Gaussian distribution.
    
    Returns : :class:`numpy.ndarray`
        Attribute value for each edge in `graph`.
    '''
    ecount = elist.shape[0] if elist is not None else graph.edge_nb()
    return np.random.normal(avg, std, ecount)

def lognormal_distrib(graph, elist=None, position=1., scale=0.2, **kwargs):
    ecount = elist.shape[0] if elist is not None else graph.edge_nb()
    return np.random.lognormal(position, scale, ecount)

def lin_correlated_distrib(correl_attribute, graph, elist=None,
                           noise_scale=None, min=0., max=2., **kwargs):
    ecount = elist.shape[0] if elist is not None else graph.edge_nb()
    ra_noise = ( 1 if noise_scale is None else np.abs(np.random.normal(1,
                 noise_scale, ecount)) )
    if correl_attribute == "betweenness":
        pass
    else:
        pass

def log_correlated_distrib(correl_attribute, graph, elist=None,
                           noise_scale=None, min=0., max=2.,
                           **kwargs):
    ecount = elist.shape[0] if elist is not None else graph.edge_nb()
    pass


di_dfunc = {
    "constant": delta_distrib,
    "uniform": uniform_distrib,
    "lognormal": lognormal_distrib,
    "gaussian": gaussian_distrib,
    "lin_corr": lin_correlated_distrib,
    "log_corr": log_correlated_distrib
}
