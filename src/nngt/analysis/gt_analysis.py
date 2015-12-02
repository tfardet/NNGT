#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Tools for graph analysis using the graph_tool library """

import scipy as sp
import scipy.sparse.linalg as spl

from ..globals import (adjacency, assort, edge_reciprocity, s_glib,
                       global_clustering, label_components, pseudo_diameter)


#-----------------------------------------------------------------------------#
# Distributions
#------------------------
#

def degree_distrib(net, deg_type="total", node_list=None, use_weights=True,
                   log=False, num_bins=30):
    '''
    Computing the degree distribution of a network.
    
    Parameters
    ----------
    net : :class:`~nngt.Graph` or subclass
        the network to analyze.
    deg_type : string, optional (default: "total")
        type of degree to consider ("in", "out", or "total").
    node_list : list or numpy.array of ints, optional (default: None)
        Restrict the distribution to a set of nodes (default: all nodes).
    use_weights : bool, optional (default: True)
        use weighted degrees (do not take the sign into account: all weights
        are positive).
    log : bool, optional (default: False)
        use log-spaced bins.
    
    Returns
    -------
    counts : :class:`numpy.array`
        number of nodes in each bin
    deg : :class:`numpy.array`
        bins
    '''
    ia_node_deg = net.get_degrees(node_list, deg_type, use_weights)
    ra_bins = sp.linspace(ia_node_deg.min(), ia_node_deg.max(), num_bins)
    if log:
        ra_bins = sp.logspace(sp.log10(sp.maximum(ia_node_deg.min(),1)),
                               sp.log10(ia_node_deg.max()), num_bins)
    counts,deg = sp.histogram(ia_node_deg, ra_bins)
    ia_indices = sp.argwhere(counts)
    return counts[ia_indices], deg[ia_indices]
            
def betweenness_distrib(net, use_weights=True, log=False):
    '''
    Computing the betweenness distribution of a network
    
    Parameters
    ----------
    net : :class:`~nngt.Graph` or subclass
        the network to analyze.
    use_weights : bool, optional (default: True)
        use weighted degrees (do not take the sign into account : all weights
        are positive).
    log : bool, optional (default: False)
        use log-spaced bins.
    
    Returns
    -------
    ncounts : :class:`numpy.array`
        number of nodes in each bin
    nbetw : :class:`numpy.array`
        bins for node betweenness
    ecounts : :class:`numpy.array`
        number of edges in each bin
    ebetw : :class:`numpy.array`
        bins for edge betweenness
    '''
    ia_nbetw, ia_ebetw = net.get_betweenness(use_weights)
    num_nbins, num_ebins = int(len(ia_nbetw) / 50), int(len(ia_ebetw) / 50)
    ra_nbins = sp.linspace(ia_nbetw.min(), ia_nbetw.max(), num_nbins)
    ra_ebins = sp.linspace(ia_ebetw.min(), ia_ebetw.max(), num_ebins)
    if log:
        ra_nbins = sp.logspace(sp.log10(sp.maximum(ia_nbetw.min(),10**-8)),
                               sp.log10(ia_nbetw.max()), num_nbins)
        ra_ebins = sp.logspace(sp.log10(sp.maximum(ia_ebetw.min(),10**-8)),
                               sp.log10(ia_ebetw.max()), num_ebins)
    ncounts,nbetw = sp.histogram(ia_nbetw, ra_nbins)
    ecounts,ebetw = sp.histogram(ia_ebetw, ra_ebins)
    return ncounts, nbetw[:-1], ecounts, ebetw[:-1]


#-----------------------------------------------------------------------------#
# Scalar properties
#------------------------
#

def assortativity(net, deg_type="total"):
    '''
    Assortativity of the graph.
    
    Parameters
    ----------
    net : :class:`~nngt.Graph` or subclass
        Network to analyze.
    deg_type : string, optional (default: 'total')
        Type of degree to take into account (among 'in', 'out' or 'total').
    
    Returns
    -------
    a float describing the network assortativity.
    '''
    return assort(net.graph,"total")[0]

def reciprocity(net):
    '''
    Returns the network reciprocity, defined as :math:`E^\leftrightarrow/E`,
    where :math:`E^\leftrightarrow` and :math:`E` are, respectively, the number
    of bidirectional edges and the total number of edges in the network.
    '''
    return edge_reciprocity(net.graph)

def clustering(net):
    '''
    Returns the global clustering coefficient of the graph, defined as
    
    .. math::
       c = 3 \times \frac{\text{number of triangles}}
                         {\text{number of connected triples}}
    '''
    return global_clustering(net.graph)[0]

def num_iedges(net):
    ''' Returns the number of inhibitory connections. '''
    num_einhib = len(net["type"].a < 0)
    return float(num_einhib)/net.edge_nb()

def num_scc(net, listing=False):
    '''
    Returns the number of strongly connected components, i.e. ensembles where 
    all nodes inside the ensemble can reach any other node in the ensemble
    using the directed edges.
    
    See also
    --------
    num_wcc
    '''
    vprop_comp, lst_histo = label_components(net.graph,directed=True)
    if listing:
        return len(lst_histo), lst_histo
    else:
        return len(lst_histo)

def num_wcc(net, listing=False):
    '''
    Connected components if the directivity of the edges is ignored (i.e. all 
    edges are considered as bidirectional).
    
    See also
    --------
    num_scc
    '''
    vprop_comp, lst_histo = label_components(net.graph,directed=False)
    if listing:
        return len(lst_histo), lst_histo
    else:
        return len(lst_histo)

def diameter(net):
    ''' Pseudo-diameter of the graph '''
    return pseudo_diameter(net.graph)[0]


#-----------------------------------------------------------------------------#
# Spectral properties
#------------------------
#

def spectral_radius(net, typed=True, weighted=True):
    '''
    Spectral radius of the graph, defined as the eigenvalue of greatest module.
    
    Parameters
    ----------
    net : :class:`~nngt.Graph` or subclass
        Network to analyze.
    typed : bool, optional (default: True)
        Whether the excitatory/inhibitory type of the connnections should be
        considered.
    weighted : bool, optional (default: True)
        Whether the weights should be taken into account.
    
    Returns
    -------
    the spectral radius as a float.
    '''
    weights = None
    if typed and "type" in net.graph.edge_attributes.keys():
        weights = net.edge_attributes["type"].copy()
    if weighted and "weight" in net.graph.edge_attributes.keys():
        if weights is not None:
            weights = sp.multiply(weights,
                                  net.graph.edge_attributes["weight"])
        else:
            weights = net.graph.edge_attributes["weight"].copy()
    matAdj = adjacency(net.graph,weights)
    eigVal = [0]
    try:
        eigVal = spl.eigs(matAdj,return_eigenvectors=False)
    except spl.eigen.arpack.ArpackNoConvergence,err:
        eigVal = err.eigenvalues
    if len(eigVal):
        return sp.amax(sp.absolute(eigVal))
    else:
        raise spl.eigen.arpack.ArpackNoConvergence()

def adjacency_matrix(net, typed=True, weighted=True, eprop=None):
    '''
    Adjacency matrix of the graph.
    
    Parameters
    ----------
    net : :class:`~nngt.Graph` or subclass
        Network to analyze.
    typed : bool, optional (default: True)
        Whether the excitatory/inhibitory type of the connnections should be
        considered.
    weighted : bool, optional (default: True)
        Whether the weights should be taken into account.
    
    Returns
    -------
    a :class:`~scipy.sparse.csr_matrix`.
    '''
    weights = None
    if typed and "type" in net.graph.edge_attributes.keys():
        weights = net.edge_attributes["type"].copy()
    if weighted and "weight" in net.graph.edge_attributes.keys():
        weights = sp.multiply(weights,
                                graph.edge_attributes["weight"])
    return adjacency(net.graph, weights)
        
