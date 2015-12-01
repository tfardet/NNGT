#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Tools for graph analysis using the graph_tool library """

import numpy as np


            
def degree_distrib(graph, deg_type="total", node_list=None, use_weights=True,
                   log=False, num_bins=30):
    '''
    Computing the degree distribution of a graph
    
    Parameters
    ----------
    graph : :class:`GraphObject`
        the graph to analyze.
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
    ia_node_deg = graph.degree_list(node_list, deg_type, use_weights)
    ra_bins = np.linspace(ia_node_deg.min(), ia_node_deg.max(), num_bins)
    if log:
        ra_bins = np.logspace(np.log10(np.maximum(ia_node_deg.min(),1)),
                               np.log10(ia_node_deg.max()), num_bins)
    counts,deg = np.histogram(ia_node_deg, ra_bins)
    ia_indices = np.argwhere(counts)
    return counts[ia_indices], deg[ia_indices]
            
def betweenness_distrib(graph, use_weights=True, log=False):
    '''
    Computing the betweenness distribution of a graph
    
    Parameters
    ----------
    graph : :class:`GraphObject`
        the graph to analyze.
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
    ia_nbetw, ia_ebetw = graph.betweenness_list(use_weights)
    num_nbins, num_ebins = int(len(ia_nbetw) / 50), int(len(ia_ebetw) / 50)
    ra_nbins = np.linspace(ia_nbetw.min(), ia_nbetw.max(), num_nbins)
    ra_ebins = np.linspace(ia_ebetw.min(), ia_ebetw.max(), num_ebins)
    if log:
        ra_nbins = np.logspace(np.log10(np.maximum(ia_nbetw.min(),10**-8)),
                               np.log10(ia_nbetw.max()), num_nbins)
        ra_ebins = np.logspace(np.log10(np.maximum(ia_ebetw.min(),10**-8)),
                               np.log10(ia_ebetw.max()), num_ebins)
    ncounts,nbetw = np.histogram(ia_nbetw, ra_nbins)
    ecounts,ebetw = np.histogram(ia_ebetw, ra_ebins)
    return ncounts, nbetw[:-1], ecounts, ebetw[:-1]
