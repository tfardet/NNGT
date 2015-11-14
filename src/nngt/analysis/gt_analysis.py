#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Tools for graph analysis using the graph_tool library """

import numpy as np

from graph_tool.centrality import betweenness


    
def degree_list(graph, deg_type, use_weights=True):
    deg_propmap = graph.degree_property_map(deg_type)
    if "weight" in graph.edge_properties.keys() and use_weights:
        deg_propmap = graph.degree_property_map(deg_type,
                            graph.edge_properties["weight"])
    return deg_propmap.a

def betweenness_list(graph, use_weights=True):
    if "weight" in graph.edge_properties.keys() and use_weights:
        weight_propmap = graph.copy_property(graph.edge_properties["weight"])
        weight_propmap.a = weight_propmap.a.max() - weight_propmap.a
        tpl = betweenness(graph, weight=weight_propmap)
        return tpl[0].a, tpl[1].a
    else:
        tpl = betweenness(graph)
        return tpl[0].a, tpl[1].a
            
def degree_distrib(graph, deg_type="total", use_weights=True, log=False):
    '''
    Computing the degree distribution of a graph
    
    Parameters
    ----------
    graph : :class:`GraphObject`
        the graph to analyze.
    deg_type : string, optional (default: "total")
        type of degree to consider ("in", "out", or "total")
    use_weights : bool, optional (default: True)
        use weighted degrees (do not take the sign into account : all weights
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
    ia_node_deg = degree_list(graph, deg_type, use_weights)
    num_bins = len(ia_node_deg)
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
    ia_nbetw, ia_ebetw = betweenness_list(graph, use_weights)
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
