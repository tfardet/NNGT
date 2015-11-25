#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Generating the weights of the graph object's connections """

import numpy as np
import scipy.sparse as ssp



def uniform_weights(graph, min_weight=0., max_weight=1.5, **kwargs):
    ecount = graph.edge_nb()
    mat_weights = graph.adjacency_matrix()
    mat_weights.data = np.random.uniform(min_weight, max_weight, ecount)
    mat_weights = mat_weights.tolil()
    mat_weights.setdiag(0)
    return mat_weights

def gaussian_weights(graph, avg_weight=1., std_dev=0.2, **kwargs):
    ecount = graph.edge_nb()
    mat_weights = graph.adjacency_matrix()
    mat_weights.data = np.random.normal(avg_weight, std_dev, ecount)
    mat_weights = mat_weights.tolil()
    mat_weights.setdiag(0)
    return mat_weights

def lognormal_weights(graph, position=1., scale=0.2, **kwargs):
    ecount = graph.edge_nb()
    mat_weights = graph.adjacency_matrix()
    mat_weights.data = np.random.lognormal(position, scale, ecount)
    mat_weights = mat_weights.tolil()
    mat_weights.setdiag(0)
    return mat_weights

def lin_correlated_weights(graph, correl_attribute, noise_scale=None,
                           min_weight=0., max_weight=2., **kwargs):
    ra_w = ( 1 if noise_scale is None else np.abs(np.random.normal(1,
                                                    noise_scale, ecount)) )
    if correl_attribute == "betweenness":
        pass
    else:
        pass

def log_correlated_weights(graph, correl_attribute, noise_scale=None, 
                           min_weight=0., max_weight=2., **kwargs):
    pass
