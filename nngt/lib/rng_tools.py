#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Generating the weights of the graph object's connections """

import numpy as np
import scipy.sparse as ssp

import nngt


# ----------- #
# Random seed #
# ----------- #

def seed(seed=None):
    '''
    Seed the random generator used by NNGT (i.e. the numpy `RandomState`: for
    details, see :class:`numpy.random.RandomState`).

    Parameters
    ----------
    seed : int or array_like, optional
        Seed for `RandomState`.
        Must be convertible to 32 bit unsigned integers.
    '''
    np.random.seed(seed)
    nngt._config["seed"] = seed


# ----------------------------- #
# Return the right distribution #
# ----------------------------- #

def _generate_random(number, instructions):
    name = instructions[0]
    if name in di_dfunc:
        return di_dfunc[name](None, None, number, *instructions[1:])
    else:
        raise NotImplementedError(
            "Unknown distribution: '{}'. Supported distributions " \
            "are {}".format(name, ", ".join(di_dfunc.keys())))


def _eprop_distribution(graph, distrib_type, matrix=False, elist=None, **kw):
    ra_values = di_dfunc[distrib_type](graph, elist=elist, **kw)
    num_edges = graph.edge_nb()
    if matrix:
        return _make_matrix(graph, num_edges, ra_values, elist)
    else:
        return ra_values


# --------------------- #
# Generating the matrix #
# --------------------- #

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


# ----------------------- #
# Distribution generators #
# ----------------------- #

def delta_distrib(graph=None, elist=None, num=None, value=1., **kwargs):
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
    num = _compute_num_prop(elist, graph, num)
    return np.repeat(value, num)


def uniform_distrib(graph, elist=None, num=None, lower=0., upper=1.5,
                    **kwargs):
    '''
    Uniform distribution for edge attributes.
    
    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph for which an edge attribute will be generated.
    elist : @todo
    lower : float, optional (default: 0.)
        Min value of the uniform distribution.
    upper : float, optional (default: 1.5)
        Max value of the uniform distribution.
    
    Returns : :class:`numpy.ndarray`
        Attribute value for each edge in `graph`.
    '''
    num = _compute_num_prop(elist, graph, num)
    return np.random.uniform(lower, upper, num)


def gaussian_distrib(graph, elist=None, num=None, avg=1., std=0.2, **kwargs):
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
    num = _compute_num_prop(elist, graph, num)
    return np.random.normal(avg, std, num)


def lognormal_distrib(graph, elist=None, num=None, position=1., scale=0.2,
                      **kwargs):
    num = _compute_num_prop(elist, graph, num)
    return np.random.lognormal(position, scale, num)


def lin_correlated_distrib(graph, elist=None, correl_attribute="betweenness",
                           noise_scale=None, lower=0., upper=2., **kwargs):
    ecount = _compute_num_prop(elist, graph)
    noise = ( 1. if noise_scale is None
                 else np.abs(np.random.normal(1, noise_scale, ecount)) )
    if correl_attribute == "betweenness":
        betw = graph.get_betweenness(kwargs["btype"], kwargs["use_weights"])
        betw *= noise
        bmax = betw.max()
        bmin = betw.min()
        return lower + (upper-lower)*(betw-bmin)/(bmax-bmin)
    else:
        raise NotImplementedError()


def log_correlated_distrib(graph, elist=None, correl_attribute="betweenness",
                           noise_scale=None, lower=0., upper=2.,
                           **kwargs):
    ecount = _compute_num_prop(elist, graph)
    raise NotImplementedError()


di_dfunc = {
    "constant": delta_distrib,
    "uniform": uniform_distrib,
    "lognormal": lognormal_distrib,
    "gaussian": gaussian_distrib,
    "normal": gaussian_distrib,
    "lin_corr": lin_correlated_distrib,
    "log_corr": log_correlated_distrib
}


# ----- #
# Tools #
# ----- #

def _compute_num_prop(elist, graph, ecount=None):
    if ecount is None:
        return len(elist) if elist is not None else graph.edge_nb()
    return ecount
