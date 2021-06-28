#-*- coding:utf-8 -*-
#
# lib/rng_tools.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Generating the weights of the graph object's connections """

import numpy as np
import scipy.sparse as ssp

import nngt

from .errors import InvalidArgument
from .test_functions import nonstring_container
from .test_functions import mpi_random


# ----------- #
# Random seed #
# ----------- #

@mpi_random
def seed(msd=None, seeds=None):
    '''
    Seed the random generator used by NNGT
    (i.e. the numpy `RandomState`: for details, see
    :class:`numpy.random.RandomState`).

    Parameters
    ----------
    msd : int, optional
        Master seed for numpy `RandomState`.
        Must be convertible to 32-bit unsigned integers.
    seeds : list of ints, optional
        Seeds for `RandomState` (when using MPI).
        Must be convertible to 32-bit unsigned integers, one entry per MPI
        process.
    '''
    # when using MPI numpy seeeds are sync-ed via the mpi_random decorator
    msd = np.random.randint(0, 2**31 - 1) if msd is None else msd

    # seed both random state and new generator
    np.random.seed(msd)
    nngt._rng = np.random.default_rng(msd)

    nngt._config['msd'] = msd

    nngt._seeded = True

    nngt._seeded_local = False

    # check subseeds
    if seeds is not None:
        with_mt = nngt.get_config('multithreading')
        with_mpi = nngt.get_config('mpi')
        err = 'Expected {} seeds.'

        if with_mpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            assert size == len(seeds), err.format(size)
            nngt._config['seeds'] = seeds
        elif with_mt:
            num_omp = nngt.get_config('omp')
            assert num_omp == len(seeds), err.format(num_omp)
            nngt._config['seeds'] = seeds

        nngt._seeded_local = True
        nngt._used_local   = False


# ----------------------------- #
# Return the right distribution #
# ----------------------------- #

def _generate_random(number, instructions):
    name = "not defined"

    if isinstance(instructions, dict):
        name = instructions["distribution"]

        instructions = {
            k: v for k, v in instructions.items() if k != "distribution"
        }

        if name in di_dfunc:
            return di_dfunc[name](None, None, number, **instructions)

        raise NotImplementedError(
            "Unknown distribution: '{}'. Supported distributions " \
            "are {}".format(name, ", ".join(di_dfunc.keys())))
    elif nonstring_container(instructions):
        name = instructions[0]

        if name in di_dfunc:
            return di_dfunc[name](None, None, number, *instructions[1:])

        raise NotImplementedError(
            "Unknown distribution: '{}'. Supported distributions " \
            "are {}".format(name, ", ".join(di_dfunc.keys())))

    raise NotImplementedError(
        "Unknown instructions: '{}'".format(instructions))


def _eprop_distribution(graph, distrib_type, matrix=False, elist=None,
                        last_edges=False, **kw):
    ra_values = di_dfunc[distrib_type](graph, elist=elist,
                                       last_edges=last_edges, **kw)
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
        mat_distrib = ssp.coo_matrix(
            (values, (elist[:, 0], elist[:, 1])), (n, n))
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
    elist : list of edges, optional (default: all edges)
        Generate values for only a subset of edges.
    value : float, optional (default: 1.)
        Value of the delta distribution.

    Returns : :class:`numpy.ndarray`
        Attribute value for each edge in `graph`.
    '''
    num = _compute_num_prop(elist, graph, num)
    return np.repeat(value, num)


def uniform_distrib(graph, elist=None, num=None, lower=None, upper=None,
                    **kwargs):
    '''
    Uniform distribution for edge attributes.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph for which an edge attribute will be generated.
    elist : list of edges, optional (default: all edges)
        Generate values for only a subset of edges.
    lower : float, optional (default: 0.)
        Min value of the uniform distribution.
    upper : float, optional (default: 1.5)
        Max value of the uniform distribution.

    Returns : :class:`numpy.ndarray`
        Attribute value for each edge in `graph`.
    '''
    num = _compute_num_prop(elist, graph, num)
    return np.random.uniform(lower, upper, num)


def gaussian_distrib(graph, elist=None, num=None, avg=None, std=None,
                     **kwargs):
    '''
    Gaussian distribution for edge attributes.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph for which an edge attribute will be generated.
    elist : list of edges, optional (default: all edges)
        Generate values for only a subset of edges.
    avg : float, optional (default: 0.)
        Average of the Gaussian distribution.
    std : float, optional (default: 1.5)
        Standard deviation of the Gaussian distribution.

    Returns : :class:`numpy.ndarray`
        Attribute value for each edge in `graph`.
    '''
    num = _compute_num_prop(elist, graph, num)
    return np.random.normal(avg, std, num)


def lognormal_distrib(graph, elist=None, num=None, position=None, scale=None,
                      **kwargs):
    '''
    Lognormal distribution for edge attributes.

    Parameters
    ----------
    graph : :class:`~nngt.Graph` or subclass
        Graph for which an edge attribute will be generated.
    elist : list of edges, optional (default: all edges)
        Generate values for only a subset of edges.
    position : float, optional (default: 0.)
        Average of the normal distribution (i.e. log of the actual mean of the
        lognormal distribution).
    scale : float, optional (default: 1.5)
        Standard deviation of the normal distribution.

    Returns : :class:`numpy.ndarray`
        Attribute value for each edge in `graph`.
    '''
    num = _compute_num_prop(elist, graph, num)
    return np.random.lognormal(position, scale, num)


def lin_correlated_distrib(graph, elist=None, correl_attribute="betweenness",
                           noise_scale=None, lower=None, upper=None,
                           slope=None, offset=0., last_edges=False, **kwargs):
    if slope is not None and (lower, upper) != (None, None):
        raise InvalidArgument('`slope` and `lower`/`upper` parameters are not '
                              'compatible, please choose one or the other.')
    elif (lower is not None or upper is not None) and None in (lower, upper):
        raise InvalidArgument('Both `lower` and `upper` should be set if one '
                              'of the two is used.')
    ecount = _compute_num_prop(elist, graph)
    noise = (1. if noise_scale is None
             else np.abs(np.random.normal(1, noise_scale, ecount)))
    data = None
    if correl_attribute == "betweenness":
        data = graph.get_betweenness(kwargs["btype"], kwargs["weights"])
    elif correl_attribute == "distance":
        assert 'distance' in graph.edge_attributes, \
            'Graph has no "distance" edge attribute.'
        if 'distance' not in kwargs:
            if last_edges:
                data = graph._eattr['distance'][-len(elist):]
            else:
                data = graph.get_edge_attributes(elist, 'distance')
        else:
            data = kwargs['distance']
    else:
        raise NotImplementedError()
    if noise_scale is not None:
        data *= noise
    if len(data):
        if slope is None:
            dmax = np.max(data)
            dmin = np.min(data)
            return lower + (upper-lower)*(data-dmin)/(dmax-dmin) + offset
        else:
            return slope*data + offset
    return np.array([])


def log_correlated_distrib(graph, elist=None, correl_attribute="betweenness",
                           noise_scale=None, lower=0., upper=2.,
                           **kwargs):
    ecount = _compute_num_prop(elist, graph)
    raise NotImplementedError()


def custom(graph, values=None, elist=None, **kwargs):
    if values is None and elist is not None:
        return np.ones(len(elist))

    return values


di_dfunc = {
    "constant": delta_distrib,
    "uniform": uniform_distrib,
    "lognormal": lognormal_distrib,
    "gaussian": gaussian_distrib,
    "normal": gaussian_distrib,
    "lin_corr": lin_correlated_distrib,
    "log_corr": log_correlated_distrib,
    "custom": custom
}


# ----- #
# Tools #
# ----- #

def _compute_num_prop(elist, graph, ecount=None):
    if ecount is None:
        return len(elist) if elist is not None else graph.edge_nb()
    return ecount
