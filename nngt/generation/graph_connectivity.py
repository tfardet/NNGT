#-*- coding:utf-8 -*-
#
# graph_connectivity.py
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
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

""" Connectivity generators for nngt.Graph """

from copy import deepcopy
import logging

import numpy as np

import nngt
from nngt.geometry.geom_utils import conversion_magnitude
from nngt.lib.connect_tools import _set_options
from nngt.lib.logger import _log_message
from nngt.lib.test_functions import (mpi_checker, mpi_random, deprecated,
                                     on_master_process)


# do default import

from .connect_algorithms import *

# try to import multithreaded or mpi algorithms

using_mt_algorithms = False

if nngt.get_config("multithreading"):
    logger = logging.getLogger(__name__)
    try:
        from .cconnect import *
        from .connect_algorithms import price_network
        using_mt_algorithms = True
        _log_message(logger, "DEBUG",
                     "Using multithreaded algorithms compiled on install.")
        nngt.set_config('multithreading', True, silent=True)
    except Exception as e:
        try:
            import cython
            import pyximport

            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                comm.bcast(pyximport.install(language_level=3))
            except:
                if nngt.get_config("mpi"):
                    raise RuntimeError("Cannot safely compile with MPI.")

                pyximport.install(language_level=3)

            # wait for compilation to finish
            nngt.lib.mpi_barrier()

            if on_master_process():
                from .cconnect import *
    
            nngt.lib.mpi_barrier()

            from .cconnect import *
            from .connect_algorithms import price_network

            using_mt_algorithms = True

            _log_message(logger, "DEBUG", str(e) + "\n\tCompiled "
                         "multithreaded algorithms on-the-run.")

            nngt.set_config('multithreading', True, silent=True)
        except Exception as e2:
            _log_message(
                logger, "WARNING", str(e) + "\n\t" + str(e2) + "\n\t"
                "Cython import failed, using non-multithreaded algorithms.")
            nngt._config['multithreading'] = False

if nngt.get_config("mpi"):
    try:
        from .mpi_connect import *
        nngt._config['mpi'] = True
    except ImportError as e:
        nngt._config['mpi'] = False
        raise e


__all__ = [
    'all_to_all',
    'circular',
	'distance_rule',
	'erdos_renyi',
    'fixed_degree',
    'from_degree_list',
    'gaussian_degree',
	'newman_watts',
	'random_scale_free',
	'price_scale_free',
    'watts_strogatz',
]


# ----------------------------- #
# Specific degree distributions #
# ----------------------------- #

def all_to_all(nodes=0, weighted=True, directed=True, multigraph=False,
               name="AllToAll", shape=None, positions=None, population=None,
               **kwargs):
    """
    Generate a graph where all nodes are connected.

    .. versionadded:: 1.0

    Parameters
    ----------
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    reciprocity : double, optional (default: -1 to let it free)
        Fraction of edges that are bidirectional  (only for directed graphs
        -- undirected graphs have a reciprocity of  1 by definition)
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment.
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).

    Note
    ----
	`nodes` is required unless `population` is provided.

    Returns
    -------
    graph_all : :class:`~nngt.Graph`, or subclass
        A new generated graph.
    """
    nodes = nodes if population is None else population.size

    graph_all = nngt.Graph(name=name, nodes=nodes, directed=directed, **kwargs)

    _set_options(graph_all, population, shape, positions)

    # add edges
    if nodes > 1:
        ids = np.arange(nodes, dtype=np.uint)
        edges = _all_to_all(ids, ids, directed=directed, multigraph=multigraph)
        graph_all.new_edges(edges, check_duplicates=False,
                            check_self_loops=False, check_existing=False)

    graph_all._graph_type = "all_to_all"

    return graph_all


@mpi_random
def from_degree_list(degrees, degree_type='in', weighted=True,
                     directed=True, multigraph=False, name="DL",
                     shape=None, positions=None, population=None,
                     from_graph=None, **kwargs):
    """
    Generate a random graph from a given list of degrees.

    Parameters
    ----------
    degrees : list
        The list of degrees for each node in the graph.
    degree_type : str, optional (default: 'in')
        The type of the fixed degree, among ``'in'``, ``'out'`` or ``'total'``.
        @todo `'total'` not implemented yet.
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        @todo: only for directed graphs for now. Whether the graph is directed
        or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment.
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_dl : :class:`~nngt.Graph`, or subclass
        A new generated graph or the modified `from_graph`.
    """
    # set node number and library graph
    graph_dl = from_graph
    nodes    = len(degrees)

    if "nodes" in kwargs:
        assert kwargs["nodes"] == nodes, \
            "Invalid `nodes` entry: the number of nodes should " \
            "be ``len(degrees)``."
        del kwargs["nodes"]

    if graph_dl is not None:
        nodes = graph_dl.node_nb()
        graph_dl.clear_all_edges()
    else:
        nodes = population.size if population is not None else nodes
        graph_dl = nngt.Graph(
            name=name, nodes=nodes, directed=directed, **kwargs)

    _set_options(graph_dl, population, shape, positions)

    # add edges
    ia_edges = None

    if nodes > 1:
        ids = np.arange(nodes, dtype=np.uint)
        ia_edges = _from_degree_list(ids, ids, degrees, degree_type,
                                     directed=directed, multigraph=multigraph)
        # check for None if MPI
        if ia_edges is not None:
            graph_dl.new_edges(ia_edges, check_duplicates=False,
                               check_self_loops=False, check_existing=False)

    graph_dl._graph_type = "from_{}_degree_list".format(degree_type)

    return graph_dl


@mpi_random
def fixed_degree(degree, degree_type='in', nodes=0, reciprocity=-1.,
                 weighted=True, directed=True, multigraph=False, name="FD",
                 shape=None, positions=None, population=None, from_graph=None,
                 **kwargs):
    """
    Generate a random graph with constant in- or out-degree.

    Parameters
    ----------
    degree : int
        The value of the constant degree.
    degree_type : str, optional (default: 'in')
        The type of the fixed degree, among ``'in'``, ``'out'`` or ``'total'``.

        @todo
			`'total'` not implemented yet.

    nodes : int, optional (default: None)
        The number of nodes in the graph.
    reciprocity : double, optional (default: -1 to let it free)
        @todo: not implemented yet. Fraction of edges that are bidirectional
        (only for directed graphs -- undirected graphs have a reciprocity of
        1 by definition)
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        @todo: only for directed graphs for now. Whether the graph is directed
        or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment.
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Note
    ----
	`nodes` is required unless `from_graph` or `population` is provided.
	If an `from_graph` is provided, all preexistant edges in the
	object will be deleted before the new connectivity is implemented.

    Returns
    -------
    graph_fd : :class:`~nngt.Graph`, or subclass
        A new generated graph or the modified `from_graph`.
    """
    # set node number and library graph
    graph_fd = from_graph

    if graph_fd is not None:
        nodes = graph_fd.node_nb()
        graph_fd.clear_all_edges()
    else:
        nodes = population.size if population is not None else nodes
        graph_fd = nngt.Graph(
            name=name, nodes=nodes, directed=directed, **kwargs)

    _set_options(graph_fd, population, shape, positions)

    # add edges
    ia_edges = None

    if nodes > 1:
        ids = np.arange(nodes, dtype=np.uint)
        ia_edges = _fixed_degree(
            ids, ids, degree, degree_type, reciprocity=reciprocity,
            directed=directed, multigraph=multigraph)
        # check for None if MPI
        if ia_edges is not None:
            graph_fd.new_edges(ia_edges, check_duplicates=False,
                               check_self_loops=False, check_existing=False)

    graph_fd._graph_type = "fixed_{}_degree".format(degree_type)

    return graph_fd


@mpi_random
def gaussian_degree(avg, std, degree_type='in', nodes=0, reciprocity=-1.,
                    weighted=True, directed=True, multigraph=False, name="GD",
                    shape=None, positions=None, population=None,
                    from_graph=None, **kwargs):
    """
    Generate a random graph with constant in- or out-degree.

    Parameters
    ----------
    avg : float
        The value of the average degree.
    std : float
		The standard deviation of the Gaussian distribution.
    degree_type : str, optional (default: 'in')
        The type of the fixed degree, among 'in', 'out' or 'total' (or the
        full version: 'in-degree'...)
        @todo: Implement 'total' degree
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    reciprocity : double, optional (default: -1 to let it free)
        @todo: not implemented yet. Fraction of edges that are bidirectional
        (only for directed graphs -- undirected graphs have a reciprocity of
        1 by definition)
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        @todo: only for directed graphs for now. Whether the graph is directed
        or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment.
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_gd : :class:`~nngt.Graph`, or subclass
        A new generated graph or the modified `from_graph`.

    Note
    ----
	`nodes` is required unless `from_graph` or `population` is provided.
	If an `from_graph` is provided, all preexistant edges in the object
	will be deleted before the new connectivity is implemented.
    """
    # set node number and library graph
    graph_gd = from_graph

    if graph_gd is not None:
        nodes = graph_gd.node_nb()
        graph_gd.clear_all_edges()
    else:
        nodes    = population.size if population is not None else nodes
        graph_gd = nngt.Graph(
            name=name, nodes=nodes, directed=directed, **kwargs)

    _set_options(graph_gd, population, shape, positions)

    # add edges
    ia_edges = None
    if nodes > 1:
        ids = np.arange(nodes, dtype=np.uint)
        ia_edges = _gaussian_degree(
            ids, ids, avg, std, degree_type, reciprocity=reciprocity,
            directed=directed, multigraph=multigraph)
        # check for None if MPI
        if ia_edges is not None:
            graph_gd.new_edges(ia_edges, check_duplicates=False,
                               check_self_loops=False, check_existing=False)

    graph_gd._graph_type = "gaussian_{}_degree".format(degree_type)

    return graph_gd


# ----------- #
# Erdos-Renyi #
# ----------- #

def erdos_renyi(density=None, nodes=0, edges=None, avg_deg=None,
                reciprocity=-1., weighted=True, directed=True,
                multigraph=False, name="ER", shape=None, positions=None,
                population=None, from_graph=None, **kwargs):
    """
    Generate a random graph as defined by Erdos and Renyi but with a
    reciprocity that can be chosen.

    Parameters
    ----------
    density : double, optional (default: -1.)
        Structural density given by `edges / nodes`:math:`^2`. It is also the
        probability for each possible edge in the graph to exist.
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    edges : int (optional)
        The number of edges between the nodes
    avg_deg : double, optional
        Average degree of the neurons given by `edges / nodes`.
    reciprocity : double, optional (default: -1 to let it free)
        Fraction of edges that are bidirectional (only for
        directed graphs -- undirected graphs have a reciprocity of 1 by
        definition)
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment.
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_er : :class:`~nngt.Graph`, or subclass
        A new generated graph or the modified `from_graph`.

    Note
    ----
	`nodes` is required unless `from_graph` or `population` is provided.
	If an `from_graph` is provided, all preexistant edges in the
	object will be deleted before the new connectivity is implemented.
    """
    # set node number and library graph
    graph_er = from_graph

    if graph_er is not None:
        nodes = graph_er.node_nb()
        graph_er.clear_all_edges()
    else:
        nodes = population.size if population is not None else nodes
        graph_er = nngt.Graph(
            name=name, nodes=nodes, directed=directed, **kwargs)

    _set_options(graph_er, population, shape, positions)

    # add edges
    ia_edges = None

    if nodes > 1:
        ids = range(nodes)
        ia_edges = _erdos_renyi(ids, ids, density, edges, avg_deg, reciprocity,
                                directed, multigraph)
        graph_er.new_edges(ia_edges, check_duplicates=False,
                           check_self_loops=False, check_existing=False)

    graph_er._graph_type = "erdos_renyi"

    return graph_er


# ----------------- #
# Scale-free models #
# ----------------- #

def random_scale_free(in_exp, out_exp, nodes=0, density=None, edges=None,
                      avg_deg=None, reciprocity=0., weighted=True,
                      directed=True, multigraph=False, name="RandomSF",
                      shape=None, positions=None, population=None,
                      from_graph=None, **kwargs):
    """
    Generate a free-scale graph of given reciprocity and otherwise
    devoid of correlations.

    Parameters
    ----------
    in_exp : float
        Absolute value of the in-degree exponent :math:`\gamma_i`, such that
        :math:`p(k_i) \propto k_i^{-\gamma_i}`
    out_exp : float
        Absolute value of the out-degree exponent :math:`\gamma_o`, such that
        :math:`p(k_o) \propto k_o^{-\gamma_o}`
    nodes : int, optional (default: 0)
        The number of nodes in the graph.
    density: double, optional
        Structural density given by `edges / (nodes*nodes)`.
    edges : int optional
        The number of edges between the nodes
    avg_deg : double, optional
        Average degree of the neurons given by `edges / nodes`.
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes. can contain multiple edges between two
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment.
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`)
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_fs : :class:`~nngt.Graph`

    Note
    ----
	As reciprocity increases, requested values of `in_exp` and `out_exp`
	will be less and less respected as the distribution will converge to a
	common exponent :math:`\gamma = (\gamma_i + \gamma_o) / 2`.
	Parameter `nodes` is required unless `from_graph` or `population` is
	provided.
    """
    # set node number and library graph
    graph_rsf = from_graph
    if graph_rsf is not None:
        nodes = graph_rsf.node_nb()
        graph_rsf.clear_all_edges()
    else:
        nodes = population.size if population is not None else nodes
        graph_rsf = nngt.Graph(
            name=name,nodes=nodes,directed=directed,**kwargs)

    _set_options(graph_rsf, population, shape, positions)

    # add edges
    if nodes > 1:
        ids = range(nodes)
        ia_edges = _random_scale_free(ids, ids, in_exp, out_exp, density,
                          edges, avg_deg, reciprocity, directed, multigraph)
        graph_rsf.new_edges(ia_edges, check_duplicates=False,
                            check_self_loops=False, check_existing=False)

    graph_rsf._graph_type = "random_scale_free"

    return graph_rsf


def price_scale_free(m, c=None, gamma=1, nodes=0, weighted=True, directed=True,
                     seed_graph=None, multigraph=False, name="PriceSF",
                     shape=None, positions=None, population=None,
                     from_graph=None, **kwargs):
    """
    @todo
    make the algorithm.

    Generate a Price graph model (Barabasi-Albert if undirected).

    Parameters
    ----------
    m : int
        The number of edges each new node will make.
    c : double
        Constant added to the probability of a vertex receiving an edge.
    gamma : double
        Preferential attachment power.
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).
    from_graph : :class:`~nngt.Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_price : :class:`~nngt.Graph` or subclass.

    Note
    ----
	`nodes` is required unless `from_graph` or `population` is provided.
    """
    nodes = ( ( population.size if population is not None else nodes )
              if from_graph is None else from_graph.node_nb() )
    #~ c = c if c is not None else 0 if directed else 1

    g = price_network(nodes, m, c, gamma, directed, seed_graph)
    graph_obj_price = nngt.Graph.from_library(g)

    graph_price = nngt.Graph.from_library(g)

    _set_options(graph_price, population, shape, positions)
    graph_price._graph_type = "price_scale_free"
    return graph_price


# -------------- #
# Circular graph #
# -------------- #

def circular(coord_nb, reciprocity=1., reciprocity_choice="random", nodes=0,
             weighted=True, directed=True, multigraph=False, name="Circular",
             shape=None, positions=None, population=None, from_graph=None,
             **kwargs):
    '''
    Generate a circular graph.

    The nodes are placed on a circle and connected to their `coord_nb` closest
    neighbours.
    If the graph is directed, the number of connections depends on the value
    of `reciprocity`: if ``reciprocity == 0.``, then only half of all possible
    connections will be created, so that no bidirectional edges exist; on the
    other hand, for ``reciprocity == 1.``, all possible edges are created; for
    intermediate values of `reciprocity`, the number of edges increases
    linearly as ``0.5*(1 + reciprocity)*nodes*coord_nb``.

    Parameters
    ----------
    coord_nb : int
        The number of neighbours for each node on the initial topological
        lattice (must be even).
    reciprocity : double, optional (default: 1.)
        Proportion of reciprocal edges in the graph.
    reciprocity_choice : str, optional (default: "random")
        How reciprocal edges should be chosen, which can be either "random" or
        "closest". If the latter option is used, then connections
        between first neighbours are rendered reciprocal first, then between
        second neighbours, etc.
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    density: double, optional (default: 0.1)
        Structural density given by `edges` / (`nodes`*`nodes`).
    edges : int (optional)
        The number of edges between the nodes
    avg_deg : double, optional
        Average degree of the neurons given by `edges` / `nodes`.
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_circ : :class:`~nngt.Graph` or subclass
    '''
    if multigraph:
        raise ValueError("`multigraph` is not supported for circular graphs.")

    # set node number and library graph
    graph_circ = from_graph

    if graph_circ is not None:
        nodes = graph_circ.node_nb()
    else:
        nodes = population.size if population is not None else nodes
        graph_circ = nngt.Graph(
            name=name, nodes=nodes, directed=directed, **kwargs)

    _set_options(graph_circ, population, shape, positions)

    # add edges
    if nodes > 1:
        ids   = range(nodes)
        edges = _circular(ids, ids, coord_nb, reciprocity, directed,
                          reciprocity_choice=reciprocity_choice)

        graph_circ.new_edges(edges, check_duplicates=False,
                             check_self_loops=False, check_existing=False)

    graph_circ._graph_type = "circular"

    return graph_circ
    

# ------------------ #
# Small-world models #
# ------------------ #

def newman_watts(coord_nb, proba_shortcut=None, reciprocity_circular=1.,
                 reciprocity_choice_circular="random", nodes=0, edges=None,
                 weighted=True, directed=True, multigraph=False, name="NW",
                 shape=None, positions=None, population=None, from_graph=None,
                 **kwargs):
    """
    Generate a (potentially small-world) graph using the Newman-Watts
    algorithm.

    For directed networks, the reciprocity of the initial circular network can
    be chosen.

    .. versionchanged:: 2.0
        Added the `reciprocity_circular` and `reciprocity_choice_circular`
        options.

    Parameters
    ----------
    coord_nb : int
        The number of neighbours for each node on the initial topological
        lattice (must be even).
    proba_shortcut : double, optional
        Probability of adding a new random (shortcut) edge for each existing
        edge on the initial lattice.
        If `edges` is provided, then will be computed automatically as
        ``edges / (coord_nb * nodes * (1 + reciprocity_circular) / 2)``
    reciprocity_circular : double, optional (default: 1.)
        Proportion of reciprocal edges in the initial circular graph.
    reciprocity_choice_circular : str, optional (default: "random")
        How reciprocal edges should be chosen in the initial circular graph.
        This can be either "random" or "closest". If the latter option
        is used, then connections between first neighbours are rendered
        reciprocal first, then between second neighbours, etc.
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    edges : int (optional)
        The number of edges between the nodes.
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_nw : :class:`~nngt.Graph` or subclass

    Note
    ----
	`nodes` is required unless `from_graph` or `population` is provided.
    """
    if multigraph:
        raise ValueError("`multigraph` is not supported for Watts-Strogatz.")

    # set node number and library graph
    graph_nw = from_graph

    if graph_nw is not None:
        nodes = graph_nw.node_nb()
    else:
        nodes = population.size if population is not None else nodes
        graph_nw = nngt.Graph(
            name=name, nodes=nodes, directed=directed, **kwargs)

    _set_options(graph_nw, population, shape, positions)

    # add edges
    if nodes > 1:
        ids = range(nodes)

        ia_edges = _newman_watts(
            ids, ids, coord_nb, proba_shortcut, reciprocity_circular,
            edges=edges, directed=directed)

        graph_nw.new_edges(ia_edges, check_duplicates=False,
                           check_self_loops=False, check_existing=False)

    graph_nw._graph_type = "watts_strogatz"

    return graph_nw


def watts_strogatz(coord_nb, proba_shortcut=None, reciprocity_circular=1.,
                   reciprocity_choice_circular="random", shuffle="random",
                   nodes=0, weighted=True, directed=True, multigraph=False,
                   name="WS", shape=None, positions=None, population=None,
                   from_graph=None,  **kwargs):
    """
    Generate a (potentially small-world) graph using the Watts-Strogatz
    algorithm.

    For directed networks, the reciprocity of the initial circular network can
    be chosen.

    .. versionadded:: 2.0

    Parameters
    ----------
    coord_nb : int
        The number of neighbours for each node on the initial topological
        lattice (must be even).
    proba_shortcut : double, optional
        Probability of adding a new random (shortcut) edge for each existing
        edge on the initial lattice.
        If `edges` is provided, then will be computed automatically as
        ``edges / (coord_nb * nodes * (1 + reciprocity_circular) / 2)``
    reciprocity_circular : double, optional (default: 1.)
        Proportion of reciprocal edges in the initial circular graph.
    reciprocity_choice_circular : str, optional (default: "random")
        How reciprocal edges should be chosen in the initial circular graph.
        This can be either "random" or "closest". If the latter option
        is used, then connections between first neighbours are rendered
        reciprocal first, then between second neighbours, etc.
    shuffle : str, optional (default: 'random')
        Whether to shuffle only 'targets' (out-degree of all nodes remains
        constant), 'sources' (in-degree remains constant), or randomly the
        source or the target for each edge ('random') in the case of directed
        graphs.
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_nw : :class:`~nngt.Graph` or subclass

    Note
    ----
	`nodes` is required unless `from_graph` or `population` is provided.
    """
    if multigraph:
        raise ValueError("`multigraph` is not supported for Newman-Watts.")

    # set node number and library graph
    graph_nw = from_graph
    if graph_nw is not None:
        nodes = graph_nw.node_nb()
    else:
        nodes = population.size if population is not None else nodes
        graph_nw = nngt.Graph(
            name=name, nodes=nodes, directed=directed, **kwargs)

    _set_options(graph_nw, population, shape, positions)

    # add edges
    if nodes > 1:
        ids = range(nodes)

        ia_edges = _watts_strogatz(
            ids, ids, coord_nb, proba_shortcut, reciprocity_circular,
            shuffle, directed=directed)

        graph_nw.new_edges(ia_edges, check_duplicates=False,
                           check_self_loops=False, check_existing=False)

    graph_nw._graph_type = "newman_watts"

    return graph_nw


# --------------------- #
# Distance-based models #
# --------------------- #

@mpi_random
def distance_rule(scale, rule="exp", shape=None, neuron_density=1000.,
                  max_proba=-1., nodes=0, density=None, edges=None,
                  avg_deg=None, unit='um', weighted=True, directed=True,
                  multigraph=False, name="DR", positions=None, population=None,
                  from_graph=None, **kwargs):
    """
    Create a graph using a 2D distance rule to create the connection between
    neurons. Available rules are linear and exponential.

    Parameters
    ----------
    scale : float
        Characteristic scale for the distance rule. E.g for linear distance-
        rule, :math:`P(i,j) \propto (1-d_{ij}/scale))`, whereas for the
        exponential distance-rule, :math:`P(i,j) \propto e^{-d_{ij}/scale}`.
    rule : string, optional (default: 'exp')
        Rule that will be apply to draw the connections between neurons.
        Choose among "exp" (exponential), "gaussian" (Gaussian), or
        "lin" (linear).
    shape : :class:`~nngt.geometry.Shape`, optional (default: None)
        Shape of the neurons' environment. If not specified, a square will be
        created with the appropriate dimensions for the number of neurons and
        the neuron spatial density.
    neuron_density : float, optional (default: 1000.)
        Density of neurons in space (:math:`neurons \cdot mm^{-2}`).
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    p : float, optional
        Normalization factor for the distance rule; it is equal to the
        probability of connection when testing a node at zero distance.
    density: double, optional
        Structural density given by `edges` / (`nodes` * `nodes`).
    edges : int, optional
        The number of edges between the nodes
    avg_deg : double, optional
        Average degree of the neurons given by `edges` / `nodes`.
    unit : string (default: 'um')
        Unit for the length `scale` among 'um' (:math:`\mu m`), 'mm', 'cm',
        'dm', 'm'.
    weighted : bool, optional (default: True)
        Whether the graph edges have weights.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "DR")
        Name of the created graph.
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D (N, 2) or 3D (N, 3) shaped array containing the positions of the
        neurons in space.
    population : :class:`~nngt.NeuralPop`, optional (default: None)
        Population of neurons defining their biological properties (to create a
        :class:`~nngt.Network`).
    from_graph : :class:`Graph` or subclass, optional (default: None)
        Initial graph whose nodes are to be connected.
    """
    distance = []
    # convert neuronal density in (mu m)^2
    neuron_density *= conversion_magnitude(unit, 'mm')**2
    # set node number and library graph
    graph_dr = from_graph
    if graph_dr is not None:
        nodes = graph_dr.node_nb()
        graph_dr.clear_all_edges()
    else:
        nodes = population.size if population is not None else nodes
    # check shape
    if shape is None:
        h = w = np.sqrt(float(nodes) / neuron_density)
        shape = nngt.geometry.Shape.rectangle(h, w)
    if graph_dr is None:
        graph_dr = nngt.SpatialGraph(
            name=name, nodes=nodes, directed=directed, shape=shape,
            positions=positions, **kwargs)
    else:
        Graph.make_spatial(graph_dr, shape, positions=positions)
    positions = np.array(graph_dr.get_positions().T, dtype=np.float32)
    # set options (graph has already been made spatial)
    _set_options(graph_dr, population, None, None)
    # add edges
    ia_edges = None
    conversion_factor = conversion_magnitude(shape.unit, unit)
    if unit != shape.unit:
        positions = np.multiply(conversion_factor, positions, dtype=np.float32)
    if nodes > 1:
        ids = np.arange(0, nodes, dtype=np.uint)
        ia_edges = _distance_rule(
            ids, ids, density, edges, avg_deg, scale, rule, max_proba, shape,
            positions, directed, multigraph, distance=distance, **kwargs)
        attr = {'distance': distance}
        # check for None if MPI
        if ia_edges is not None:
            graph_dr.new_edges(ia_edges, attributes=attr,
                               check_duplicates=False, check_self_loops=False,
                               check_existing=False)

    graph_dr._graph_type = "{}_distance_rule".format(rule)
    return graph_dr


# -------------------- #
# Polyvalent generator #
# -------------------- #

_di_generator = {
    "all_to_all": all_to_all,
    "circular": circular,
    "distance_rule": distance_rule,
    "erdos_renyi": erdos_renyi,
    "fixed_degree": fixed_degree,
    "from_degree_list": from_degree_list,
    "gaussian_degree": gaussian_degree,
    "newman_watts": newman_watts,
    "price_scale_free": price_scale_free,
    "random_scale_free": random_scale_free,
    "watts_strogatz": watts_strogatz,
}


def generate(di_instructions, **kwargs):
    '''
    Generate a :class:`~nngt.Graph` or one of its subclasses from a ``dict``
    containing all the relevant informations.

    Parameters
    ----------
    di_instructions : ``dict``
        Dictionary containing the instructions to generate the graph. It must
        have at least ``"graph_type"`` in its keys, with a value among
        ``"distance_rule", "erdos_renyi", "fixed_degree", "newman_watts",
        "price_scale_free", "random_scale_free"``. Depending on the type,
        `di_instructions` should also contain at least all non-optional
        arguments of the generator function.

    See also
    --------
    :mod:`~nngt.generation`
    '''
    graph_type = di_instructions["graph_type"]
    instructions = deepcopy(di_instructions)
    instructions.update(kwargs)
    return _di_generator[graph_type](**instructions)
