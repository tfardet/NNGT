#-*- coding:utf-8 -*-
#
# core/connections.py
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

""" Graph data strctures in NNGT """

import numpy as np
from numpy.random import randint, uniform

from ..lib import (InvalidArgument, nonstring_container, is_integer,
                   default_neuron, default_synapse, POS, WEIGHT, DELAY, DIST,
                   TYPE, BWEIGHT)
from ..lib.rng_tools import _eprop_distribution


# ----------- #
# Connections #
# ----------- #

class Connections:

    """
    The basic class that computes the properties of the connections between
    neurons for graphs.
    """

    #-------------------------------------------------------------------------#
    # Class methods

    @staticmethod
    def distances(graph, elist=None, pos=None, dlist=None, overwrite=False):
        '''
        Compute the distances between connected nodes in the graph. Try to add
        only the new distances to the graph. If they overlap with previously
        computed distances, recomputes everything.

        Parameters
        ----------
        graph : class:`~nngt.Graph` or subclass
            Graph the nodes belong to.
        elist : class:`numpy.array`, optional (default: None)
            List of the edges.
        pos : class:`numpy.array`, optional (default: None)
            Positions of the nodes; note that if `graph` has a "position"
            attribute, `pos` will not be taken into account.
        dlist : class:`numpy.array`, optional (default: None)
            List of distances (for user-defined distances)

        Returns
        -------
        new_dist : class:`numpy.array`
            Array containing *ONLY* the newly-computed distances.
        '''
        elist = graph.edges_array if elist is None else elist

        if dlist is not None:
            dlist = np.array(dlist)
            graph.set_edge_attribute(DIST, value_type="double", values=dlist)
            return dlist
        else:
            pos = graph._pos if hasattr(graph, "_pos") else pos

            # compute the new distances
            if graph.edge_nb():
                ra_x = pos[elist[:,0], 0] - pos[elist[:,1], 0]
                ra_y = pos[elist[:,0], 1] - pos[elist[:,1], 1]

                ra_dist = np.sqrt( np.square(ra_x) + np.square(ra_y) )

                # update graph distances
                graph.set_edge_attribute(DIST, value_type="double",
                                         values=ra_dist, edges=elist)
                return ra_dist
            else:
                return []

    @staticmethod
    def delays(graph=None, dlist=None, elist=None, distribution="constant",
               parameters=None, noise_scale=None):
        '''
        Compute the delays of the neuronal connections.

        Parameters
        ----------
        graph : class:`~nngt.Graph` or subclass
            Graph the nodes belong to.
        dlist : class:`numpy.array`, optional (default: None)
            List of user-defined delays).
        elist : class:`numpy.array`, optional (default: None)
            List of the edges which value should be updated.
        distribution : class:`string`, optional (default: "constant")
            Type of distribution (choose among "constant", "uniform",
            "lognormal", "gaussian", "user_def", "lin_corr", "log_corr").
        parameters : class:`dict`, optional (default: {})
            Dictionary containing the distribution parameters.
        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.

        Returns
        -------
        new_delays : class:`scipy.sparse.lil_matrix`
            A sparse matrix containing *ONLY* the newly-computed weights.
        '''
        elist = np.array(elist) if elist is not None else elist
        if dlist is not None:
            dlist = np.array(dlist)
            num_edges = graph.edge_nb() if elist is None else elist.shape[0]
            if len(dlist) != num_edges:
                raise InvalidArgument("`dlist` must have one entry per edge.")
        else:
            parameters["btype"] = parameters.get("btype", "edge")
            parameters["weights"] = parameters.get("weights", None)
            dlist = _eprop_distribution(graph, distribution, elist=elist,
                                        **parameters)
        # add to the graph container
        if graph is not None:
            graph.set_edge_attribute(
                DELAY, value_type="double", values=dlist, edges=elist)
        return dlist

    @staticmethod
    def weights(graph=None, elist=None, wlist=None, distribution="constant",
                parameters={}, noise_scale=None):
        '''
        Compute the weights of the graph's edges.

        Parameters
        ----------
        graph : class:`~nngt.Graph` or subclass
            Graph the nodes belong to.
        elist : class:`numpy.array`, optional (default: None)
            List of the edges (for user defined weights).
        wlist : class:`numpy.array`, optional (default: None)
            List of the weights (for user defined weights).
        distribution : class:`string`, optional (default: "constant")
            Type of distribution (choose among "constant", "uniform",
            "lognormal", "gaussian", "user_def", "lin_corr", "log_corr").
        parameters : class:`dict`, optional (default: {})
            Dictionary containing the distribution parameters.
        noise_scale : class:`int`, optional (default: None)
            Scale of the multiplicative Gaussian noise that should be applied
            on the weights.

        Returns
        -------
        new_weights : class:`scipy.sparse.lil_matrix`
            A sparse matrix containing *ONLY* the newly-computed weights.
        '''
        parameters["btype"] = parameters.get("btype", "edge")
        parameters["weights"] = parameters.get("weights", None)

        elist = np.array(elist) if elist is not None else elist

        if wlist is not None:
            wlist = np.array(wlist)
            num_edges = graph.edge_nb() if elist is None else elist.shape[0]
            if len(wlist) != num_edges:
                raise InvalidArgument("`wlist` must have one entry per edge.")
        else:
            wlist = _eprop_distribution(graph, distribution, elist=elist,
                                        **parameters)

        # normalize by the inhibitory weight factor
        if graph is not None and graph.is_network():
            if not np.isclose(graph._iwf, 1.):
                adj = graph.adjacency_matrix(types=True, weights=False)
                keep = (adj[elist[:, 0], elist[:, 1]] < 0).A1
                wlist[keep] *= graph._iwf

        if graph is not None:
            graph.set_edge_attribute(
                WEIGHT, value_type="double", values=wlist, edges=elist)

        return wlist

    @staticmethod
    def types(graph, inhib_nodes=None, inhib_frac=None, values=None):
        '''
        Define the type of a set of neurons.
        If no arguments are given, all edges will be set as excitatory.

        Parameters
        ----------
        graph : :class:`~nngt.Graph` or subclass
            Graph on which edge types will be created.
        inhib_nodes : int or list, optional (default: `None`)
            If `inhib_nodes` is an int, number of inhibitory nodes in the graph
            (all connections from inhibitory nodes are inhibitory); if it is a
            float, ratio of inhibitory nodes in the graph; if it is a list, ids
            of the inhibitory nodes.
        inhib_frac : float, optional (default: `None`)
            Fraction of the selected edges that will be set as refractory (if
            `inhib_nodes` is not `None`, it is the fraction of the nodes' edges
            that will become inhibitory, otherwise it is the fraction of all
            the edges in the graph).

        Returns
        -------
        t_list : :class:`~numpy.ndarray`
            List of the edges' types.
        '''
        num_inhib = 0
        idx_inhib = []

        if values is not None:
            graph.new_edge_attribute("type", "int", values=values)
            return values
        elif inhib_nodes is None and inhib_frac is None:
            graph.new_edge_attribute("type", "int", val=1)
            return np.ones(graph.edge_nb())
        else:
            t_list = np.repeat(1, graph.edge_nb())
            n = graph.node_nb()

            if inhib_nodes is None:
                # set inhib_frac*num_edges random inhibitory connections
                num_edges = graph.edge_nb()
                num_inhib = int(num_edges*inhib_frac)
                num_current = 0
                while num_current < num_inhib:
                    new = randint(0, num_edges, num_inhib-num_current)
                    idx_inhib = np.unique(np.concatenate((idx_inhib, new)))
                    num_current = len(idx_inhib)
                t_list[idx_inhib.astype(int)] *= -1
            else:
                edges  = graph.edges_array
                # get the dict of inhibitory nodes
                num_inhib_nodes = 0
                idx_nodes = {}
                if nonstring_container(inhib_nodes):
                    idx_nodes = {i: -1 for i in inhib_nodes}
                    num_inhib_nodes = len(idx_nodes)
                if is_integer(inhib_nodes):
                    num_inhib_nodes = int(inhib_nodes)
                while len(idx_nodes) != num_inhib_nodes:
                    indices = randint(0,n,num_inhib_nodes-len(idx_nodes))
                    di_tmp  = {i: -1 for i in indices}
                    idx_nodes.update(di_tmp)
                for v in edges[:, 0]:
                    if v in idx_nodes:
                        idx_inhib.append(v)
                idx_inhib = np.unique(idx_inhib)

                # set the inhibitory edge indices
                for v in idx_inhib:
                    idx_edges = np.argwhere(edges[:, 0] == v)

                    n = len(idx_edges)

                    if inhib_frac is not None:
                        idx_inh = []
                        num_inh = n*inhib_frac
                        i = 0
                        while i != num_inh:
                            ids = randint(0, n, num_inh-i)
                            idx_inh = np.unique(np.concatenate((idx_inh,ids)))
                            i = len(idx_inh)
                        t_list[idx_inh] *= -1
                    else:
                        t_list[idx_edges] *= -1

            graph.set_edge_attribute("type", value_type="int", values=t_list)

            return t_list
