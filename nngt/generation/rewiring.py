#-*- coding:utf-8 -*-
#
# grewiring.py
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

""" Rewiring functions """

from copy import deepcopy

import numpy as np

import nngt
from nngt.generation import graph_connectivity as gc
from nngt.lib import nonstring_container


__all__ = [
    "lattice_rewire",
    "random_rewire"
]


def lattice_rewire(g, target_reciprocity=1., node_attr_constraints=None,
                   edge_attr_constraints=None,  weight=None,
                   weight_constraint=None, distance_sort="inverse"):
    '''
    Build a (generally irregular) lattice by rewiring the edges of a graph.

    The lattice is based on a circular graph, meaning that the nodes are placed
    on a circle and connected based on the topological distance between them,
    the distance being defined through the positive modulo:

    .. math::

        d_{ij} = (i - j) % N

    with :math:`N` the number of nodes in the graph.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph based on which the lattice will be generated.
    target_reciprocity : float, optional (default: 1.)
        Value of reciprocity that should be aimed at. Depending on the number
        of edges, it may not be possible to reach this value exactly.
    node_attr_constraints : str, optional (default: randomize all attributes)
        Whether attribute randomization is constrained: either "preserve",
        where all nodes keep their attributes, or "together", where attributes
        are randomized by groups (all attributes of a given node are sent to
        the same new node). By default, attributes are completely and
        separately randomized.
    edge_attr_constraints : str, optional (default: randomize all but `weight`)
        Whether attribute randomization is constrained.
        If "distance" is used, then all number attributes (float or int) are
        sorted and are first associated to the shortest or longest edges
        depending on the value of `distance_sort`. Note that, for directed
        graphs, if a reciprocal edge exists, it is immediately assigned the
        next highest (respectively lowest) attribute after that of its directed
        couterpart.
        If "together" is used, edges attributes are randomized by groups (all
        attributes of a given edge are sent to the same new edge) either
        randomly if `weight` is None, or following the constrained `weight`
        attribute. By default, attributes are completely and separately
        randomized (except for `weight` if it has been provided).
    weight : str, optional (default: None)
        Whether a specific edge attribute should play the role of weight and
        have special constraints.
    weight_constraint : str, optional (default: "distance")
        Same as `edge_attr_constraints`` but only applies to `weight` and can
        only be "distance" or None since "together" was related to `weight`.
    distance_sort : str, optional (default: "inverse")
        How attributes are sorted with edge distance: either "inverse", with
        the shortest edges being assigned the largest weights, or with a
        "linear" sort, where shortest edges are assigned the lowest weights.
    '''
    directed  = g.is_directed()
    num_nodes = g.node_nb()
    num_edges = g.edge_nb()

    if directed and target_reciprocity != 1:
        raise ValueError("Reciprocity is always 1 for undirected graphs.")

    new_graph = nngt.Graph(nodes=num_nodes, directed=directed,
                           name=g.name + "_latticized")

    ia_edges = np.full((num_edges, 2), -1, dtype=np.int64)

    # compute the coodination number of the closest regular lattice
    coord_nb, e_reglat = None, None

    if directed:
        coord_nb = int(2*num_edges / (num_nodes*(1 + target_reciprocity)))
        e_reglat = int(0.5*num_nodes*(1 + target_reciprocity)*coord_nb)
    else:
        coord_nb = int(num_edges / num_nodes)
        e_reglat = num_nodes*coord_nb

    # generate the edges of the regular lattice
    ids = range(num_nodes)

    ia_edges[:e_reglat] = gc._circular(
        ids, ids, coord_nb, target_reciprocity, directed=directed,
        reciprocity_choice="closest-ordered")

    # add the remaining edges (remaining edges strictly smaller than num_nodes)
    e_remaining = num_edges - e_reglat

    if e_remaining:
        last_edges = np.full((e_remaining, 2), -1, dtype=np.int64)

        if directed:
            # make reciprocal edges
            num_recip  = int(0.5*target_reciprocity*e_remaining)

            last_edges[:num_recip] = [(i, i + dist) for i in range(num_recip)]
            last_edges[num_recip:2*num_recip] = last_edges[:num_recip, ::-1]

            # make remaning non-reciprocal edges
            e_final = e_remaining - 2*num_recip

            if e_final:
                last_edges[2*num_recip:] = \
                    [(i, i + dist) for i in range(num_recip, num_recip + e_final)]
        else:
            dist = int(0.5*coord_nb) + 1

            last_edges = [(i, i + dist) for i in range(e_remaning)]

            # put targets back into [0, num_nodes[
            last_edges[:, last_edges[:, 1] >= num_nodes] -= num_nodes

        ia_edges[e_reglat:] = last_edges

    # add the edges
    new_graph.new_edges(ia_edges, check_edges=False)

    # set the node attributes
    _set_node_attributes(g, new_graph, node_attr_constraints, num_nodes)

    # edge attributes @todo

    return new_graph


def random_rewire(g, constraints=None, node_attr_constraints=None,
                  edge_attr_constraints=None):
    '''
    Generate a new rewired graph from `g`.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Base graph based on which a new rewired graph will be generated.
    constraints : str, optional (default: no constraints)
        Defines which properties of `g` will be maintained in the rewired
        graph. By default, the graph is completely rewired into an Erdos-Renyi
        model. Available constraints are "in-degree", "out-degree",
        "total-degree", "all-degrees", and "clustering".
    node_attr_constraints : str, optional (default: randomize all attributes)
        Whether attribute randomization is constrained: either "preserve",
        where all nodes keep their attributes, or "together", where attributes
        are randomized by groups (all attributes of a given node are sent to
        the same new node). By default, attributes are completely and
        separately randomized.
    edge_attr_constraints : str, optional (default: randomize all attributes)
        Whether attribute randomization is constrained.
        If `constraints` is "in-degree" (respectively "out-degree") or
        "degrees", this can be "preserve_in" (respectively "preserve_out"),
        in which case all attributes of a given edge are moved together to a
        new incoming (respectively outgoing) edge of the same node.
        Regardless of `constraints`, "together" can be used so that edges
        attributes are randomized by groups (all attributes of a given edge are
        sent to the same new edge). By default, attributes are completely and
        separately randomized.
    '''
    directed  = g.is_directed()
    num_nodes = g.node_nb()
    num_edges = g.edge_nb()

    new_graph = None

    # check compatibility between `constraints` and `edge_attr_constraints`
    valid_e = (None, "preserve_in", "preserve_out", "together")

    if edge_attr_constraints not in valid_e:
        raise ValueError(
            "`edge_attr_constraints` must be in {}.".format(valid_e))
    elif edge_attr_constraints == "preserve_in":
        assert constraints in ("in-degree", "all-degrees"), \
            "Can only use 'preserve_in' if `constraints` is 'in-degree' or " \
            "'all-degrees'."
    elif edge_attr_constraints == "preserve_out":
        assert constraints in ("out-degree", "all-degrees"), \
            "Can only use 'preserve_out' if `constraints` is 'out-degree' " \
            "or 'all-degrees'."

    # generate rewired graph
    if constraints is None:
        new_graph = gc.erdos_renyi(edges=num_edges, nodes=num_nodes,
                                   directed=directed)
    elif constraints == "all-degrees":
        raise NotImplementedError("Full degrees constraints is not yet "
                                  "implemented.")
    elif "degree" in constraints:
        degrees   = g.get_degrees(constraints)
        new_graph = gc.from_degree_list(degrees, constraints,
                                        directed=directed)
    elif constraints == "clustering":
        raise NotImplementedError("Rewiring with constrained clustering is "
                                  "not yet available.")

    rng = nngt._rng

    # node attributes
    _set_node_attributes(g, new_graph, node_attr_constraints, num_nodes)

    # edge attributes
    order = np.arange(num_edges, dtype=int)

    if edge_attr_constraints == "together":
        rng.shuffle(order)
    elif edge_attr_constraints == "preserve_in":
        for i in range(num_nodes):
            old_edges = g.get_edges(target_node=i)
            new_edges = new_graph.get_edges(target_node=i)

            if len(new_edges):
                old_ids = g.edge_id(old_edges)
                new_ids = new_graph.edge_id(new_edges)

                order[new_ids] = old_ids
    elif edge_attr_constraints == "preserve_out":
        for i in range(num_nodes):
            old_edges = g.get_edges(source_node=i)
            new_edges = new_graph.get_edges(source_node=i)

            if len(new_edges):
                old_ids = g.edge_id(old_edges)
                new_ids = new_graph.edge_id(new_edges)

                order[new_ids] = old_ids

    for k in g.edge_attributes:
        v = deepcopy(g.get_edge_attributes(name=k))

        if edge_attr_constraints is None:
            rng.shuffle(v)
        else:
            v = v[order]

        dtype = g.get_attribute_type(k, attribute_class="edge")

        new_graph.new_edge_attribute(k, dtype, values=v)

    # set spatial/network properties
    if g.is_spatial():
        nngt.Graph.make_spatial(new_graph, shape=g.shape.copy(),
                                positions=g.get_positions().copy())
    if g.is_network():
        nngt.Graph.make_network(new_graph, neural_pop=g.population.copy())

    new_graph._name = g.name + "_rewired"

    return new_graph


# ----- #
# Tools #
# ----- #

def _set_node_attributes(old_graph, new_graph, constraints, num_nodes):
    ''' Reassign node attributes '''
    order = None

    if constraints == "together":
        order = [i for i in range(num_nodes)]
        nngt._rng.shuffle(order)  # shuffled order for "together"

    if constraints not in (None, "preserve", "together"):
        raise ValueError("`node_attr_constraints` must be either None, "
                         "'preserve', or 'together'.")
    else:
        for k, v in old_graph.node_attributes.items():
            values = None

            if constraints is None:
                values = deepcopy(v)
                nngt._rng.shuffle(values)
            elif constraints == "together":
                values = v[order]

            dtype = old_graph.get_attribute_type(k, attribute_class="node")

            new_graph.new_node_attribute(k, dtype, values=values)


def _lattice_shuffle_eattr(name, old_graph, new_graph, edges, coord_nb,
                           target_recip, order, distance_sort):
    '''
    Reassign edge attributes based on a constraint or a pre-defined
    order for the lattice rewiring.

    Parameters
    ----------
    name : str
        Name of the edge attribute.
    old_graph : :class:`~nngt.Graph`
        The old graph.
    new_graph : :class:`~nngt.Graph`
        The new graph.
    edges : array of dimension (E, 2)
        Edges of the new graph.
    coord_nb : int
        Coordination number of the lattice.
    target_recip : float
        Target reciprocity of the lattice.
    order : array of indices, distance", or None
        Constraint on edge reassignment: either a precomputed order, "distance"
        if we perform a distance-based shuffle, or None if we randomly
        shuffle the attributes.
    distance_sort : str
        How the attributes should correlate with the distance (linearly or
        inversely) if `order` is "distance".

    Returns
    -------
    order : array of indices
        The order in which the edge attributes have been shuffled.
    '''
    num_nodes = new_graph.node_nb()
    num_edges = len(edges)

    if order is None:
        order = np.arange(num_edges, dtype=int)
        nngt._rng.shuffle(order)
    elif order == "distance":
        directed = new_graph.is_directed()

        if directed:
            # we need to find the reciprocal edges for the attribute
            # assignment (this relies on the precise implementation of the
            # function _circular_directed_recip that the closest distances
            # come first, then the reciprocal edges are at the end in the
            # same order)
            init_edges = int(0.5*num_nodes*(1 + target_recip)*coord_nb)
            num_recip  = init_edges - num_nodes*int(0.5*coord_nb)

            # fill the order list in the following order
            order = np.zeros(num_edges, dtype=int)
            # every other first entries are the initial edges that got a
            # reciprocal connection
            order[:2*num_recip:2] = np.arange(num_recip)
            # we enter the index of the reciprocal connection after each
            order[1:2*num_recip + 1:2] = np.arange(init_edges,
                                                   init_edges + num_recip)
            # then we fill the last entries with the initial edges that did
            # not get a reciprocal connection
            order[2*num_recip + 1:] = np.arange(num_recip, init_edges)

            if distance_sort == "inverse":
                order = order[::-1]
        else:
            # we don't need to sort the new edges because they are ordered by
            # distance by default in the circular algorithm
            if distance_sort == "linear":
                order = slice(num_edges)
            else:
                order = slice(0, num_edges, -1)

    # set the new attributes
    value_type = old_graph.get_attribute_type(name, "edge")

    values = old_graph.edge_attributes[name]

    new_graph.new_edge_attribute(name, value_type, values=values[order])

    return order
