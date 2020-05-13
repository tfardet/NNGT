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


def lattice_rewire(g, target_reciprocity=1., weight=None,
                   weight_constraint=None, node_attr_constraints=None,
                   edge_attr_constraints=None, distance_sort="inverse"):
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
    weight : str, optional (default: None)
        Whether a specific edge attribute should play the role of weight and
        have special constraints.
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
        depending on the value of `distance_sort`.
        If "together" is used, edges attributes are randomized by groups (all
        attributes of a given edge are sent to the same new edge) either
        randomly if `weight` is None, or following the constrained `weight`
        attribute. By default, attributes are completely and separately
        randomized (except for `weight` if it has been provided).
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

    ia_edges[:e_reglat] = gc._circular(ids, ids, coord_nb, target_reciprocity,
                                       directed=directed)

    # add the remaining edges (remaining edges strictly smaller than num_nodes)
    e_remaining = num_edges - e_reglat

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
        assert constraints in ("in-degree", "all-degrees"), \
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
    nattr = deepcopy(g.get_node_attributes())

    order = [i for i in range(num_nodes)]
    rng.shuffle(order)  # shuffled order for "together"

    if node_attr_constraints not in (None, "preserve", "together"):
        raise ValueError("`node_attr_constraints` must be either None, "
                         "'preserve', or 'together'.")
    else:
        for k, v in nattr.items():
            if node_attr_constraints is None:
                rng.shuffle(v)
            elif node_attr_constraints == "together":
                v = v[order]

            dtype = g.get_attribute_type(k, attribute_class="node")

            new_graph.new_node_attribute(k, dtype, values=v)

    # edge attributes
    eattr = deepcopy(g.get_edge_attributes())

    order = np.arange(num_edges, dtype=int)

    if edge_attr_constraints == "together":
        rng.shuffle(order)
    elif edge_attr_constraints == "preserve_in":
        for i in range(num_nodes):
            old_edges = g.get_edges(source=i)
            new_edges = new_graph.edge_edges(source=i)

            old_ids = g.edge_id(old_edges)
            new_ids = new_graph.edge_id(new_edges)

            order[new_ids] = old_ids
    elif edge_attr_constraints == "preserve_out":
        for i in range(num_nodes):
            old_edges = g.get_edges(target=i)
            new_edges = new_graph.edge_edges(target=i)

            old_ids = g.edge_id(old_edges)
            new_ids = new_graph.edge_id(new_edges)

            order[new_ids] = old_ids

    for k, v in eattr.items():
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
