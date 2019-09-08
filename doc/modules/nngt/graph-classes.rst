Graph classes
=============

.. autosummary::

    nngt.Graph
    nngt.SpatialGraph
    nngt.Network
    nngt.SpatialNetwork


Details
-------

.. currentmodule:: nngt
.. autoclass:: Graph
   :inherited-members:
   :no-undoc-members:
   :exclude-members: neighbors, nbunch_iter, nattr_class, in_edges, in_degree,
                     has_successor, has_predecessor, has_node, has_edge,
                     get_edge_data, fresh_copy, edges, edge_subgraph,
                     edge_attr_dict_factory, eattr_class, degree, clear,
                     adjlist_outer_dict_factory, adjlist_inner_dict_factory,
                     adjacency, adj, add_weighted_edges_from, add_nodes_from,
                     add_node, add_edges_from, add_edge, node,
                     node_dict_factory, nodes, number_of_edges, number_of_nodes,
                     order, out_degree, out_edges, pred, predecessors,
                     remove_edges_from, remove_node, remove_nodes_from,
                     reverse, size, subgraph, succ, successors, remove_edge,
                     remove_edges_from, to_directed, to_undirected

.. currentmodule:: nngt
.. autoclass:: SpatialGraph

.. currentmodule:: nngt
.. autoclass:: Network

.. currentmodule:: nngt
.. autoclass:: SpatialNetwork
