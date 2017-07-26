==================
Detailed structure
==================

.. warning ::
    This section is not up to date anymore!

Here is a small bottom-up approach of the library to justify its structure.


Rationale for the structure
===========================

The basis: a graph
------------------

The core object is :class:`nngt.core.GraphObject` that inherits from either :class:`gt.Graph` or :class:`snap.TNEANet` and :class:`Shape` that encodes the spatial structure of the neurons' environment.
The purpose of :class:`GraphObject` is simple: implementing a library independant object with a unique set of functions to interact with graphs.

.. warning ::
	This object should never be directly modified through its methods but rather using those of the four containing classes. The only reason to access this object should be to perform graph-theoretical measurements on it which do not modify its structure; any other action will lead to undescribed behaviour.

Frontend
--------

Detailed neural networks contain properties that the :class:`~nngt.core.GraphObject` does not know about; because of this, direct modification of the structure can lead to nodes or edges missing properties or to properties assigned to nonexistent nodes or edges.

The user can safely interact with the graph using one of the following classes:

- :class:`~nngt.Graph`: container for simple topological graphs with no spatial embedding, nor biological properties
- :class:`~nngt.SpatialGraph`: container for spatial graphs without biological properties
- :class:`~nngt.Network`: container for topological graphs with biological properties (to interact with NEST)
- :class:`~nngt.SpatialNetwork`: container with spatial and biological properties (to interact with NEST)

The reason behind those four objects is to ensure coherence in the properties: either nodes/edges all have a given property or they all don't.
Namely:

- adding a node will always require a position parameter when working with a spatial graph,
- adding a node or a connection will always require biological parameters when working with a network.

Moreover, these classes contain the :class:`GraphObject` in their ``graph`` attribute and do not inherit from it. The reason for this is to make it easy to maintain different addition/deletion functions for the topological and spatial container by keeping independant of the graph library. (otherwise overwriting one of these function would require the use of library-dependant features).
