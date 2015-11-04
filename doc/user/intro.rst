Introduction
============

Yet another graph library?
--------------------------

It is not ;)

This library is based on existing graph libraries (such as `graph_tool <https://graph-tool.skewed.de>`_, and possibly soon `SNAP <http://snap.stanford.edu/snap/>`_) and acts as a convenient interface to build various networks from efficient and verified algorithms.

Moreover, it also acts as an interface between those graph libraries and the NEST simulator.


Rationale for the structure
---------------------------

The main objects are :class:`GraphObject` that inherits from either :class:`graph_tool.Graph` or :class:`snap.TNEANet` and :class:`Shape` that encodes the spatial structure of the neurons' environment.
The purpose of :class:`GraphObject` is simple: implementing a library independant object with a unique set of functions to interact with graphs.
The user should (in general) not interact directly with this class, but rather with one of the four containing classes in the core module:

- :class:`~nngt.core.GraphClass`: container for simple topological graphs with no spatial structure, nor biological properties
- :class:`~nngt.core.SpatialGraph`: container for spatial graphs without biological properties
- :class:`~nngt.core.Network`: container for topological graphs with biological properties (to interact with NEST)
- :class:`~nngt.core.SpatialNetwork`: container with spatial and biological properties (to interact with NEST)

The reason for those four classes is to ensure coherence in the properties: either nodes/edges all have a given property or they all don't.
Namely:
- adding a node will always require a position parameter when working with a spatial graph,
- adding a node or a connection will always require biological parameters when working with a network.

Moreover, these classes contain the :class:`GraphObject` in their `graph` attribute and do not inherit from it. The reason for this is to make it easy to maintain different addition/deletion functions for the topological and spatial container by keeping independant of the graph library. (otherwise overwriting one of these function would require the use of library-dependant features).


Graph-theoretical models
^^^^^^^^^^^^^^^^^^^^^^^^
Several classical graphs are efficiently implemented and the generation procedures are detailed in the documentation.

.. toctree::
   :maxdepth: 2
   
   ../modules/nngt
