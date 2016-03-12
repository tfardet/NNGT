============
Introduction
============

Yet another graph library?
==========================

It is not ;)

This library is based on existing graph libraries (such as `graph_tool <https://graph-tool.skewed.de>`_, and possibly soon `SNAP <http://snap.stanford.edu/snap/>`_) and acts as a convenient interface to build various networks from efficient and verified algorithms.

Moreover, it also acts as an interface between those graph libraries and the NEST simulator.


Description
===========

Neural networks are described by four graph classes which inherit from the main class of the chosen graph library (:class:`graph_tool.Graph`, :class:`igraph.Graph` or :class:`networkx.DiGraph`):
	- :class:`~nngt.Graph`: base for simple topological graphs with no spatial structure, nor biological properties
	- :class:`~nngt.SpatialGraph`: subclass for spatial graphs without biological properties
	- :class:`~nngt.Network`: subclass for topological graphs with biological properties (to interact with NEST)
	- :class:`~nngt.SpatialNetwork`: subclass with spatial and biological properties (to interact with NEST)

Using these objects, the user can access to the topological structure of the network (including the connections' type -- inhibitory or excitatory -- and its weight which is always positive)

.. warning ::
	This object should never be directly modified through its methods but rather using those of the previously listed classes. If for some reason you should directly use the methods from the graph library on the graph, make sure they do not modify its structure; any modification performed from a method other than those of the :class:`~nngt.Graph` subclasses will lead to undescribed behaviour.

Nodes/neurons are defined by a unique index which can be used to access their properties and those of the connections between them.

In addition to ``graph``, the containers can have other attributes, such as:
	- ``shape`` for :class:`~nngt.SpatialGraph`: and :class:`~nngt.SpatialNetwork`:, which describes the spatial delimitations of the neurons' environment (e.g. many *in vitro* culture are contained in circular dishes).
	- ``population`` which contains informations on the various groups of neurons that exist in the network (for instance inhibitory and excitatory neurons can be grouped together)
	- ``connections`` which stores the informations about the synaptic connections between the neurons


Graph-theoretical models
------------------------

Several classical graphs are efficiently implemented and the generation procedures are detailed in the documentation.

.. toctree::
   :maxdepth: 2
   
   ../modules/nngt
