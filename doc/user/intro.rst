Introduction
============

Yet another graph library?
--------------------------

It is not ;)

This library is based on existing graph libraries (such as `graph_tool <https://graph-tool.skewed.de>`_, and possibly soon `SNAP <http://snap.stanford.edu/snap/>`_) and acts as a convenient interface to build various networks from efficient and verified algorithms.

Moreover, it also acts as an interface between those graph libraries and the NEST simulator.


Description
-----------

Neural networks are described by four container classes:
	- :class:`~nngt.Graph`: container for simple topological graphs with no spatial structure, nor biological properties
	- :class:`~nngt.SpatialGraph`: container for spatial graphs without biological properties
	- :class:`~nngt.Network`: container for topological graphs with biological properties (to interact with NEST)
	- :class:`~nngt.SpatialNetwork`: container with spatial and biological properties (to interact with NEST)

Using these objects, the user can access to a the ``graph`` attribute, which contains the topological structure of the network (including the connections' type -- inhibitory or excitatory -- and its weight which is always positive)

.. warning ::
	This object should never be directly modified through its methods but rather using those of the four containing classes. The only reason to access this object should be to perform graph-theoretical measurements on it which do not modify its structure; any other action will lead to undescribed behaviour.

Nodes/neurons are defined by a unique index which can be used to access their properties and those of the connections between them.

In addition to ``graph``, the containers can have other attributes, such as:
	- ``shape`` for :class:`~nngt.SpatialGraph`: and :class:`~nngt.SpatialNetwork`:, which describes the spatial delimitations of the neurons' environment (e.g. many *in vitro* culture are contained in circular dishes).
	- ``population`` which contains informations on the various groups of neurons that exist in the network (for instance inhibitory and excitatory neurons can be grouped together)
	- ``connections`` which stores the informations about the synaptic connections between the neurons


Graph-theoretical models
^^^^^^^^^^^^^^^^^^^^^^^^
Several classical graphs are efficiently implemented and the generation procedures are detailed in the documentation.

.. toctree::
   :maxdepth: 2
   
   ../modules/nngt
