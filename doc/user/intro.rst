.. _intro:

===================
Intro & user manual
===================

Yet another graph library?
==========================

It is not ;)

This library is based on existing graph libraries (such as
`graph-tool`_, igraph_, networkx_, and possibly soon
`SNAP <http://snap.stanford.edu/snap/>`_) and acts as a convenient interface to
build various networks from efficient and verified algorithms.
Most importantly, it provides a series of analysis functions that are
guaranteed to provide the same results with all backends, enabling fully
portable codes (see :ref:`graph-analysis`).

Moreover, it also acts as an interface between those graph libraries and the
NEST_ and DeNSE_ simulators.


Documentation structure
-----------------------

For users that are in a hurry, you can go directly to the Tutorial_ section.
For more specific and detailed examples, several topics are then detailed
separately in the following pages:

.. toctree::
   :maxdepth: 1

   graph-generation
   component-properties
   graph-analysis
   parallelism
   neural-groups
   nest-interaction
   activity-analysis

.. note ::
  This library provides many tools which will (or not) be loaded on startup
  depending on the python packages available on your computer.
  The default behaviour of those tools is set in the `~/.nngt/nngt.conf` file
  (see Configuration_).
  Moreover, to see all potential messages related to the import of those tools,
  you can use the logging function of NNGT, either by setting the `log_level`
  value to `INFO`, or by setting `log_to_file` to True, and having a look
  at the log file in `~/.nngt/log/`.


Description
===========

The graph objects
-----------------

Neural networks are described by four graph classes which inherit from the main
class of the chosen graph library (:class:`gt.Graph`,
:class:`igraph.Graph` or :class:`networkx.DiGraph`):

- :class:`~nngt.Graph`: base for simple topological graphs with no spatial
  structure, nor biological properties
- :class:`~nngt.SpatialGraph`: subclass for spatial graphs without
  biological properties
- :class:`~nngt.Network`: subclass for topological graphs with biological
  properties (to interact with NEST)
- :class:`~nngt.SpatialNetwork`: subclass with spatial and biological
  properties (to interact with NEST)

Using these objects, the user can access to the topological structure of the
network (for neuroscience, this includes the connections' type -- inhibitory or
excitatory -- and its synaptic weight, which is always positive)


Additional properties
---------------------

Nodes/neurons are defined by a unique index which can be used to access their
properties and those of the connections between them.

The graph objects can have other attributes, such as:

- ``shape`` for :class:`~nngt.SpatialGraph` and :class:`~nngt.SpatialNetwork`,
  which describes the spatial delimitations of the neurons' environment (e.g.
  many *in vitro* culture are contained in circular dishes),
- ``population``, for :class:`~nngt.Network`, which contains informations on
  the various groups of neurons that exist in the network (for instance
  inhibitory and excitatory neurons can be grouped together),
- ``connections`` which stores the informations about the synaptic connections
  between the neurons.


Graph-theoretical models
------------------------

Several classical graphs are efficiently implemented and the generation
procedures are detailed in the documentation.

.. toctree::
   :maxdepth: 1

   graph-generation
   ../modules/nngt


Known bugs
----------

* erratic key release issue with animation of spiking networks,
* see `the issue tracker <https://github.com/Silmathoron/NNGT/issues>`_ for an
  updated list.


.. _DeNSE: https://dense.readthedocs.io
.. _`graph-tool`: http://graph-tool.skewed.de
.. _igraph: http://igraph.org/
.. _NEST: http://www.nest-simulator.org/
.. _networkx: https://networkx.github.io/

.. _Configuration: install#configuration
.. _Tutorial: tutorial
