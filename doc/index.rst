.. NNGT documentation master file

================================
Welcome to NNGT's documentation!
================================

.. image:: https://builds.sr.ht/~tfardet/nngt/commits.svg
    :target: https://builds.sr.ht/~tfardet/nngt
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3402493.svg
   :target: https://doi.org/10.5281/zenodo.3402493


Overview
========

The Neural Networks and Graphs' Topology (NNGT) module provides a unified
interface to access, generate, and analyze networks via any of the well-known
Python graph libraries: networkx_, igraph_, and `graph-tool`_.

For people in neuroscience, the library also provides tools to grow and
study detailed biological networks by interfacing efficient graph libraries
with highly distributed activity simulators.

The library has two main targets:

* people looking for a unifying interface for these three graph libraries,
  allowing to run and share a single code on different platforms
* neuroscience people looking for an easy way to generate complex networks
  while keeping track of neuronal populations and their biological properties


Main classes
------------

NNGT provides four main classes.
The two first are aimed at the graph-theoretical community, the third and
fourth are more for the neuroscience community.
Additional details are provided on the :ref:`main_api` page.

:class:`~nngt.Graph`
  provides a simple implementation to access and analyse topological graphs by
  wrapping any graph object from other graph libraries.
:class:`~nngt.SpatialGraph`
  a Graph embedded in space (nodes have positions and connections are
  associated to a distance).
:class:`~nngt.Network`
  provides more detailed characteristics to emulate biological neural
  networks, such as classes of inhibitory and excitatory neurons, synaptic
  properties...
:class:`~nngt.SpatialNetwork`
  combines spatial embedding and biological properties.


Generation of graphs
--------------------

Structured graphs and connectivity:
  connectivity between the nodes can be chosen from various well-known graph
  models, specific groups and structures can be generated to simplify edge
  generation
Populations:
  populations of neurons can be used and be set to respect various constraints
  (for instance a given fraction of inhibitory neurons), they simplify
  network generation and make it highly efficient to interact with the NEST_
  simulator
Synaptic properties:
  synaptic weights and delays can be set from various distributions or
  correlated to edge properties


Interacting with NEST
---------------------

The generated graphs can be used to easily create complex networks using the
NEST_ simulator, on which you can then simulate their activity.


The docs
========

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   user/install
   user/intro
   user/tutorial
   gallery/gallery

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   developer/contributing

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Modules

   modules/nngt
   modules/analysis
   modules/database
   modules/generation
   modules/geospatial
   modules/geometry
   modules/lib
   modules/plot
   modules/simulation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. Links

.. _`graph-tool`: http://graph-tool.skewed.de
.. _igraph: http://igraph.org/
.. _networkx: https://networkx.github.io/
.. _NEST: nest-simulator.readthedocs.io/
