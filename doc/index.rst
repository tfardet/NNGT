.. NNGT documentation master file

================================
Welcome to NNGT's documentation!
================================

.. image:: https://travis-ci.org/Silmathoron/NNGT.svg?branch=master
    :target: https://travis-ci.org/Silmathoron/NNGT
    :height: 20px
    :width: 90px
.. image:: https://coveralls.io/repos/github/Silmathoron/NNGT/badge.svg?branch=master
    :target: https://coveralls.io/github/Silmathoron/NNGT?branch=master
    :height: 20px
    :width: 99px


Overview
========

The Neural Network Growth and Topology (NNGT) module provides tools to grow and
study detailed biological networks by interfacing efficient graph libraries
with highly distributed activity simulators.


Main classes
------------

NNGT uses four main classes:

:class:`~nngt.Graph`
  provides a simple implementation over graphs objects from graph libraries
  (namely the addition of a name, management of detailed nodes and connection
  properties, and simple access to basic graph measurements).
:class:`~nngt.SpatialGraph`
  a Graph embedded in space (neurons have positions and connections are
  associated to a distance)
:class:`~nngt.Network`
  provides more detailed characteristics to emulate biological neural
  networks, such as classes of inhibitory and excitatory neurons, synaptic
  properties...
:class:`~nngt.SpatialNetwork`
  combines spatial embedding and biological properties


Generation of graphs
--------------------

Structured connectivity:
  connectivity between the nodes can be chosen from various well-known graph
  models
Populations:
  populations of neurons are distributed afterwards on the structured
  connectivity, and can be set to respect various constraints (for instance a
  given fraction of inhibitory neurons and synapses)
Synaptic properties:
  synaptic weights and delays can be set from various distributions or
  correlated to edge properties


Interacting with NEST
---------------------

The generated graphs can be used to easily create complex networks using the
NEST simulator, on which you can then simulate their activity.


The docs
========

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   user/install
   user/intro
   user/tutorial

.. ~ .. toctree::
.. ~    :glob:
.. ~    :maxdepth: 2
.. ~    :caption: Developer space

.. ~    developer/detailed-structure
.. ~    developer/graph-attributes

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Modules

   modules/nngt
   modules/analysis
   modules/generation
   modules/geometry
   modules/lib
   modules/plot
   modules/simulation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

