.. NNGT documentation master file, created by
   sphinx-quickstart on Fri Oct 30 15:32:21 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NNGT's documentation!
=================================

.. toctree::
   :hidden:

   self

.. user:

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   user/intro
   user/install
   user/graph-generation


Overview
---------

The Neural Network Growth and Topology (NNGT) module provides tools to grow and study detailed biological networks by interfacing efficient graph libraries with highly distributed activity simulators. 


Main classes
------------

NNGT uses three main classes:

:class:`~nngt.core.GraphClass`
	provides a very simple implementation over `graph_tool.Graph` (namely the addition of a name, management of excitatory and inhibitory connections, and simple access to basic graph properties
:class:`~nngt.core.NeuralNetwork`
	provides more detailed characteristics to emulate biological neural networks, such as classes of inhibitory and excitatory neurons, synaptic properties...
:class:`~nngt.core.InputConnect`
	is a basic connectivity to feed external signals to a network

Generation of graphs
--------------------

Structured connectivity:
	connectivity between the nodes can be chosen from various well-known graph models
Populations:
	populations of neurons are distributed afterwards on the structured connectivity, and can be set to respect various constraints (for instance a given fraction of inhibitory neurons and synapses)
Synaptic properties:
	synaptic weights and delays can be set from various distributions or correlated to edge properties

Interacting with NEST
---------------------

The generated graphs can be used to easily create complex networks using the NEST simulator, on which you can then simulate their activity.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

