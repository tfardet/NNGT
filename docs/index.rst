.. AGNet documentation master file, created by
   sphinx-quickstart on Fri Oct 30 15:32:21 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AGNet's documentation!
=================================

Contents:

.. toctree::
   :maxdepth: 2
   
   user/intro
   user/install
   user/graph-generation


Main classes
------------

AGNet uses three main classes:

* :class:`GraphClass`, which provides a very simple implementation over `graph_tool.Graph` (namely the addition of a name, management of excitatory and inhibitory connections, and simple access to basic graph properties
* :class:`NeuralNetwork`, which provides more detailed characteristics to emulate biological neural networks, such as classes of inhibitory and excitatory neurons, synaptic properties...
* :class:`InputConnect`, which is a basic connectivity to feed external signals to a network

Generation of graphs
--------------------

* **Structured connectivity:** connectivity between the nodes can be chosen from various well-known graph models
* **Populations:** populations of neurons are distributed afterwards on the structured connectivity, and can be set to respect various constraints (for instance a given fraction of inhibitory neurons and synapses)
* **Synaptic properties:** synaptic weights and delays can be set from various distributions or correlated to edge properties

Interacting with NEST
---------------------

The generated graphs can be used to easily create complex networks using the NEST simulator, on which you can then simulate their activity.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

