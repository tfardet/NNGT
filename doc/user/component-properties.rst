.. graph-prop:

==============================
Properties of graph components
==============================

.. warning ::
    This section is not up to date anymore!

Components
==========

In the graph libraries used by NNGT, the main components of a graph are *nodes* (also called *vertices* in graph theory), which correspond to *neurons* in neural networks, and *edges*, which link *nodes* and correspond to synaptic connections between neurons in biology.

The library supposes for now that nodes/neurons and edges/synapses are always added and never removed. Because of this, we can attribute indices to the nodes and the edges which will be directly related to the order in which they have been created (the first node will have index 0, .


Node properties
===============

If you are just working with basic graphs (for instance looking at the influence of topology with purely excitatory networks), then your nodes do not need to have properties. This is the same if you consider only the average effect of inhibitory neurons by including inhibitory connections between the neurons but not a clear distinction between populations of purely excitatory and purely inhibitory neurons.
To model more realistic networks, however, you might want to define these two types of populations and connect them in specific ways.


Two types of node properties
----------------------------

In the library, there is a difference between:
	- spatial properties (the positions of the neurons), which are stored in a specific :class:`numpy.array`,
	- biological/group properties, which define assemblies of nodes sharing common properties, and are stored inside a :class:`~nngt.properties.NeuralPop` object.

Biological/group properties
---------------------------

.. note ::
	All biological/group properties are stored in a :class:`~nngt.properties.NeuralPop` object inside a :class:`~nngt.Network` instance (let us call it ``graph`` in this example); this attribute can be accessed using ``graph.population``.
	:class:`~nngt.properties.NeuralPop` objects can also be created from a :class:`~nngt.Graph` or :class:`~nngt.SpatialGraph` but they will not be stored inside the object.

The :class:`~nngt.properties.NeuralPop` class allows you to define specific groups of neurons (described by a :class:`~nngt.properties.NeuralGroup`). Once these populations are defined, you can constrain the connections between those populations.
If the connectivity already exists, you can use the :class:`~nngt.properties.GroupProperties` class to create a population with groups that respect specific constraints.

.. warning ::
	The implementation of this library has been optimized for generating an arbitrary number of neural populations where neurons share common properties; this implies that accessing the properties of one specific neuron will take O(N) operations, where N is the number of neurons. This might change in the future if such operations are judged useful enough.


Edge properties
===============

In the library, there is a difference between the (synaptic) weights and types (excitatory or inhibitory) and the other biological properties (delays, synaptic models and synaptic parameters).
This is because the weights and types are directly involved in many measurements in graph theory and are therefore directly stored inside the :class:`~nngt.core.GraphObject`.
