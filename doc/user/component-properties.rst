.. _graph-prop:

==============================
Properties of graph components
==============================

This section details the different attributes and properties which can be
associated to nodes/neurons and connections in graphs and networks.

**Content:**

.. contents::
   :local:


Components of a graph
=====================

In the graph libraries used by NNGT, the main components of a graph are *nodes*
(also called *vertices* in graph theory), which correspond to *neurons* in
neural networks, and *edges*, which link *nodes* and correspond to synaptic
connections between neurons in biology.

The library supposes for now that nodes/neurons and edges/synapses are always
added and never removed. Because of this, we can attribute indices to the nodes
and the edges which will be directly related to the order in which they have
been created (the first node will have index 0, the second index 1, etc).

The source file for the examples given here can be found at
:source:`doc/examples/attributes.py`.


Node attributes
===============

If you are just working with basic graphs (for instance looking at the
influence of topology with purely excitatory networks), then your nodes do not
necessarily need to have attributes.
This is the same if you consider only the average effect of inhibitory neurons
by including inhibitory connections between the neurons but not a clear
distinction between populations of purely excitatory and purely inhibitory
neurons.
However, if you want to include additional information regarding the nodes, to
account for specific differences in their properties, then node attributes
are what you need. They are stored in :attr:`~nngt.Graph.nodes_attributes`.
Furthermore, to model more realistic neuronal networks, you might also want to
define different groups and types of neurons, then connect them in specific
ways. This specific feature will be provides by :class:`~nngt.NeuralGroup`
objects.


Three types of node attributes
------------------------------

In the library, there is a difference between:

- standard attributes, which are stored in any type of :class:`~nngt.Graph`
  and can be created, modified, and accessed via the
  :func:`~nngt.Graph.new_node_attribute`,
  :func:`~nngt.Graph.set_node_attribute`, and
  :func:`~nngt.Graph.get_node_attributes` functions.
- spatial properties (the positions of the neurons), which are stored in a
  specific ``positions`` :class:`numpy.ndarray` and can be accessed using the
  :func:`~nngt.SpatialGraph.get_positions` function,
- biological/group properties, which define assemblies of nodes sharing common
  properties, and are stored inside a :class:`~nngt.NeuralPop` object.


Standard attributes
-------------------

Standard attributes can be any given label that might vary among the nodes in
the network and will be attached to each node.

Users can define any attribute, through the
:func:`~nngt.Graph.new_node_attribute` function.

.. literalinclude:: ../examples/attributes.py
   :lines: 28-49

Attributes can have different types:

* ``"double"`` for floating point numbers
* ``"int``" for integers
* ``"string"`` for strings
* ``"object"`` for any other python object

Here we create a second node attribute of type ``"double"``:

.. literalinclude:: ../examples/attributes.py
   :lines: 54-78


Biological/group properties
---------------------------

.. note::
    All biological/group properties are stored in a
    :class:`~nngt.NeuralPop` object inside a :class:`~nngt.Network`
    instance; this attribute can be accessed through
    :attr:`~nngt.Network.population`.
    :class:`~nngt.NeuralPop` objects can also be created from a
    :class:`~nngt.Graph` or :class:`~nngt.SpatialGraph` but they will not be
    stored inside the object.

The :class:`~nngt.NeuralPop` class allows you to define specific
groups of neurons (described by a :class:`~nngt.NeuralGroup`).
Once these populations are defined, you can constrain the connections between
those populations.
If the connectivity already exists, you can use the
:class:`~nngt.GroupProperty` class to create a population with
groups that respect specific constraints.

For more details on biological properties, see :ref:`neural_groups`.


Edge attributes
===============

Like nodes, edges can also be attributed specific values to characterize them.
However, where nodes are directly numbered and can be indexed and accessed
easily, accessing edges is more complicated, especially since, usually, not all
possible edges are present in a graph.

To easily access the desired edges, it is thus recommended to use the
:func:`~nngt.Graph.get_edges` function.


Weights and delays
------------------

By default, graphs in NNGT are weighted: each edge is associated a "weight"
value (this behavior can be changed by setting ``weighted=False`` upon
creation).

Similarly, :class:`~nngt.Network` objects always have a "delay" associated to
their connections.

Both attributes can either be set upon graph creation, through the ``weights``
and ``delays`` keyword arguments, or any any time using
:func:`~nngt.Graph.set_weights` and :func:`~nngt.Graph.set_delays`.

Let us see how the :func:`~nngt.Graph.get_edges` function can be used to
facilitate the creation of various weight patterns:

.. literalinclude:: ../examples/attributes.py
   :lines: 119-149


Custom edge attributes
----------------------

Non-default edge attributes (besides "weights" or "delays") can also be created
through smilar functions as node attributes:


**Go to other tutorials:**

* :ref:`intro`
* :ref:`graph_gen`
* :ref:`multithread`
* :ref:`neural_groups`
* :ref:`nest_int`
* :ref:`activ_analysis`
