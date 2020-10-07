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
are what you need. They are stored in :attr:`~nngt.Graph.node_attributes`.
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

Edge attributes can then be created and recovered using similar functions as
node attributes, namely :func:`~nngt.Graph.new_edge_attribute`,
:func:`~nngt.Graph.set_edge_attribute`, and
:func:`~nngt.Graph.get_edge_attributes`.


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

.. note::
    When working with NEST and using excitatory and inhibitory neurons via
    groups (see :ref:`neural_groups`), the weight of all connections
    (including inhibitory connections) should be positive: the excitatory or
    inhibitory type of the synapses will be set automatically when the NEST
    network is created based on the type of the source neuron.

    In general, it is also not a good idea to use negative weights directly
    since standard graph analysis methods cannot handle them.
    If you are not working with biologically realistic neurons and want to
    set some inhibitory connections that do not depend on a "neuronal type",
    use the :func:`~nngt.Graph.set_types` method.

Let us see how the :func:`~nngt.Graph.get_edges` function can be used to
facilitate the creation of various weight patterns:

.. literalinclude:: ../examples/attributes.py
   :lines: 119-149

Note that here, the weights were generated randomly from specific
distributions; for more details on the available distributions and their
parameters, see `Attributes and distributions`_.


Custom edge attributes
----------------------

Non-default edge attributes (besides "weights" or "delays") can also be created
through smilar functions as node attributes:

.. literalinclude:: ../examples/attributes.py
   :lines: 84-89,157-184


.. _distributions:

Attributes and distributions
============================

Node and edge attributes can be generated based on the following
distributions:

uniform
   - a flat distribution with identical probability for all values,
   - parameters: ``"lower"`` and ``"upper"`` values.

delta
   - the Dirac delta "distribution", where a single value can be drawn,
   - parameters: ``"value"``.

Gaussian
   - the normal distribution :math:`P(x) = P_0 e^{(x - \mu)^2/(2\sigma^2)}`
   - parameters: ``"avg"`` (:math:`\mu`) and ``"std"`` (:math:`\sigma`).

lognormal
   - :math:`P(x) = P_0 e^{(\log(x) - \mu)^2/(2\sigma^2)}`
   - parameters: ``"position"`` (:math:`\mu`) and ``"scale"`` (:math:`\sigma`).

linearly correlated
   - distribution name: ``"lin_corr"``
   - a distribution which evolves linearly between two values depending on the
     value of a reference variable
   - parameters: ``"correl_attribute"`` (the reference variable, usually
     another attribute), ``"lower"`` and ``"upper"``, the minimum and maximum
     values.


Example
-------

Generating a graph with delays that are linearly correlated to the distance
between nodes.

.. code-block:: python

    dmin = 1.
    dmax = 8.

    d = {
        "distribution": "lin_corr", "correl_attribute": "distance",
        "lower": dmin, "upper": dmax
    }

    g = nngt.generation.distance_rule(200., nodes=100, avg_deg=10, delays=d)


----

**Go to other tutorials:**

* :ref:`intro`
* :ref:`graph_gen`
* :ref:`parallelism`
* :ref:`neural_groups`
* :ref:`nest_int`
* :ref:`activ_analysis`
