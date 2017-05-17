================
Graph attributes
================

.. warning ::
    This section is not up to date anymore!
    
The :class:`~nngt.Graph` class and its subclasses contain several attributes
regarding the properties of the edges and nodes. Edges attributes are contained
in the graph dictionary; more complex properties about the biological details
of the nodes/neurons are contained in the NeuralPop member of the
:class:`~nngt.Graph`. These are briefly described in
:doc:`../user/component-properties`; a more detailed description is provided
here.


Attributes and graph libraries
==============================

Usual graph libraries can store node and edge properties; as an example, many
graphs are weighted and these weights can then be used to compute other
properties such as weighted centralities, which is why it is interesting to
have those properties stored in the basic graph library class.

The `graph_object.py` file contains the ``_GtEProperty`` and ``_GtNProperty``
classes which allow a generic interactions with the various libraries ways of
storing properties.

However, several problems occurs:

* for `graph_tool <https://graph-tool.skewed.de>`_, the edge properties are
  stored in a linear array that is not directly related to the adjacency matrix,
  thus difficult to handle; this could however be avoided by multiplying the
  adjacency matrix by the property of interest...
* but for `igraph <http://igraph.org/>`_, there is not straightforward way to
  obtain a scipy adjacency matrix multiplied by an edge property...

To get rid of those problems, the (possibly temporary) solution adopted is to
have the weights (synaptic strength) and types (inhibitory or excitatory)
attributes stored both in the graph library object and in the
:class:`~nngt.Graph` container.

The libraries indices the edges in the order they are created; because of this,
weights must be added to the library using the edge list, which is stored
inside the :class:`~nngt.Graph` container (access it through the ``''edges''``
key). The addition is performed in the following way: let
``lil_matrix_attribute`` contain the attribute of interest and ``network`` be
the graph container to which we want to add the property, then the following
code is used,

.. code-block:: python

  sources, targets = network["edges"][:,0], network["edges"][:,1]
  list_ordered_weights = lil_matrix_attribute[sources,targets].data[0]
  network.graph.new_edge_attribute(
      "weight", "double", values=list_ordered_weights)


Use of attributes in a graph object
===================================

This allows for fast graph filtering: we can keep only the edges or nodes we
are interested in.

This property is invaluable if you want to study the graph properties of only
the inhibitory network or look a the squeleton of the strongest synapses in the
graph...

.. note ::

  This mixed format is not too good... I should either store everything in
  the container or in the library graph.

  Library graph:
      * difficult to manage
      * but users can use the library on the graph
      * if I cannot provide a fast conversion, it will be bad to interact
        with NEST

  Container:
      * easier to manage
      * but need to convert for the analysis functions
      * users cannot use the library as the graph misses its attributes
      * optimized for NEST interactions
	
