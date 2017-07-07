========
Tutorial
========

This page provides a step-by-step walkthrough of the basic features of NNGT.

To run this tutorial, it is recommended to use either IPython_ or Jupyter_,
since they will provide automatic autocompletion of the various functions, as
well as easy access to the docstring help.

First, import the NNGT package:

>>> import nngt

Then, you will be able to use the help from IPython by typing, for instance:

>>> nngt.Graph?

In Jupyter, the docstring can be viewed using Shift+Tab.


The ``Graph`` object
====================

Basic functions
---------------

Let's create an empty :class:`~nngt.Graph`:

>>> g = nngt.Graph()

We can then add some nodes to it

>>> g.new_nodes(10)  # create nodes 0, 1, ... to 9
>>> g.node_nb()       # returns 10

And create edges between these nodes:

>>> g.new_edge(1, 4)  # create on connection going from 11 to 56
>>> g.edge_nb()       # returns 1
>>> g.new_edges([(0, 3), (5, 9), (9, 3)])
>>> g.edge_nb()       # returns 4


Node and edge properties
------------------------

@todo


Using the graph library of the NNGT object
==========================================

As mentionned in the installation and introduction, NNGT uses existing graph
library objects to store the graph.
The library was designed so that most of the functions of the underlying graph
library can be used directly on the :class:`~nngt.Graph` object.

.. warning::
    One notable exception to this behaviour relates to the creation and
    deletion of nodes or edges, for which you have to use the functions
    provided by NNGT.
    As a general rule, any operation that might alter the graph structure
    should be done thourgh NNGT and never directly using the underlying
    library.

Apart from this, you can use any analysis or drawing tool from the graph
library.


Example using graph-tool
------------------------

>>> import graph_tool as gt
>>> import matplotlib.pyplot as plt
>>> print(gt.centrality.closeness(g, harmonic=True))
>>> gt.draw.graph_draw(g)
>>> nngt.plot.draw_network(g)
>>> plt.show()


Example using igraph
--------------------

>>> import igraph as ig
>>> import matplotlib.pyplot as plt
>>> print(g.closeness(mode='out'))
>>> ig.plot(g)
>>> nngt.plot.draw_network(g)
>>> plt.show()


Example using networkx
----------------------

>>> import networkx as nx
>>> import matplotlib.pyplot as plt
>>> print(nx.closeness_centrality(g))
>>> nx.draw(g)
>>> nngt.plot.draw_network(g)
>>> plt.show()


.. note::
    People testing these 3 codes will notice that all closeness results are
    different (though I made sure the functions of each libraries worked
    on the same outgoing edges)!
    This example is given voluntarily to remind you, when using these
    libraries, to check that they indeed compute what you think they do.
    And even when they compute it, check how they do it!


NNGT configuration status
=========================

>>> nngt.get_config()


.. References

.. _IPython: http://ipython.org/
.. _Jupyter: https://jupyter.org/
