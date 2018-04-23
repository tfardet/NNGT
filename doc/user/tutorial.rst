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

**Content:**

.. contents::
   :local:


NNGT properties and configuration
=================================

Upon loading, NNGT will display its current configuration, e.g.: ::

    # ----------- #
    # NNGT loaded #
    # ----------- #
    Graph library:  igraph 0.7.1
    Multithreading: True (1 thread)
    MPI:            False
    Plotting:       True
    NEST support:   NEST 2.14.0
    Shapely:        1.6.1
    SVG support:    True
    DXF support:    False
    Database:       False

Let's walk through this configuration:

* the backend used here is ``igraph``, so all graph-theoretical tools will be
  derived from those of the igraph_ library and we're using version 0.7.1.
* Multithreaded algorithms will be used, currently running on only one thread
  (see :ref:`Multithreading` for more details)
* MPI algorithms are not in use (you cannot use both MT and MPI at the same
  time)
* Plotting is available because the matplotlib_ library is installed
* NEST is installed on the machine (version 2.14), so NNGT automatically
  loaded it
* Shapely_ is also available, which allows the creation of complex structures
  for space-embedded networks (see :ref:`Geometry module` for more details)
* Importing SVG files to generate spatial structures is possible, meaning that
  the `svg.path`_ module is installed.
* Importing DXF files to generate spatial structures is not possible because
  the dxfgrabber_ module is not installed.
* Using the database is not possible because peewee_ is not installed.

In general, most of NNGT options can be found/set through the
:func:`~nngt.get_config`/:func:`~nngt.set_config` functions, or made permanent
by modifying the ``~/.nngt/nngt.conf`` configuration file.


The ``Graph`` object
====================

Basic functions
---------------

Let's create an empty :class:`~nngt.Graph`:

>>> g = nngt.Graph()

We can then add some nodes to it

>>> g.new_node(10)  # create nodes 0, 1, ... to 9
>>> g.node_nb()     # returns 10

And create edges between these nodes:

>>> g.new_edge(1, 4)  # create on connection going from 11 to 56
>>> g.edge_nb()       # returns 1
>>> g.new_edges([(0, 3), (5, 9), (9, 3)])
>>> g.edge_nb()       # returns 4


Node and edge attributes
------------------------

Adding a node with specific attributes: ::

    g2 = nngt.Graph()
    g2.new_node(attributes={'size': 2., 'color': 'blue'},
                value_types={'size': 'double', 'color': 'string'})
    print(g2.node_attributes)

Adding several: ::

    g2.new_node(3, attributes={'size': [4., 5., 1.], 'color': ['r', 'g', 'b']},
                value_types={'size': 'double', 'color': 'string'})
    print(g2.node_attributes)

Attributes can also be created afterwards: ::

    import numpy as np
    g3 = nngt.Graph(nodes=100)
    g3.new_node_attribute('size', 'double',
                          values=np.random.uniform(0, 20, 100))
    g3.node_attributes

All the previous techniques can also be used with :func:`~nngt.Graph.new_edge`
or :func:`~nngt.Graph.new_edges`, and :func:`~nngt.Graph.new_edge_attribute`.
Note that attributes can also be set selectively: ::

    edges = g3.new_edges(np.random.randint(0, 100, (50, 2)))
    g3.new_edge_attribute('rank', 'int', val=0)
    g3.set_edge_attribute('rank', val=2, edges=edges[:3, :])
    g3.edge_attributes


Generating and analyzing more complex networks
==============================================

NNGT provides a whole set of methods to connect nodes in specific fashions
inside a graph.
These methods are present in the :mod:`nngt.generation` module, and the network
properties can then be plotted and analyzed via the tools present in the
:mod:`nngt.plot` and :mod:`nngt.analysis` modules. ::

    from nngt import generation as ng
    from nngt import analysis as na
    from nngt import plot as nplt

NNGT implements some fast generation tools to create several of the standard
networks, such as Erdős-Rényi ::

    g = ng.erdos_renyi(nodes=1000, avg_deg=100)
    nplt.degree_distribution(g, ('in', 'total'))
    print(na.clustering(g))

More heterogeneous networks, with scale-free degree distribution (but no
correlations like in Barabasi-Albert networks and user-defined exponents) are
also implemented: ::

    g = ng.random_scale_free(1.8, 3.2, nodes=1000, avg_deg=100)
    nplt.degree_distribution(g, ('in', 'out'), num_bins=30, logx=True,
                             logy=True, show=True)
    print("Clustering: {}".format(na.clustering(g)))


Using random numbers
====================

By default, NNGT uses the `numpy` random-number generators (RNGs) which are
seeded automatically when `numpy` is loaded.

However, you can seed the RNGs manually using the following command: ::

    nngt.set_config("msd", 0)

which will seed the master seed to 0 (or any other value you enter).
Once seeded manually, a NNGT script will always give the same results provided
the same number of thread is being used.

Indeed, when using multithreading, sub-RNGs are used (one per thread). By
default, these RNGs are seeded from the master seed as `msd + n + 1` where `n`
is the thread number, starting from zero.
If needed, these sub-RNGs can also be seeded manually using (for 4 threads) ::

    nngt.set_config("seeds", [1, 2, 3, 4])

.. warning ::
When using NEST, the simulator's RNGs must be seeded separately using the NEST
commands; see the
`NEST user manual <http://www.nest-simulator.org/random-numbers/>`_ for
details.


Complex populations: :class:`~nngt.NeuralGroup` and :class:`~nngt.NeuralPop`
============================================================================

The :class:`~nngt.NeuralGroup` allows the creation of nodes that belong
together. You can then make a population from these groups and connect them
with specific connectivities using the
:func:`~nngt.generation.connect_neural_groups` function.

.. literalinclude:: ../examples/multi_groups_network.py
   :lines: 32-63


Real neuronal culture and NEST interaction: the :class:`~nngt.Network`
======================================================================

Besides connectivity, the main interest of the :class:`~nngt.NeuralGroup` is
that you can pass it the biological properties that the neurons belonging to
this group will share.

Since we are using NEST, these properties are:

* the model's name
* its non-default properties
* the synapses that the neurons have and their properties
* the type of the neurons (``1`` for excitatory or ``-1`` for inhibitory)

.. literalinclude:: ../examples/nest_network.py
   :lines: 29-68

Once this network is created, it can simply be sent to nest through the
command: ``gids = net.to_nest()``, and the NEST gids are returned.

In order to access the gids from each group, you can do: ::

    oscill_gids = net.nest_gid[oscill.ids]



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
    should be done through NNGT and never directly using the underlying
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


.. References

.. _IPython: http://ipython.org/
.. _Jupyter: https://jupyter.org/
.. _Shapely: http://toblerity.org/shapely/manual.html
.. _dxfgrabber: https://pythonhosted.org/dxfgrabber/
.. _igraph: http://igraph.org/
.. _matplotlib: https://matplotlib.org/
.. _peewee: http://docs.peewee-orm.com/en/latest/
.. _`svg.path`: https://pypi.python.org/pypi/svg.path
