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

In Jupyter, the docstring can be viewed using :kbd:`Shift` + :kbd:`Tab`.

The source file for the tutorial can be found here:
:source:`doc/examples/introductory_tutorial.py`.

.. note ::
    For a list of example files, see the `'examples' directory on GitHub
    <https://github.com/Silmathoron/NNGT/tree/master/doc/examples>`_.

    For specific tutorials see also:

    * :ref:`graph_gen`
    * :ref:`parallelism`
    * :ref:`neural_groups`
    * :ref:`nest_int`
    * :ref:`activ_analysis`
    * :ref:`graph-prop`


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
  (see :ref:`parallelism` for more details)
* MPI algorithms are not in use (you cannot use both MT and MPI at the same
  time)
* Plotting is available because the matplotlib_ library is installed
* NEST is installed on the machine (version 2.14), so NNGT automatically
  loaded it
* Shapely_ is also available, which allows the creation of complex structures
  for space-embedded networks (see :ref:`geometry` for more details)
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

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 33

We can then add some nodes to it

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 36-37

And create edges between these nodes:

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 40-43


Node and edge attributes
------------------------

Adding a node with specific attributes:

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 50-68

By default, nodes that are added without specifying attribute values will get
their attributes filled with default values which depend on the type:

* ``NaN`` for "double"
* 0 for "int"
* ``""`` for "string"
* ``None`` for "object"

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 71-79

Adding several nodes and attributes at the same time:

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 82-85

Attributes can also be created afterwards:

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 88-92

All the previous techniques can also be used with :func:`~nngt.Graph.new_edge`
or :func:`~nngt.Graph.new_edges`, and :func:`~nngt.Graph.new_edge_attribute`.
Note that attributes can also be set selectively:

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 95-98


Generating and analyzing more complex networks
==============================================

NNGT provides a whole set of methods to connect nodes in specific fashions
inside a graph.
These methods are present in the :mod:`nngt.generation` module, and the network
properties can then be plotted and analyzed via the tools present in the
:mod:`nngt.plot` and :mod:`nngt.analysis` modules.

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 110-112

NNGT implements some fast generation tools to create several of the standard
networks, such as Erdős-Rényi:

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 115-120

More heterogeneous networks, with scale-free degree distribution (but no
correlations like in Barabasi-Albert networks and user-defined exponents) are
also implemented:

.. literalinclude:: ../examples/introductory_tutorial.py
   :lines: 123-129

For more details, see the full page on :ref:`graph_gen`.


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
    When using NEST, the simulator's RNGs must be seeded separately using the
    NEST commands; see the
    `NEST user manual <http://www.nest-simulator.org/random-numbers/>`_ for
    details.


Structuring nodes: :class:`~nngt.Group` and :class:`~nngt.Structure`
====================================================================

The :class:`~nngt.Group` allows the creation of nodes that belong
together. You can then make a complex :class:`~nngt.Structure` from these
groups and connect them with specific connectivities using the
:func:`~nngt.generation.connect_groups` function.

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 51-84

For more details, see the full page on :ref:`neural_groups`.


The same with neurons: :class:`~nngt.NeuralGroup`, :class:`~nngt.NeuralPop`
===========================================================================

The :class:`~nngt.NeuralGroup` allows the creation of nodes that belong
together. You can then make a population from these groups and connect them
with specific connectivities using the
:func:`~nngt.generation.connect_groups` function.

.. literalinclude:: ../examples/introduction_to_groups.py
   :lines: 103-127

For more details, see the full page on :ref:`neural_groups`.


Real neuronal networks and NEST interaction: the :class:`~nngt.Network`
=======================================================================

Besides connectivity, the main interest of the :class:`~nngt.NeuralGroup` is
that you can pass it the biological properties that the neurons belonging to
this group will share.

Since we are using NEST, these properties are:

* the model's name
* its non-default properties
* the synapses that the neurons have and their properties
* the type of the neurons (``1`` for excitatory or ``-1`` for inhibitory)

.. literalinclude:: ../examples/nest_network.py
   :lines: 29-52, 62-83

Once this network is created, it can simply be sent to nest through the
command: ``gids = net.to_nest()``, and the NEST gids are returned.

In order to access the gids from each group, you can do: ::

    oscill_gids = net.nest_gid[oscill.ids]

For more details to use NNGT with NEST, see :ref:`nest_int`.


.. _graph_attr:

Underlying graph objects and libraries
======================================

Starting with version 2.0 of NNGT, the library no longer uses inheritance but
composition to provide access to the underlying graph object, which is stored
in the :attr:`~nngt.Graph.graph` attribute of the :class:`~nngt.Graph` class.

It can simply be accessed via: ::

    g = nngt.Graph()

    library_graph = g.graph

Using :attr:`~nngt.Graph.graph` attribute, on can directly use functions of the
underlying graph library (networkx, igraph, or graph-tool) if their equivalent
is not yet provided in NNGT -- see :ref:`graph-analysis` for implemented
functions.

.. warning::
    One notable exception to this behaviour relates to the creation and
    deletion of nodes or edges, for which you have to use the functions
    provided by NNGT.
    As a general rule, any operation that might alter the graph structure
    should be done through NNGT and never directly by calling functions or
    methods on the :attr:`~nngt.Graph.graph` attribute.

Apart from this, you can use any analysis or drawing tool from the graph
library.


Example using graph-tool
------------------------

>>> import graph_tool as gt
>>> import matplotlib.pyplot as plt
>>> print(gt.centrality.closeness(g.graph))
>>> gt.draw.graph_draw(g.graph)
>>> nngt.plot.draw_network(g)
>>> plt.show()


Example using igraph
--------------------

>>> import igraph as ig
>>> import matplotlib.pyplot as plt
>>> print(g.graph.closeness(mode='out'))
>>> ig.plot(g.graph)
>>> nngt.plot.draw_network(g)
>>> plt.show()


Example using networkx
----------------------

>>> import networkx as nx
>>> import matplotlib.pyplot as plt
>>> print(nx.closeness_centrality(g.graph.reverse()))
>>> nx.draw(g.graph)
>>> nngt.plot.draw_network(g)
>>> plt.show()


.. note::
    People testing these 3 codes will notice that all closeness results are
    different (though I made sure the functions of each libraries worked
    on the same outgoing edges)!
    This example is given voluntarily to remind you, when using these
    libraries, to check that they indeed compute what you think they do and
    what are the underlying hypotheses or definitions.

    To avoid such issues and make sure that results are the same with all
    libraries, use the functions provided in :ref:`graph-analysis`.


**Go to other tutorials:**

* :ref:`intro`
* :ref:`graph_gen`
* :ref:`parallelism`
* :ref:`neural_groups`
* :ref:`nest_int`
* :ref:`activ_analysis`
* :ref:`graph-prop`


.. References

.. _IPython: http://ipython.org/
.. _Jupyter: https://jupyter.org/
.. _Shapely: http://toblerity.org/shapely/manual.html
.. _dxfgrabber: https://pythonhosted.org/dxfgrabber/
.. _igraph: http://igraph.org/
.. _matplotlib: https://matplotlib.org/
.. _peewee: http://docs.peewee-orm.com/en/latest/
.. _`svg.path`: https://pypi.python.org/pypi/svg.path
