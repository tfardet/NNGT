==============
Multithreading
==============

.. warning:
  When using ``graph-tool``, read carefully the `Graph-tool caveat`_ section
  before playing with multiple threads!


Principle
=========

The NNGT package provides the possibility to use multithreaded algorithms to
generate networks.
This feature means that the computation is distributed on several CPUs and can
be useful for:

- machines with several cores but low frequency
- generation functions requiring large amounts of computation
- very large graphs

However, the multithreading part concerns only the generation of the edges; if
a graph library such as ``graph-tool``, ``igraph``, or ``networkx`` is used,
the building process of the graph object will be taken care of by this library.
Since this process is not multithreaded, obtaining the graph object can be much
longer than the actual generation process.


Use
===

Setting multithreading
----------------------

Multithreading in NNGT can be set via

>>> nngt.set_config({"multithreading": True, "omp": num_omp_threads})

and you can then switch it off using

>>> nngt.set_config("multithreading", False)

This will automatically switch between the standard and multithreaded
algorithms for graph generation.


Random seeds for multithreading
-------------------------------

@Todo


Graph-tool caveat
-----------------

The ``graph-tool`` library also provides some multithreading capabilities,
using

>>> graph_tool.openmp_set_num_threads(num_omp_threads)

However, this sets the number of OpenMP threads session-wide, which means that
**it will interfere with the ``NEST`` setup!**
Hence, if you are working with both ``NEST`` and ``graph-tool``, **you have
to use the same number of OpenMP threads in both libraries**.

To prevent bad surprises as much as possible, NNGT will raise an error if
a value of ``"omp"`` is provided, which differs from the current NEST
configuration.
Regardless of this precaution, keeping only one value for the number of threads
and using it consistently throughout the code is strongly advised.

