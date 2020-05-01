.. _graph-analysis:

===================================
Consistent tools for graph analysis
===================================

NNGT provides several functions for topological analysis that return consistent
results for all backends (the results will always be the same regardless of
which library is used under the hood).
This section describes these functions and gives an overview of the currently
supported methods.

.. note::
    It is of course possible to use any function from the library on the
    :py:attr:`~nngt.Graph.graph` attribute; however, not using one of the
    supported NNGT functions below will usually return results that are not
    consistent between libraries (and the code will obviously no longer be
    portable).


Supported functions
===================

The following table details which functions are supported for directed and
undirected networks, and whether they also work with weighted edges.

The test file where these functions are checked can be found here:
:source:`testing/library_compatibility.py`.

For each type of graph, the table tells which libraries are supported for the
given function (graph-tool is `gt`, networkx is `nx` and igraph is `ig`).
Custom implementation of a function is denoted by `nngt`, meaning that the
function can be used even if no graph library is installed.
A library marked between parentheses denotes partial support and additional
explanation is usually given in the footnotes.
A cross means that no consistent implementation is currently provided and
the function will raise an error if one tries to use it on such graphs.

+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
|  Method                                            | Unweighted undirected | Unweighted directed | Weighted undirected | Weighted directed  |
+====================================================+=======================+=====================+=====================+====================+
| :func:`~nngt.analysis.assortativity` [1]_          |    gt, nx, ig         |   gt, nx, ig        |   gt, ig            |   gt, ig           |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.betweenness`                 |    gt, nx, ig         |   gt, nx, ig        |   gt, nx, ig        |   gt, nx, ig       |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.betweenness_distrib`         |    gt, nx, ig         |   gt, nx, ig        |   gt, nx, ig        |   gt, nx, ig       |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.closeness` [2]_              |    gt, nx, (ig)       |   gt, nx, (ig)      |   gt, nx, (ig)      |   gt, nx, (ig)     |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.connected_components`        |    gt, nx, ig         |   gt, nx, ig        |   gt, nx, ig        |   gt, nx, ig       |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.degree_distrib`              |    gt, nx, ig, nngt   |   gt, nx, ig, nngt  |   gt, nx, ig, nngt  |   gt, nx, ig, nngt |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.diameter` [3]_               |    gt, nx, ig         |   gt, nx, ig        |   gt, nx, ig        |   gt, nx, ig       |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.global_clustering`           |    gt, nx, ig         |   x                 |   x                 |   x                |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.local_clustering` [4]_       |    gt, nx, ig         |   x                 |   x                 |   x                |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.reciprocity`                 |    gt, nx, ig         |   gt, nx, ig        |   gt, nx, ig        |   gt, nx, ig       |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.spectral_radius`             |    nngt               |   nngt              |   nngt              |   nngt             |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.subgraph_centrality`         |    nngt               |   nngt              |   nngt              |   nngt             |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+
| :func:`~nngt.analysis.transitivity` [5]_           |    gt, nx, ig         |   x                 |   x                 |   x                |
+----------------------------------------------------+-----------------------+---------------------+---------------------+--------------------+


.. [1] networkx could be used via a workaround but `an issue
       <https://github.com/networkx/networkx/issues/3917>`_ has been raised to
       find out how to best deal with this.
.. [2] since definitions of the maximum distances differ between libraries,
       igraph is currently not usable if the in- or out-degree of any of the
       nodes is zero; it also does not provide an implementation for the
       harmonic closeness.
.. [3] the implementation of the diameter for graph-tool is approximmate so
       results may occasionaly be inexact with this backend.
.. [4] for directed and weighted networks, definitions and implementations
       differ between graph libraries, generic implementation in NNGT will
       come soon.
.. [5] identical to ``global_clustering``.

----


**Go to other tutorials:**

* :ref:`intro`
* :ref:`graph_gen`
* :ref:`multithread`
* :ref:`neural_groups`
* :ref:`nest_int`
* :ref:`activ_analysis`
