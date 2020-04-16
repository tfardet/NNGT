.. _graph-analysis:

==============
Graph analysis
==============

NNGT provides several functions for topological analysis that return consistent
results for all backends.
This section describes these functions and gives an overview of the currently
supported methods.

**Content:**

.. contents::
   :local:


Supported functions
===================

The following table details which functions are supported for directed and
undirected networks, and whether they also work with weighted edges.

+----------------------------------------------------+------------------------+---------------------+---------------------+-------------------+
| Method                                             | Unweighted, undirected | Unweighted directed | Weighted undirected | Weighted directed |
+====================================================+========================+=====================+=====================+===================+
| :func:`~nngt.analysis.global_clustering`           |       gt, nx, ig       |         x           |   x                 |  x                |
+----------------------------------------------------+------------------------+---------------------+---------------------+-------------------+
| :func:`~nngt.analysis.undirected_local_clustering` |       gt, nx, ig       |         x           |   x                 |  x                |
+----------------------------------------------------+------------------------+---------------------+---------------------+-------------------+
| :func:`~nngt.analysis.assortativity`               |       gt, nx, ig       |     gt, nx, ig      |    gt, ig [1]_      |    gt, ig         |
+----------------------------------------------------+------------------------+---------------------+---------------------+-------------------+
| :func:`~nngt.analysis.reciprocity`                 |       gt, nx, ig       |     gt, nx, ig      |     gt, nx, ig      |    gt, nx, ig     |
+----------------------------------------------------+------------------------+---------------------+---------------------+-------------------+
| :func:`~nngt.analysis.closeness` [2]_              |       gt, nx, (ig)     |     gt, nx, (ig)    |         x           |         x         |
+----------------------------------------------------+------------------------+---------------------+---------------------+-------------------+


.. [1] networkx could be used via a workaround but `an issue
       <https://github.com/networkx/networkx/issues/3917>`_ has been raised to
       find out how to best deal with this
.. [2] since definitions of the distances differ for the libraries, igraph is
       currently not usable if the in- or out-degree of any of the nodes is
       zero and does not provide an implementation for the harmonic closeness


----


**Go to other tutorials:**

* :ref:`intro`
* :ref:`graph_gen`
* :ref:`multithread`
* :ref:`neural_groups`
* :ref:`nest_int`
* :ref:`activ_analysis`
