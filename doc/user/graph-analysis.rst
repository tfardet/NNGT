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


----


**Go to other tutorials:**

* :ref:`intro`
* :ref:`graph_gen`
* :ref:`multithread`
* :ref:`neural_groups`
* :ref:`nest_int`
* :ref:`activ_analysis`


.. Links

.. _global_clustering: 
.. _global_clustering: 
