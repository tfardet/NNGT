#-*- coding:utf-8 -*-
#
# clustering.py
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2019  Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Tools for directed/weighted clsutering analysis """

import numpy as np

import nngt


__all__ = [
	"global_clustering",
    "global_clustering_binary_undirected",
    "local_clustering",
    "local_clustering_binary_undirected",
    "triplet_count",
    "triangle_count",
]


def global_clustering_binary_undirected(g):
    '''
    Returns the undirected global clustering coefficient.

    This corresponds to the ratio of undirected triangles to the number of
    undirected triads.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    '''
    # Note, this function is overloaded by the library-specific version
    # if igraph, graph-tool, or networkx is used
    triangles = triangle_count(g, weights=None, directed=False)
    triplets  = triplet_count(g, weights=None, directed=False)

    return np.sum(triangles) / np.sum(triplets)


def global_clustering(g, directed=True, weights=None, method="continuous",
                      combine_weights="mean"):
    '''
    Returns the global clustering coefficient.

    This corresponds to the ratio of triangles to the number of triplets.
    For directed and weighted cases, see definitions of generalized triangles
    and triplets in the associated functions below.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    directed : bool, optional (default: True)
        Whether to compute the directed clustering if the graph is directed.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    method : str, optional (default: 'continuous')
        Method used to compute the weighted clustering, either 'barrat'
        [Barrat2004]_, 'continuous', or 'onnela' [Onnela2005]_.
    combine_weights : str, optional (default: 'mean')
        How to combine the weights of reciprocal edges if the graph is directed
        but `directed` is set to False. It can be:

        * "sum": the sum of the edge attribute values will be used for the new
          edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge.

    References
    ----------
    .. [gt-global-clustering] :gtdoc:`clustering.global_clustering`
    .. [ig-global-clustering] :igdoc:`transitivity_undirected`
    .. [nx-global-clustering] :nxdoc:`algorithms.cluster.transitivity`
    .. [Barrat2004] Barrat, Barthelemy, Pastor-Satorras, Vespignani. The
        Architecture of Complex Weighted Networks. PNAS 2004, 101 (11).
        :doi:`10.1073/pnas.0400087101`.
    .. [Onnela2005] Onnela, Saramäki, Kertész, Kaski. Intensity and Coherence
        of Motifs in Weighted Complex Networks. Phys. Rev. E 2005, 71 (6),
        065103. :doi:`10.1103/physreve.71.065103`, arxiv:`cond-mat/0408629`.

    See also
    --------
    :func:`~nngt.analysis.triplet_count`
    :func:`~nngt.analysis.triangle_count`
    '''
    # check directivity and weights
    directed *= g.is_directed()
    weighted  = weights not in (False, None)

    if not directed and not weighted:
        return global_clustering_binary_undirected(g)
    elif not weighted:
        # directed clustering
        triangles = triangle_count(g)
        triplets  = triplet_count(g)

        return np.sum(triangles) / np.sum(triplets)

    triangles, triplets = _triangles_and_triplets(g, directed, weights, method,
                                                  combine_weights, None)

    return np.sum(triangles) / np.sum(triplets)


def local_clustering_binary_undirected(g, nodes=None):
    r'''
    Returns the undirected local clustering coefficient of some `nodes`.

    .. math::

        C_i = \frac{A^3_{ii}}{d_i(d_i - 1)} = \frac{\Delta_i}{T_i}

    with :math:`A` the adjacency matrix, :math:`d_i` the degree of node
    :math:`i`, :math:`\Delta_i` is the number of triangles, and :math:`T_i` is
    the number of triplets to which :math:`i` belongs.

    If `g` is directed, then it is converted to a simple undirected graph
    (no parallel edges), both directed and reciprocal edges are merged into
    a single edge.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    nodes : list, optional (default: all nodes)
        The list of nodes for which the clustering will be returned

    Returns
    -------
    lc : :class:`numpy.ndarray`
        The list of clustering coefficients, on per node.

    References
    ----------
    .. [gt-local-clustering] :gtdoc:`clustering.local_clustering`
    .. [ig-local-clustering] :igdoc:`transitivity_local_undirected`
    .. [nx-local-clustering] :nxdoc:`algorithms.cluster.clustering`
    '''
    # Note, this function is overloaded by the library-specific version
    # if igraph, graph-tool, or networkx is used
    triangles = triangle_count(g, weights=None, nodes=nodes, directed=False)
    triplets  = triplet_count(g, weights=None, nodes=nodes, directed=False)

    triplets[triangles == 0] = 1

    return triangles / triplets


def local_clustering(g, nodes=None, directed=True, weights=None,
                     method="continuous", combine_weights="mean"):
    r'''
    Local (weighted directed) clustering coefficient of the nodes.

    If no weights are requested and the graph is undirected, returns the
    undirected binary clustering.

    For all weighted cases, the weights are assumed to be positive and they are
    normalized to dimensionless values between 0 and 1 through a division by
    the highest weight.

    The default `method` for weighted networks is based on a modification of
    the proposal in [Zhang2005]_ with:

    .. math::

        C_i = \frac{\left(W^{\left[\frac{2}{3}\right]}\right)^3_{ii}}
                   {\left(s^{\left[\frac{1}{2}\right]}_i\right)^2 - s_i}

    for undirected networks, with :math:`\tilde{W}` the adjacency matrix
    (normalized to have its highest weight set to 1 if weighted), :math:`s_i`
    the strength of node :math:`i`, and
    :math:`s^{[\frac{1}{2}]}_i = \sum_k \sqrt{w_{ik}}`
    the strength associated to the matrix
    :math:`W^{[\frac{1}{2}]} = \{\sqrt{w_{ij}}\}`.

    For directed networks, we used the total clustering defined in
    [Fagiolo2007]_, hence the second equation becomes:

    .. math::

        C_i = \frac{\frac{1}{2}\left(W^{\left[\frac{2}{3}\right]}
                    + W^{\left[\frac{2}{3}\right],T}\right)^3_{ii}}
                   {\left(s^{\left[\frac{1}{2}\right]}_i\right)^2
                    - 2s^{\leftrightarrow}_i - s_i}

    with :math:`s^{\leftrightarrow} = \sum_k \sqrt{w_{ik}w_{ki}}` the
    reciprocal strength (associated to reciprocal connections).

    Contrary to 'barrat' and 'onnela' [Saramaki2007]_, this method displays
    *all* following properties:

    * fully continuous (no jump in clustering when weights go to zero),
    * equivalent to binary clustering when all weights are 1,
    * equivalence between no-edge and zero-weight edge cases,
    * normalized (always between zero and 1).

    Parameters
    ----------
    g : :class:`~nngt.Graph` object
        Graph to analyze.
    nodes : array-like container with node ids, optional (default = all nodes)
        Nodes for which the local clustering coefficient should be computed.
    directed : bool, optional (default: True)
        Whether to compute the directed clustering if the graph is directed.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    method : str, optional (default: 'continuous')
        Method used to compute the weighted clustering, either 'barrat'
        [Barrat2004]_/[Clemente2018]_, 'continuous', or 'onnela' [Onnela2005]_.
    combine_weights : str, optional (default: 'mean')
        How to combine the weights of reciprocal edges if the graph is directed
        but `directed` is set to False. It can be:

        * "sum": the sum of the edge attribute values will be used for the new
          edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge.

    Returns
    -------
    lc : :class:`numpy.ndarray`
        The list of clustering coefficients, on per node.

    References
    ----------
    .. [Barrat2004] Barrat, Barthelemy, Pastor-Satorras, Vespignani. The
        Architecture of Complex Weighted Networks. PNAS 2004, 101 (11).
        :doi:`10.1073/pnas.0400087101`.
    .. [Clemente2018] Clemente, Grassi. Directed Clustering in Weighted
        Networks: A New Perspective. Chaos, Solitons & Fractals 2018, 107,
        26–38. :doi:`10.1016/j.chaos.2017.12.007`, :arxiv:`1706.07322`.
    .. [Fagiolo2007] Fagiolo. Clustering in Complex Directed Networks.
        Phys. Rev. E 2007, 76, (2), 026107. :doi:`10.1103/PhysRevE.76.026107`,
        :arxiv:`physics/0612169`.
    .. [Onnela2005] Onnela, Saramäki, Kertész, Kaski. Intensity and Coherence
        of Motifs in Weighted Complex Networks. Phys. Rev. E 2005, 71 (6),
        065103. :doi:`10.1103/physreve.71.065103`, :arxiv:`cond-mat/0408629`.
    .. [Saramaki2007] Saramäki, Kivelä, Onnela, Kaski, Kertész. Generalizations
        of the Clustering Coefficient to Weighted Complex Networks.
        Phys. Rev. E 2007, 75 (2), 027105. :doi:`10.1103/PhysRevE.75.027105`,
        :arxiv:`cond-mat/0608670`.
    .. [Zhang2005] Zhang, Horvath. A General Framework for Weighted Gene
        Co-Expression Network Analysis. Statistical Applications in Genetics
        and Molecular Biology 2005, 4 (1). :doi:`10.2202/1544-6115.1128`,
        `PDF <https://dibernardo.tigem.it/files/papers/2008/
        zhangbin-statappsgeneticsmolbio.pdf>`_.

    See also
    --------
    :func:`undirected_binary_clustering`
    :func:`global_clustering`
    '''
    # check directivity and weights
    directed *= g.is_directed()
    weighted  = weights not in (None, False)

    if not directed and not weighted:
        # undirected binary clustering uses the library method
        return local_clustering_binary_undirected(g, nodes=nodes)
    elif not weighted:
        # directed clustering
        triangles = triangle_count(g, nodes=nodes)
        triplets  = triplet_count(g, nodes).astype(float)

        triplets[triangles == 0] = 1

        return triangles / triplets

    triangles, triplets = _triangles_and_triplets(g, directed, weights, method,
                                                  combine_weights, nodes)

    triplets[triangles == 0] = 1

    return triangles / triplets


def triangle_count(g, nodes=None, directed=True, weights=None,
                   method="normal", combine_weights="mean"):
    '''
    Returns the number or the strength (also called intensity) of triangles
    for each node.

    Parameters
    ----------
    g : :class:`~nngt.Graph` object
        Graph to analyze.
    nodes : array-like container with node ids, optional (default = all nodes)
        Nodes for which the local clustering coefficient should be computed.
    directed : bool, optional (default: True)
        Whether to compute the directed clustering if the graph is directed.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    method : str, optional (default: 'normal')
        Method used to compute the weighted triangles, either 'normal', where
        the weights are directly used, or the definitions associated to the
        weighted clustering: 'barrat' [Barrat2004]_, 'continuous', or 'onnela'
        [Onnela2005]_.
    combine_weights : str, optional (default: 'mean')
        How to combine the weights of reciprocal edges if the graph is directed
        but `directed` is set to False. It can be:

        * "sum": the sum of the edge attribute values will be used for the new
          edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge.

    Returns
    -------
    tr : array
        Number or weight of triangles to which each node belongs.
    '''
    directed *= g.is_directed()
    weighted  = weights not in (False, None)

    exponent = None

    if method == "onnela":
        exponent = 1/3
    elif method == "continuous":
        exponent = 2/3

    # get relevant matrices (use directed=False to get both dir/undir mat)
    _, matsym = _get_matrices(g, directed, weights, weighted, combine_weights,
                              exponent=exponent)

    # if unweighted, adjsym is matsym
    adjsym = matsym

    # for barrat, we need both weighted and binary matrices
    if method == "barrat" and weighted:
        _, adjsym = _get_matrices(g, directed, None, False, combine_weights)

    return _triangle_count(matsym, adjsym, method, weighted, directed, nodes)


def triplet_count(g, nodes=None, directed=True, weights=None,
                method="normal", combine_weights="mean"):
    r'''
    Returns the number or the strength (also called intensity) of triplets for
    each node.

    For binary networks, the triplets of node :math:`i` are defined as:

    .. math::

        T_i = \sum_{j,k} a_{ij}a_{ik}

    Parameters
    ----------
    g : :class:`~nngt.Graph` object
        Graph to analyze.
    nodes : array-like container with node ids, optional (default = all nodes)
        Nodes for which the local clustering coefficient should be computed.
    directed : bool, optional (default: True)
        Whether to compute the directed clustering if the graph is directed.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    method : str, optional (default: 'continuous')
        Method used to compute the weighted triplets, either 'normal', where
        the edge weights are directly used, or the definitions used for
        weighted clustering coefficients, 'barrat' [Barrat2004]_ or
        'continuous'.
    combine_weights : str, optional (default: 'mean')
        How to combine the weights of reciprocal edges if the graph is directed
        but `directed` is set to False. It can be:

        * "sum": the sum of the edge attribute values will be used for the new
          edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge.

    Returns
    -------
    tr : array
        Number or weight of triplets to which each node belongs.
    '''
    directed *= g.is_directed()
    weighted  = weights not in (False, None)

    # simple binary cases
    if not weighted or method == "onnela":
        # undirected
        if not directed:
            deg = None

            if g.is_directed():
                _, adjsym = _get_matrices(g, directed, None, False,
                                          combine_weights)

                deg = adjsym.sum(axis=0).A1
            else:
                deg = g.get_degrees(nodes=nodes)

            return (0.5*deg*(deg - 1)).astype(int)

        # directed
        adj = g.adjacency_matrix()

        d_recip = (adj*adj).diagonal()
        dtot    = g.get_degrees("total", nodes=nodes)

        tr = dtot*(dtot - 1) - 2*d_recip

        if nodes is None:
            return tr

        return tr[nodes]

    # check method for weighted
    W, Wu, A, Au = None, None, None, None

    if method in ("continuous", "normal"):
        # we need only the weighted matrices
        W, Wu = _get_matrices(g, directed, weights, weighted,
                              combine_weights=combine_weights)
    elif method == "barrat":
        # we need only the (potentially) directed matrices
        W = g.adjacency_matrix(weights=weights)
        A = g.adjacency_matrix()
    else:
        raise ValueError("`method` must be either 'barrat', 'continuous' "
                         "or 'normal' (identical, recommended options).")

    return _triplet_count_weighted(
        g, W, Wu, A, Au, method, directed, weights, nodes)


# ---------------------------------------------------------- #
# Overwrite binary clusterings with library-specific version #
# ---------------------------------------------------------- #

if nngt._config["backend"] == "networkx":
    from .nx_functions import (global_clustering_binary_undirected,
                               local_clustering_binary_undirected)

if nngt._config["backend"] == "igraph":
    from .ig_functions import (global_clustering_binary_undirected,
                               local_clustering_binary_undirected)

if nngt._config["backend"] == "graph-tool":
    from .gt_functions import (global_clustering_binary_undirected,
                               local_clustering_binary_undirected)


# -------------- #
# Tool functions #
# -------------- #


def _triangles_and_triplets(g, directed, weights, method, combine_weights,
                            nodes):
    ''' Return the triangles and triplets '''
    # weighted clustering
    W, Wu, A, Au = None, None, None, None
    triplets = None

    # check the method to get the relevant matrices
    if method == "continuous":
        W, Wu = _get_matrices(g, directed, weights, True, combine_weights,
                              exponent=2/3)

        Wtr, Wtru = _get_matrices(g, directed, weights, True, combine_weights)

        triplets = _triplet_count_weighted(
            g, Wtr, Wtru, A, Au, method, directed, weights, nodes)
    elif method == "onnela":
        W, Wu = _get_matrices(g, directed, weights, True, combine_weights,
                              exponent=1/3)

        # onnela uses the binary triplets
        triplets = triplet_count(g, nodes=nodes, directed=directed,
                                 weights=None)
    elif method == "barrat":
        # we need all matrices
        W, Wu = _get_matrices(g, directed, weights, True, combine_weights)
        A, Au = _get_matrices(g, directed, None, False, combine_weights)

        triplets = _triplet_count_weighted(
            g, W, Wu, A, Au, method, directed, weights, nodes)

    # get triangles and triplet strength
    triangles = _triangle_count(Wu, Au, method, weighted=True,
                                directed=directed, nodes=nodes)

    return triangles, triplets


def _triangle_count(matsym, adjsym, method, weighted, directed, nodes):
    '''
    (Un)weighted (un)directed triangle count.
    '''
    tr = None

    if not weighted:
        tr = (0.5*(adjsym*adjsym*adjsym).diagonal()).astype(int)
    elif method in ("continuous", "normal", "onnela"):
        tr = 0.5*(matsym*matsym*matsym).diagonal()
    elif method == "barrat":
        tr = 0.5*(matsym*adjsym*adjsym).diagonal()
    else:
        raise ValueError("Invalid `method`: '{}'".format(method))

    if nodes is None:
        return tr

    return tr[nodes]


def _triplet_count_weighted(g, mat, matsym, adj, adjsym, method, directed,
                            weights, nodes):
    '''
    triplet count, weighted only.
    '''
    tr = None

    if method == "normal":
        pass
    elif method == "continuous":
        if directed:
            sqmat = mat.sqrt()

            s2_sq_tot = np.square(sqmat.sum(axis=0).A1 + sqmat.sum(axis=1).A1)
            s_tot     = matsym.sum(axis=0).A1
            s_recip   = 2*(sqmat*sqmat).diagonal()

            tr = s2_sq_tot - s_tot - s_recip
        else:
            sqmat = matsym.sqrt()

            s2_sq = np.square(sqmat.sum(axis=0).A1)
            s     = mat.sum(axis=0).A1

            tr = 0.5*(s2_sq - s)
    elif method == "barrat":
        if directed:
            # specifc definition of the reciprocal strength from Clemente
            s_recip = 0.5*(mat*adj + adj*mat).diagonal()

            dtot = g.get_degrees("total")
            wmax = np.max(g.get_weights())
            stot = g.get_degrees("total", weights=weights) / wmax

            tr = stot*(dtot - 1) - 2*s_recip
        elif g.is_directed():
            d = adjsym.sum(axis=0).A1
            s = matsym.sum(axis=0).A1

            tr = 0.5*s*(d - 1)
        else:
            d = g.get_degrees()
            s = matsym.sum(axis=0).A1

            tr = 0.5*s*(d - 1)
    else:
        raise ValueError(
            "Invalid `method` for triplet count: '{}'".format(method))

    if nodes is None:
        return tr

    return tr[nodes]


def _get_matrices(g, directed, weights, weighted, combine_weights,
                  exponent=None):
    '''
    Return the relevant matrices:
    * W, Wu if weighted
    * A, Au otherwise
    '''
    if weighted:
        # weighted undirected
        W  = g.adjacency_matrix(weights=weights)
        W /= W.max()

        # remove potential self-loops
        W.setdiag(0.)

        if exponent is not None:
            W = W.power(exponent)

        Wu = W

        if g.is_directed():
            Wu = W + W.T

            if not directed:
                # find reciprocal edges
                if combine_weights != "sum":
                    recip = (W*W).nonzero()

                    if combine_weights == "mean":
                        Wu[recip] *= 0.5
                    elif combine_weights == "min":
                        Wu[recip] = mat[recip].minimum(mat.T[recip])
                    elif combine_weights == "max":
                        Wu[recip] = mat[recip].maximum(mat.T[recip])

        return W, Wu

    # binary undirected
    A  = g.adjacency_matrix()

    # remove potential self-loops
    A.setdiag(0.)

    Au = A

    if g.is_directed():
        Au = Au + Au.T

        if not directed:
            Au.data = np.ones(len(Au.data))

    return A, Au
