#-*- coding:utf-8 -*-
#
# analysis/clustering.py
#
# This file is part of the NNGT project, a graph-library for standardized and
# and reproducible graph analysis: generate and analyze networks with your
# favorite graph library (graph-tool/igraph/networkx) on any platform, without
# any change to your code.
# Copyright (C) 2015-2021 Tanguy Fardet
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Tools for directed/weighted clsutering analysis """

import numpy as np

import nngt
from nngt.lib import nonstring_container
from nngt.lib.graph_helpers import _get_matrices


__all__ = [
	"global_clustering",
    "global_clustering_binary_undirected",
    "local_closure",
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
                      mode="total", combine_weights="mean"):
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
        [Barrat2004]_, 'continuous' [Fardet2021]_, 'onnela' [Onnela2005]_, or
        'zhang' [Zhang2005]_.
    mode : str, optional (default: "total")
        Type of clustering to use for directed graphs, among "total", "fan-in",
        "fan-out", "middleman", and "cycle" [Fagiolo2007]_.
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
    .. [Fagiolo2007] Fagiolo. Clustering in Complex Directed Networks.
        Phys. Rev. E 2007, 76 (2), 026107. :doi:`10.1103/PhysRevE.76.026107`,
        :arxiv:`physics/0612169`.
    .. [Zhang2005] Zhang, Horvath. A General Framework for Weighted Gene
        Co-Expression Network Analysis. Statistical Applications in Genetics
        and Molecular Biology 2005, 4 (1). :doi:`10.2202/1544-6115.1128`,
        `PDF <https://dibernardo.tigem.it/files/papers/2008/
        zhangbin-statappsgeneticsmolbio.pdf>`_.
    .. [Fardet2021] Fardet, Levina. Weighted directed clustering:
        interpretations and requirements for heterogeneous, inferred, and
        measured networks. 2021. :arxiv:`2105.06318`.

    See also
    --------
    :func:`~nngt.analysis.triplet_count`
    :func:`~nngt.analysis.triangle_count`
    '''
    assert method in ("barrat", "continuous", "onnela", "zhang"), \
        "Unknown method '{}'".format(method)

    # check directivity and weights
    directed *= g.is_directed()
    weighted  = weights not in (False, None)

    if not directed and not weighted:
        return global_clustering_binary_undirected(g)
    elif not weighted:
        # directed clustering
        triangles = triangle_count(g, mode=mode)
        triplets  = triplet_count(g, mode=mode)

        return np.sum(triangles) / np.sum(triplets)

    triangles, triplets = _triangles_and_triplets(g, directed, weights, method,
                                                  mode, combine_weights, None)

    return np.sum(triangles) / np.sum(triplets)


def local_closure(g, directed=True, weights=None, method=None,
                  mode="cycle-out", combine_weights="mean"):
    r'''
    Compute the local closure for each node, as defined in [Yin2019]_ as the
    fraction of 2-walks that are closed.

    For undirected binary or weighted adjacency matrices
    :math:`W = \{ w_{ij} \}`, the normal (or Zhang-like) definition is given
    by:

    .. math::

        H_i^0 = \frac{\sum_{j\neq k} w_{ij} w_{jk} w_{ki}}
                     {\sum_{j\neq k\neq i} w_{ij}w_{jk}}
              = \frac{W^3_{ii}}{\sum_{j \neq i} W^2_{ij}}

    While a continuous version of the local closure is also proposed as:

    .. math::

        H_i = \frac{\sum_{j\neq k} \sqrt[3]{w_{ij} w_{jk} w_{ki}}^2}
                   {\sum_{j\neq k\neq i} \sqrt{w_{ij}w_{jk}}}
            = \frac{\left( W^{\left[ \frac{2}{3} \right]} \right)_{ii}^3}
                   {\sum_{j \neq i} \left( W^{\left[ \frac{1}{2} \right]}
                                    \right)^2_{ij}}

    with :math:`W^{[\alpha]} = \{ w^\alpha_{ij} \}`.

    Directed versions of the local closure where defined as follow for a node
    :math:`i` connected to nodes :math:`j` and :math:`k`:

    * "cycle-out" is given by the pattern [(i, j), (j, k), (k, i)],
    * "cycle-in" is given by the pattern [(k, j), (j, i), (i, k)],
    * "fan-in" is given by the pattern [(k, j), (j, i), (k, i)],
    * "fan-out" is given by the pattern [(i, j), (j, k), (i, k)].

    See [Fardet2021]_ for more details.

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
        Method used to compute the weighted clustering, either 'normal'/'zhang'
        or 'continuous'.
    mode : str, optional (default: "circle-out")
        Type of clustering to use for directed graphs, among "circle-out",
        "circle-in", "fan-in", or "fan-out".
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
    .. [Yin2019] Yin, Benson, and Leskovec. The Local Closure Coefficient: A
        New Perspective On Network Clustering. Proceedings of the Twelfth ACM
        International Conference on Web Search and Data Mining 2019, 303-311.
        :doi:`10.1145/3289600.3290991`, `PDF <https://www.cs.cornell.edu/~arb/
        papers/closure-coefficients-WSDM-2019.pdf>`_.
    .. [Fardet2021] Fardet, Levina. Weighted directed clustering:
        interpretations and requirements for heterogeneous, inferred, and
        measured networks. 2021. :arxiv:`2105.06318`.
    '''
    directed *= g.is_directed()
    weighted  = weights not in (False, None)

    mat, numer, denom = None, None, None

    if not directed and g.is_directed():
        _, mat = _get_matrices(g, directed, weights, weighted, combine_weights,
                               normed=True)
    else:
        mat = g.adjacency_matrix(weights=weights).astype(float)
        mat /= mat.max()
        mat.setdiag(0)

    mat2, mat3 = None, None

    if directed:
        # set correct matrix
        if mode.endswith("-in"):
            mat = mat.T

        if method == "continuous" and weights is not None:
            sqmat = mat.sqrt()
            cbmat = mat.power(2/3)

            mat2 = sqmat*sqmat

            if mode in ("cycle-in", "cycle-out"):
                mat3 = cbmat*cbmat*cbmat
            elif mode in ("fan-in", "fan-out"):
                mat3 = cbmat*cbmat*cbmat.T
            else:
                raise ValueError("Unknown `mode`: '" + mode + "'.'")
        elif method in ("normal", "zhang", None):
            mat2 = mat*mat

            if mode in ("cycle-in", "cycle-out"):
                mat3 = mat2*mat
            elif mode in ("fan-in", "fan-out"):
                mat3 = mat2*mat.T
            else:
                raise ValueError("Unknown `mode`: '" + mode + "'.'")
        else:
            raise ValueError("Unknown `method`: '" + method + "'.'")
    else:
        # undirected
        if method == "continuous" and weights is not None:
            sqmat = mat.sqrt()
            cbmat = mat.power(2/3)

            mat2 = sqmat*sqmat
            mat3 = cbmat*cbmat*cbmat
        elif method in ("normal", "zhang", None):
            mat2 = mat*mat
            mat3 = mat2*mat
        else:
            raise ValueError("Unknown `method`: '" + method + "'.'")

    numer = mat3.diagonal()
    denom = mat2.sum(axis=1).A1 - mat2.diagonal()

    denom[denom == 0] = 1

    return numer / denom


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

    if nonstring_container(triangles):
        triplets[triangles == 0] = 1
    elif triangles == 0:
        return 0

    return triangles / triplets


def local_clustering(g, nodes=None, directed=True, weights=None,
                     method="continuous", mode="total", combine_weights="mean"):
    r'''
    Local (weighted directed) clustering coefficient of the nodes, ignoring
    self-loops.

    If no weights are requested and the graph is undirected, returns the
    undirected binary clustering.

    For all weighted cases, the weights are assumed to be positive and they are
    normalized to dimensionless values between 0 and 1 through a division by
    the highest weight.

    The default `method` for weighted networks is the continuous definition
    [Fardet2021]_ and is defined as:

    .. math::

        C_i = \frac{\sum_{jk} \sqrt[3]{w_{ij} w_{ik} w_{jk}}}
                   {\sum_{j\neq k} \sqrt{w_{ij} w_{ik}}}
            = \frac{\left(W^{\left[\frac{2}{3}\right]}\right)^3_{ii}}
                   {\left(s^{\left[\frac{1}{2}\right]}_i\right)^2 - s_i}

    for undirected networks, with
    :math:`W = \{ w_{ij}\} = \tilde{W} / \max(\tilde{W})` the normalized
    weight matrix, :math:`s_i` the normalized strength of node :math:`i`, and
    :math:`s^{[\frac{1}{2}]}_i = \sum_k \sqrt{w_{ik}}` the strength associated
    to the matrix :math:`W^{[\frac{1}{2}]} = \{\sqrt{w_{ij}}\}`.

    For directed networks, we used the total clustering defined in
    [Fagiolo2007]_ by default, hence the second equation becomes:

    .. math::

        C_i = \frac{\frac{1}{2}\left(W^{\left[\frac{2}{3}\right]}
                    + W^{\left[\frac{2}{3}\right],T}\right)^3_{ii}}
                   {\left(s^{\left[\frac{1}{2}\right]}_i\right)^2
                    - 2s^{\leftrightarrow}_i - s_i}

    with :math:`s^{\leftrightarrow} = \sum_k \sqrt{w_{ik}w_{ki}}` the
    reciprocal strength (associated to reciprocal connections).

    For the other modes, see the generalized definitions in [Fagiolo2007]_.

    Contrary to 'barrat' and 'onnela' [Saramaki2007]_, this method displays
    *all* following properties:

    * fully continuous (no jump in clustering when weights go to zero),
    * equivalent to binary clustering when all weights are 1,
    * equivalence between no-edge and zero-weight edge cases,
    * normalized (always between zero and 1).

    Using either 'continuous' or 'zhang' is usually recommended for weighted
    graphs, see the discussion in [Fardet2021]_ for details.

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
        [Barrat2004]_/[Clemente2018]_, 'continuous' [Fardet2021]_, 'onnela'
        [Onnela2005]_/[Fagiolo2007]_, or 'zhang' [Zhang2005]_.
    mode : str, optional (default: "total")
        Type of clustering to use for directed graphs, among "total", "fan-in",
        "fan-out", "middleman", and "cycle" [Fagiolo2007]_.
    combine_weights : str, optional (default: 'mean')
        How to combine the weights of reciprocal edges if the graph is directed
        but `directed` is set to False. It can be:

        * "min": the minimum of the edge attribute values will be used for the
          new edge.
        * "max": the maximum of the edge attribute values will be used for the
          new edge.
        * "mean": the mean of the edge attribute values will be used for the
          new edge.
        * "sum": equivalent to mean due to weight normalization.

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
    .. [Fardet2021] Fardet, Levina. Weighted directed clustering:
        interpretations and requirements for heterogeneous, inferred, and
        measured networks. 2021. :arxiv:`2105.06318`.

    See also
    --------
    :func:`undirected_binary_clustering`
    :func:`global_clustering`
    '''
    # check directivity and weights
    directed *= g.is_directed()
    weighted  = weights not in (None, False)

    triplets, triangles = None, None

    if not directed and not weighted:
        # undirected binary clustering uses the library method
        return local_clustering_binary_undirected(g, nodes=nodes)
    elif not weighted:
        # directed clustering
        triangles = triangle_count(g, nodes=nodes, mode=mode)
        triplets  = triplet_count(g, nodes, mode=mode).astype(float)
    else:
        triangles, triplets = _triangles_and_triplets(
            g, directed, weights, method, mode, combine_weights, nodes)

    if nonstring_container(triplets):
        triplets[triangles == 0] = 1
    elif triangles == 0:
        return 0

    return triangles / triplets


def triangle_count(g, nodes=None, directed=True, weights=None,
                   method="normal", mode="total", combine_weights="mean"):
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
        weighted clustering: 'barrat' [Barrat2004]_, 'continuous', 'onnela'
        [Onnela2005]_, or 'zhang' [Zhang2005]_.
    mode : str, optional (default: "total")
        Type of clustering to use for directed graphs, among "total", "fan-in",
        "fan-out", "middleman", and "cycle" [Fagiolo2007]_.
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

    References
    ----------
    .. [Barrat2004] Barrat, Barthelemy, Pastor-Satorras, Vespignani. The
        Architecture of Complex Weighted Networks. PNAS 2004, 101 (11).
        :doi:`10.1073/pnas.0400087101`.
    .. [Fagiolo2007] Fagiolo. Clustering in Complex Directed Networks.
        Phys. Rev. E 2007, 76, (2), 026107. :doi:`10.1103/PhysRevE.76.026107`,
        :arxiv:`physics/0612169`.
    .. [Onnela2005] Onnela, Saramäki, Kertész, Kaski. Intensity and Coherence
        of Motifs in Weighted Complex Networks. Phys. Rev. E 2005, 71 (6),
        065103. :doi:`10.1103/physreve.71.065103`, :arxiv:`cond-mat/0408629`.
    .. [Zhang2005] Zhang, Horvath. A General Framework for Weighted Gene
        Co-Expression Network Analysis. Statistical Applications in Genetics
        and Molecular Biology 2005, 4 (1). :doi:`10.2202/1544-6115.1128`,
        `PDF <https://dibernardo.tigem.it/files/papers/2008/
        zhangbin-statappsgeneticsmolbio.pdf>`_.
    '''
    directed *= g.is_directed()
    weighted  = weights not in (False, None)

    exponent = None

    if method == "onnela":
        exponent = 1/3
    elif method == "continuous":
        exponent = 2/3

    # get relevant matrices (use directed=False to get both dir/undir mat)
    mat, matsym = _get_matrices(
        g, directed, weights, weighted, combine_weights, exponent=exponent,
        normed=True)

    # if unweighted, adj is mat, adjsym is matsym
    adj, adjsym = mat, matsym

    # for barrat, we need both weighted and binary matrices
    if method == "barrat" and weighted:
        adj, adjsym = _get_matrices(g, directed, None, False, combine_weights)

    return _triangle_count(mat, matsym, adj, adjsym, method, mode, weighted,
                           directed, nodes)


def triplet_count(g, nodes=None, directed=True, weights=None,
                  method="normal", mode="total", combine_weights="mean"):
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
        weighted clustering coefficients, 'barrat' [Barrat2004]_,
        'continuous', 'onnela' [Onnela2005]_, or 'zhang' [Zhang2005]_.
    mode : str, optional (default: "total")
        Type of clustering to use for directed graphs, among "total", "fan-in",
        "fan-out", "middleman", and "cycle" [Fagiolo2007]_.
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

    References
    ----------
    .. [Barrat2004] Barrat, Barthelemy, Pastor-Satorras, Vespignani. The
        Architecture of Complex Weighted Networks. PNAS 2004, 101 (11).
        :doi:`10.1073/pnas.0400087101`.
    .. [Fagiolo2007] Fagiolo. Clustering in Complex Directed Networks.
        Phys. Rev. E 2007, 76, (2), 026107. :doi:`10.1103/PhysRevE.76.026107`,
        :arxiv:`physics/0612169`.
    .. [Zhang2005] Zhang, Horvath. A General Framework for Weighted Gene
        Co-Expression Network Analysis. Statistical Applications in Genetics
        and Molecular Biology 2005, 4 (1). :doi:`10.2202/1544-6115.1128`,
        `PDF <https://dibernardo.tigem.it/files/papers/2008/
        zhangbin-statappsgeneticsmolbio.pdf>`_.
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

                if nodes is None:
                    deg = adjsym.sum(axis=0).A1
                else:
                    deg = adjsym.sum(axis=0).A1[nodes]
            else:
                deg = g.get_degrees(nodes=nodes)

            if nodes is None or nonstring_container(nodes):
                return (0.5*deg*(deg - 1)).astype(int)

            return 0.5*deg*(deg - 1)

        # directed
        if mode in ("total", "cycle", "middleman"):
            adj = g.adjacency_matrix()

            d_recip = (adj*adj).diagonal()

            if nodes is not None:
                d_recip = d_recip[nodes]

            din  = g.get_degrees("in", nodes=nodes)
            dout = g.get_degrees("out", nodes=nodes)

            if mode == "total":
                dtot = din + dout

                return dtot*(dtot - 1) - 2*d_recip

            return din*dout - d_recip
        else:
            assert mode in ("fan-in", "fan-out"), \
                "Unknown mode '{}'".format(mode)

            deg = g.get_degrees(mode[4:], nodes=nodes)

            return deg*(deg - 1)

    # check method for weighted
    W, Wu, A, Au = None, None, None, None

    if method in ("continuous", "normal", "zhang"):
        # we need only the weighted matrices
        W, Wu = _get_matrices(g, directed, weights, weighted,
                              combine_weights=combine_weights, normed=True)
    elif method == "barrat":
        # we need only the (potentially) directed matrices
        W = g.adjacency_matrix(weights=weights)
        A = g.adjacency_matrix()
    else:
        raise ValueError("`method` must be either 'barrat', 'onnela', "
                         "'zhang', or 'continuous'/'normal' (identical "
                         "options).")

    return _triplet_count_weighted(
        g, W, Wu, A, Au, method, mode, directed, weights, nodes)


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


def _triangles_and_triplets(g, directed, weights, method, mode,
                            combine_weights, nodes):
    ''' Return the triangles and triplets '''
    # weighted clustering
    W, Wu, A, Au = None, None, None, None
    triplets = None

    # check the method to get the relevant matrices
    if method == "continuous":
        W, Wu = _get_matrices(g, directed, weights, True, combine_weights,
                              exponent=2/3, normed=True)

        Wtr, Wtru = _get_matrices(g, directed, weights, True, combine_weights,
                                  normed=True)

        triplets = _triplet_count_weighted(
            g, Wtr, Wtru, A, Au, method, mode, directed, weights, nodes)
    if method == "zhang":
        W, Wu = _get_matrices(g, directed, weights, True, combine_weights,
                              normed=True)

        triplets = _triplet_count_weighted(
            g, W, Wu, A, Au, method, mode, directed, weights, nodes)
    elif method == "onnela":
        W, Wu = _get_matrices(g, directed, weights, True, combine_weights,
                              exponent=1/3, normed=True)

        # onnela uses the binary triplets
        triplets = triplet_count(g, nodes=nodes, directed=directed,
                                 mode=mode, weights=None)
    elif method == "barrat":
        # we need all matrices
        W, Wu = _get_matrices(g, directed, weights, True, combine_weights,
                              normed=True)
        A, Au = _get_matrices(g, directed, None, False, combine_weights)

        triplets = _triplet_count_weighted(
            g, W, Wu, A, Au, method, mode, directed, weights, nodes)

    # get triangles and triplet strength
    triangles = _triangle_count(W, Wu, A, Au, method, mode, weighted=True,
                                directed=directed, nodes=nodes)

    return triangles, triplets


def _triangle_count(mat, matsym, adj, adjsym, method, mode, weighted, directed,
                    nodes):
    '''
    (Un)weighted (un)directed triangle count.
    '''
    tr = None

    if method == "barrat":
        if mode == "total":
            tr = 0.5*(matsym*adjsym*adjsym).diagonal()
        elif mode == "cycle":
            tr = 0.5*(mat*adj*adj + mat.T*adj.T*adj.T).diagonal()
        elif mode == "middleman":
            tr = 0.5*(mat.T*adj*adj.T + mat*adj.T*adj).diagonal()
        elif mode == "fan-in":
            tr = 0.5*(mat.T*adjsym*adj).diagonal()
        elif mode == "fan-out":
            tr = 0.5*(mat*adjsym*adj.T).diagonal()
        else:
            raise ValueError("Unknown mode ''.".format(mode))
    else:
        if not weighted:
            mat, matsym = adj, adjsym
        elif method not in ("continuous", "zhang", "normal", "onnela"):
            raise ValueError("Invalid `method`: '{}'".format(method))

        if mode == "total":
            tr = 0.5*(matsym*matsym*matsym).diagonal()
        elif mode == "cycle":
            tr = (mat*mat*mat).diagonal()
        elif mode == "middleman":
            tr = (mat*mat.T*mat).diagonal()
        elif mode == "fan-in":
            tr = (mat.T*mat*mat).diagonal()
        elif mode == "fan-out":
            tr = (mat*mat*mat.T).diagonal()
        else:
            raise ValueError("Unknown mode ''.".format(mode))

    if nodes is None:
        return tr

    return tr[nodes]


def _triplet_count_weighted(g, mat, matsym, adj, adjsym, method, mode,
                            directed, weights, nodes):
    '''
    triplet count, weighted only.
    '''
    tr = None

    if method == "normal":
        pass
    elif method == "continuous":
        if directed:
            sqmat = mat.sqrt()

            if mode == "total":
                s2_sq_tot = np.square(sqmat.sum(axis=0).A1 +
                                      sqmat.sum(axis=1).A1)
                s_tot     = mat.sum(axis=0).A1 + mat.sum(axis=1).A1
                s_recip   = 2*(sqmat*sqmat).diagonal()

                tr = s2_sq_tot - s_tot - s_recip
            elif mode in ("cycle", "middleman"):
                s_sq_out = sqmat.sum(axis=0).A1
                s_sq_in  = sqmat.sum(axis=1).A1
                s_recip  = (sqmat*sqmat).diagonal()

                tr = s_sq_in*s_sq_out - s_recip
            elif mode in ("fan-in", "fan-out"):
                axis  = 0 if mode == "fan-in" else 1
                s2_sq = np.square(sqmat.sum(axis=axis).A1)
                sgth  = mat.sum(axis=axis).A1

                tr = s2_sq - sgth
            else:
                raise ValueError("Unknown mode ''.".format(mode))
        else:
            sqmat = matsym.sqrt()

            s2_sq = np.square(sqmat.sum(axis=0).A1)
            s     = matsym.sum(axis=0).A1

            tr = 0.5*(s2_sq - s)
    elif method == "zhang":
        if directed:
            mat2 = mat.power(2)

            if mode == "total":
                s2_sq_tot = np.square(mat.sum(axis=0).A1 +
                                      mat.sum(axis=1).A1)
                s_tot     = mat2.sum(axis=0).A1 + mat2.sum(axis=1).A1
                s_recip   = 2*(mat*mat).diagonal()

                tr = s2_sq_tot - s_tot - s_recip
            elif mode in ("cycle", "middleman"):
                s_sq_out = mat.sum(axis=0).A1
                s_sq_in  = mat.sum(axis=1).A1
                s_recip  = (mat*mat).diagonal()

                tr = s_sq_in*s_sq_out - s_recip
            elif mode in ("fan-in", "fan-out"):
                axis  = 0 if mode == "fan-in" else 1
                s2_sq = np.square(mat.sum(axis=axis).A1)
                sgth  = mat2.sum(axis=axis).A1

                tr = s2_sq - sgth
            else:
                raise ValueError("Unknown mode ''.".format(mode))
        else:
            mat2 = matsym.power(2)

            s2_sq = np.square(matsym.sum(axis=0).A1)
            s     = mat2.sum(axis=0).A1

            tr = 0.5*(s2_sq - s)
    elif method == "barrat":
        if directed:
            # specifc definition of the reciprocal strength from Clemente
            if mode == "total":
                s_recip = 0.5*(mat*adj + adj*mat).diagonal()

                dtot = g.get_degrees("total")
                wmax = np.max(g.get_weights())
                stot = g.get_degrees("total", weights=weights) / wmax

                tr = stot*(dtot - 1) - 2*s_recip
            elif mode in ("cycle", "middleman"):
                s_recip = 0.5*(mat*adj + adj*mat).diagonal()
                s_in    = mat.sum(axis=0).A1
                s_out   = mat.sum(axis=1).A1
                d_in    = g.get_degrees("in")
                d_out   = g.get_degrees("out")

                tr = 0.5*(s_in*d_out + s_out*d_in) - s_recip
            elif mode in ("fan-in", "fan-out"):
                axis = 0 if mode == "fan-in" else 1
                sgth = mat.sum(axis=axis).A1
                deg  = g.get_degrees(mode[4:])

                tr = sgth*(deg - 1)
            else:
                raise ValueError("Unknown mode ''.".format(mode))
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
