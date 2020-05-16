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

__all__ = [
	"global_clustering",
    "local_clustering",
    "triad_count",
    "triangle_count",
]


def global_clustering(g, weights=None, method="continuous", directed=True):
    '''
    Returns the undirected global clustering coefficient.

    This corresponds to the ratio of triangles to the number of triads.
    For directed and weighted cases, see definitions of generalized triangles
    and triads in the associated functions below.

    Parameters
    ----------
    g : :class:`~nngt.Graph`
        Graph to analyze.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.

    References
    ----------
    .. [gt-global-clustering] :gtdoc:`clustering.global_clustering`
    .. [ig-global-clustering] :igdoc:`transitivity_undirected`
    .. [nx-global-clustering] :nxdoc:`algorithms.cluster.transitivity`

    See also
    --------
    :func:`~nngt.analysis.triad_count`
    :func:`~nngt.analysis.triangle_count`
    '''
    directed = g.is_directed()
    weighted = weights not in (False, None)

    if not directed and not weighted:
        return undirected_binary_global_clustering(g)

    # weighted or directed cases


def undirected_binary_clustering(g, nodes=None):
    r'''
    Returns the undirected local clustering coefficient of some `nodes`.

    .. math::

        C_i = \frac{A^3_{ii}}{d_i(d_i - 1)} = \frac{\Delta_i}{T_i}

    with :math:`A` the adjacency matrix, :math:`d_i` the degree of node
    :math:`i`, :math:`\Delta_i` is the number of triangles, and :math:`T_i` is
    the number of triads to which :math:`i` belongs.

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
    triads    = triad_count(g, weights=None, nodes=nodes, directed=False)
    triangles = triangle_count(g, weights=None, nodes=nodes, directed=False)

    return triads / triangles


def local_clustering(g, weights=None, nodes=None, method="continuous",
                     directed=True, combine_weights="mean"):
    '''
    Local (weighted directed) clustering coefficient of the nodes.

    If no weights are requested and the graph is undirected, returns the
    undirected binary clustering.

    For all weighted cases, the weights are assumed to be positive and they are
    normalized to dimensionless values between 0 and 1 through a division by
    the highest weight.

    The default `method` for weighted networks is based on a modification of
    the proposal in [Zhang2005] with:

    .. math::

        C_i = \frac{W^3_{ii}}{\left(s^{[\frac{1}{2}]}_i\right)^2 - s_i}

    for undirected networks, with :math:`\tilde{W}` the adjacency matrix
    (normalized to have its highest weight set to 1 if weighted), :math:`s_i`
    the strength of node :math:`i`, and
    :math:`s^{[\frac{1}{2}]}_i = \sum_k \sqrt{w_{ik}}`
    the strength associated to the matrix
    :math:`W^{[\frac{1}{2}]} = \{\sqrt{w_{ij}}\}`.

    For directed networks, we used the total clustering defined in
    [Fagiolo2007], hence the second equation becomes:

    .. math::

        C_i = \frac{\frac{1}{2}\left(W + W^T\right)^3_{ii}}{
                    \left(s^{[\frac{1}{2}]}_i\right)^2 - 2s^{\leftrightarrow}_i
                          - s_i}

    with :math:`s^{\leftrightarrow} = \sum_k \sqrt{w_{ik}w_{ki}}` the
    reciprocal strength (associated to reciprocal connections).

    Contrary to 'barrat' and 'onella', this method displays *all* following
    properties:

    * fully continuous (no jump in clustering when weights go to zero),
    * equivalent to binary clustering when all weights are 1,
    * equivalence between no-edge and zero-weight edge cases,
    * normalized (always between zero and 1).

    Parameters
    ----------
    g : :class:`~nngt.Graph` object
        Graph to analyze.
    weights : bool or str, optional (default: binary edges)
        Whether edge weights should be considered; if ``None`` or ``False``
        then use binary edges; if ``True``, uses the 'weight' edge attribute,
        otherwise uses any valid edge attribute required.
    nodes : array-like container with node ids, optional (default = all nodes)
        Nodes for which the local clustering coefficient should be computed.
    method : str, optional (default: 'continuous')
        Method used to compute the weighted clustering, either 'barrat'
        [Barrat2004], 'continuous', or 'onella' [Saramaki2007].
    directed : bool, optional (default: True)
        Whether to compute the directed clustering if the graph is directed.
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
    .. [Fagiolo2007] Fagiolo. Clustering in Complex Directed Networks.
        Phys. Rev. E 2007, 76, (2), 026107. :doi:`10.1103/PhysRevE.76.026107`,
        :arxiv:`physics/0612169`.
    .. [Saramaki2007] Saramäki, Kivelä, Onnela, Kaski, Kertész. Generalizations
        of the Clustering Coefficient to Weighted Complex Networks.
        Phys. Rev. E 2007, 75 (2), 027105. :doi:`10.1103/PhysRevE.75.027105`,
        :arxiv:`cond-mat/0608670`.
    .. [Zhang2005] Zhang, Horvath. A General Framework for Weighted Gene
        Co-Expression Network Analysis. Statistical Applications in Genetics
        and Molecular Biology 2005, 4 (1). :doi:`10.2202/1544-6115.1128`,
        `pdf <https://dibernardo.tigem.it/files/papers/2008/
        zhangbin-statappsgeneticsmolbio.pdf>`_.

    See also
    --------
    :func:`undirected_binary_clustering`
    :func:`global_clustering`
    '''
    # check directivity and weights
    directed = g.is_directed()

    weighted = weights not in (None, False)

    # undirected binary clustering uses the library method
    if not directed and not weighted:
        return undirected_local_clustering(g, weights=weights, nodes=nodes)

    # directed or weighted clustering
    mat = g.adjacency_matrix(weights=weights)

    if weighted:
        weights = 'weight' if weights is True else weights
        wmax = np.max(g.edge_attributes[weights])
        mat /= wmax

    matsym = mat + mat.T if directed else mat

    numer, denom = None, None

    if method == "continuous":
        sqmat = mat.sqrt() if weighted else mat

        numer = 0.5*(matsym*matsym*matsym).diagonal()

        # strength for denom
        if directed:
            s_cyc1 = np.square(sqmat.sum(axis=0).A1 + sqmat.sum(axis=1).A1)
            s_cyc2 = matsym.sum(axis=0).A1
            s_recp = 2*(sqmat*sqmat).diagonal()

            denom = s_cyc1 - s_cyc2 - s_recp
        else:
            s_cyc1 = np.square(sqmat.sum(axis=0).A1)
            s_cyc2 = mat.sum(axis=0).A1
            denom  = s_cyc1 - s_cyc2
    elif method == "onnela":
        matsym = mat + mat.T

        numer = 0.5*np.power((matsym*matsym*matsym).diagonal(), 1/3)

        # degrees for denom
        dtot = g.get_degrees("total")

        adj = g.adjacency_matrix(weights=None)

        d_recp = (adj*adj).diagonal() if directed else 0.

        denom = dtot(dtot - 1) - 2*d_recp
    elif method == "barrat":
        adj = g.adjacency_matrix(weights=None)
        s_recp = (mat*adj).diagonal() if directed else 0.

        adj = adj + adj.T

        numer = 0.25*(matsym*adj*adj + adj*adj*matsym)

        # degrees and strength
        dtot = g.get_degrees("total")
        stot = g.get_degrees("total", weights=weights)

        denom = stot*(dtot - 1) - 2*s_recp

    denom[numer == 0] = 1.

    return numer/denom


def triad_count(g, nodes=None, directed=True, weights=None,
                method="normal", combine_weights="mean"):
    '''
    Returns the number or strength of triads for each node.

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
        Method used to compute the weighted triads, either 'normal', where
        the edge weights are directly used, or the definitions used for
        weighted clustering coefficients, 'barrat' [Barrat2004] or
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
        Number or weight of triads to which each node belongs.
    '''
    directed *= g.is_directed()
    weighted  = weights not in (False, None)

    # simple binary cases
    if not weighted or method == "onella":
        # undirected
        if not g.is_directed():
            deg = g.get_degrees(nodes=nodes)

            return deg*(deg - 1)

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
        W, Wu = _get_matrices(g, weights, weighted,
                              combine_weights=combine_weights)
    elif method == "onella":
        
    if method == "barrat":
        raise ValueError("`method` must be either 'barrat', 'continuous' "
                         "or 'normal' (identical, recommended option), "
                         "or 'onella'.")

    return _triad_count_weighted(g, W, Wu, A, Au, method, directed, nodes)


def triangle_count(g, nodes=None, directed=True, weights=None,
                   method="normal", combine_weights="mean"):
    '''
    Returns the number or strength of triangles 

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
        weighted clustering: 'barrat' [Barrat2004], 'continuous', or 'onella'
        [Saramaki2007].
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

    # get relevant matrices (use directed=False to get both dir/undir mat)
    _, matsym = _get_matrices(g, weights, weighted, False, combine_weights)

    adjsym = None

    if method == "barrat" and weighted:
        _, adjsym = _get_matrices(g, None, False, False, combine_weights)

    return _triangle_count_weighted_directed(matsym, adjsym, method, weighted,
                                             directed, nodes)


# -------------- #
# Tool functions #
# -------------- #


def _triad_count_weighted(g, mat, matsym, adj, adjsym, method, directed,
                          nodes):
    '''
    Triad count, weighted only.
    '''
    tr = None

    if method == "normal":
        pass
    elif method == "continuous":
        sqmat = mat.sqrt()

        if directed:
            s_cyc1 = np.square(sqmat.sum(axis=0).A1 + sqmat.sum(axis=1).A1)
            s_cyc2 = matsym.sum(axis=0).A1
            s_recp = 2*(sqmat*sqmat).diagonal()

            tr = s_cyc1 - s_cyc2 - s_recp
        else:
            s_cyc1 = np.square(sqmat.sum(axis=0).A1)
            s_cyc2 = mat.sum(axis=0).A1
            denom  = s_cyc1 - s_cyc2
    elif method == "barrat":
        s_recp = (mat*adj).diagonal() if directed else 0.

        dtot = g.get_degrees("total")
        stot = g.get_degrees("total", weights=weights)

        tr = stot*(dtot - 1) - 2*s_recp
    else:
        raise ValueError(
            "Invalid `method` for triad count: '{}'".format(method))

    if nodes is None:
        return tr

    return tr[nodes]


def _triangle_count(matsym, adjsym, method, weighted, directed, nodes):
    '''
    (Un)weighted (un)directed triangle count.
    '''
    tr = None

    if not weighted and not directed:
        tr = (matsym*matsym*matsym).diagonal()
    elif not weighted or method in ("continuous", "normal"):
        tr = 0.5*(matsym*matsym*matsym).diagonal()
    elif method == "onnela":
        tr = 0.5*np.power((matsym*matsym*matsym).diagonal(), 1/3)
    elif method == "barrat":
        tr = 0.25*(matsym*adjsym*adjsym + adjsym*adjsym*matsym)
    else:
        raise ValueError("Invalid `method`: '{}'".format(method))

    if nodes is None:
        return tr

    return tr[nodes]


def _get_matrices(g, weights, weighted, combine_weights):
    '''
    Return the relevant matrices:
    * W, Wu if weighted
    * A, Au otherwise
    '''
    if weighted:
        # weighted undirected
        W  = g.adjacency_matrix(weights=weights)
        Wu = W

        if g.is_directed():
            Wu = mat + mat.T

            # find reciprocal edges
            if combine_weights != "sum":
                recip = (Wu*Wu).nonzero()

                if combine_weights == "mean":
                    Wu[recip] *= 0.5
                elif combine_weights == "min":
                    Wu[recip] = mat[recip].minimum(mat.T[recip])
                elif combine_weights == "max":
                    Wu[recip] = mat[recip].maximum(mat.T[recip])
        return W, Wu

    # binary undirected
    A  = g.adjacency_matrix()
    Au = A

    if g.is_directed():
        Au = Au + Au.T
        Au.data = np.ones(len(Au.data))

    return A, Au
