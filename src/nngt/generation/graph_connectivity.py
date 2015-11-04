#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Connectivity generators for GraphClass """

import numpy as np

from .. import GraphClass, SpatialGraph, GraphObject
from ..lib.utils import (delete_self_loops, delete_parallel_edges,
                         adjacency_matrix, make_spatial)
from ..lib.connect_tools import *



n_MAXTESTS = 10000 # ensure that generation will finish


#
#---
# Erdos-Renyi
#------------------------

def erdos_renyi(nodes=None, density=0.1, edges=-1, avg_deg=-1.,
                reciprocity=-1., directed=True, multigraph=False,
                name="ER", shape=None, positions=None, initial_graph=None):
    """
    Generate a random graph as defined by Erdos and Renyi but with a
    reciprocity that can be chosen.

    Parameters
    ----------
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    density : double, optional (default: 0.1)
        Structural density given by `edges` / `nodes`:math:`^2`.
    edges : int (optional)
        The number of edges between the nodes
    avg_deg : double, optional
        Average degree of the neurons given by `edges` / `nodes`.
    reciprocity : double, optional (default: -1 to let it free)
        Fraction of edges that are bidirectional (only for
        directed graphs -- undirected graphs have a reciprocity of 1 by
        definition)
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.core.Shape`, optional (default: None)
        Shape of the neurons' environment
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space
    initial_graph : :class:`GraphClass` or :class:`NeuralNetwork`, optional (default: None)
        Initial graph whose nodes are to be connected.

    Returns
    -------
    graph_er : :class:`~nngt.GraphClass`, or subclass
        A new generated graph or the modified `initial_graph`.

    Notes
    -----
    `nodes` is required unless `initial_graph` is provided.
    If an `initial_graph` is provided, all preexistant edges in the
    object will be deleted before the new connectivity is implemented.
    """

    np.random.seed()
    nodes = nodes if initial_graph is None else initial_graph.node_nb()
    edges = _compute_connections(nodes, density, edges, avg_deg)
    frac_recip = 0.
    pre_recip_edges = edges
    if not directed:
        edges = int(edges/2)
    elif reciprocity > 0.:
        frac_recip = reciprocity/(2.0-reciprocity)
        pre_recip_edges = edges / (1+frac_recip)

    # generate graph object
    
    graph_obj_er = (GraphObject(nodes, directed=directed) if
                    initial_graph == None else initial_graph.graph)
    graph_obj_er.clear_edges()
    num_test,num_current_edges = 0,0
    
    while num_current_edges != pre_recip_edges and num_test < n_MAXTESTS:
        ia_edges = np.random.randint(0,nodes,(pre_recip_edges-num_current_edges,2))
        graph_obj_er.add_edge_list(ia_edges)
        delete_self_loops(graph_obj_er)
        if not multigraph:
            delete_parallel_edges(graph_obj_er)
        num_current_edges = graph_obj_er.edge_nb()
        num_test += 1
        
    if directed and reciprocity > 0.:
        coo_adjacency = adjacency_matrix(graph_er).tocoo()
        while num_current_edges != edges and num_test < n_MAXTESTS:
            ia_indices = np.random.randint(0, num_current_edges,
                                           edges-num_current_edges)
            ia_edges = np.array([coo_adjacency.col[ia_indices],
                                 coo_adjacency.row[ia_indices]])
            graph_er.add_edge_list(ia_edges)
            delete_self_loops(graph_er)
            if not multigraph:
                delete_parallel_edges(graph_er)
            num_current_edges = graph_er.edge_nb()
            num_test += 1
    
    graph_er = (GraphClass(name=name, graph=graph_obj_er) if
                initial_graph is None else initial_graph)
    if shape is not None:
        make_spatial(graph_er, shape, positions)
    return graph_er


#
#---
# Scale-free models
#------------------------

def random_free_scale(in_exp, out_exp, nodes=None, density=0.1, edges=-1,
                      avg_deg=-1, reciprocity=0., directed=True,
                      multigraph=False, name="ER", shape=None, positions=None,
                      initial_graph=None):
    """
    @todo
    Generate a free-scale graph of given reciprocity and otherwise
    devoid of correlations.

    Parameters 
    ----------
    nodes : int, optional (default: None)
        The number of nodes in the graph.
    density: double, optional (default: 0.1)
        Structural density given by `edges` / (`nodes`*`nodes`).
    edges : int (optional)
        The number of edges between the nodes
    avg_deg : double, optional
        Average degree of the neurons given by `edges` / `nodes`.
    directed : bool, optional (default: True)
        Whether the graph is directed or not.
    multigraph : bool, optional (default: False)
        Whether the graph can contain multiple edges between two
        nodes.

    Returns
    -------
    graph_fs : :class:`~nngt.GraphClass`
    
    Notes
    -----
    `nodes` is required unless `initial_graph` is provided.
    """
    np.random.seed()
    nodes = nodes if initial_graph is None else initial_graph.node_nb() 
    edges = _compute_connections(nodes, density, edges, avg_deg)
    frac_recip = 0.
    pre_recip_edges = edges
    if not directed:
        edges = int(edges/2)
    elif reciprocity > 0.:
        frac_recip = reciprocity/(2.0-reciprocity)
        pre_recip_edges = edges / (1+frac_recip)

    # define necessary constants

    # for probability functions F(x) = c * x^{-tau} (for in- and out-degrees)
    c_in = edges*(in_exp-1)/(nodes)
    c_out = edges*(out_exp-1)/(nNodes)
    # for the average of the pareto 2 = lomax distribution
    avg_in = 1/(in_exp-2.)
    avg_out = 1/(out_exp-2.)
    # lists containing the in/out-degrees for nodes i
    lst_in_deg = np.random.pareto(in_exp,nodes)+1
    lst_out_deg = np.random.pareto(out_exp,nodes)+1
    lst_in_deg = np.floor(np.multiply(c_in/np.mean(lst_in_deg),
                                      lst_in_deg)).astype(int)
    lst_out_deg = np.floor(np.multiply(c_out/np.mean(lst_out_deg),
                                       lst_out_deg)).astype(int)

    # count and generate the stubs (half edges)
    num_in_stubs = int(np.sum(lst_in_deg))
    num_out_stubs = int(np.sum(lst_out_deg))
    lstInStubs = np.zeros(num_in_stubs)
    lstOutStubs = np.zeros(num_out_stubs)
    nStartIn = 0
    nStartOut = 0
    for node in range(nNodes):
        nInDegVert = lstInDeg[node]
        nOutDegVert = lstOutDeg[node]
        for j in range(np.max([nInDegVert,nOutDegVert])):
            if j < nInDegVert:
                lstInStubs[nStartIn+j] += node
            if j < nOutDegVert:
                lstOutStubs[nStartOut+j] += node
        nStartOut+=nOutDegVert
        nStartIn+=nInDegVert
    # on vérifie qu'on a à peu près le nombre voulu d'edges
    while nInStubs*(1+rFracRecip)/float(nArcs) < 0.95 :
        node = np.random.randint(0,nNodes)
        nAddInStubs = int(np.floor(Ai/rMi*(np.random.pareto(rInDeg)+1)))
        lstInStubs = np.append(lstInStubs,np.repeat(node,nAddInStubs)).astype(int)
        nInStubs+=nAddInStubs
    while nOutStubs*(1+rFracRecip)/float(nArcs) < 0.95 :
        nAddOutStubs = int(np.floor(Ao/rMo*(np.random.pareto(rOutDeg)+1)))
        lstOutStubs = np.append(lstOutStubs,np.repeat(node,nAddOutStubs)).astype(int)
        nOutStubs+=nAddOutStubs
    # on s'assure d'avoir le même nombre de in et out stubs (1.13 is an experimental correction)
    nMaxStubs = int(1.13*(2.0*nArcs)/(2*(1+rFracRecip)))
    if nInStubs > nMaxStubs and nOutStubs > nMaxStubs:
        np.random.shuffle(lstInStubs)
        np.random.shuffle(lstOutStubs)
        lstOutStubs.resize(nMaxStubs)
        lstInStubs.resize(nMaxStubs)
        nOutStubs = nInStubs = nMaxStubs
    elif nInStubs < nOutStubs:
        np.random.shuffle(lstOutStubs)
        lstOutStubs.resize(nInStubs)
        nOutStubs = nInStubs
    else:
        np.random.shuffle(lstInStubs)
        lstInStubs.resize(nOutStubs)
        nInStubs = nOutStubs
    # on crée le graphe, les noeuds et les stubs
    nRecip = int(np.floor(nInStubs*rFracRecip))
    nEdges = nInStubs + nRecip +1
    # les stubs réciproques
    np.random.shuffle(lstInStubs)
    np.random.shuffle(lstOutStubs)
    lstInRecip = lstInStubs[0:nRecip]
    lstOutRecip = lstOutStubs[0:nRecip]
    lstEdges = np.array([np.concatenate((lstOutStubs,lstInRecip)),np.concatenate((lstInStubs,lstOutRecip))]).astype(int)
    # add edges
    graphFS.add_edge_list(np.transpose(lstEdges))
    delete_self_loops(graphFS)
    delete_parallel_edges(graphFS)
    graphFS.reindex_edges()
    nNodes = graphFS.node_nb()
    nEdges = graphFS.edge_nb()
    rDens = nEdges / float(nNodes**2)
    # generate types
    rInhibFrac = dicProperties["InhibFrac"]
    lstTypesGen = np.random.uniform(0,1,nEdges)
    lstTypeLimit = np.full(nEdges,rInhibFrac)
    lstIsExcitatory = np.greater(lstTypesGen,lstTypeLimit)
    nExc = np.count_nonzero(lstIsExcitatory)
    epropType = graphFS.new_edge_property("int",np.multiply(2,lstIsExcitatory)-np.repeat(1,nEdges)) # excitatory (True) or inhibitory (False)
    graphFS.edge_properties["type"] = epropType
    # and weights
    if dicProperties["Weighted"]:
        lstWeights = dicGenWeights[dicProperties["Distribution"]](graphFS,dicProperties,nEdges,nExc) # generate the weights
        epropW = graphFS.new_edge_property("double",lstWeights) # crée la propriété pour stocker les poids
        graphFS.edge_properties["weight"] = epropW
    return graphFS

def price_free_scale(m, c=None, gamma=1, nodes=None, directed=True,
                     seed_graph=None, multigraph=False, name="ER", shape=None,
                     positions=None, initial_graph=None):
    """
    Generate a Price graph model (Barabasi-Albert if undirected).
    @todo: make the algorithm.
	
	Parameters 
	----------
	nodes : int, optional (default: None)
		The number of nodes in the graph.
    m : int
        The number of edges each new node will make.
    c : double
        Constant added to the probability of a vertex receiving an edge.
    gamma : double
        Preferential attachment power.
	directed : bool, optional (default: True)
		Whether the graph is directed or not.
	multigraph : bool, optional (default: False)
		Whether the graph can contain multiple edges between two
		nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.core.Shape`, optional (default: None)
        Shape of the neurons' environment
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space
    initial_graph : :class:`GraphClass` or :class:`NeuralNetwork`, optional (default: None)
        Initial graph whose nodes are to be connected.
	
	Returns
	-------
	graph_price : :class:`~nngt.GraphClass` or subclass.
    
    Notes
    -----
    `nodes` is required unless `initial_graph` is provided.
	"""
    np.random.seed()
    nodes = nodes if initial_graph is None else initial_graph.node_nb()
    #~ c = c if c is not None else 0 if directed else 1
    
    graph_obj_price = GraphObject.to_graph_object(
                        price_network(nodes,m,c,gamma,directed,seed_graph))
    if initial_graph is not None:
        initial_graph.graph = graph_obj_price
    
    graph_price = (GraphClass(name=name, graph=graph_obj_price) if
                initial_graph == None else initial_graph)
    if shape is not None:
        make_spatial(graph_price, shape, positions)
    return graph_price

#
#---
# Small-world models
#------------------------

def newman_watts(coord_nb, proba_shortcut, nodes=None, density=0.1, edges=-1,
                  avg_deg=-1., directed=True, multigraph=False, name="ER",
                  shape=None, positions=None, initial_graph=None):
    """
    Generate a small-world graph using the Newman-Watts algorithm.
    @todo: generate the edges of a circular graph to not replace the graph
    of the `initial_graph` and implement chosen reciprocity.
	
	Parameters 
	----------
	nodes : int, optional (default: None)
		The number of nodes in the graph.
    coord_nb : int
        The number of neighbours for each node on the initial topological 
        lattice.
    proba_shortcut : double
        Probability of adding a new random (shortcut) edge for each existing 
        edge on the initial lattice.
	density: double, optional (default: 0.1)
		Structural density given by `edges` / (`nodes`*`nodes`).
	edges : int (optional)
		The number of edges between the nodes
	avg_deg : double, optional
		Average degree of the neurons given by `edges` / `nodes`.
	directed : bool, optional (default: True)
		Whether the graph is directed or not.
	multigraph : bool, optional (default: False)
		Whether the graph can contain multiple edges between two
		nodes.
    name : string, optional (default: "ER")
        Name of the created graph.
    shape : :class:`~nngt.core.Shape`, optional (default: None)
        Shape of the neurons' environment
    positions : :class:`numpy.ndarray`, optional (default: None)
        A 2D or 3D array containing the positions of the neurons in space
    initial_graph : :class:`GraphClass` or :class:`NeuralNetwork`, optional (default: None)
        Initial graph whose nodes are to be connected.
	
	Returns
	-------
	graph_nw : :class:`~nngt.GraphClass` or subclass
    
    Notes
    -----
    `nodes` is required unless `initial_graph` is provided.
	"""
    np.random.seed()
    nodes = nodes if initial_graph is None else initial_graph.node_nb() 
    edges = _compute_connections(nodes, density, edges, avg_deg,
                           (coord_nb,proba_shortcut))
    #~ frac_recip = 0.
    #~ pre_recip_edges = edges
    #~ if not directed:
        #~ edges = int(edges/2)
    #~ elif reciprocity > 0.:
        #~ frac_recip = reciprocity/(2.0-reciprocity)
        #~ pre_recip_edges = edges / (1+frac_recip)
    
    graph_obj_nw = GraphObject.to_graph_object(circular_graph(
                        nodes,k=coord_nb/2,directed=directed))
    if initial_graph is not None:
        initial_graph.graph = graph_obj_nw
    
    num_test,num_current_edges = 0,graph_obj_nw.edge_nb()
    while num_current_edges != edges and num_test < n_MAXTESTS:
        ia_edges = np.random.randint(0,nodes, (edges-num_current_edges,2))
        graph_obj_nw.add_edge_list(ia_edges)
        if not multigraph:
            delete_parallel_edges(graph_obj_nw)
        delete_self_loops(graph_obj_nw)
        num_current_edges = graph_obj_nw.edge_nb()
        num_test += 1
        
    graph_nw = (GraphClass(name=name, graph=graph_obj_nw) if
                initial_graph == None else initial_graph)
    if shape is not None:
        make_spatial(graph_nw, shape, positions)
    return graph_nw

#---------------------------#
# Exponential Distance Rule #
#---------------------------#

def gen_edr(dicProperties):
	np.random.seed()
	# on définit toutes les grandeurs de base
	rRho2D = dicProperties["Rho"]
	rLambda = dicProperties["Lambda"]
	nNodes = 0
	nEdges = 0
	rDens = 0.0
	if "Nodes" in dicProperties.keys():
		nNodes = dicProperties["Nodes"]
		if "Edges" in dicProperties.keys():
			nEdges = dicProperties["Edges"]
			rDens = nEdges / float(nNodes**2)
			dicProperties["Density"] = rDens
		else:
			rDens = dicProperties["Density"]
			nEdges = int(np.floor(rDens*nNodes**2))
			dicProperties["Edges"] = nEdges
	else:
		nEdges = dicProperties["Edges"]
		rDens = dicProperties["Density"]
		nNodes = int(np.floor(np.sqrt(nEdges/rDens)))
		dicProperties["Nodes"] = nNodes
	rSideLength = np.sqrt(nNodes/rRho2D)
	rAverageDistance = np.sqrt(2)*rSideLength / 3
	# generate the positions of the neurons
	lstPos = np.array([np.random.uniform(0,rSideLength,nNodes),np.random.uniform(0,rSideLength,nNodes)])
	lstPos = np.transpose(lstPos)
	numDesiredEdges = int(float(rDens*nNodes**2))
	graphEDR,pos = geometric_graph(lstPos,0)
	graphEDR.set_directed(True)
	graphEDR.vertex_properties["pos"] = pos
	# test edges building on random neurons
	nEdgesTot = graphEDR.edge_nb()
	numTest = 0
	while nEdgesTot < numDesiredEdges and numTest < n_MAXTESTS:
		nTests = int(np.minimum(1.1*np.ceil(numDesiredEdges-nEdgesTot)*np.exp(np.divide(rAverageDistance,rLambda)),1e7))
		lstVertSrc = np.random.randint(0,nNodes,nTests)
		lstVertDest = np.random.randint(0,nNodes,nTests)
		lstDist = np.linalg.norm(lstPos[lstVertDest]-lstPos[lstVertSrc],axis=1)
		lstDist = np.exp(np.divide(lstDist,-rLambda))
		lstCreateEdge = np.random.uniform(size=nTests)
		lstCreateEdge = np.greater(lstDist,lstCreateEdge)
		nEdges = np.sum(lstCreateEdge)
		if nEdges+nEdgesTot > numDesiredEdges:
			nEdges = numDesiredEdges - nEdgesTot
			lstVertSrc = lstVertSrc[lstCreateEdge][:nEdges]
			lstVertDest = lstVertDest[lstCreateEdge][:nEdges]
			lstEdges = np.array([lstVertSrc,lstVertDest]).astype(int)
		else:
			lstEdges = np.array([lstVertSrc[lstCreateEdge],lstVertDest[lstCreateEdge]]).astype(int)
		graphEDR.add_edge_list(np.transpose(lstEdges))
		# make graph simple and connected
		delete_self_loops(graphEDR)
		delete_parallel_edges(graphEDR)
		nEdgesTot = graphEDR.edge_nb()
		numTest += 1
	graphEDR.reindex_edges()
	nNodes = graphEDR.node_nb()
	nEdges = graphEDR.edge_nb()
	rDens = nEdges / float(nNodes**2)
	# generate types
	rInhibFrac = dicProperties["InhibFrac"]
	lstTypesGen = np.random.uniform(0,1,nEdges)
	lstTypeLimit = np.full(nEdges,rInhibFrac)
	lstIsExcitatory = np.greater(lstTypesGen,lstTypeLimit)
	nExc = np.count_nonzero(lstIsExcitatory)
	epropType = graphEDR.new_edge_property("int",np.multiply(2,lstIsExcitatory)-np.repeat(1,nEdges)) # excitatory (True) or inhibitory (False)
	graphEDR.edge_properties["type"] = epropType
	# and weights
	if dicProperties["Weighted"]:
		lstWeights = dicGenWeights[dicProperties["Distribution"]](graphEDR,dicProperties,nEdges,nExc) # generate the weights
		epropW = graphEDR.new_edge_property("double",lstWeights) # crée la propriété pour stocker les poids
		graphEDR.edge_properties["weight"] = epropW
	return graphEDR


#
#---
# Generating weights
#------------------------

def gaussian_weights(graph,dicProperties,nEdges,nExc):
	rMeanExc = dicProperties["MeanExc"]
	rMeanInhib = dicProperties["MeanInhib"]
	rVarExc = dicProperties["VarExc"]
	rVarInhib = dicProperties["VarInhib"]
	lstWeightsExc = np.random.normal(rMeanExc,rVarExc,nExc)
	lstWeightsInhib = np.random.normal(rMeanInhib, rVarInhib, nEdges-nExc)
	lstWeights = np.concatenate((np.absolute(lstWeightsExc), np.absolute(lstWeightsInhib)))
	return lstWeights

def lognormal_weights(graph,dicProperties,nEdges,nExc):
	rScaleExc = dicProperties["ScaleExc"]
	rLocationExc = dicProperties["LocationExc"]
	rScaleinHib = dicProperties["ScaleInhib"]
	rLocationInhib = dicProperties["LocationInhib"]
	lstWeightsExc = np.random.lognormal(rLocationExc,rScaleExc,nExc)
	lstWeightsInhib = np.random.lognormal(rLocationInhib,rScaleInhib,nEdges-nExc)
	lstWeights = np.concatenate((np.absolute(lstWeightsExc), np.absolute(lstWeightsInhib)))
	return lstWeights

def betweenness_correlated_weights(graph,dicProperties,nEdges,nExc):
	lstWeights = np.zeros(nEdges)
	rMin = dicProperties["Min"]
	rMax = dicProperties["Max"]
	vpropBetw,epropBetw = betweenness(graph)
	arrBetw = epropBetw.a.copy()
	arrLogBetw = np.log10(arrBetw)
	rMaxLogBetw = arrLogBetw.max()
	rMinLogBetw = arrLogBetw.min()
	arrLogBetw = -5 + 2 * (arrLogBetw - rMinLogBetw ) / (rMaxLogBetw - rMinLogBetw)
	arrBetw = np.exp(np.log(10) * arrLogBetw)
	rMaxBetw = arrBetw.max()
	rMinBetw = arrBetw.min()
	lstWeights = np.multiply(arrBetw-rMinBetw,rMax/rMaxBetw) + rMin
	return lstWeights

def degree_correlated_weights(graph,dicProperties,nEdges,nExc):
	lstWeights = np.repeat(1, nEdges)
	return lstWeights


dicGenWeights = {	"Gaussian": gaussian_weights,
					"Lognormal": lognormal_weights,
					"Betweenness": betweenness_correlated_weights,
					"Degree": degree_correlated_weights	}
