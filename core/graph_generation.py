#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Structure generators for GraphClass """

import numpy as np

from graph_tool import Graph
from graph_tool.generation import geometric_graph
from graph_tool.centrality import betweenness
from graph_tool.stats import *
from graph_tool.util import *



#
#---
# Generating graphs
#------------------------

n_MAXTESTS = 10000 # ensure that generation will finish

#-------------#
# Erdos-Renyi #
#-------------#

def erdos_renyi(dicProperties):
	np.random.seed()
	# initialize graph
	graphER = Graph()
	nNodes = 0
	nEdges = 0
	rDens = 0.0
	if "Nodes" in dicProperties.keys():
		nNodes = dicProperties["Nodes"]
		graphER.add_vertex(nNodes)
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
		graphER.add_vertex(nNodes)
		dicProperties["Nodes"] = nNodes
	# generate edges
	numTest,numCurrentEdges = 0,0
	while numCurrentEdges != nEdges and numTest < n_MAXTESTS:
		lstEdges = np.random.randint(0,nNodes,(nEdges-numCurrentEdges,2))
		graphER.add_edge_list(lstEdges)
		# remove loops and duplicate edges
		remove_self_loops(graphER)
		remove_parallel_edges(graphER)
		numCurrentEdges = graphER.num_edges()
		numTest += 1
	graphER.reindex_edges()
	nEdges = graphER.num_edges()
	rDens = nEdges / float(nNodes**2)
	# generate types
	rInhibFrac = dicProperties["InhibFrac"]
	lstTypesGen = np.random.uniform(0,1,nEdges)
	lstTypeLimit = np.full(nEdges,rInhibFrac)
	lstIsExcitatory = np.greater(lstTypesGen,lstTypeLimit)
	nExc = np.count_nonzero(lstIsExcitatory)
	epropType = graphER.new_edge_property("int",np.multiply(2,lstIsExcitatory)-np.repeat(1,nEdges)) # excitatory (True) or inhibitory (False)
	graphER.edge_properties["type"] = epropType
	# and weights
	if dicProperties["Weighted"]:
		lstWeights = dicGenWeights[dicProperties["Distribution"]](graphER,dicProperties,nEdges,nExc) # generate the weights
		epropW = graphER.new_edge_property("double",lstWeights) # crée la propriété pour stocker les poids
		graphER.edge_properties["weight"] = epropW
	return graphER

#------------#
# Free-scale #
#------------#

def gen_fs(dicProperties):
	np.random.seed()
	graphFS = Graph()
	# on définit la fraction des arcs à utiliser la réciprocité
	f = dicProperties["Reciprocity"]
	rFracRecip =  f/(2.0-f)
	# on définit toutes les grandeurs de base
	rInDeg = dicProperties["InDeg"]
	rOutDeg = dicProperties["OutDeg"]
	nNodes = 0
	nEdges = 0
	rDens = 0.0
	if "Nodes" in dicProperties.keys():
		nNodes = dicProperties["Nodes"]
		graphFS.add_vertex(nNodes)
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
		graphFS.add_vertex(nNodes)
		dicProperties["Nodes"] = nNodes
	# on définit le nombre d'arcs à créer
	nArcs = int(np.floor(rDens*nNodes**2)/(1+rFracRecip))
	# on définit les paramètres fonctions de probabilité associées F(x) = A x^{-tau}
	Ai = nArcs*(rInDeg-1)/(nNodes)
	Ao = nArcs*(rOutDeg-1)/(nNodes)
	# on définit les moyennes des distributions de pareto 2 = lomax
	rMi = 1/(rInDeg-2.)
	rMo = 1/(rOutDeg-2.)
	# on définit les trois listes contenant les degrés sortant/entrant/bidirectionnels associés aux noeuds i in range(nNodes)
	lstInDeg = np.random.pareto(rInDeg,nNodes)+1
	lstOutDeg = np.random.pareto(rOutDeg,nNodes)+1
	lstInDeg = np.floor(np.multiply(Ai/np.mean(lstInDeg), lstInDeg)).astype(int)
	lstOutDeg = np.floor(np.multiply(Ao/np.mean(lstOutDeg), lstOutDeg)).astype(int)
	# on génère les stubs qui vont être nécessaires et on les compte
	nInStubs = int(np.sum(lstInDeg))
	nOutStubs = int(np.sum(lstOutDeg))
	lstInStubs = np.zeros(np.sum(lstInDeg))
	lstOutStubs = np.zeros(np.sum(lstOutDeg))
	nStartIn = 0
	nStartOut = 0
	for vert in range(nNodes):
		nInDegVert = lstInDeg[vert]
		nOutDegVert = lstOutDeg[vert]
		for j in range(np.max([nInDegVert,nOutDegVert])):
			if j < nInDegVert:
				lstInStubs[nStartIn+j] += vert
			if j < nOutDegVert:
				lstOutStubs[nStartOut+j] += vert
		nStartOut+=nOutDegVert
		nStartIn+=nInDegVert
	# on vérifie qu'on a à peu près le nombre voulu d'edges
	while nInStubs*(1+rFracRecip)/float(nArcs) < 0.95 :
		vert = np.random.randint(0,nNodes)
		nAddInStubs = int(np.floor(Ai/rMi*(np.random.pareto(rInDeg)+1)))
		lstInStubs = np.append(lstInStubs,np.repeat(vert,nAddInStubs)).astype(int)
		nInStubs+=nAddInStubs
	while nOutStubs*(1+rFracRecip)/float(nArcs) < 0.95 :
		nAddOutStubs = int(np.floor(Ao/rMo*(np.random.pareto(rOutDeg)+1)))
		lstOutStubs = np.append(lstOutStubs,np.repeat(vert,nAddOutStubs)).astype(int)
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
	remove_self_loops(graphFS)
	remove_parallel_edges(graphFS)
	lstIsolatedVert = find_vertex(graphFS, graphFS.degree_property_map("total"), 0)
	graphFS.remove_vertex(lstIsolatedVert)
	graphFS.reindex_edges()
	nNodes = graphFS.num_vertices()
	nEdges = graphFS.num_edges()
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
	nEdgesTot = graphEDR.num_edges()
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
		remove_self_loops(graphEDR)
		remove_parallel_edges(graphEDR)
		nEdgesTot = graphEDR.num_edges()
		numTest += 1
	lstIsolatedVert = find_vertex(graphEDR, graphEDR.degree_property_map("total"), 0)
	graphEDR.remove_vertex(lstIsolatedVert)
	graphEDR.reindex_edges()
	nNodes = graphEDR.num_vertices()
	nEdges = graphEDR.num_edges()
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
