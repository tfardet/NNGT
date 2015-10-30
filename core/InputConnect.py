#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Input connectivity class """

import numpy as np
import scipy.sparse as ssp

from graph_measure import betweenness_list, degree_list
from random_gen import rand_int_trunc_exp



#
#---
# Connectivity class
#------------------------

class InputConnect:
	
	numConnectivities = 0

	#-----------#
	# Generator #
	#-----------#
	
	def __init__ (self, matrix=[], network=None, dicProp={}):
		self.__matConnect = matrix
		self.lstBetweenness = [] # heavy so store it if it's already been computed
		self.dicProperties= {}
		# counter
		InputConnect.numConnectivities += 1
		if dicProp == {}:
			self.dicProperties["Name"] = "Connectivity{}".format(Connectivity.numConnectivities)
			self.dicProperties["IODim"] = 0
			self.dicProperties["ReservoirDim"] = 0
			self.dicProperties["Type"] = None
		else:
			self.dicProperties = dicProp
			if network is not None:
				strCorr = dicProp["Type"]
				bAntiCorr = dicProp["AntiCorr"]
				self.gen_matrix(dicProp, network, strCorr, bAntiCorr)
				self.set_name()

	#---------------#
	# Set functions #
	#---------------#

	def set_name(self,name=""):
		''' set graph name '''
		if name != "":
			self.dicProperties["Name"] = name
		else:
			strName = self.dicProperties["Type"]
			tplIgnore = ("Type", "Name")
			for key,value in self.dicProperties.items():
				if key not in tplIgnore and (value.__class__ != dict):
					strName += '_' + key[0] + str(value)
			self.dicProperties["Name"] = strName

	def set_dimensions(self,dimensions):
		self.dicProperties["IODim"] = dimensions[0]
		self.dicProperties["ReservoirDim"] = dimensions[1]
		if self.__matConnect:
			self.__matConnect = np.zeros(dimensions)

	def set_matrix(npMatrix):
		self.__matConnect=npMatrix
		self.setDimensions(npMatrix.shape())

	def set_type(strType):
		self.dicProperties["Type"] = strType

	#---------------#
	# Get functions #
	#---------------#
	
	def get_name(self):
		return self.dicProperties["Name"]

	def get_dimensions(self):
		return self.dicProperties["IODim"], self.dicProperties["ReservoirDim"]

	def get_mat_connect(self):
		''' return the connectivity matrix in dense format '''
		return self.__matConnect

	def as_csr(self):
		''' return a copy of the connectivity matrix in csr format '''
		return ssp.csr_matrix(self.__matConnect)

	def get_list_neighbours(self):
		strNeighbours = "#{}\n".format(self.get_dimensions())
		idxLast = self.dicProperties["IODim"]-1
		for i in range(idxLast+1):
			strNeighbours += "{}".format(i)
			lstNeighbours = np.nonzero(self.__matConnect[i,:])[0]
			for j in range(len(lstNeighbours)):
				strNeighbours += " {};{}".format(lstNeighbours[j],self.__matConnect[i,lstNeighbours[j]])
			if i != idxLast:
				strNeighbours += "\n"
		return strNeighbours

	def get_type(self):
		return self.dicProperties["Type"]

	#-------------------#
	# Matrix generation #
	#-------------------#

	def gen_matrix(self, dicProp, network, strCorr, bAntiCorr=False):
		self.dicProperties.update(dicProp)
		nConnectPerInput = int(np.floor(self.dicProperties["ReservoirDim"] * self.dicProperties["Density"]))
		self.__matConnect = np.zeros((self.dicProperties["IODim"],self.dicProperties["ReservoirDim"]))
		# chose to which nodes each input neuron will connect
		matTargetNeurons = self.target_neurons(nConnectPerInput,network,strCorr,bAntiCorr).astype(int)
		# assign the weights based on the chosen weight rule
		self.assign_weights(network,matTargetNeurons)

	def target_neurons(self,nConnectPerInput,network,strCorr,bAntiCorr=False):
		numInput = self.dicProperties["IODim"]
		numNodesReservoir = self.dicProperties["ReservoirDim"]
		matTargetNeurons = np.zeros((numInput,nConnectPerInput))
		if strCorr == "Betweenness":
			self.lstBetweenness = betweenness_list(network)[0].a #get edge betweenness array
			lstSortedNodes = np.argsort(self.lstBetweenness)
			if not bAntiCorr:
				lstSortedNodes = lstSortedNodes[::-1]
			for i in range(numInput):
				lstRandIdx = rand_int_trunc_exp(0,numNodesReservoir,0.2,nConnectPerInput) # characteristic exponential decay is a fifth of the reservoir's size
				matTargetNeurons[i,:] = lstSortedNodes[lstRandIdx]
		elif "degree" in strCorr:
			# get the degree type
			idxDash = strCorr.find("-")
			strDegType = strCorr[:idxDash].lower()
			lstDegrees = degree_list(network,strDegType)
			# sort the nodes by their importance
			lstSortedNodes = np.argsort(lstDegrees)
			if not bAntiCorr:
				lstSortedNodes = lstSortedNodes[::-1]
			for i in range(numInput):
				lstRandIdx = rand_int_trunc_exp(0,numNodesReservoir,0.2,nConnectPerInput) # characteristic exponential decay is a fifth of the reservoir's size
				matTargetNeurons[i,:] = lstSortedNodes[lstRandIdx]
		else:
			matTargetNeurons = np.random.randint(0,numNodesReservoir,(numInput,nConnectPerInput))
		return matTargetNeurons.astype(int)

	def assign_weights(self,network,matTargetNeurons):
		numInput = self.dicProperties["IODim"]
		numNodesReservoir = self.dicProperties["ReservoirDim"]
		numInhib = numInput*numNodesReservoir*self.dicProperties["InhibFrac"]
		nRowLength = len(matTargetNeurons[0])
		numInhibPerRow = int(np.floor(nRowLength*self.dicProperties["InhibFrac"]))
		if self.dicProperties["Distribution"] == "Betweenness":
			if self.lstBetweenness == []:
				self.lstBetweenness = betwCentrality(network)[0].a
			rMaxBetw = self.lstBetweenness.max()
			rMinBetw = self.lstBetweenness.min()
			rMaxWeight = self.dicProperties["Max"]
			rMinWeight = self.dicProperties["Min"]
			for i in range(self.dicProperties["IODim"]):
				self.lstBetweenness = np.multiply(np.add(self.lstBetweenness,-rMinBetw+rMinWeight*rMaxBetw/(rMaxWeight-rMinWeight)),(rMaxWeight-rMinWeight)/rMaxBetw)
				self.__matConnect[i,matTargetNeurons[i]] = self.lstBetweenness[matTargetNeurons[i]] # does not take duplicate indices into account... never mind
			# generate the necessary inhibitory connections
			lstNonZero = np.nonzero(self.__matConnect)
			lstInhib = np.random.randint(0,len(lstNonZero),numInhib)
			self.__matConnect[lstInhib] = -self.__matConnect[lstInhib]
			rFactor = (self.dicProperties["Max"]-self.dicProperties["Min"])/(rMaxBetw-rMinBetw) # entre 0 et Max-Min
			self.__matConnect = np.add(np.multiply(self.__matConnect,rFactor),self.dicProperties["Min"]) # entre Min et Max
		elif self.dicProperties["Distribution"] == "Gaussian":
			for i in range(self.dicProperties["IODim"]):
				self.__matConnect[i,matTargetNeurons[i,:numInhibPerRow]] = -np.random.normal(self.dicProperties["MeanInhib"],self.dicProperties["VarInhib"],numInhibPerRow)
				self.__matConnect[i,matTargetNeurons[i,numInhibPerRow:]] = np.random.normal(self.dicProperties["MeanExc"],self.dicProperties["VarExc"],nRowLength-numInhibPerRow)
		elif self.dicProperties["Distribution"] == "Lognormal":
			for i in range(self.dicProperties["IODim"]):
				self.__matConnect[i,matTargetNeurons[i][:numInhibPerRow]] = -np.random.lognormal(self.dicProperties["LocationInhib"],self.dicProperties["ScaleInhib"],numInhibPerRow)
				self.__matConnect[i,matTargetNeurons[i][numInhibPerRow:]] = np.random.lognormal(self.dicProperties["LocationExc"],self.dicProperties["ScaleExc"],nRowLength-numInhibPerRow)
		else:
			None # I don't know what to do for the degree correlations yet

	#------------#
	# Destructor #
	#------------#

	def __del__(self):
		InputConnect.numConnectivities -= 1
		print('Connectivity died')
