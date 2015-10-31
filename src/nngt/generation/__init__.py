"""
GENERATION MODULE
=================

Functions that generates the underlying connectivity of graphs, as well
as the synaptic properties (weight/strength and delay).

====================== =================================================
Functions (connectivity)
========================================================================
erdos_renyi				Random graph studied by Erdos and Renyi
random_free_scale		An uncorrelated free-scale graph
price_free_scale		Price's network (Barabasi-Albert if undirected)
newman_watts			Newman-Watts small world network
distance_rule			Distance-dependent connection probability
====================== =================================================

====================== =================================================
Functions (weigths/delays)
========================================================================
gaussian_eprop			Random gaussian distribution
lognormal_eprop			Random lognormal distribution
uniform_eprop			Random uniform distribution
custom_eprop			User defined distribution
correlated_fixed_eprop	Computed from an edge property
correlated_proba_eprop	Randomly drawn, correlated to an edge property
====================== =================================================

Summary
-------

.. autosummary::
	:toctree:	
	nngt.generation.erdos_renyi
	nngt.generation.random_free_scale

Contents
--------

"""

from __future__ import absolute_import
from .graph_connectivity import *
#~ from .synaptic_properties import *
#~ from .utilities import *

depends = ['graph_tool']

__all__ = [
	'erdos_renyi',
	'random_free_scale',
	'price_free_scale',
	'newman_watts',
	'distance_rule',
	'gaussian_eprop',
	'lognormal_eprop',
	'uniform_eprop',
	'custom_eprop',
	'correlated_fixed_eprop',
	'correlated_proba_eprop'
]
