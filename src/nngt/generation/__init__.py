"""
Functions that generates the underlying connectivity of graphs, as well
as the synaptic properties (weight/strength and delay).

Content
=======

"""

from __future__ import absolute_import

from .graph_connectivity import *


depends = ['graph_tool','NeuralNetwork']

__all__ = [
	'erdos_renyi',
	'random_free_scale',
	'price_free_scale',
	'newman_watts',
    'connect_neural_types',
    'connect_neural_groups',
	'distance_rule',
	'gaussian_eprop',
	'lognormal_eprop',
	'uniform_eprop',
	'custom_eprop',
	'correlated_fixed_eprop',
	'correlated_proba_eprop'
]
