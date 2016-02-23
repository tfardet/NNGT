"""
Functions that generates the underlying connectivity of graphs, as well
as the synaptic properties (weight/strength and delay).

Content
=======

"""

from __future__ import absolute_import

from .graph_connectivity import *


__all__ = [
    'connect_neural_groups',
    'connect_neural_types',
	'distance_rule',
	'erdos_renyi',
    'fixed_degree',
	'random_scale_free',
	'price_scale_free',
	'newman_watts'
]
