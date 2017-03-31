"""
Functions that generates the underlying connectivity of graphs, as well
as the synaptic properties (weight/strength and delay).

Content
=======

"""

from .graph_connectivity import *
from .connect_tools import _compute_connections


__all__ = [
    'connect_neural_groups',
    'connect_neural_types',
	'distance_rule',
	'erdos_renyi',
    'fixed_degree',
    'gaussian_degree',
	'random_scale_free',
	'price_scale_free',
	'newman_watts'
]
