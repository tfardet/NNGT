#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
===============
Plotting module
===============
"""

import matplotlib
matplotlib.use('GTK3Agg')
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

from ..globals import s_glib


# module import

from .custom_plt import palette
from .plt_properties import degree_distribution, betweenness_distribution
if s_glib == 'graph_tool':
    from .plt_networks import draw_network
else:
    warning.warn("Graph drawing is only available with graph_tool at the \
moment. When using {}, all graph drawing functions will be \
disabled".format(s_glib))
