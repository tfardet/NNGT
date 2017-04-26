#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
Functions for plotting graphs and graph properties.

.. note ::
For now, graph plotting is only supported when using the
`graph_tool <https://graph-tool.skewed.de>`_ library.

Content
=======
"""

import matplotlib as _mpl

import nngt as _nngt

if _nngt._config["backend"] is not None:
    _mpl.use(_nngt._config["backend"])
else:
    sav_backend = _mpl.get_backend()
    backends = [ 'GTK3Agg', 'Qt4Agg', 'Qt5Agg' ]
    keep_trying = True
    while backends and keep_trying:
        try:
            backend = backends.pop()
            _mpl.use(backend)
            keep_trying = False
        except:
            _mpl.use(sav_backend)


import warnings as _warn
_warn.filterwarnings("ignore", module="matplotlib")


# module import

from .custom_plt import palette
from .animations import Animation2d, AnimationNetwork
from .plt_properties import *
from . import plt_properties as _plt_prop


__all__ = [
    "Animation2d",
    "AnimationNetwork",
    "palette",
]
__all__.extend(_plt_prop.__all__)


if _nngt._config["graph_library"] == 'graph-tool':
    from .plt_networks import draw_network
    __all__.append("draw_network")
else:
    _warn.warn("Graph drawing is only available with graph_tool at the "
               "moment. As {} is currently being used, all graph drawing "
               "functions will be disabled.".format(
               _nngt._config["graph_library"]))
