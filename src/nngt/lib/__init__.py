"""
========================
Lib module
========================

==================== =========================================================
Errors
==============================================================================
InvalidArgument		Argument passed to the function are not valid
==================== =========================================================

"""

from __future__ import absolute_import

from .errors import *
from .decorators import *
from .utils import *
from .connect_tools import *
from .lil_object import ObjectLil

#~ __all__ = ['InvalidArgument', 'valid_arguments']
