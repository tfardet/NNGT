# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/lib/db_tools.py

""" Database tools for NNGT """

import platform as pfm
from collections import namedtuple


try:
    import psutil
except ImportError:

    class MockPsutil:
        ''' Mock class for psutil '''

        @staticmethod
        def cpu_count():
            return -1

        @staticmethod
        def virtual_memory():
            attr = ('total', 'available', 'percent', 'used', 'free', 'active',
                    'inactive', 'buffers', 'cached')
            VM = namedtuple('VM', attr)
            return VM( ( -1 for _ in attr ) )

    psutil = MockPsutil
