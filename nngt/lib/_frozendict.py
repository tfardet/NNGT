# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/lib/_frozendict.py

""" Read-only dict """

class _frozendict(dict):

    def __init__(self, *args, **kwargs):
        self.message = 'Cannot set items of frozendict.'
        if "message" in kwargs:
            self.message = kwargs["message"]
            del kwargs["message"]
        super(_frozendict, self).__init__(*args, **kwargs)

    def __setitem__(self, name, value):
        '''
        Prevent item setting
        '''
        raise RuntimeError(self.message)
