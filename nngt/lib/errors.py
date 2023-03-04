# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/lib/errors.py

""" Errors for the NGT module """


class InvalidArgument(ValueError):

    ''' Error raised when an argument is invalid. '''

    pass


def not_implemented(*args, **kwargs):
    return NotImplementedError("Not implemented.")
