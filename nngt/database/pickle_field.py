# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# nngt/database/pickle_field.py

""" Pickle field for peewee """

import sqlite3
import pickle

from playhouse.fields import BlobField


class PickledField(BlobField):

    def python_value(self, value):
        if isinstance(value, (bytearray, sqlite3.Binary)):
            value = bytes(value)
        return pickle.loads(value)

    def db_value(self, value):
        return sqlite3.Binary(pickle.dumps(value, 2))
