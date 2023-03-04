# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2015-2023 Tanguy Fardet
# SPDX-License-Identifier: GPL-3.0-or-later
# extra/format_headers.py

"""
Header generation for all python files

NB: this file must be run from inside the NNGT folder via
``python extra/format_headers.py``.
"""

import datetime
import os
import re


# check current working directory
cwd = os.path.abspath(os.path.curdir)

assert cwd.endswith("NNGT"), \
    "Script must be called via `python extra/format_headers.py`."


def update_headers(filetype, header, year, ignore=None):
    ''' Update file headers '''
    ignore = ignore or "$^"

    for (dirpath, dirnames, filenames) in os.walk(cwd):
        for f in filenames:
            abspath = os.path.join(dirpath, f)

            if f.endswith(filetype) and not re.search(ignore, abspath):
                with open(abspath, "r") as rf:
                    data = rf.readlines()

                with open(abspath, "w") as wf:
                    start = dirpath.find("NNGT")
                    start += 6 if dirpath.endswith("/") else 5

                    filename = os.path.join(dirpath[start:], f)

                    wf.write(header.format(filename=filename, year=year))

                    is_header = True

                    for line in data:
                        starter = "^#" if ".py" in filetype else "^(..\n|\s+)"

                        if not re.match(starter, line) or not is_header:
                            wf.write(line)
                            is_header = False


''' Python header '''

# load header string
headerfile = os.path.join(cwd, "extra/header.txt")

with open(headerfile, "r") as h:
    header = h.read()

# get year
year = datetime.datetime.now().year

# ignored files
ignore = "decorator.py|linksourcecode.py|extlinks_fancy.py|nngt/geometry/|" \
         "nngt/plot/chord_diag/|setup.py"

update_headers(".py", header, year, ignore)


''' Documentation header '''

headerfile = os.path.join(cwd, "extra/header_rst.txt")

with open(headerfile, "r") as h:
    header = h.read()

update_headers(".rst.in", header, year)
update_headers(".rst", header, year)
