#!/usr/bin/env python
#-*- coding:utf-8 -*-

''' Remove cython header from _pygrowth.cpp '''


def clean_header(filename):
    with open(filename, "r+") as f:
        data = [line for line in f]

        # get the header
        i, idx_end = 0, 0
        searching = True
        while not idx_end and i < len(data):
            if "END: Cython Metadata */" in data[i]:
                idx_end = i + 1
            i += 1

        # remove header and overwrite
        output = "".join(data[idx_end:])
        f.seek(0)
        f.write(output)
        f.truncate()
