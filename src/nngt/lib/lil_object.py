#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Object lil_matrix """

import bisect


#
#---
# ObjectLil
#------------------------

class ObjectLil:
    
    def __init__(self, shape):
        M, N = shape
        self.shape = (M,N)
        self.rows = [ [] for _ in range(M) ]
        self.data = [ [] for _ in range(M) ]
    
    def __getitem__(self, index):
        """
        Return the element(s) index=int or (int i, int j), slices are not
        supported.
        """
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            if ((isinstance(i, int) or isinstance(i, np.integer)) and
                    (isinstance(j, int) or isinstance(j, np.integer))):
                try:
                    idx_j = self.rows[i].index(j)
                    return self.data[i][idx_j]
                except ValueError:
                    return None
        else:
            return self.data[index]

    def __setitem__(self, index, x):
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            if ((isinstance(i, int) or isinstance(i, np.integer)) and
                    (isinstance(j, int) or isinstance(j, np.integer))):
                if i >= self.shape[0] or j >= self.shape[1]:
                    raise IndexError("LilObject index out of range")
                else:
                    idx_j = bisect.bisect(self.rows[i],j)
                    self.rows[i].insert(idx_j,j)
                    self.data[i].insert(idx_j,x)
        else:
            raise ArgumentError("Index should be a (int i, int j) tuple")
