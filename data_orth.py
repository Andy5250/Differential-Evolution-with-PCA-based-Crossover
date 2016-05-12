# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 20:12:57 2015

@author: ZDZ
"""

import numpy as np
#from scipy import linalg


class OrthA:
    def __init__(self, n):
        self.n = n
#        A = np.random.rand(self.n, self.n)
#        A = linalg.orth(A)
#        A = linalg.qr(A)[0]
#        A = np.eye(self.n)
#        self.A = A
        self.A = np.eye(self.n)
#class OrthA:
#    def __init__(self, n):
#        self.n = n/5
#        M = np.random.rand(self.n, self.n)
#        M = linalg.qr(M)[0]
#        A = linalg.block_diag(M,M,M,M,M)
##        A = linalg.block_diag(M,M,M,M,M,M,M,M,M,M, M,M,M,M,M,M,M,M,M,M)
#        self.A = A