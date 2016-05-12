# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 20:12:57 2015

@author: ZDZ
"""

import numpy as np
from scipy import linalg


#class OrthA:
#    def __init__(self):
#        M = np.random.rand(25,25)
#        M = linalg.qr(M)[0]
#        A = linalg.block_diag(M,M,M,M)
#        print A
##        A = linalg.block_diag(M,M,M,M,M,M,M,M,M,M, M,M,M,M,M,M,M,M,M,M)
#        self.A = A
        
        
class OrthA:
    def __init__(self):
        M = np.random.rand(5,5)
        M = linalg.qr(M)[0]
        A = linalg.block_diag(M,M,M,M)
        A = linalg.block_diag(M,M,M,M,M,M,M,M,M,M, M,M,M,M,M,M,M,M,M,M)
        self.A = A
