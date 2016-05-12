# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 17:12:46 2014

@author: ZDZ
"""

import numpy as np
import data_DE_ori
import data_orth
import data_Func_Dim


class test_rosenbrock_function(object):
  def __init__(self, dim=5, func=None, domain=None, index = 0):
#    self.data = xlwt.Workbook() 
#    self.table = self.data.add_sheet('pca', cell_overwrite_ok=True)      
    self.fileHandle = open('D:/Users/zdz/data_cec13_30/dataOLD%d.txt'%index, 'w')
  
    self.x = None
    self.n = dim
    self.func = func
    self.dim = dim
    self.domain = domain
    self.orthmat = data_orth.OrthA(self.n)
    self.optimizer = data_DE_ori.differential_evolution_optimizer(
                        self,population_size=self.n*4,n_cross=0,cr=0.5, max_iter=300000,
                         eps=1e-8, monitor_cycle=100000, show_progress=True)



    
  def target(self, vector):
    self.vectorOrth = np.dot(self.orthmat.A, vector)
    result = self.func(self.vectorOrth)
    return result

  def print_status(self, mins,means,vector,txt):
    print txt,'\t', mins,'\t', means#, vector
  
    self.fileHandle.write('%d %.9f %.9f\n' %(txt, mins, means))  
    

def run(i):
    
  fc = data_Func_Dim.FD()
  fc.prob_index = i
  tmp = test_rosenbrock_function(fc.dimension, fc.func, fc.domain, i) 
  tmp.fileHandle.close()
  print "OK"


if __name__ == "__main__":
  for i in [6,9,12]:
      run(i)