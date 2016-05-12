# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 17:12:46 2014

@author: ZDZ
"""

import numpy as np
import data_orth
import data_DEaxisPCA
import data_Func_Dim




class test_rosenbrock_function(object):
  def __init__(self, dim=5, func=None, domain=None, index = 0):
#    self.data = xlwt.Workbook() 
##    self.data = xlrd.open_workbook('C:\Users\ZDZ\Documents\programs\Result\PCA//data.xlsx')
##    self.table = self.data.sheets()[0] 
#    self.table = self.data.add_sheet('pca', cell_overwrite_ok=True)
    
    self.fileHandle = open('D:/Users/zdz/cec13_10/dataPCA_CR5%d.txt'%index, 'w')
    
    self.x = None
    self.n = dim
    self.func = func
    self.dim = dim
    self.domain = domain
    self.orthmat = data_orth.OrthA(self.n)
    self.optimizer = data_DEaxisPCA.differential_evolution_optimizer(
                                self,population_size=self.n*10, max_iter=30000,
                                n_cross=0,cr=0.5, eps=1e-8, monitor_cycle=100000,
                                show_progress=True)

  def target(self, vector):
    self.vectorOrth = np.dot(self.orthmat.A, vector)
    result = self.func(self.vectorOrth)
    return result

  def print_status(self, mins,means,vector,txt):
    print txt,'\t', mins,'\t', means#,'\t' ,  np.dot(self.orthmat.A, vector)

    self.fileHandle.write('%d %f %f\n' %(txt, mins, means))    


def run(i):
    
  fc = data_Func_Dim.FD()
  fc.prob_index = 11+1
  tmp = test_rosenbrock_function(fc.dimension, fc.func, fc.domain, i) 
  tmp.fileHandle.close()
  print "OK"


if __name__ == "__main__":
  for i in xrange(20):
      run(i)