# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 17:12:46 2014

@author: ZDZ
"""

import numpy as np
import copy
import math
import CC_orth
import data_orth
import pywt
import CC_DEaxisPCA
import deap.benchmarks
import data_Func_Dim
import xlwt




class test_rosenbrock_function(object):
  def __init__(self, dim=5, func=None, domain=None):
    self.data = xlwt.Workbook() 
#    self.data = xlrd.open_workbook('C:\Users\ZDZ\Documents\programs\Result\PCA//data.xlsx')
#    self.table = self.data.sheets()[0] 
    self.table = self.data.add_sheet('pca',
                                     cell_overwrite_ok=True)


    self.x = None
    self.n = dim
    self.func = func
    self.dim = dim
    self.domain = domain
    self.orthmat = data_orth.OrthA(self.n)
    self.GroupNum = 5
    for i in xrange(10000):
        self.optimizer = CC_DEaxisPCA.differential_evolution_optimizer(
                                self, population_size=min(self.n*10,100), Round = i%5,
                                n_cross=0,cr=0.5, eps=1e-8, monitor_cycle=50000,
                                show_progress=True)
#    print list(self.x)
#    for x in self.x:
#      assert abs(x-1.0)<1e-2


  def target2(self, vector):

    fr = 4000
    s = 0.0
    p = 1.0
    for j in xrange(self.n):
        s = s + vector[j]**2
    for j in xrange(self.n):                     
        p = p * np.cos(vector[j]/np.sqrt(j+1)) #float?    #未加1导致除以0得NAN 只有此处加1 别处不加
    y = s/fr-p+1
    return y

#  def target(self, vector):
#    y = 0
#    for j in xrange(self.n-1):
#         
#        y = y + (vector[j]-1)**2 - 10*np.cos(2*np.pi*(vector[j]-1))+10 
#    y = y + (vector[self.n-1]-2)**2 - 10*np.cos(2*np.pi*(vector[self.n-1]-2))+10 
#    return y    
  def target4(self, vector):
#    self.idwted = pywt.idwt(vector[:(len(vector)/2)], vector[(len(vector)/2):],'db1')
#    self.idwted = map(lambda x: pywt.idwt(x[:(len(vector)/2)], x[(len(vector)/2):],'haar'), vector)
    self.vectorOrth = np.dot(self.orthmat.A, vector)
#    print self.vectorOrth[5]
    y = 0
    for j in xrange(self.n):
        if (j%2):
            y = y + (self.vectorOrth[j]-1)**2 - 10*np.cos(2*np.pi*(self.vectorOrth[j]-1))+10 
        else:
            y = y + (self.vectorOrth[j]-2)**2 - 10*np.cos(2*np.pi*(self.vectorOrth[j]-2))+10 
    return y    
  
  def target3(self, vector):
#    self.idwted = pywt.idwt(vector[:(len(vector)/2)], vector[(len(vector)/2):],'db1')
#    self.vectorOrth = np.dot(self.orthmat.A, np.array(self.idwted))
    self.vectorOrth = np.dot(self.orthmat.A, vector)
#    self.vectorOrth = vector
    x_vec = self.vectorOrth[0:len(self.vectorOrth)/2]
    y_vec = self.vectorOrth[len(self.vectorOrth)/2:]
    result=0
    for x,y in zip(x_vec,y_vec):
      result+=100.0*((y-x*x)**2.0) + (1-x)**2.0
    return result  

  def target(self, vector):
    self.vectorOrth = np.dot(self.orthmat.A, vector)
    result = self.func(self.vectorOrth)
    return result

  def print_status(self, mins,means,vector,txt):
    print txt,'\t', mins,'\t', means#,'\t' ,  np.dot(self.orthmat.A, vector)
#    self.table.put_cell(int(txt), 0, 2, mins,  0 ) 
#    self.table.put_cell(int(txt), 1, 2, means,  0 ) 
#        self.table.put_cell(txt, 1, ctype=2, value = means, xf = 0 ) 
    self.table.write(txt, 0, mins)
    self.table.write(txt, 1, means)     
    

def run(i):
  fc = data_Func_Dim.FD()
  tmp = test_rosenbrock_function(fc.dimension, fc.func, fc.domain) 
  print "OK"
  tmp.data.save('C:\Users\ZDZ\Documents\programs\Result\PCA//dataPCA%d.xls'%i)

if __name__ == "__main__":
  for i in xrange(5):
      run(i)