# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 17:12:46 2014

@author: ZDZ
"""

import numpy as np
import data_DE_ultimate
import data_orth
import data_Func_Dim
import grouping2

class test_rosenbrock_function(object):
  def __init__(self, dim=5, func=None, domain=None, index = 0):

    self.fileHandle = open('D:/Users/zdz/cec13_10/dataUlt%d.txt'%index, 'w')     
    self.x = None
    self.n = dim
    self.func = func
    self.dim = dim
    self.domain = domain
    self.orthmat = data_orth.OrthA(self.n)

    group = grouping2.grouping()
    xbest = [0] * self.n
    for i in xrange(self.n):
        xbest[i] = self.domain[i][0] + (self.domain[i][1] - self.domain[i][0])*np.random.random()
    xnew = [0] * self.n
    for i in xrange(self.n):
        xnew[i] = self.domain[i][0] + (self.domain[i][1] - self.domain[i][0])*np.random.random()    
        
    group.fit(xnew, xbest, self.target)
    grouplist = group.group                 
    print   grouplist   

    self.optimizer = data_DE_ultimate.differential_evolution_optimizer(
                        self,population_size=self.n*10,n_cross=0,cr=0.5, max_iter=100000,
                         eps=1e-8, monitor_cycle=100000, show_progress=True, grouplist = grouplist)
           
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

#    self.iffted = np.real(np.fft.ifft(vector))   
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
    print txt,'\t', mins,'\t', means#, vector
#    self.table.write(txt, 0, mins)
#    self.table.write(txt, 1, means)     
    self.fileHandle.write('%d %f %f\n' %(txt, mins, means))  

def run(i):
  fc = data_Func_Dim.FD()
  fc.prob_index = i+1
  tmp = test_rosenbrock_function(fc.dimension, fc.func, fc.domain, i) 
  tmp.fileHandle.close()

if __name__ == "__main__":
  for i in xrange(20):
      run(i)