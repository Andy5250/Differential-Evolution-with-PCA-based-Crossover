# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:57:53 2015

@author: ZDZ
"""
import numpy as np
import orth
import copy

import deap.benchmarks

#输入n 提供target(x[n])
class grouping(object):
    def __init__(self):
        self.elenum = 0
        self.gpnum = 0
        self.group = []
        self.vector1 = 0
        self.vector2 = 0  
        self.groupinfo = []
        self.target = []
        self.counter = 0
        
    def fit(self, vector1, vector2, target):
        assert len(vector1) == len(vector2)
        self.vector1 = vector1
        self.vector2 = vector2 
        self.target = target
        self.elenum = len(vector1) 
        rawdata = range(self.elenum)

        while len(rawdata)!=0:
            temp = rawdata[0]
            self.group.append([temp])
            rawdata.remove(temp)
            for j in rawdata[:]:
                if self.detect(temp, j):
                    self.group[-1] = self.group[-1] + [j]
                    rawdata.remove(j)
                    
#        for i in rawdata:
#
#                self.group.append([i])
#                rawdata.remove(i)
#                for j in rawdata[:]:
#             
#                        if self.detect(i, j):
#                            self.group[-1] = self.group[-1] + [j]
#                            rawdata.remove(j)

                
    def detect(self, i, j):
        x11 = copy.deepcopy(self.vector1)
    
        x21 = copy.deepcopy(self.vector1)
        x21[i] = self.vector2[i]
    
        x12 = copy.deepcopy(self.vector1)
        x12[j] = self.vector2[j]
    
        x22 = copy.deepcopy(self.vector1)
        x22[i] = self.vector2[i]
        x22[j] = self.vector2[j]
        
        global counter
        self.counter += 1
        print self.counter, '%.2f%%'%(self.counter/1000.0**2*100)        
        
        if abs(self.target(x11) + self.target(x22) - self.target(x12) - self.target(x21)) > 1e-3:
            return 1
        else:
            return 0        

#
#
#counter = 0
#n = 100
#orthmat = orth.OrthA()
#def target2(vector):
#    vectorOrth = np.dot(orthmat.A, np.array(vector))
#    y = 0
#    for j in xrange(n):
#        if (j%2):
#            y = y + (vectorOrth[j]-1)**2 - 10*np.cos(2*np.pi*(vectorOrth[j]-1)) + 10 
#        else:
#            y = y + (vectorOrth[j]-2)**2 - 10*np.cos(2*np.pi*(vectorOrth[j]-2)) + 10 
#            
#    global counter
#    counter += 1
#    print counter, '%.2f%%'%(counter/40000.0*100)
#    
#    return y    
#    
#def target(vector):
##    bench = Benchmark()
##    deap.benchmarks.rastrigin
##    fun_fitness = bench.get_function(5)
#    vectorOrth = np.dot(orthmat.A, np.array(vector))
#    result = deap.benchmarks.rastrigin(vectorOrth)#[0]
##    result = fun_fitness(np.array(vectorOrth))
#    global counter
#    counter += 1
#    print counter, '%.2f%%'%(counter/1000.0**2*100)
#    return result
#    
#domain = [-5.12,5.12]
#
#xbest = [0] * n
#for i in xrange(n):
#    xbest[i] = domain[0] + (domain[1] - domain[0])*np.random.random()
#xbestf = target(xbest)
##输入n 最佳个体xbest[n] vector(x[n]) domain=[]只有一个
#
###产生相关矩阵
##RM = np.zeros((n,n))
##
###产生两两变量函数值矩阵
##VAF = np.zeros((n,n))
#
##随机产生xi'
#xnew = [0] * n
#for i in xrange(n):
#    xnew[i] = domain[0] + (domain[1] - domain[0])*np.random.random()
#
#
#
#group = grouping()
#group.fit(xnew, xbest, target)
#print group.group
#==============================================================================
#x1=xbest[:]
#x2=xbest[:]
#x1[0] = xnew[0]
#x2[0] = xnew[0]
#x2[1] = xnew[1]
#print target(x1),target(x2),xbestf
#==============================================================================
#计算VAF
#var = copy.deepcopy(xbest)
#for i in xrange(n):
#    vari = copy.deepcopy(xbest)
#    vari[i] = xnew[i]
#    for j in xrange(n):
#        varj = copy.deepcopy(vari)
#        varj[j] = xnew[j]
#        VAF[i][j] = target(varj)
#        
#print VAF[5][5],VAF[0][0],VAF[0][5],xbestf
#
#def detect(i, j):
#    x11 = copy.deepcopy(xnew)
#    
#    x21 = copy.deepcopy(xnew)
#    x21[i] = xbest[i]
#    
#    x12 = copy.deepcopy(xnew)
#    x12[j] = xbest[j]
#    
#    x22 = copy.deepcopy(xnew)
#    x22[i] = xbest[i]
#    x22[j] = xbest[j]
#    
#    if abs(x11 + x22 - x21 - x12) > 0.000001:
#        return 1
#    else:
#        return 0
#    
##计算RM
#for i in xrange(n):
#    for j in xrange(n):
#        if abs(VAF[i][i] + VAF[j][j] - VAF[i][j] - xbestf) > 0.000001:    #DONT FORGET ABS
#            RM[i][j] = 1
#        else:
#            RM[i][j] = 0
#        
#print RM[20]
#print RM
#RMlist = RM.tolist() 
#data = open('C:\Users\ZDZ\Documents\programs\CC/data.txt','w')
#data.write(RMlist)
#RMlist.tofile(data,sep=',',format='%s')

