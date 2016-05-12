# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 20:12:57 2015

@author: ZDZ
"""


import _get_target_13


        
class FD:
    def __init__(self):
#        self.func = deap.benchmarks.rosenbrock
#        self.dimension = 30
#        self.domain = [ (-100, 100) ]*self.dimension
        
        
        self.func = self.cec15
        self.prob_index = 1
        self.dimension = 30
        self.domain = [ (-100, 100) ]*self.dimension
        
    def cec15(self, individual):
        
        return _get_target_13.getTarget(individual, self.prob_index) - (self.prob_index-15)*100
        
