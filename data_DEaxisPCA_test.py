# -*- coding: utf-8 -*-
"""
Created on Wed Jan 07 11:11:29 2015

@author: ZDZ
"""
import numpy as np
from sklearn.decomposition import PCA

class differential_evolution_optimizer(object):
     
  def __init__(self,
               evaluator,           #哪个测试函数调用时，若是self，则等于测试函数
               population_size=50,
               f=None,
               cr=0.9,
               eps=1e-5,
               n_cross=1,
               max_iter=300000,
               monitor_cycle=100,
               out=None,
               show_progress=False,
               show_progress_nth_cycle=1,
               insert_solution_vector=None,
               dither_constant=0.4):
    self.dither=dither_constant
    self.show_progress=show_progress
    self.show_progress_nth_cycle=show_progress_nth_cycle
    self.evaluator = evaluator
    self.population_size = population_size
    self.f = f
    self.cr = cr
    self.n_cross = n_cross
    self.max_iter = max_iter
    self.monitor_cycle = monitor_cycle
    self.vector_length = evaluator.n
    self.eps = eps
    self.population = []
    self.seeded = False
    if insert_solution_vector is not None:
      assert len( insert_solution_vector )==self.vector_length
      self.seeded = insert_solution_vector
    for ii in xrange(self.population_size):
      self.population.append([0]*self.vector_length)


    self.scores = [1000]*self.population_size
    self.optimize()
    self.best_score = np.min( self.scores )
    self.best_vector = self.population[ np.argmin( self.scores ) ]
    self.evaluator.x = self.best_vector
    if self.show_progress:
      self.evaluator.print_status(
            np.min(self.scores),
            np.mean(self.scores),
            self.population[ np.argmin( self.scores ) ],
            max_iter)
    #返回最优值及最优解
    

  def optimize(self):
    # initialise the population please
    self.make_random_population()
#    self.population = np.fft.fft(self.population)
    # score the population please
    self.score_population()
    converged = False
    monitor_score = np.min( self.scores )
    self.count = 0
    while not converged:
      self.pca = PCA(n_components = self.vector_length)#, whiten = True )
      tmp_sorted = sorted(zip(self.scores, self.population), key=lambda x: x[0])
      self.population_sorted = np.array(map(lambda x: x[1], tmp_sorted))
      
      if self.count<1:
          self.pca.fit(self.population_sorted[0 : self.evaluator.n])  #不能低于30 否则无法产生30维pca
          self.transMat = np.array( self.pca.components_ )

      self.evolve() 

      location = np.argmin( self.scores )
      if self.show_progress:
        if self.count%self.show_progress_nth_cycle==0:

          self.evaluator.print_status(
            np.min(self.scores),
            np.mean(self.scores),
            self.population[ np.argmin( self.scores ) ],
            self.count)

      self.count += 1
      if self.count%self.monitor_cycle==0:
        if (monitor_score-np.min(self.scores) ) < self.eps:
          converged = True
        else:
         monitor_score = np.min(self.scores)

      if self.count>=self.max_iter:
        converged =True

  def make_random_population(self):
    #初始化

    self.FL = np.random.rand(self.vector_length)/2 + 0.5
    for ii in xrange(self.vector_length):
      delta  = self.evaluator.domain[ii][1]-self.evaluator.domain[ii][0]
      offset = self.evaluator.domain[ii][0]
      random_values = np.random.random(self.population_size)
      random_values = random_values*delta+offset
      # now please place these values ni the proper places in the
      # vectors of the population we generated
      for vector, item in zip(self.population,random_values):
        vector[ii] = item
    if self.seeded is not False:
      self.population[0] = self.seeded

  def score_population(self):
    for vector,ii in zip(self.population,xrange(self.population_size)):
      tmp_score = self.evaluator.target(vector)
      self.scores[ii]=tmp_score #需要更新 append不便更新

  def evolve(self):
    for ii in xrange(self.population_size):
      rnd = np.random.random(self.population_size-1)
      permut = np.argsort(rnd)
      # make parent indices
      i1=permut[0]
      if (i1>=ii):
        i1+=1
      i2=permut[1]
      if (i2>=ii):
        i2+=1
      i3=permut[2]
      if (i3>=ii):
        i3+=1


      x1 = self.population[ i1 ]
      x2 = self.population[ i2 ]
      x3 = self.population[ i3 ]
      xbest = self.population[ np.argmin(self.scores)]
      #刷新CR及F

      use_f = np.random.random()/2 + 0.5

      vi = np.add(x1 , np.multiply(use_f, np.subtract(x2,x3)))

      #判定是否越界
      for jj in xrange( self.vector_length  ):
          if vi[jj]<self.evaluator.domain[jj][0] or vi[jj]>self.evaluator.domain[jj][1]:
              vi[jj] = self.evaluator.domain[jj][0]+np.random.random()*(self.evaluator.domain[jj][1]-self.evaluator.domain[jj][0])
      
      # prepare the offspring vector please
      dif = vi - self.population[ii]
      dif_transed = np.dot(self.transMat, dif)
      rnd = np.random.random(self.vector_length)
      permut = np.argsort(rnd)
      coef = np.zeros(self.vector_length)
      # first the parameters that sure cross over
      for jj in xrange( self.vector_length  ):
        if (jj<self.n_cross):
          coef[ permut[jj] ] = 1
        else:
          if (rnd[jj]<self.cr): #改为小于
            coef[jj] = 1
      test_vector = self.population[ii] + np.dot(self.transMat.T, coef*dif_transed)
      test_score = self.evaluator.target( test_vector )  #产生过程中的随进向量库 


      
      if test_score < self.scores[ii] :
        self.scores[ii] = test_score
        self.population[ii] = test_vector

 

  def show_population(self):
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    for vec in self.population:
      print list(vec)       ##转化为list （若之前是（））
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
  def returnresult(self):
    return self.best_score, self.best_vector
