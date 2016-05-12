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
      self.pca = PCA(n_components = self.vector_length, whiten = True )
      tmp_sorted = sorted(zip(self.scores, self.population), key=lambda x: x[0])
      self.population_sorted = np.array(map(lambda x: x[1], tmp_sorted))
      
      if self.count%1==0:
          self.pca.fit(self.population_sorted[0 : self.population_size])  #不能低于30 否则无法产生30维pca
          self.transMat = np.array( self.pca.components_ )
      else:
          self.transMat = np.eye(self.evaluator.n)
#          np.random.shuffle(self.transMat)
#      if self.count%10==0:
#          print np.dot(self.evaluator.orthmat.A, np.linalg.inv(self.transMat))
#      checkpoint1 = self.population[ np.argmin( self.scores ) ]
#      if self.count%100==0:
#          checkpoint2 = self.population[ np.argmin( self.scores ) ]
#          self.FL = np.random.rand(self.vector_length)/2 + 0.5
#      if self.count%100==51:
#          if (checkpoint2 == checkpoint1).all():
##              self.transMat = orth.OrthA(self.evaluator.n).A
#              self.FL = np.random.normal(0, 0.1, self.vector_length)
#              for item in self.FL:
#                  if item < 0:
#                      item = item + 1
#              print 'check'
#      print np.array(self.population).T[29]
#      self.transMat = self.transMat
#      print self.transMat[0], self.transMat[29]
#      self.transMat = np.eye(30)
#      self.transMat = orth.OrthA(self.evaluator.n).A
      self.evolve() 
#      dif_of_trans = ((self.transMatl - self.evaluator.orthmat.A)**2).sum()
#      print dif_of_trans
      location = np.argmin( self.scores )
      if self.show_progress:
        if self.count%self.show_progress_nth_cycle==0:
          # make here a call to a custom print_status function in the evaluator function
          # the function signature should be (min_target, mean_target, best vector)
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
#收敛判定2
#      rd = (np.mean(self.scores) - np.min(self.scores) )        ####收敛判定2 待
#      rd = rd*rd/(np.min(self.scores)*np.min(self.scores) + self.eps )
#      if ( rd < self.eps ):
#        converged = True


      if self.count>=self.max_iter:
        converged =True

  def make_random_population(self):
    #初始化
#    self.tao1 = 0
#    self.tao2 = 0
#    self.CRL = [self.cr]*self.population_size
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
      #防止加1后等于其他向量，所以大于等于ii时全部加1？
      #防止加1后等于其他向量，所以大于等于ii时全部加1？
      #原本的i1 i2 i3不可能互相相等，但可能等于ii， 如果简单的在i1等于ii时使i1加1，就可能打破原有i1 i2 i3秩序使其发生
      #互相相等的情况，若使i2,i2,i3同时加1，又可能使i2,i3发生等于ii的情况；故将[0,population_size]分为ii左边和ii右边
      #所有大于等于ii的i1,i2,i3都要同时加1，小于ii的保持原值。
      #rnd = np.random.random_integers(0,self.population_size-1,3)
      #i1,i2,i3 = rnd[0],rnd[1],rnd[2]

      x1 = self.population[ i1 ]
      x2 = self.population[ i2 ]
      x3 = self.population[ i3 ]
      xbest = self.population[ np.argmin(self.scores)]
      #刷新CR及F
#      if np.random.random() < self.tao1:
#          self.CRL[ii] = np.random.uniform(0.5,1)
#      if np.random.random() < self.tao2:
#          self.FL[ii] = np.random.uniform(0.1,1)
#      if self.f is None:
#        self.f = np.random.random()/1.2 + 0.2/1.2
      use_f = np.random.random()/2 + 0.5
#      use_f = (np.random.random()/2 + 0.5)*(self.evaluator.target(np.array(x3))-self.evaluator.target(np.array(x2)))/np.linalg.norm(np.array(x3)-np.array(x2))
#      else:
#        use_f = self.f
      vi = np.add(x1 , np.multiply(use_f, np.subtract(x2,x3)))
#      vi = np.add(x1 , np.multiply(use_f,np.dot(self.transMatl[i1], np.subtract(x2,x3))))
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
