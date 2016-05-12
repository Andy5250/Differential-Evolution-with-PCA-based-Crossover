# -*- coding: utf-8 -*-
"""
Created on Wed Jan 07 11:11:29 2015

@author: ZDZ
"""
import numpy as np
import copy
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
               out=None,
               monitor_cycle=5000,
               show_progress=False,
               show_progress_nth_cycle=1,
               insert_solution_vector=None,
               dither_constant=0.4,
               grouplist=None):
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
    self.population_list = []
    self.seeded = False
    self.group_num = len(grouplist)
    self.grouplist = grouplist
    self.representative = [50]*self.vector_length
    self.population_size_list = [0] * self.group_num

    
    if insert_solution_vector is not None:
      assert len( insert_solution_vector )==self.vector_length
      self.seeded = insert_solution_vector
    for ii in xrange(self.group_num):
      self.population_list.append([])
      self.population_size_list[ii] = 10 * len(self.grouplist[ii])
      for jj in xrange(self.population_size_list[ii]):
        self.population_list[ii].append([0]*len(self.grouplist[ii]))

    self.scores_list = []
    for ii in xrange(self.group_num):
        self.scores_list.append([0]*self.population_size_list[ii])
        
    self.optimize()
    self.best_score = self.evaluator.target(self.representative)
    self.best_vector = self.representative
    self.evaluator.x = self.best_vector
    if self.show_progress:
      self.evaluator.print_status(
            self.best_score,
            self.best_score,
            self.representative ,
            max_iter)
    #返回最优值及最优解
    

  def optimize(self):
    # initialise the population please
    self.make_random_population()
#    for ii, index in enumerate(self.grouplist):
#        self.representative[index] = best_individual_piece[ii]

    # score the population please
    self.score_population()
    converged = False
    monitor_score = self.evaluator.target(self.representative)
    self.best_score = monitor_score
    self.count = 0
    while not converged:
      self.evolve() #evolve2

#      update self.representative
      self.count += 1
      if self.count%self.monitor_cycle==0:
        if (monitor_score - self.evaluator.target(self.representative) ) < self.eps:
          converged = True
        else:
         monitor_score = self.evaluator.target(self.representative)
         
      if self.show_progress:
        if self.count%self.show_progress_nth_cycle==0:
          # make here a call to a custom print_status function in the evaluator function
          # the function signature should be (min_target, mean_target, best vector)
          self.evaluator.print_status(
            self.evaluator.target(self.representative),
            monitor_score,
            self.representative,
            self.count)
#收敛判定2
#      rd = (np.mean(self.scores) - np.min(self.scores) )        ####收敛判定2 待
#      rd = rd*rd/(np.min(self.scores)*np.min(self.scores) + self.eps )
#      if ( rd < self.eps ):
#        converged = True


      if self.count>=self.max_iter:
        converged =True

  def make_random_population(self):
    #初始化
    self.tao1 = 0
    self.tao2 = 0
    self.CRL = [self.cr]*self.population_size
    self.FL = [0.5]*self.population_size
    for jj in xrange(self.group_num):
        for ii in xrange(len(self.grouplist[jj])):
            delta  = self.evaluator.domain[self.grouplist[jj][ii]][1]-self.evaluator.domain[self.grouplist[jj][ii]][0]    #有漏洞！！！！！
            offset = self.evaluator.domain[self.grouplist[jj][ii]][0]
            random_values = np.random.random(self.population_size_list[jj])
            random_values = random_values*delta+offset
      # now please place these values ni the proper places in the
      # vectors of the population we generated
            for vector, item in zip(self.population_list[jj], random_values):
                vector[ii] = item
    if self.seeded is not False:
      self.population[0] = self.seeded

  def score_population(self):
    for ii in xrange(self.group_num):
      for jj in xrange(self.population_size_list[ii]):
          self.scores_list[ii][jj] = self.get_value(self.population_list[ii][jj], self.grouplist[ii])


  def get_value(self, individual, pop_index_list):
      temp_representative = copy.deepcopy(self.representative)
      for ii, index in enumerate(pop_index_list):
          temp_representative[index] = individual[ii]
      return self.evaluator.target(temp_representative)
      
  def evolve(self):
    for jj in xrange(self.group_num):
        self.population_list[jj]
        self.grouplist[jj]
        
        self.pca = PCA(n_components = len(self.grouplist[jj]))#, whiten = True )
#        tmp_sorted = sorted(zip(self.scores, self.population_list[jj]), key=lambda x: x[0])
#        self.population_sorted = np.array(map(lambda x: x[1], tmp_sorted))
#       self.pca.fit(self.population_sorted[0 : (self.evaluator.n*2)])  #不能低于30 否则无法产生30维pca
        self.pca.fit(self.population_list[jj])  #不能低于30 否则无法产生30维pca
        self.transMat = np.array( self.pca.components_ )
        for ii in xrange(self.population_size_list[jj]):
            rnd = np.random.random(self.population_size_list[jj]-1)
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
            x1 = self.population_list[jj][ i1 ]
            x2 = self.population_list[jj][ i2 ]
            x3 = self.population_list[jj][ i3 ]
            xbest = self.representative
      #刷新CR及F
#      if np.random.random() < self.tao1:
#          self.CRL[ii] = np.random.uniform(0.5,1)
#      if np.random.random() < self.tao2:
#          self.FL[ii] = np.random.uniform(0.1,1)
#      if self.f is None:
            use_f = np.random.random()/2.0 + 0.5
#      else:
#        use_f = self.f

            vi = np.add(x1 , np.multiply(use_f,np.subtract(x2,x3)))
      #判定是否越界
            for kk in xrange( len(self.grouplist[jj])  ):
                if vi[kk]<self.evaluator.domain[jj][0] or vi[kk]>self.evaluator.domain[jj][1]:
                    vi[kk] = self.evaluator.domain[jj][0]+np.random.random()*(self.evaluator.domain[jj][1]-self.evaluator.domain[jj][0])
      
      # prepare the offspring vector please
            rnd = np.random.random(len(self.grouplist[jj]))
            permut = np.argsort(rnd)
            test_vector_prototype = copy.deepcopy(self.population_list[jj][ii])
            
            dif = vi - test_vector_prototype
            dif_transed = np.dot(self.transMat, dif)

            coef = np.zeros(len(self.grouplist[jj]))
      # first the parameters that sure cross over
            for kk in xrange(len(self.grouplist[jj])):
                    if (rnd[kk]<self.cr): #改为小于
                        coef[kk] = 1
            test_vector = test_vector_prototype + np.dot(self.transMat.T, coef*dif_transed)
         
            
            
#      # first the parameters that sure cross over
#            for kk in xrange( len(self.grouplist[jj])  ):
#                if (rnd[kk]<self.cr): #改为小于
#                    test_vector[kk] = vi[kk]
      # get the score please     每产生一个新向量便与父代对应个体比较，若优则替换，并进入下一个体
            test_score = self.get_value( test_vector, self.grouplist[jj] )  #产生过程中的随进向量库 
      # check if the score if lower #不同于书上的全部产生完再比较替换。
            self.scores_list[jj][ii] = self.get_value( self.population_list[jj][ii], self.grouplist[jj] )   #老score 未随rep更新
            if test_score < self.scores_list[jj][ii] :
                self.scores_list[jj][ii] = test_score
                self.population_list[jj][ii] = test_vector
#                print self.population_list[jj][ii], self.scores_list[jj][ii]
                
        best_individual_piece = self.population_list[jj][np.argmin(self.scores_list[jj])]
        
#        print "haha", best_individual_piece, 

        for ii, index in enumerate(self.grouplist[jj]):

            self.representative[index] = best_individual_piece[ii]



        

  def show_population(self):
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    for vec in self.population:
      print list(vec)       ##转化为list （若之前是（））
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
  def returnresult(self):
    return self.best_score, self.best_vector
