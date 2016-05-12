# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:45:18 2016

@author: v-siz
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fileHandle = open('D:/Users/zdz/cec15_10/input_data/M_2_D100.txt', 'r')
data = fileHandle.readlines()
table = []
dim = 100
for item in data:
    table.append([float(i) for i in item.strip().split()])

for i in xrange(len(table)):
    table[i] = map(lambda x: x!=0, table[i])
#grouplist = [[0, 9], [1, 7, 11, 21], [2, 3, 4, 6, 15, 16, 22, 23, 27], [5, 18, 20], [8, 13, 17, 24, 26], [10, 12, 14, 19, 25, 28, 29]]
#grouplist = [[0, 12, 17, 20, 21, 26, 29], [1, 4, 5, 6, 9, 10, 11, 13, 28], [2, 14, 22, 23, 25], [3], [7], [8], [15, 19, 27], [16], [18], [24]]
grouplist = [[0, 5, 11, 27, 32, 37, 39, 67], [1], [2], [3], [4], [6, 12, 24, 34, 42, 43, 46, 47, 53, 56, 64, 66, 86, 94], [7, 8, 33, 38, 65, 68, 70, 78, 81, 84, 88, 96], [9, 10, 15, 26, 28, 31, 35, 54, 95, 97], [13], [14], [16, 50, 69, 75, 92, 99], [17], [18], [19], [20], [21], [22], [23], [25], [29], [30], [36], [40], [41], [44], [45], [48], [49], [51], [52], [55], [57], [58], [59], [60], [61], [62], [63], [71], [72], [73], [74], [76], [77], [79], [80], [82], [83], [85], [87], [89], [90], [91], [93], [98]]
#grouplist = [[0, 5, 14, 20, 21, 22, 23, 24, 26], [1, 3, 11, 12, 15, 16, 18, 27, 29], [2, 4, 6, 7, 8, 9, 10, 13, 17, 19, 25, 28]]
table_detected = np.array([[0]*dim]*dim)

for item in grouplist:
    for element in item:
        for element2 in item:
            table_detected[element][element2] = 1

a=0
b=0
c=0
d=0

   
for i in xrange(dim):
    for j in range(i+1 , dim):
        if table[i][j] == 1 and table_detected[i][j] == 1:
            a = a+1
        elif table[i][j] == 0 and table_detected[i][j] == 0:
            b=b+1
        elif table[i][j] == 1 and table_detected[i][j] == 0:
            c=c+1
        else:
            d=d+1
print a, b, c, d
beta = 8
pa = a/(a+c)
na = b/(b+d)
Fb = (1+beta**2)*pa*na/(pa+ beta**2*na)
print Fb
cmap = sns.diverging_palette(220, 10)
sns.heatmap(np.array(table), alpha=.5)
sns.heatmap(np.array(table_detected), alpha=.5)
plt.show()