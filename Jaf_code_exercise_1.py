# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:07:31 2022

@author: unimi
"""

from sklearn . datasets import load_digits
digits = load_digits ()
print ( digits . keys ())

data = digits["data"]
images = digits["images"]
target = digits["target"]
target_names = digits ["target_names"]

import numpy as np
con = np.column_stack((data, target))
print(con.shape)

a= con[con[:, 64] == 8]
a.shape
b= con[con[:, 64] == 3]
b.shape
c= np.concatenate((a,b), axis=0)
c.shape
minitarget= c[:,64]
minitarget.shape
minidata= c[:,0:64]
minidata
minidata.shape
intercept= np.full(shape=357,fill_value=1,dtype=np.float64())
print(intercept)

feature_matrix = np.column_stack((minidata, intercept))
feature_matrix
feature_matrix.shape

#### the dataasets were made
for i in range(len(minitarget)):
    if minitarget[i]== 3:
        minitarget[i]=1
    else:
        minitarget[i]=-1
