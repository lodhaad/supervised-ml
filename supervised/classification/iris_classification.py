# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:37:39 2018

@author: Aditya
"""

from sklearn import datasets  

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

plt.style.use ('ggplot')

iris = datasets.load_iris()

print (type (iris))

print (iris.keys())

print (type (iris.data))

print (iris.data.shape)

print (iris.target_names)


X = iris.data 

y = iris.target 

df = pd.DataFrame (X, columns = iris.feature_names)

print (df)


pd.scatter_matrix (df, c = y, s = 150 , marker = 'D', figsize = [8, 8])

plt.show ()


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier (n_neighbors = 6)

knn.fit (X , y )

X_new = [[ 5.9   ,            3.0   ,             5.1     ,          1.8 ]]

prediction = knn.predict (X_new)

print (prediction)


from sklearn.model_selection import train_test_split


X_train, X_test , y_train, y_test = train_test_split (X, y , test_size = 0.3 ,
                                                      random_state = 42 , stratify = y)


knn = KNeighborsClassifier (n_neighbors=6)

knn.fit (X_train, y_train)

prediction = knn.predict (X_test)

print (prediction)


print (knn.score (X_test, y_test))







