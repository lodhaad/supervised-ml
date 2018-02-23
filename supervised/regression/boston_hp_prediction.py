# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:50:47 2018

@author: Aditya
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import pandas as pd 
from  sklearn.metrics import mean_squared_error
import numpy as np 



import matplotlib.pyplot as plt 


boston = datasets.load_boston()

print (type (boston))

print (boston.keys())

df = pd.DataFrame(boston.data, columns = boston.feature_names)



df['Price'] = boston.target

print (df.head())

X_rooms = df ['RM' ].values.reshape (-1,1)

y = df['Price'].values.reshape (-1,1)

X_train, X_test , y_train, y_test = train_test_split (X_rooms , y )

lm = LinearRegression ()

lm.fit (X_train, y_train)

pred_range = np.linspace (min (X_rooms), max (X_rooms)).reshape (-1,1)

pred = lm.predict(X_test)

score = lm.score (X_test, y_test)

print (score)

plt.scatter (x = X_rooms, y = y , color = 'blue')

plt.plot (pred_range, lm.predict (pred_range),color = 'black', linewidth = 3)

plt.show ()



