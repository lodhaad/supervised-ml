# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 00:34:39 2018

@author: Aditya
"""

import pandas as pd 

import matplotlib.pyplot as plt 


df = pd.read_csv ('../data/boston.csv')

print (df)

print (df.head ())

y = df['MEDV'].values

X = df.drop ('MEDV', axis = 1).values 

X_rooms = X[: , 5 ]

print (X_rooms.shape)

y= y.reshape (-1,1)

X_rooms = X_rooms.reshape (-1,1)

print (X_rooms.shape)


#plt.scatter (X_rooms, y)

#plt.show ()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# for one variable 


X_train, X_test, y_train, y_test = train_test_split (X_rooms, y , test_size = .3 , random_state = 42)

lm = LinearRegression()

lm.fit (X_train, y_train)

prediction = lm.predict (X_test)


print (lm.score (X_test, y_test))


pred_range = np.linspace (min (X_rooms), max (X_rooms)).reshape (-1,1)

pred = lm.predict(X_test)

score = lm.score (X_test, y_test)

print (score)

plt.scatter (x = X_rooms, y = y )

plt.plot (pred_range, lm.predict (pred_range),color = 'black', linewidth = 3)

plt.show ()

### for all the variables

X_train, X_test, y_train, y_test = train_test_split (X, y , test_size = .3 , random_state = 42)

lm = LinearRegression()

lm.fit (X_train, y_train)

prediction = lm.predict (X_test)


print (lm.score (X_test, y_test))


###### cross validation 

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#print (X)
# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score (reg, X,y , cv = 5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


# Perform 3-fold CV
cvscores_3 = cross_val_score (reg, X, y , cv = 3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score (reg, X, y , cv = 10)
print(np.mean(cvscores_10))



#### lasso regression 

from sklearn.linear_model import Lasso

lasso = Lasso (alpha = 0.1 )

names = df.drop ('MEDV' , axis = 1).columns

lasso_coef = lasso.fit (X_train, y_train).coef_

print (lasso.score(X_test, y_test))

plt.plot (range (len(names)), lasso_coef)

plt.xticks(range (len(names)), names , rotation =60)

plt.show ()



#### ridge regression 

from sklearn.linear_model import Ridge



ridge = Ridge (alpha = 0.1 , normalize = True )

names = df.drop ('MEDV' , axis = 1).columns


ridge.fit (X_train, y_train)

print (ridge.score(X_test, y_test))





 

