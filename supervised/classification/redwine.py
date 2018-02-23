# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:28:13 2018

@author: Aditya
"""

import pandas as pd 
import numpy as np 

df = pd.read_csv ('../data/red_wine.txt', delimiter = ';')

print (df.describe())


df['quality']  = np.where(df['quality']>= 5, 0, 1)

print (df.describe())


from sklearn.preprocessing import scale



y = df ['quality']

X = df.drop ('quality', axis = 1)

col_names = X.columns

df = scale (X)

df = pd.DataFrame (df, columns = col_names)  

print (df.describe ())


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


scaler = StandardScaler ()

knn = KNeighborsClassifier()

steps = [('scaler',scaler), ('knn', knn)]

pipeline = Pipeline (steps)


# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split (X, y, random_state = 42, test_size = .3)


knn_scaled = pipeline.fit (X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))


from sklearn.metrics import classification_report
from sklearn.svm import SVC
from  sklearn.model_selection import GridSearchCV

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))





# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline (steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split (X, y, random_state = 42, test_size = .3)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit (X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))





