# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:47:53 2018

@author: Aditya
"""

import pandas as pd 
import numpy as np 

df = pd.read_csv ('../data/1984-Us-elections.csv')

column_names = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
       'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
       'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']

df.columns = column_names

print (df.head ())


df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))




# df.column.replace (0,np.nan,inplace = True )


df.infants = df.infants.map(dict(y=1, n=0 ))
df.water = df.water.map(dict(y=1, n=0))
df.budget = df.budget.map(dict(y=1, n=0))
df.physician = df.physician.map(dict(y=1, n=0))
df.salvador = df.salvador.map(dict(y=1, n=0))
df.religious = df.religious.map(dict(y=1, n=0))
df.satellite = df.satellite.map(dict(y=1, n=0))
df.aid = df.aid.map(dict(y=1, n=0))
df.missile = df.missile.map(dict(y=1, n=0))
df.immigration = df.immigration.map(dict(y=1, n=0))
df.synfuels = df.synfuels.map(dict(y=1, n=0))
df.education = df.education.map(dict(y=1, n=0))
df.superfund = df.superfund.map(dict(y=1, n=0))
df.crime = df.crime.map(dict(y=1, n=0))
df.duty_free_exports = df.duty_free_exports.map(dict(y=1, n=0))
df.eaa_rsa = df.eaa_rsa.map(dict(y=1, n=0))
df.party = df.party.map(dict(republican=1, democrat=0))


print (df.head ())

df1 = df





y = df['party']

print (y.head())

X = df.drop('party', axis=1)

print (X.head ())


from sklearn.metrics import  confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


knn = KNeighborsClassifier (    n_neighbors= 8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit (X_train, y_train)

y_predict = knn.predict (X_test)

print (confusion_matrix (y_test, y_predict))

print (classification_report (y_test, y_predict))

print ('##################### logistic #########################')
      
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression ( )

lr.fit (X_train,y_train)

y_predict = lr.predict (X_test)

print (confusion_matrix (y_test, y_predict))

from  sklearn.metrics import roc_curve

y_predict_prob = lr.predict_proba (X_test)[: , 1]

fpr , tpr , thresh = roc_curve (y_test, y_predict_prob)

import matplotlib.pyplot as plt

plt.plot ([0,1], [0,1], 'k--')
plt.plot (fpr,tpr, label = 'Logistic')

plt.xlabel ('false positive rates ')

plt.ylabel ('true positive rates')
plt.show()

from sklearn.metrics import roc_auc_score

print (roc_auc_score (y_test, y_predict_prob))


print ('################ end Logistic ############')

from  sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report



from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))

print (confusion_matrix (y_test, y_pred))


print ('############################### Pipeline ######################')
       
       
       
       
# Convert '?' to NaN
#df1[df1 == '?'] = np.nan

# Print the number of NaNs
#print(df1.isnull().sum())

# Print shape of original DataFrame
#print("Shape of Original DataFrame: {}".format(df1.shape))

# Drop missing values and print shape of new DataFrame
#df1 = df1.dropna()

# Print shape of new DataFrame
#print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df1.shape))


# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC ()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]


# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split (X, y , random_state = 42, test_size = 0.3)

# Fit the pipeline to the train set
pipeline.fit (X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict (X_test)

# Compute metrics
print(classification_report(y_test, y_pred))




