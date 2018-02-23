# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:34:44 2018

@author: Aditya
"""

import pandas as pd 
import numpy as np
from sklearn.preprocessing import Imputer

df = pd.read_csv ('../data/diabetes.csv')

df.info()

df.SkinThickness.replace (0, np.nan, inplace = True)
df.Insulin.replace (0, np.nan, inplace = True)
df.BMI.replace (0, np.nan, inplace = True)

print (df)

col_names = df.columns
#new_df = df.dropna()

print (col_names)

print (df.shape)

#print (new_df.shape)

#imp = Imputer (missing_values='NaN', strategy='mean',axis=0 )    

#imp.fit (df)

#df_new = imp.transform(df)

#df_new = pd.DataFrame (df,columns = col_names )



#mean

y = df ['Outcome']
X = df.drop ('Outcome' , axis = 1)

print (X)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split (X, y , random_state = 42)

imp = Imputer (missing_values='NaN', strategy='mean', axis=0 ) 

log = LogisticRegression()

steps = [('Imputing Mean', imp), 
         ('Logistic regression' ,log )]

pipeline = Pipeline(steps)

pipeline.fit (X_train, y_train)

y_pred = pipeline.predict (X_test)

print (pipeline.score (X_test, y_test))