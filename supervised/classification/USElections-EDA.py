# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:50:39 2018

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


df.infants = df.infants.map(dict(y=1, n=0))
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

y = df['party']

print (y.head())

X = df.drop('party', axis=1)

print (X.head ())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()







# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

#print (df)

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

print (y)

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier (n_neighbors = 6 )

# Fit the classifier to the data
knn.fit (X, y)


prediction = knn.predict (X)

print (prediction)


