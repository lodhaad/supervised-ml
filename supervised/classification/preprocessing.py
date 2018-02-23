# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:10:08 2018

@author: Aditya
"""

import pandas as pd 

import matplotlib.pyplot as plt

mpg = pd.read_csv('../data/mpg.csv')

print (mpg)

mpg_origin = pd.get_dummies (mpg)


#df_region = pd.get_dummies(df, drop_first = True)

mpg.boxplot ('mpg', 'origin', rot = 60)

plt.show()





mpg_origin = mpg_origin.drop ('origin_Asia', axis = 1)

print (mpg_origin)

y = mpg_origin ['mpg']
X = mpg_origin.drop ('mpg', axis = 1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

X_train,X_test, y_train, y_test = train_test_split (X, y , random_state = 42)

ridge = Ridge (alpha = 0.5, normalize= True)

ridge.fit (X_train,y_train)

print (ridge.score (X_test,y_test ))



print ('#####DC ###########################')

# Import pandas
import pandas as pd 

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv ('../data/gapminder.csv')

print (df.head())
# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()



# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first = True)

# Print the new columns of df_region
print(df_region.columns)


# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


# Instantiate a ridge regressor: ridge
ridge = Ridge (alpha = 0.5 , normalize = True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score (ridge,X, y,  cv = 5)

# Print the cross-validated scores
print(ridge_cv)



