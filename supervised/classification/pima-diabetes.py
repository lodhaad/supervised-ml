# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 22:39:59 2018

@author: Aditya
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np



df = pd.read_csv ('../data/diabetes.csv')

print (df)



sns.pairplot (df)

plt.show ()

sns.heatmap (df.corr() , annot = True)

plt.show ()

sns.heatmap (df.corr ().abs () > 0.6)
plt.show ()

sns.boxplot (data = df )
plt.xticks(rotation=45)

plt.show ()



print (df.columns)

y = df ['Outcome'].values
X = df.drop ('Outcome' , axis = 1).values



# Import necessary modules
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Create training and test set

X_train, X_test, y_train, y_test = train_test_split (X, y , test_size = 0.4 , random_state = 42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier (n_neighbors = 6) 

# Fit the classifier to the training data
knn.fit (X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict (X_test)

# Generate the confusion matrix and classification report
#print (y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print ('#####################################Hyerparameter value for KNN #########')

from sklearn.model_selection import GridSearchCV
#import numpy as np

print ()

param_grid = {'n_neighbors' : np.arange(1, 50)}



grid_cv = GridSearchCV (knn, param_grid , cv = 5)

grid_cv.fit (X_train, y_train)

print (grid_cv.best_score_)
print (grid_cv.best_params_)


print ('##### datacamps #############################')

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit (X,y)


# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


print ('############### randomized Search CV #####################')
       
# Import necessary modules
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit (X,y)


# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))



print ("##################### logistic regression ########################")

       
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit (X_train, y_train)
y_pred = lr.predict(X_test)

print (confusion_matrix (y_test, y_pred))

print (classification_report (y_test, y_pred))



# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = lr.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import roc_auc_score

print (roc_auc_score (y_test, y_pred_prob))


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score (lr , X_train, y_train , cv = 5 ,scoring = 'roc_auc' )

print (cv_scores)


print ('##############hold out #########################')
       
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV




# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))






