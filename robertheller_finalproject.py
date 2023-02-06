# -*- coding: utf-8 -*-
"""
Robert Heller
Machine Learning Final Project

Dataset can be found here:
https://www.kaggle.com/datasets/codefantasy/list-of-best-selling-nintendo-games

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import csv
import io
import sys
from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

from dateutil import parser

file = files.upload()

df = pd.read_csv( io.StringIO(file['List-of-best-selling-videogames.csv'].decode('utf-8')))

# Jonathan Bouchet's Kaggle code to fix some minor issues.

df = df.iloc[:,0:7] # Remove all null columns
df['Platform'] = df['Platform'].apply(lambda x: "Super NES" if x == "Nintendo Entertainment System (NES)" else x) # Fix platform name mistake
df['Platform'] = df['Platform'].apply(lambda x: "Game Boy Advance" if x == "Game Boy Advanced" else x) # Fix platform name mistake
df['Release date'] = np.where((df['Release date'] == "33635"),"June 21, 1993",df['Release date']) # Fix date format

# Convert the dates to a proper format
def get_year(x):
    x = x.split(",")
    if len(x)>1:
        return int(x[1].strip())
    else:
        return np.nan

# End of Jonathan Bouchet's Kaggle code.

df['Release date'] = df['Release date'].apply(lambda x: get_year(x))

df.head()

plt.figure(figsize=(16,7))
plt.bar( df['Platform'], df['Sales'], width = 0.3)
plt.xlabel('Platform')
plt.ylabel('Sales of Best-selling Game (In 10\'s of millions)')

df.describe()

print("Maximum sales number: " + str(df['Sales'].max()))
print("Minimum sales number: " + str(df['Sales'].min()))
print("Average sales number: " + str(df['Sales'].mean()))

plt.bar(df['Release date'], df['Sales'])
plt.xlabel('Year of Release')
plt.ylabel('Highest Sales Number (in 10\'s of millions)')

from sklearn.preprocessing import OneHotEncoder

#df = df.drop(columns=['Publisher'])
#df = df.drop(columns=['Game'])

#df = df.drop(columns=['Platform'])
#df = df.drop(columns=['Genre'])
#df = df.drop(columns=['Developer'])

dummies_platform = pd.get_dummies(df['Platform'],drop_first=True)
dummies_genre = pd.get_dummies(df['Genre'],drop_first=True)
dummies_developer = pd.get_dummies(df['Developer'],drop_first=True)

#df = pd.concat([df.drop('Developer', axis=1), dummies_developer], axis=1)
#df = pd.concat([df.drop('Platform', axis=1), dummies_platform], axis=1)
#df = pd.concat([df.drop('Genre', axis=1), dummies_genre], axis=1)

df.head()

df.describe()

plt.scatter(df['Release date'], df['Sales'], c=dummies_platform['Wii'])
plt.colorbar()

# Test data

X = df.drop(columns=['Sales'], axis=1)
y = df['Sales']

from sklearn.decomposition import PCA
pca = PCA(0.99)
pca.fit(pd.get_dummies(X,drop_first=True))
X_trans = pca.transform(pd.get_dummies(X,drop_first=True))

pca.components_

pca.explained_variance_ratio_

from sklearn.model_selection import train_test_split
import sklearn.tree as tree

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Parul Pandey's code from Towards Data Science. Solves the issue of the "Dummy variable trap."

# Dummy encoding Training set
X_train_encoded = pd.get_dummies(X_train, drop_first=True)

# Saving the columns in a list
cols = X_train_encoded.columns.tolist()

X_test_encoded = pd.get_dummies(X_test, drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=cols).fillna(0)

# End of Parul Pandey's code.

X_train_encoded

myTree = tree.DecisionTreeRegressor(max_depth = 3)
myTree.fit(X_train_encoded, y_train)
y_pred = myTree.predict(X_test_encoded)

plt.figure( figsize=(30,10))

tree.plot_tree(myTree)

from sklearn.linear_model import LinearRegression

est = LinearRegression()
est.fit(X_train_encoded,y_train)

y_predict = est.predict(X_test_encoded)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_predict))

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor( max_depth=3 )

rf.fit(X_train_encoded, y_train)
y_pred_rf = rf.predict(X_test_encoded)
print("Random Forest Regressor r2 score: ", r2_score(y_test,y_pred_rf))

from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor()

ada.fit(X_train_encoded, y_train)
y_pred_ada = ada.predict(X_test_encoded)
print("AdaBoost Regressor r2 score: ", r2_score(y_test,y_pred_ada))

from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()

gb.fit(X_train_encoded, y_train)
y_pred_gb = gb.predict(X_test_encoded)
print("Gradient Boost Regressor r2 score: ", r2_score(y_test,y_pred_gb))

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(20,70), activation="relu",
                   alpha=10,
                   solver='lbfgs',
                   max_iter=1000)
mlp.fit(X_train_encoded, y_train)
y_pred_mlp = mlp.predict(X_test_encoded)
print("MLP Regressor r2 score: ", r2_score(y_test, y_pred_mlp))

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, LeaveOneOut

cv = KFold(n_splits=10, shuffle = True)
#model = GradientBoostingRegressor()
model = MLPRegressor()

cv_score = cross_val_score(model, pd.get_dummies(X,drop_first=True), y, cv=cv)
print("Mean: ", np.mean(cv_score))