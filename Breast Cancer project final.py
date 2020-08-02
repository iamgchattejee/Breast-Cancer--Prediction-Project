# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:23:22 2020

@author: Gaurav
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,2:32].values
y = dataset.iloc[:,1:2].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc1= StandardScaler()
X_train=sc1.fit_transform(X_train)
X_test=sc1.transform(X_test)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(bootstrap=True, class_weight=None,
                       criterion='gini', max_depth=6, max_features=0.5,
                       max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=3,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
