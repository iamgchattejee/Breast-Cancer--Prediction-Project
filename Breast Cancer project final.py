# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:23:22 2020

@author: Gaurav
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.write("""
# Breast Cancer Prediction.
This app predicts the **Breast Cancer** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    radius_mean = st.sidebar.slider('Radius Mean', 6.98,28.1, 8.4)
    texture_mean = st.sidebar.slider('Texture Mean', 9.71, 39.3, 15.6)
    perimeter_mean = st.sidebar.slider('Perimeter Mean', 43.8, 189.0, 70.2)
    area_mean = st.sidebar.slider('Area Mean', 144.0,2500.0, 20.50)
    smoothness_mean = st.sidebar.slider('Smoothness Mean', 0.05, 0.16, .08)
    compactness_mean = st.sidebar.slider('Compactness Mean', 0.02, 0.35, 0.18)
    concavity_mean = st.sidebar.slider('Concavity Mean', 0.0, 0.43, 0.2)
    concave_points_mean = st.sidebar.slider('Concave Points Mean', 0.0, 0.2, 0.1)
    symmetry_mean= st.sidebar.slider('Symmetry Mean', 0.11, 0.3, 0.2)
    fractal_dimension_mean = st.sidebar.slider('Fractal Dimensions Mean', 0.05, 0.1, 0.76)
    radius_se = st.sidebar.slider('Radius Se', 0.11, 0.287, 0.15)
    texture_se = st.sidebar.slider('Texture Se', 0.36, 4.88, 2.38)
    perimeter_se= st.sidebar.slider('Perimeter Se', 0.76, 22.0, 2.8)
    area_se= st.sidebar.slider('Area Se', 6.8, 542.0, 28.4)
    smoothness_se= st.sidebar.slider('Smoothness Se', 0.0, 0.03, 0.012)
    compactness_se= st.sidebar.slider('Compactness Se', 0.0,0.14, 0.022)
    concavity_se= st.sidebar.slider('Concavity Se', 0.0, 0.4, 0.2)
    concave_points_se = st.sidebar.slider('Concave Points Se', 0.0, 0.05, 0.02)
    symmetry_se= st.sidebar.slider('Symmetry Se', 0.01, 0.08, 0.032)
    fractal_dimension_se= st.sidebar.slider('Fractal Dimensions Se', 0.0, 0.03, 0.012)
    radius_worst= st.sidebar.slider('Radius Worst', 7.93, 36.0, 10.6)
    texture_worst= st.sidebar.slider('Texture Worst', 12.0, 49.5, 22.2)
    perimeter_worst= st.sidebar.slider('Perimeter Worst', 50.4, 251.0, 62.7)
    area_worst= st.sidebar.slider('Area Worst', 185.0, 4250.0, 255.67)
    smoothness_worst= st.sidebar.slider('Smoothness Worst', 0.07, 0.22, 0.12)
    compactness_worst= st.sidebar.slider('Compactness Worst', 0.03, 1.06, 0.082)
    concavity_worst= st.sidebar.slider('Concavity Worst', 0.0, 1.25, 0.25)
    concave_points_worst = st.sidebar.slider('Concave Points Worst', 0.0, 0.29, 0.125)
    symmetry_worst= st.sidebar.slider('Symmetry Worst', 0.16, 0.66, 0.3)
    fractal_dimension_worst= st.sidebar.slider('Fractal Dimensions Worst', 0.6, 0.21, 0.8)
    data = {'Radius Mean':radius_mean,
            'Texture Mean':texture_mean,
            'Perimeter Mean':perimeter_mean,
            'Area Mean':area_mean,
            'Smoothness Mean':smoothness_mean,
            'Compactness Mean':compactness_mean,
            'Concavity Mean':concavity_mean,
            'Concave Points Mean':concave_points_mean,
            'Symmetry Mean': symmetry_mean,
            'Fractal Dimension Mean':fractal_dimension_mean,
            'Radius Se':radius_se,
            'Texture Se':texture_se,
            'Perimeter Se':perimeter_se,
            'Area Se':area_se,
            'Smoothness Se':smoothness_se,
            'Compactness Se':compactness_se,
            'Concavity Se':concavity_se,
            'Concave Points Se':concave_points_se,
            'Symmetry Se':symmetry_se,
            'Fractal Dimensions Se':fractal_dimension_se,
            'Radius Worst':radius_worst,
            'Texture Worst':texture_worst,
            'Perimeter Worst':perimeter_worst,
            'Area Worst':area_worst,
            'Smoothness Worst':smoothness_worst,
            'Compactness Worst':compactness_worst,
            'Concavity Worst':concavity_worst,
            'Concave Points Worst':concave_points_worst,
            'Symmetry Worst':symmetry_worst,
            'Fractal Dimensions Worst':fractal_dimension_worst,
           }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

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
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)



rfbase = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=3,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)


prediction = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)
cancer_type="Benign"
if(prediction==1):
    cancer_type="Malignant"

st.subheader('**Prediction**')
st.write(cancer_type)

st.subheader('Prediction Probability')
st.write(prediction_proba)