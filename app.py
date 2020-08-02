# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:40:16 2020

@author: Gaurav
"""

import streamlit as st
import pickle
import pandas as pd

pickle_in = open("model.pickle","rb")
classifier=pickle.load(pickle_in)

import numpy as np
import matplotlib.pyplot as plt

st.write("""
# Breast Cancer Prediction.
This app predicts the **Breast Cancer** type!
""")

st.sidebar.header('User Input Parameters')

def output(pred):
      st.text("""""")
      st.write(
              """
              ## 🎯 RESULT
              """
              )
      if pred == 0:
          st.text("""""")
          st.write(
                  """
                  ## 👁️ Model Predicts That The Patient Has A BENIGN Tumor.
                  """
                  )
      else:
          st.text("""""")
          st.write(
                  """
                  ## 👁️ Model Predicts That The Patient Has A MALIGNANT Tumor.
                  """
                  )
      st.write(
        """
    ### ©️ Created By Gourab Chatterjee
        """
    )    
      
      
def main():
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
    features = pd.DataFrame(data)
    prediction = classifier.predict(features.iloc[:1, :])
    output(prediction[0])

if __name__=="__main__":
    main()
          
   
          