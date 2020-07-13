import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

st.write("""
# Simple Wine Quality Prediction App
This app predicts **wine quality**!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.6, 15.9, 12.6)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.12, 1.58, 0.31)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.72)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.9, 15.5, 2.2)
    chlorides = st.sidebar.slider('chlorides', 0.012, 0.611, 0.072)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1.0, 72.0, 6.0)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6.0, 289.0, 29.0)
    density = st.sidebar.slider('Density', 0.99007, 1.00369, 0.9987)
    pH = st.sidebar.slider('pH', 2.74, 4.01, 2.88)
    sulphates = st.sidebar.slider('sulphates', 0.33, 2.0, 0.82)
    alcohol = st.sidebar.slider('alcohol', 8.4, 14.9, 9.8)
    data = {'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('wine_model.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Index for Wine Quality')
st.write("0: Your wine is not so good...")
st.write("1: Your wine is good!")

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Prediction')
st.write(prediction)