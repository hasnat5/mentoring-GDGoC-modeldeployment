import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("iris_model.pkl")

# App title
st.title("Iris Classification Model")

# User input
st.sidebar.header("User Input Features")
def user_input():
    sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.1)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.3)
    data = {'sepal_length': sepal_length, 'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width}
    return pd.DataFrame([data])

# Get input
input_df = user_input()
st.write("### User Input:")
st.write(input_df)

# Predict
prediction = model.predict(input_df)
st.write("### Prediction:", prediction[0])