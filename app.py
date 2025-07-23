import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open("model.pkl", "rb"))

st.title("Movie Interest Predictor")

# User 1
st.subheader("User 1")
age1 = st.number_input("Age (User 1)", min_value=1, max_value=100, value=25)
gender1 = st.radio("Gender (User 1)", ["Male", "Female"])
gender1_val = 1 if gender1 == "Male" else 0

# User 2
st.subheader("User 2")
age2 = st.number_input("Age (User 2)", min_value=1, max_value=100, value=30)
gender2 = st.radio("Gender (User 2)", ["Male", "Female"])
gender2_val = 1 if gender2 == "Male" else 0

if st.button("Predict"):
    input_data = np.array([[age1, gender1_val], [age2, gender2_val]])
    prediction = model.predict(input_data)
    st.success(f"Prediction for User 1: {prediction[0]}")
    st.success(f"Prediction for User 2: {prediction[1]}")
