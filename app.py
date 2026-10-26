import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_wine

# Load model and class names
model = joblib.load("wine_model.joblib")
wine = load_wine()

st.title("🍷 Wine Classification App")
st.write("Enter chemical features of the wine to predict its class (Class 0, 1, or 2).")

# Create input fields for key features
alcohol = st.number_input("Alcohol", 11.0, 15.0, 13.0)
malic_acid = st.number_input("Malic Acid", 0.7, 5.8, 2.3)
ash = st.number_input("Ash", 1.3, 3.2, 2.3)
alcalinity = st.number_input("Alcalinity of Ash", 10.6, 30.0, 19.5)
magnesium = st.number_input("Magnesium", 70.0, 162.0, 99.0)

if st.button("Predict Wine Class"):
    # Prepare features (padding remaining features with their average values to fit the 13-feature input shape)
    features = [alcohol, malic_acid, ash, alcalinity, magnesium] + [0.0] * 8
    prediction = model.predict([features])[0]
    target_name = wine.target_names[prediction]
    st.success(f"Prediction: {target_name.upper()}")
