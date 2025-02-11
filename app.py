#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
model = pickle.load(open("svm_model_new.pkl", "rb"))
scaler = pickle.load(open("scaler_new.pkl", "rb"))

st.title("Betel Vine Soil Fertility Prediction 🌱")

# Define the input fields
pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
electrical_conductivity = st.number_input("Electrical Conductivity", min_value=0.0, value=0.5)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=1.0)
available_nitrogen = st.number_input("Available Nitrogen", min_value=0, value=300)
available_phosphorus = st.number_input("Available Phosphorus", min_value=0, value=30)
available_potassium = st.number_input("Available Potassium", min_value=0, value=200)
exchangeable_calcium = st.number_input("Exchangeable Calcium", min_value=0.0, value=5.0)
exchangeable_magnesium = st.number_input("Exchangeable Magnesium", min_value=0.0, value=1.0)
available_sulfur = st.number_input("Available Sulfur", min_value=0, value=10)
iron = st.number_input("Iron", min_value=0.0, value=10.0)
manganese = st.number_input("Manganese", min_value=0.0, value=10.0)
zinc = st.number_input("Zinc", min_value=0.0, value=1.0)
copper = st.number_input("Copper", min_value=0.0, value=0.5)
soil_depth = st.selectbox("Soil Depth", options=["Subsurface", "Surface"])
soil_type = st.selectbox("Soil Type", options=["Loamy", "Sandy", "Clayey", "Mixed"])

# Add a Predict button
if st.button("Predict"):
    # Convert categorical variables into one-hot encoding
    soil_depth_encoded = [1 if soil_depth == "Surface" else 0]  # Assuming 'Surface' is encoded as 1
    soil_type_encoded = [
        1 if soil_type == "Loamy" else 0,
        1 if soil_type == "Mixed" else 0,
        1 if soil_type == "Sandy" else 0,
    ]  # Assuming Loamy is the base category (drop_first=True was used)

    # Combine all features into a DataFrame with correct column names
    feature_names = [
        "pH", "Electrical_Conductivity", "Organic_Carbon", "Available_Nitrogen",
        "Available_Phosphorus", "Available_Potassium", "Exchangeable_Calcium",
        "Exchangeable_Magnesium", "Available_Sulfur", "Iron", "Manganese",
        "Zinc", "Copper", "Soil_Depth_Surface", "Soil_Type_Loamy",
        "Soil_Type_Mixed", "Soil_Type_Sandy"
    ]

    input_data = pd.DataFrame(
        [[pH, electrical_conductivity, organic_carbon, available_nitrogen, available_phosphorus,
          available_potassium, exchangeable_calcium, exchangeable_magnesium, available_sulfur,
          iron, manganese, zinc, copper] + soil_depth_encoded + soil_type_encoded],
        columns=feature_names
    )

    # Apply scaling
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_label = "✅ Suitable for Betel Vine" if prediction[0] == 1 else "❌ Not Suitable for Betel Vine"

    # Display result
    st.subheader("Prediction Result")
    st.write(f"### {prediction_label}")


# In[ ]:




