import os
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
# Assuming the model uploaded by train.py is for tourism package prediction
model_path = hf_hub_download(repo_id="kutkarsh200/tourism-model", filename="tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("This app predicts whether a customer will purchase the newly introduced Wellness Tourism Package based on their details.")
st.write("Please enter the customer details to check their likelihood of purchasing the package.")

# Collect user input for Tourism dataset features
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting (including customer)", min_value=1, max_value=10, value=1)
PreferredPropertyStar = st.number_input("Preferred Hotel Rating (1-5 stars)", min_value=1, max_value=5, value=3)
NumberOfTrips = st.number_input("Average Number of Trips Annually", min_value=0, max_value=50, value=1)
NumberOfChildrenVisiting = st.number_input("Number of Children (below age 5) Visiting", min_value=0, max_value=5, value=0)
MonthlyIncome = st.number_input("Monthly Income (gross income)", min_value=0.0, value=30000.0)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
NumberOfFollowups = st.number_input("Number of Follow-ups by Salesperson", min_value=0, max_value=10, value=1)
DurationOfPitch = st.number_input("Duration of Sales Pitch (in minutes)", min_value=0.0, value=10.0)

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier (1 = highest, 3 = lowest)", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer", "Government Service"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Passport = st.selectbox("Holds a valid Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a car?", ["Yes", "No"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "President"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

# Convert categorical inputs to match model training expectations
input_data = pd.DataFrame([{
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'Designation': Designation,
    'ProductPitched': ProductPitched
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the Wellness Tourism Package" if prediction == 1 else "not purchase the Wellness Tourism Package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
