import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer choosing a tourism product based on its customer parameters.
Enter the customer and their related data below to get a prediction.
""")

Age = st.number_input("Age", min_value=18, max_value=61, value=35)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=130.0, value=15.0, step=1.0)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=3)
NumberOfFollowups = st.number_input("Number of Followups", min_value=1.0, max_value=6.0, value=4.0, step=1.0)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips (per year)", min_value=1.0, max_value=8.0, value=3.0, step=1.0)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=3.0, value=1.0, step=1.0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=23000.0, step=500.0)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

# Set the classification threshold
classification_threshold = 0.45

if st.button("Predict Customer purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer will purchase" if prediction == 1 else "Customer won't purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
