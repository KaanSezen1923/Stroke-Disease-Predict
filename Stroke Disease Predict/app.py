import streamlit as st
import joblib
import numpy as np

st.title("Stroke Disease Prediction with Logistic Regression")
st.write("This is a simple web application to predict stroke disease using Logistic Regression.")

# Encoding functions
def encode_gender(gender):
    return 0 if gender == "Male" else 1

def encode_hypertension(hypertension):
    return 0 if hypertension == "Yes" else 1

def encode_heart_disease(heart_disease):
    return 0 if heart_disease == "Yes" else 1

def encode_work_type(work_type):
    mapping = {
        "Private": 0,
        "Self-employed": 1,
        "Govt_job": 2,
        "children": -1,
        "Never_worked": -2
    }
    return mapping.get(work_type, -2)

def encode_smoking_status(smoking_status):
    mapping = {
        "never smoked": 0,
        "formerly smoked": 1,
        "smokes": 2,
        "Unknown": -1
    }
    return mapping.get(smoking_status, -1)

def encode_ever_married(ever_married):
    return 0 if ever_married == "Yes" else 1

# Input fields
gender = st.selectbox("Select a gender", options=["Male", "Female"])
age = st.number_input("Enter your age", min_value=0, max_value=120, value=0)
hypertension = st.selectbox("Do you have hypertension?", options=["Yes", "No"])
heart_disease = st.selectbox("Do you have heart disease?", options=["Yes", "No"])
work_type = st.selectbox("Select your work type", options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
avg_glucose_level = st.number_input("Enter your average glucose level", min_value=0.0, max_value=300.0, value=0.0)
bmi = st.number_input("Enter your BMI", min_value=0.0, max_value=50.0, value=0.0)
smoking_status = st.selectbox("Select your smoking status", options=["never smoked", "formerly smoked", "smokes", "Unknown"])
ever_married = st.selectbox("Are you married?", options=["Yes", "No"])

if st.button("Predict"):
    try:
        # Encode all inputs
        encoded_data = [
            encode_gender(gender),
            age,
            encode_hypertension(hypertension),
            encode_heart_disease(heart_disease),
            encode_work_type(work_type),
            avg_glucose_level,
            bmi,
            encode_smoking_status(smoking_status),
            encode_ever_married(ever_married)
        ]

        # Load the model
        model = joblib.load("Stroke Disease Predict/stroke_model.pkl")

        # Ensure data is in the correct format for prediction
        data = np.array([encoded_data])

        # Make prediction
        prediction = model.predict(data)
        prob=model.predict_proba(data)[0]

        # Display result
        st.write(f"The prediction is: {prediction[0]}")
        if prediction[0] == 1:
            prob=prob[1].round(2)*100
            st.error(f"You are at risk of  %{prob} probability stroke disease.")
        else:
            prob=prob[0].round(2)*100
            st.success(f"You are not at risk of %{prob} probability stroke disease.")

    except FileNotFoundError:
        st.error("Model file 'log_reg_model.pkl' not found. Please ensure the file is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
