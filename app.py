import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Load trained model
heart_model_bundle = joblib.load("heart_model_bundle.joblib")
pipeline = heart_model_bundle['pipeline']
features_names = heart_model_bundle['features_names']


# Prediction Function
def heart_prediction(user_input):
    user_input_df = pd.DataFrame([user_input], columns=features_names)
    prediction = pipeline.predict(user_input_df)
    return prediction[0]


# Main App
def main():

    # Title
    st.markdown("<h1 style='text-align: center; color: #ff4d6d;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #adb5bd;'>ML-based Health Assistant</h4>", unsafe_allow_html=True)

    # Divider Line
    st.markdown("""
    <div style="height:4px;
        background: linear-gradient(to right, #ff4d6d, #ff758f, #ffb3c1);
        border-radius: 10px;
        margin: 10px 0 25px 0;">
    </div>
    """, unsafe_allow_html=True)

    # Section Heading
    st.subheader("📋 Enter Patient Clinical Details")
    st.markdown("---")

    # Layout
    col1, col2, col3 = st.columns(3)

    # Column 1
    with col1:
        age = st.number_input("Age", min_value=1, value=34, help="Age of patient in years")
        sex = st.selectbox("Sex", ["Male", "Female"], index=1)
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], index=1, help="0=Typical, 3=Asymptomatic")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=118)
        chol = st.number_input("Cholesterol (mg/dl)", value=210)

    # Column 2
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True)", [0, 1], index=0)
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2], index=1)
        thalach = st.number_input("Max Heart Rate Achieved", value=192)
        exang = st.selectbox("Exercise Induced Angina (1=Yes)", [0, 1], index=0)
        oldpeak = st.number_input("Oldpeak (ST depression)", value=0.7)

    # Column 3
    with col3:
        slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2], index=2)
        ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3], index=0)
        thal = st.selectbox("Thal (0=Normal,1=Fixed,2=Reversible)", [0, 1, 2], index=2)

    # Encoding
    sex = 1 if sex == "Male" else 0

    # Prediction Button
    if st.button("🔍 Predict"):

        user_input = [
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]

        prediction = heart_prediction(user_input)

        # Result Display
        if prediction == 1:
            st.error("🚨 High Risk: The patient is likely to have Heart Disease.")
        else:
            st.success("✅ Low Risk: The patient is unlikely to have Heart Disease.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<center style='color: grey;'>Developed by Rahul Kumar | Final Year Project</center>",
        unsafe_allow_html=True
    )


# Run App
if __name__ == "__main__":
    main()