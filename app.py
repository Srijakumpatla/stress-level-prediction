import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------
st.set_page_config(page_title="Stress Predictor", page_icon="🧠", layout="wide")

st.title("🧠 Stress Level Prediction App")
st.write("Predict your stress level based on lifestyle & health factors.")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl","rb") as f:
        model, encoders, feature_order = pickle.load(f)
    return model, encoders, feature_order

model, encoders, feature_order = load_model()

# ---------------------------------------------------
# USER INPUT UI
# ---------------------------------------------------
st.header("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    age = st.slider("Age",18,80,30)
    occupation = st.selectbox("Occupation", encoders['Occupation'].classes_)
    sleep_duration = st.slider("Sleep Duration",0.0,12.0,7.0,0.1)
    quality = st.slider("Quality of Sleep",1,10,7)
    activity = st.slider("Physical Activity Level",0,100,50)

with col2:
    bmi = st.selectbox("BMI Category", encoders['BMI Category'].classes_)
    heart_rate = st.slider("Heart Rate",50,130,75)
    steps = st.slider("Daily Steps",1000,20000,6000)
    sleep_disorder = st.selectbox("Sleep Disorder", encoders['Sleep Disorder'].classes_)
    systolic = st.slider("Systolic BP",90,180,120)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("Predict Stress Level"):

    input_dict = {
        "Gender": encoders['Gender'].transform([gender])[0],
        "Age": age,
        "Occupation": encoders['Occupation'].transform([occupation])[0],
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality,
        "Physical Activity Level": activity,
        "BMI Category": encoders['BMI Category'].transform([bmi])[0],
        "Heart Rate": heart_rate,
        "Daily Steps": steps,
        "Sleep Disorder": encoders['Sleep Disorder'].transform([sleep_disorder])[0],
        "Systolic BP": systolic
    }

    # Convert input → dataframe
    input_df = pd.DataFrame([input_dict])

    # Force same column order as training
    input_df = input_df[feature_order]

    # Predict
    prediction = round(model.predict(input_df)[0])

    # Stress level category
    if prediction <= 3:
        level = "Low"
        color = "green"
    elif prediction <= 7:
        level = "Moderate"
        color = "orange"
    else:
        level = "High"
        color = "red"

    # Display Result
    st.markdown(
        f"<h2 style='color:{color}; text-align:center;'>Predicted Stress Level: {prediction} ({level})</h2>",
        unsafe_allow_html=True
    )

    # Suggestions
    st.subheader("Recommendations")

    if level == "Low":
        st.success("Maintain your healthy lifestyle 👍")
    elif level == "Moderate":
        st.warning("Try relaxation techniques and improve sleep habits.")
    else:
        st.error("High stress detected. Consider professional consultation.")
