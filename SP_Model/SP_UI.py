import joblib
import streamlit as st
import numpy as np
import pandas as pd


pipeline = joblib.load('Student_Performance.pkl')



# Set up page
st.set_page_config(page_title="📊 Student Math Score Predictor", layout="centered")

# Title and intro
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🎓 Student Math Score Predictor 📈</h1>", unsafe_allow_html=True)
st.markdown("#### 📋 Fill in your student details to predict your **Math Test Score**.")

# User input fields
Gender = st.selectbox('👧👦 Gender:', ['male', 'female'])
Race = st.selectbox('🌍 Race/Ethnicity:', ['group A', 'group B', 'group C', 'group D', 'group E'])
ParentalEducation = st.selectbox('🎓 Parental Level of Education:', [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
Lunch = st.selectbox('🥪 Lunch Type:', ['standard', 'free/reduced'])
TestPrep = st.selectbox('📚 Test Preparation Course:', ['none', 'completed'])
ReadingScore = st.slider('📖 Reading Score:', 0, 100, 70)
WritingScore = st.slider('✍️ Writing Score:', 0, 100, 70)

# Create DataFrame from inputs
new_input = pd.DataFrame([{
    'gender': Gender,
    'race/ethnicity': Race,
    'parental level of education': ParentalEducation,
    'lunch': Lunch,
    'test preparation course': TestPrep,
    'reading score': ReadingScore,
    'writing score': WritingScore
}])

# Prediction button
if st.button('🚀 Predict Math Score'):
    prediction = pipeline.predict(new_input)[0]
    rounded_score = round(prediction)

    # Show result
    if rounded_score >= 60:
        st.markdown(f"<h2 style='color: green;'>✅ Predicted Math Score: {rounded_score} — Pass 🎉</h2>", unsafe_allow_html=True)
        st.success("👏 Great job! The student passed the math test.")
        st.balloons()
    else:
        st.markdown(f"<h2 style='color: red;'>❌ Predicted Math Score: {rounded_score} — Fail</h2>", unsafe_allow_html=True)
        st.warning("📌 The student might need extra help to pass.")
