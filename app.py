
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('final_student_model.pkl')

st.title("🎓 Student Success Predictor")

st.markdown("Enter the student information below:")

# Input fields
age = st.number_input("Age", min_value=10, max_value=25, value=16)
study_hours = st.slider("Study Hours per Day", 0.0, 10.0, 2.0)
attendance = st.slider("Attendance (%)", 0, 100, 85)
gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)
motivation = st.slider("Motivation (1 - Low, 5 - High)", 1, 5, 3)
cluster_avg_gpa = st.number_input("Cluster Average GPA", min_value=0.0, max_value=4.0, value=2.8)

# Convert to DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'study_hours_per_day': study_hours,
    'attendance': attendance,
    'gpa': gpa,
    'motivation': motivation,
    'cluster_avg_gpa': cluster_avg_gpa
}])

# Predict button
if st.button("Predict Result"):
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Result: {prediction[0]}")
