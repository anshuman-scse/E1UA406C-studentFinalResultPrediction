import streamlit as st
import pandas as pd
import joblib
model = joblib.load(
"student_result_model.pkl"
)
st.title(
"Student Result Prediction System"
)
study_hours = st.number_input(
"Study Hours"
)
attendance = st.number_input(
"Attendance"
)
assignment_score = st.number_input(
"Assignment Score"
)
cgpa = st.number_input(
"CGPA"
)
participation = st.selectbox(
"Participation",
[0,1]
)
having_laptop = st.selectbox("Having Laptop",
[0,1]
)
performance_index = (
study_hours + cgpa
)/2
input_data = pd.DataFrame({
"Study_Hours":[study_hours],
"Attendance":[attendance],
"Assignment_Score":[assignment_score],
"CGPA":[cgpa],
"Participation":[participation],
"Having_Laptop":[having_laptop],
"Performance_Index":[performance_index]
})
if st.button("Predict"):
	input_data = input_data.reindex(
    columns=model.feature_names_in_,
    fill_value=0)
	prediction = model.predict(
	input_data
	)
	if prediction[0]==1:
		st.success("PASS")
	else:
		st.error("FAIL")
