import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Attrition Predictor", page_icon="📊", layout="wide")
st.title("📊 Employee Attrition Prediction App")

# Load the champion model
if os.path.exists("best_model.pkl"):
    model = joblib.load("best_model.pkl")
else:
    st.error("Model not found! Please run 'employee_attrition.py' to generate 'best_model.pkl'.")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Employee Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 18, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance = st.number_input("Distance From Home", 1, 50, 5)
        number_of_dependents = st.number_input("Number of Dependents", 0, 10, 0)
        
    with col2:
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician", 
            "Manufacturing Director", "Healthcare Representative", "Manager", 
            "Sales Representative", "Research Director", "Human Resources"
        ])
        income = st.number_input("Monthly Income", 1000, 25000, 5000)
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
        remote_work = st.selectbox("Remote Work", ["Yes", "No"])
        
    with col3:
        company_tenure = st.number_input("Company Tenure (Total Years)", 0, 40, 5)
        years_at_company = st.number_input("Years At Current Company", 0, 40, 3)
        number_of_promotions = st.number_input("Number of Promotions", 0, 10, 1)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])

    st.markdown("---")
    st.subheader("Satisfaction & Environment")
    c1, c2, c3 = st.columns(3)
    with c1: 
        work_life = st.slider("Work-Life Balance", 1, 4, 3)
        job_sat = st.slider("Job Satisfaction", 1, 4, 3)
    with c2: 
        performance_rating = st.slider("Performance Rating", 1, 4, 3)
        leadership_opps = st.selectbox("Leadership Opportunities", ["Low", "Medium", "High"])
    with c3:
        innovation_opps = st.selectbox("Innovation Opportunities", ["Low", "Medium", "High"])
        company_rep = st.selectbox("Company Reputation", ["Poor", "Good", "Excellent"])
        employee_recog = st.selectbox("Employee Recognition", ["Low", "Medium", "High"])

    submit = st.form_submit_button("Predict Attrition", use_container_width=True)

if submit:
    # This dictionary exactly matches the 22 columns the pipeline demands
    input_dict = {
        'age': age,
        'years_at_company': years_at_company,
        'monthly_income': income,
        'number_of_promotions': number_of_promotions,
        'distance_from_home': distance,
        'number_of_dependents': number_of_dependents,
        'company_tenure': company_tenure,
        'gender': gender,
        'job_role': job_role,
        'work-life_balance': work_life,
        'job_satisfaction': job_sat,
        'performance_rating': performance_rating,
        'overtime': overtime,
        'education_level': education_level,
        'marital_status': marital_status,
        'job_level': job_level,
        'company_size': company_size,
        'remote_work': remote_work,
        'leadership_opportunities': leadership_opps,
        'innovation_opportunities': innovation_opps,
        'company_reputation': company_rep,
        'employee_recognition': employee_recog
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Run the model
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 **High Risk of Attrition!** (Probability: {prob*100:.1f}%)")
    else:
        st.success(f"✅ **Employee is likely to stay.** (Probability of leaving: {prob*100:.1f}%)")