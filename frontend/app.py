import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000"

def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except:
        return False

def get_available_models():
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
        return {"available_models": [], "total_models": 0}
    except:
        return {"available_models": [], "total_models": 0}

def make_prediction(data):
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def make_batch_prediction(data_list):
    try:
        response = requests.post(f"{API_BASE_URL}/predict/batch", json={"data": data_list})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def main():
    st.title("🔍 Fraud Detection System")
    st.markdown("Vehicle Loan Default Prediction")
    
    if not check_api_health():
        st.error("❌ API is not running. Please start the FastAPI server first.")
        st.code("cd api && python main.py")
        return
    
    st.success("✅ API is running")
    
    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Information"])
    
    with tab1:
        st.header("Single Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            unique_id = st.number_input("Unique ID", min_value=1, value=12345)
            disbursed_amount = st.number_input("Disbursed Amount", min_value=0, value=50000)
            asset_cost = st.number_input("Asset Cost", min_value=0, value=60000)
            ltv = st.number_input("LTV Ratio", min_value=0.0, max_value=1.0, value=0.8)
            
            st.subheader("Location & Branch")
            branch_id = st.number_input("Branch ID", min_value=1, value=1)
            supplier_id = st.number_input("Supplier ID", min_value=1, value=1)
            manufacturer_id = st.number_input("Manufacturer ID", min_value=1, value=1)
            current_pincode_id = st.number_input("Current Pincode ID", min_value=1, value=1)
            state_id = st.number_input("State ID", min_value=1, value=1)
            
            st.subheader("Personal Information")
            date_of_birth = st.date_input("Date of Birth", value=date(1985, 1, 1))
            employment_type = st.selectbox("Employment Type", 
                ["Salaried", "Self employed", "Missing"])
            disbursal_date = st.date_input("Disbursal Date", value=date(2023, 1, 1))
            
        with col2:
            st.subheader("Employee Information")
            employee_code_id = st.number_input("Employee Code ID", min_value=1, value=1)
            
            st.subheader("Document Flags")
            mobile_flag = st.selectbox("Mobile Available", [0, 1], index=1)
            aadhar_flag = st.selectbox("Aadhar Available", [0, 1], index=1)
            pan_flag = st.selectbox("PAN Available", [0, 1], index=1)
            voterid_flag = st.selectbox("Voter ID Available", [0, 1], index=0)
            driving_flag = st.selectbox("Driving License Available", [0, 1], index=1)
            passport_flag = st.selectbox("Passport Available", [0, 1], index=0)
            
            st.subheader("Credit Score")
            perform_cns_score = st.number_input("CNS Score", min_value=0, max_value=900, value=750)
            perform_cns_score_desc = st.selectbox("CNS Score Description", 
                ["A-Very Low Risk", "B-Very Low Risk", "C-Very Low Risk", "D-Very Low Risk",
                 "E-Low Risk", "F-Low Risk", "G-Low Risk", "H-Medium Risk",
                 "I-Medium Risk", "J-High Risk", "K-High Risk", "L-Very High Risk",
                 "M-Very High Risk", "No Bureau History Available", "Not Scored: Sufficient History Not Available"])
        
        st.subheader("Account Information")
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**Primary Accounts**")
            pri_no_of_accts = st.number_input("Number of Primary Accounts", min_value=0, value=3)
            pri_active_accts = st.number_input("Active Primary Accounts", min_value=0, value=2)
            pri_overdue_accts = st.number_input("Overdue Primary Accounts", min_value=0, value=0)
            pri_current_balance = st.number_input("Primary Current Balance", min_value=0, value=10000)
            pri_sanctioned_amount = st.number_input("Primary Sanctioned Amount", min_value=0, value=50000)
            pri_disbursed_amount = st.number_input("Primary Disbursed Amount", min_value=0, value=45000)
            pri_instal_amt = st.number_input("Primary Installment Amount", min_value=0, value=5000)
        
        with col4:
            st.write("**Secondary Accounts**")
            sec_no_of_accts = st.number_input("Number of Secondary Accounts", min_value=0, value=1)
            sec_active_accts = st.number_input("Active Secondary Accounts", min_value=0, value=1)
            sec_overdue_accts = st.number_input("Overdue Secondary Accounts", min_value=0, value=0)
            sec_current_balance = st.number_input("Secondary Current Balance", min_value=0, value=5000)
            sec_sanctioned_amount = st.number_input("Secondary Sanctioned Amount", min_value=0, value=20000)
            sec_disbursed_amount = st.number_input("Secondary Disbursed Amount", min_value=0, value=18000)
            sec_instal_amt = st.number_input("Secondary Installment Amount", min_value=0, value=2000)
        
        st.subheader("Recent Activity")
        new_accts_six_months = st.number_input("New Accounts in Last 6 Months", min_value=0, value=0)
        delinquent_accts_six_months = st.number_input("Delinquent Accounts in Last 6 Months", min_value=0, value=0)
        no_of_inquiries = st.number_input("Number of Inquiries", min_value=0, value=2)
        
        col5, col6 = st.columns(2)
        with col5:
            avg_acct_age_years = st.number_input("Average Account Age (Years)", min_value=0, max_value=50, value=5)
        with col6:
            avg_acct_age_months = st.number_input("Average Account Age (Months)", min_value=0, max_value=11, value=6)
        
        col7, col8 = st.columns(2)
        with col7:
            credit_history_years = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, value=8)
        with col8:
            credit_history_months = st.number_input("Credit History Length (Months)", min_value=0, max_value=11, value=3)
        
        if st.button("Predict Loan Default", type="primary"):
            with st.spinner("Making prediction..."):
                input_data = {
                    "UNIQUEID": unique_id,
                    "DISBURSED_AMOUNT": disbursed_amount,
                    "ASSET_COST": asset_cost,
                    "LTV": ltv,
                    "BRANCH_ID": branch_id,
                    "SUPPLIER_ID": supplier_id,
                    "MANUFACTURER_ID": manufacturer_id,
                    "CURRENT_PINCODE_ID": current_pincode_id,
                    "DATE_OF_BIRTH": date_of_birth.strftime("%d-%m-%Y"),
                    "EMPLOYMENT_TYPE": employment_type,
                    "DISBURSAL_DATE": disbursal_date.strftime("%d-%m-%Y"),
                    "STATE_ID": state_id,
                    "EMPLOYEE_CODE_ID": employee_code_id,
                    "MOBILENO_AVL_FLAG": mobile_flag,
                    "AADHAR_FLAG": aadhar_flag,
                    "PAN_FLAG": pan_flag,
                    "VOTERID_FLAG": voterid_flag,
                    "DRIVING_FLAG": driving_flag,
                    "PASSPORT_FLAG": passport_flag,
                    "PERFORM_CNS_SCORE": perform_cns_score,
                    "PERFORM_CNS_SCORE_DESCRIPTION": perform_cns_score_desc,
                    "PRI_NO_OF_ACCTS": pri_no_of_accts,
                    "PRI_ACTIVE_ACCTS": pri_active_accts,
                    "PRI_OVERDUE_ACCTS": pri_overdue_accts,
                    "PRI_CURRENT_BALANCE": pri_current_balance,
                    "PRI_SANCTIONED_AMOUNT": pri_sanctioned_amount,
                    "PRI_DISBURSED_AMOUNT": pri_disbursed_amount,
                    "SEC_NO_OF_ACCTS": sec_no_of_accts,
                    "SEC_ACTIVE_ACCTS": sec_active_accts,
                    "SEC_OVERDUE_ACCTS": sec_overdue_accts,
                    "SEC_CURRENT_BALANCE": sec_current_balance,
                    "SEC_SANCTIONED_AMOUNT": sec_sanctioned_amount,
                    "SEC_DISBURSED_AMOUNT": sec_disbursed_amount,
                    "PRIMARY_INSTAL_AMT": pri_instal_amt,
                    "SEC_INSTAL_AMT": sec_instal_amt,
                    "NEW_ACCTS_IN_LAST_SIX_MONTHS": new_accts_six_months,
                    "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS": delinquent_accts_six_months,
                    "AVERAGE_ACCT_AGE": f"{avg_acct_age_years}yrs {avg_acct_age_months}mon",
                    "CREDIT_HISTORY_LENGTH": f"{credit_history_years}yrs {credit_history_months}mon",
                    "NO_OF_INQUIRIES": no_of_inquiries
                }
                
                result = make_prediction(input_data)
                
                if result:
                    st.success("Prediction completed!")
                    
                    col_pred, col_conf = st.columns(2)
                    
                    with col_pred:
                        if result['prediction'] == 0:
                            st.success(f"✅ **No Default Expected** (Prediction: {result['prediction']})")
                        else:
                            st.error(f"❌ **Default Risk Detected** (Prediction: {result['prediction']})")
                    
                    with col_conf:
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                    
                    st.subheader("Detailed Results")
                    col_prob1, col_prob2 = st.columns(2)
                    
                    with col_prob1:
                        st.metric("No Default Probability", f"{result['probability']['class_0']:.2%}")
                    with col_prob2:
                        st.metric("Default Probability", f"{result['probability']['class_1']:.2%}")
                    
                    st.info(f"Model used: {result['model_used']}")
    
    with tab2:
        st.header("Batch Prediction")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded data preview:")
                st.dataframe(df.head())
                
                if st.button("Predict Batch", type="primary"):
                    with st.spinner("Processing batch prediction..."):
                        data_list = df.to_dict('records')
                        results = make_batch_prediction(data_list)
                        
                        if results:
                            st.success(f"Batch prediction completed for {results['total_count']} records")
                            
                            results_df = pd.DataFrame(results['predictions'])
                            st.dataframe(results_df)
                            
                            download_df = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=download_df,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("Model Information")
        
        models_info = get_available_models()
        
        if models_info['total_models'] > 0:
            st.success(f"✅ {models_info['total_models']} models loaded")
            st.write("Available models:")
            for model in models_info['available_models']:
                st.write(f"- {model}")
        else:
            st.error("❌ No models available")

if __name__ == "__main__":
    main()
