#!/usr/bin/env python3

import requests
import json
import time

def test_api():
    print("Testing Fraud Detection API...")
    
    # Wait a moment for services to start
    time.sleep(5)
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API is healthy!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
        
        # Test models endpoint
        print("\n2. Testing models endpoint...")
        response = requests.get("http://localhost:8000/models")
        if response.status_code == 200:
            print("✅ Models endpoint working!")
            print(f"Available models: {response.json()}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
        
        # Test prediction endpoint
        print("\n3. Testing prediction endpoint...")
        sample_data = {
            "UNIQUEID": 12345,
            "DISBURSED_AMOUNT": 50000,
            "ASSET_COST": 60000,
            "LTV": 0.8,
            "BRANCH_ID": 1,
            "SUPPLIER_ID": 1,
            "MANUFACTURER_ID": 1,
            "CURRENT_PINCODE_ID": 1,
            "DATE_OF_BIRTH": "01-01-1985",
            "EMPLOYMENT_TYPE": "Salaried",
            "DISBURSAL_DATE": "01-01-2023",
            "STATE_ID": 1,
            "EMPLOYEE_CODE_ID": 1,
            "MOBILENO_AVL_FLAG": 1,
            "AADHAR_FLAG": 1,
            "PAN_FLAG": 1,
            "VOTERID_FLAG": 0,
            "DRIVING_FLAG": 1,
            "PASSPORT_FLAG": 0,
            "PERFORM_CNS_SCORE": 750,
            "PERFORM_CNS_SCORE_DESCRIPTION": "A-Very Low Risk",
            "PRI_NO_OF_ACCTS": 3,
            "PRI_ACTIVE_ACCTS": 2,
            "PRI_OVERDUE_ACCTS": 0,
            "PRI_CURRENT_BALANCE": 10000,
            "PRI_SANCTIONED_AMOUNT": 50000,
            "PRI_DISBURSED_AMOUNT": 45000,
            "SEC_NO_OF_ACCTS": 1,
            "SEC_ACTIVE_ACCTS": 1,
            "SEC_OVERDUE_ACCTS": 0,
            "SEC_CURRENT_BALANCE": 5000,
            "SEC_SANCTIONED_AMOUNT": 20000,
            "SEC_DISBURSED_AMOUNT": 18000,
            "PRIMARY_INSTAL_AMT": 5000,
            "SEC_INSTAL_AMT": 2000,
            "NEW_ACCTS_IN_LAST_SIX_MONTHS": 0,
            "DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS": 0,
            "AVERAGE_ACCT_AGE": "5yrs 6mon",
            "CREDIT_HISTORY_LENGTH": "8yrs 3mon",
            "NO_OF_INQUIRIES": 2
        }
        
        response = requests.post("http://localhost:8000/predict", json=sample_data)
        if response.status_code == 200:
            print("✅ Prediction endpoint working!")
            result = response.json()
            print(f"Prediction: {result['prediction']}")
            print(f"Probability: {result['probability']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Model used: {result['model_used']}")
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
        
        print("\n🎉 API testing completed!")
        print("\n📱 Streamlit frontend should be available at: http://localhost:8501")
        print("🔗 API documentation available at: http://localhost:8000/docs")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on port 8000")
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")

if __name__ == "__main__":
    test_api()
