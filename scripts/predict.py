#!/usr/bin/env python3

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import ModelPredictor
from src.utils.logger import setup_logger

def main():
    logger = setup_logger("prediction")
    
    try:
        logger.info("Loading models and making prediction...")
        
        predictor = ModelPredictor()
        predictor.load_models()
        
        sample_input = {
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
        
        result = predictor.predict_best_model(sample_input)
        logger.info(f"Prediction result: {result}")
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
