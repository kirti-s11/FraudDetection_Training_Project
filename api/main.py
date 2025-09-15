from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uvicorn
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import ModelPredictor
from src.utils.logger import setup_logger
from src.utils.exceptions import ModelPredictionError
from src.config import API_CONFIG

logger = setup_logger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="API for vehicle loan default prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    UNIQUEID: int = Field(..., description="Unique identifier")
    DISBURSED_AMOUNT: int = Field(..., description="Amount disbursed")
    ASSET_COST: int = Field(..., description="Asset cost")
    LTV: float = Field(..., description="Loan to value ratio")
    BRANCH_ID: int = Field(..., description="Branch ID")
    SUPPLIER_ID: int = Field(..., description="Supplier ID")
    MANUFACTURER_ID: int = Field(..., description="Manufacturer ID")
    CURRENT_PINCODE_ID: int = Field(..., description="Current pincode ID")
    DATE_OF_BIRTH: str = Field(..., description="Date of birth (DD-MM-YYYY)")
    EMPLOYMENT_TYPE: str = Field(..., description="Employment type")
    DISBURSAL_DATE: str = Field(..., description="Disbursal date (DD-MM-YYYY)")
    STATE_ID: int = Field(..., description="State ID")
    EMPLOYEE_CODE_ID: int = Field(..., description="Employee code ID")
    MOBILENO_AVL_FLAG: int = Field(..., description="Mobile number availability flag")
    AADHAR_FLAG: int = Field(..., description="Aadhar availability flag")
    PAN_FLAG: int = Field(..., description="PAN availability flag")
    VOTERID_FLAG: int = Field(..., description="Voter ID availability flag")
    DRIVING_FLAG: int = Field(..., description="Driving license availability flag")
    PASSPORT_FLAG: int = Field(..., description="Passport availability flag")
    PERFORM_CNS_SCORE: int = Field(..., description="CNS score")
    PERFORM_CNS_SCORE_DESCRIPTION: str = Field(..., description="CNS score description")
    PRI_NO_OF_ACCTS: int = Field(..., description="Primary number of accounts")
    PRI_ACTIVE_ACCTS: int = Field(..., description="Primary active accounts")
    PRI_OVERDUE_ACCTS: int = Field(..., description="Primary overdue accounts")
    PRI_CURRENT_BALANCE: int = Field(..., description="Primary current balance")
    PRI_SANCTIONED_AMOUNT: int = Field(..., description="Primary sanctioned amount")
    PRI_DISBURSED_AMOUNT: int = Field(..., description="Primary disbursed amount")
    SEC_NO_OF_ACCTS: int = Field(..., description="Secondary number of accounts")
    SEC_ACTIVE_ACCTS: int = Field(..., description="Secondary active accounts")
    SEC_OVERDUE_ACCTS: int = Field(..., description="Secondary overdue accounts")
    SEC_CURRENT_BALANCE: int = Field(..., description="Secondary current balance")
    SEC_SANCTIONED_AMOUNT: int = Field(..., description="Secondary sanctioned amount")
    SEC_DISBURSED_AMOUNT: int = Field(..., description="Secondary disbursed amount")
    PRIMARY_INSTAL_AMT: int = Field(..., description="Primary installment amount")
    SEC_INSTAL_AMT: int = Field(..., description="Secondary installment amount")
    NEW_ACCTS_IN_LAST_SIX_MONTHS: int = Field(..., description="New accounts in last 6 months")
    DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS: int = Field(..., description="Delinquent accounts in last 6 months")
    AVERAGE_ACCT_AGE: str = Field(..., description="Average account age (Xyrs Ymon)")
    CREDIT_HISTORY_LENGTH: str = Field(..., description="Credit history length (Xyrs Ymon)")
    NO_OF_INQUIRIES: int = Field(..., description="Number of inquiries")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Prediction (0: No default, 1: Default)")
    probability: Dict[str, float] = Field(..., description="Class probabilities")
    confidence: float = Field(..., description="Confidence score")
    model_used: str = Field(..., description="Model used for prediction")

class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest] = Field(..., description="List of prediction requests")

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")

predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = ModelPredictor()
        predictor.load_models()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": len(predictor.models) if predictor else 0}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if not predictor:
            raise HTTPException(status_code=500, detail="Predictor not initialized")
        
        input_data = request.dict()
        result = predictor.predict_best_model(input_data)
        
        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            confidence=result['confidence'],
            model_used=result['model_name']
        )
        
    except ModelPredictionError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    try:
        if not predictor:
            raise HTTPException(status_code=500, detail="Predictor not initialized")
        
        predictions = []
        for item in request.data:
            input_data = item.dict()
            result = predictor.predict_best_model(input_data)
            predictions.append({
                'unique_id': input_data['UNIQUEID'],
                'prediction': result['prediction'],
                'probability': result['probability'],
                'confidence': result['confidence'],
                'model_used': result['model_name']
            })
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/all-models")
async def predict_all_models(request: PredictionRequest):
    try:
        if not predictor:
            raise HTTPException(status_code=500, detail="Predictor not initialized")
        
        input_data = request.dict()
        results = predictor.predict_all_models(input_data)
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"All models prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/models")
async def get_available_models():
    try:
        if not predictor:
            raise HTTPException(status_code=500, detail="Predictor not initialized")
        
        return {
            "available_models": list(predictor.models.keys()),
            "total_models": len(predictor.models)
        }
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=API_CONFIG['debug']
    )
