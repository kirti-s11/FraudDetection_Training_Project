import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List
from src.utils.logger import setup_logger
from src.utils.exceptions import ModelPredictionError
from src.config import MODELS_DIR
from src.features.feature_engineering import FeatureEngineer

logger = setup_logger(__name__)

class ModelPredictor:
    def __init__(self):
        self.logger = logger
        self.models_dir = MODELS_DIR
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.best_model_name = None
        
    def load_models(self) -> None:
        """Load all trained models"""
        try:
            self.logger.info("Loading models...")
            
            # Load feature engineering artifacts
            self.feature_engineer.load_artifacts()
            
            # Load all model files
            model_files = list(self.models_dir.glob("*.joblib"))
            
            if not model_files:
                raise ModelPredictionError("No trained models found. Please train models first.")
            
            for model_file in model_files:
                model_name = model_file.stem
                model = joblib.load(model_file)
                self.models[model_name] = model
                self.logger.info(f"Loaded {model_name}")
            
            # Determine best model based on saved metrics
            self.best_model_name = self._get_best_model_name()
            
            self.logger.info(f"Loaded {len(self.models)} models. Best model: {self.best_model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise ModelPredictionError(f"Failed to load models: {str(e)}")
    
    def _get_best_model_name(self) -> str:
        """Get the best model name from saved metrics"""
        try:
            best_model = None
            best_score = -1
            
            for model_name in self.models.keys():
                metrics_file = self.models_dir / f"{model_name}_metrics.json"
                if metrics_file.exists():
                    import json
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        score = metrics.get('roc_auc', 0)
                        if score > best_score:
                            best_score = score
                            best_model = model_name
            
            return best_model or list(self.models.keys())[0]
            
        except Exception as e:
            self.logger.warning(f"Could not determine best model: {str(e)}")
            return list(self.models.keys())[0]
    
    def _prepare_input_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data for prediction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([input_data])
            
            # Apply feature engineering
            processed_df = self.feature_engineer.transform_new_data(df)
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Failed to prepare input data: {str(e)}")
            raise ModelPredictionError(f"Failed to prepare input data: {str(e)}")
    
    def predict_single_model(self, input_data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Make prediction using a specific model"""
        try:
            if model_name not in self.models:
                raise ModelPredictionError(f"Model {model_name} not found")
            
            # Prepare input data
            processed_df = self._prepare_input_data(input_data)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(processed_df)[0]
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_df)[0]
                prob_dict = {
                    'class_0': float(probabilities[0]),
                    'class_1': float(probabilities[1])
                }
            else:
                prob_dict = {
                    'class_0': 0.5,
                    'class_1': 0.5
                }
            
            # Calculate confidence (max probability)
            confidence = max(prob_dict.values())
            
            return {
                'prediction': int(prediction),
                'probability': prob_dict,
                'confidence': float(confidence),
                'model_name': model_name
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_name}: {str(e)}")
            raise ModelPredictionError(f"Prediction failed for {model_name}: {str(e)}")
    
    def predict_best_model(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using the best model"""
        try:
            if not self.models:
                raise ModelPredictionError("No models loaded")
            
            if not self.best_model_name:
                self.best_model_name = list(self.models.keys())[0]
            
            return self.predict_single_model(input_data, self.best_model_name)
            
        except Exception as e:
            self.logger.error(f"Best model prediction failed: {str(e)}")
            raise ModelPredictionError(f"Best model prediction failed: {str(e)}")
    
    def predict_all_models(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using all models"""
        try:
            if not self.models:
                raise ModelPredictionError("No models loaded")
            
            results = {}
            
            for model_name in self.models.keys():
                try:
                    result = self.predict_single_model(input_data, model_name)
                    results[model_name] = result
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {model_name}: {str(e)}")
                    results[model_name] = {
                        'error': str(e),
                        'prediction': 0,
                        'probability': {'class_0': 0.5, 'class_1': 0.5},
                        'confidence': 0.0,
                        'model_name': model_name
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"All models prediction failed: {str(e)}")
            raise ModelPredictionError(f"All models prediction failed: {str(e)}")
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for a batch of inputs"""
        try:
            if not self.models:
                raise ModelPredictionError("No models loaded")
            
            results = []
            
            for i, input_data in enumerate(input_data_list):
                try:
                    result = self.predict_best_model(input_data)
                    result['unique_id'] = input_data.get('UNIQUEID', i)
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Batch prediction failed for item {i}: {str(e)}")
                    results.append({
                        'unique_id': input_data.get('UNIQUEID', i),
                        'error': str(e),
                        'prediction': 0,
                        'probability': {'class_0': 0.5, 'class_1': 0.5},
                        'confidence': 0.0,
                        'model_name': 'error'
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {str(e)}")
            raise ModelPredictionError(f"Batch prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        try:
            info = {
                'total_models': len(self.models),
                'model_names': list(self.models.keys()),
                'best_model': self.best_model_name,
                'feature_engineer_loaded': hasattr(self.feature_engineer, 'scaler') and self.feature_engineer.scaler is not None
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {str(e)}")
            raise ModelPredictionError(f"Failed to get model info: {str(e)}")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data format"""
        try:
            required_fields = [
                'UNIQUEID', 'DISBURSED_AMOUNT', 'ASSET_COST', 'LTV',
                'BRANCH_ID', 'SUPPLIER_ID', 'MANUFACTURER_ID', 'CURRENT_PINCODE_ID',
                'DATE_OF_BIRTH', 'EMPLOYMENT_TYPE', 'DISBURSAL_DATE', 'STATE_ID',
                'EMPLOYEE_CODE_ID', 'MOBILENO_AVL_FLAG', 'AADHAR_FLAG', 'PAN_FLAG',
                'VOTERID_FLAG', 'DRIVING_FLAG', 'PASSPORT_FLAG', 'PERFORM_CNS_SCORE',
                'PERFORM_CNS_SCORE_DESCRIPTION', 'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS',
                'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT',
                'PRI_DISBURSED_AMOUNT', 'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS',
                'SEC_OVERDUE_ACCTS', 'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT',
                'SEC_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT', 'SEC_INSTAL_AMT',
                'NEW_ACCTS_IN_LAST_SIX_MONTHS', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS',
                'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH', 'NO_OF_INQUIRIES'
            ]
            
            missing_fields = [field for field in required_fields if field not in input_data]
            
            if missing_fields:
                raise ModelPredictionError(f"Missing required fields: {missing_fields}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {str(e)}")
            raise ModelPredictionError(f"Input validation failed: {str(e)}")
