import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import pickle
from pathlib import Path
from src.utils.logger import setup_logger
from src.utils.exceptions import FeatureEngineeringError
from src.config import MODEL_CONFIG, ARTIFACTS_DIR
import re

logger = setup_logger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.logger = logger
        self.target_column = MODEL_CONFIG['target_column']
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_columns = ['EMPLOYMENT_TYPE', 'PERFORM_CNS_SCORE_DESCRIPTION']
        
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            
            df_copy['DISBURSED_TO_ASSET_RATIO'] = df_copy['DISBURSED_AMOUNT'] / df_copy['ASSET_COST']
            df_copy['PRI_UTILIZATION_RATIO'] = df_copy['PRI_DISBURSED_AMOUNT'] / (df_copy['PRI_SANCTIONED_AMOUNT'] + 1)
            df_copy['SEC_UTILIZATION_RATIO'] = df_copy['SEC_DISBURSED_AMOUNT'] / (df_copy['SEC_SANCTIONED_AMOUNT'] + 1)
            df_copy['ACTIVE_TO_TOTAL_PRIMARY'] = df_copy['PRI_ACTIVE_ACCTS'] / (df_copy['PRI_NO_OF_ACCTS'] + 1)
            df_copy['ACTIVE_TO_TOTAL_SECONDARY'] = df_copy['SEC_ACTIVE_ACCTS'] / (df_copy['SEC_NO_OF_ACCTS'] + 1)
            
            self.logger.info("Ratio features created successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Failed to create ratio features: {str(e)}")
            raise FeatureEngineeringError(f"Failed to create ratio features: {str(e)}")
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            
            df_copy['CREDIT_SCORE_AGE_INTERACTION'] = df_copy['PERFORM_CNS_SCORE'] * df_copy['DATE_OF_BIRTH']
            df_copy['AMOUNT_CREDIT_INTERACTION'] = df_copy['DISBURSED_AMOUNT'] * df_copy['PERFORM_CNS_SCORE']
            df_copy['LTV_AMOUNT_INTERACTION'] = df_copy['LTV'] * df_copy['DISBURSED_AMOUNT']
            
            self.logger.info("Interaction features created successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Failed to create interaction features: {str(e)}")
            raise FeatureEngineeringError(f"Failed to create interaction features: {str(e)}")
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        try:
            df_copy = df.copy()

            for col in self.categorical_columns:
                if col in df_copy.columns:
                    values = df_copy[col].astype(str).fillna('Missing')
                    if fit:
                        if col not in self.label_encoders:
                            self.label_encoders[col] = LabelEncoder()
                        # Fit on observed values + a reserved 'Unknown' class
                        unique_vals = pd.Index(values.unique()).tolist()
                        if 'Unknown' not in unique_vals:
                            unique_vals.append('Unknown')
                        self.label_encoders[col].fit(unique_vals)
                        # Transform current values
                        df_copy[col] = self.label_encoders[col].transform(values)
                    else:
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            # Map unseen to 'Unknown' before transform
                            known = set(le.classes_)
                            mapped = values.where(values.isin(known), other='Unknown')
                            df_copy[col] = le.transform(mapped)
            self.logger.info("Categorical features encoded successfully")
            return df_copy

        except Exception as e:
            self.logger.error(f"Failed to encode categorical features: {str(e)}")
            raise FeatureEngineeringError(f"Failed to encode categorical features: {str(e)}")
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            
            numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != self.target_column]
            
            for col in numeric_columns:
                if col in df_copy.columns:
                    Q1 = df_copy[col].quantile(0.25)
                    Q3 = df_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
            
            self.logger.info("Outliers handled successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Failed to handle outliers: {str(e)}")
            raise FeatureEngineeringError(f"Failed to handle outliers: {str(e)}")
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            
            numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != self.target_column]
            
            if fit:
                df_copy[numeric_columns] = self.scaler.fit_transform(df_copy[numeric_columns])
            else:
                # Ensure all columns that were used during fit are present
                if hasattr(self.scaler, 'feature_names_in_'):
                    missing_cols = set(self.scaler.feature_names_in_) - set(df_copy.columns)
                    if missing_cols:
                        for col in missing_cols:
                            df_copy[col] = 0
                    # Reorder columns to match training data
                    df_copy = df_copy[self.scaler.feature_names_in_]
                else:
                    df_copy[numeric_columns] = self.scaler.transform(df_copy[numeric_columns])
            
            self.logger.info("Features scaled successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Failed to scale features: {str(e)}")
            raise FeatureEngineeringError(f"Failed to scale features: {str(e)}")
    
    def engineer_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None, fit: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            self.logger.info("Starting feature engineering...")
            
            train_processed = self.convert_tenure_columns(train_df)
            train_processed = self.create_ratio_features(train_processed)
            train_processed = self.create_interaction_features(train_processed)
            train_processed = self.encode_categorical_features(train_processed, fit=fit)
            train_processed = self.handle_outliers(train_processed)
            train_processed = self.scale_features(train_processed, fit=fit)
            
            if test_df is not None:
                test_processed = self.convert_tenure_columns(test_df)
                test_processed = self.create_ratio_features(test_processed)
                test_processed = self.create_interaction_features(test_processed)
                test_processed = self.encode_categorical_features(test_processed, fit=False)
                test_processed = self.handle_outliers(test_processed)
                test_processed = self.scale_features(test_processed, fit=False)
            else:
                test_processed = None
            
            self.feature_names = [col for col in train_processed.columns if col != self.target_column]
            
            self.logger.info("Feature engineering completed successfully")
            return train_processed, test_processed
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise FeatureEngineeringError(f"Feature engineering failed: {str(e)}")
    
    def prepare_training_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
        try:
            train_processed, test_processed = self.engineer_features(train_df, test_df, fit=True)
            
            X = train_processed.drop(columns=[self.target_column])
            y = train_processed[self.target_column]
            
            self.logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
            
            if test_processed is not None:
                self.logger.info(f"Test data shape: {test_processed.shape}")
            
            return train_processed, test_processed, X, y
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {str(e)}")
            raise FeatureEngineeringError(f"Failed to prepare training data: {str(e)}")
    
    def save_artifacts(self):
        try:
            artifacts_path = ARTIFACTS_DIR / "feature_engineering"
            artifacts_path.mkdir(exist_ok=True)
            
            with open(artifacts_path / "label_encoders.pkl", "wb") as f:
                pickle.dump(self.label_encoders, f)
            
            with open(artifacts_path / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            
            with open(artifacts_path / "feature_names.pkl", "wb") as f:
                pickle.dump(self.feature_names, f)
            
            self.logger.info("Feature engineering artifacts saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save feature engineering artifacts: {str(e)}")
            raise FeatureEngineeringError(f"Failed to save feature engineering artifacts: {str(e)}")
    
    def load_artifacts(self):
        try:
            artifacts_path = ARTIFACTS_DIR / "feature_engineering"
            
            with open(artifacts_path / "label_encoders.pkl", "rb") as f:
                self.label_encoders = pickle.load(f)
            
            with open(artifacts_path / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            
            with open(artifacts_path / "feature_names.pkl", "rb") as f:
                self.feature_names = pickle.load(f)
            
            self.logger.info("Feature engineering artifacts loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load feature engineering artifacts: {str(e)}")
            raise FeatureEngineeringError(f"Failed to load feature engineering artifacts: {str(e)}")
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # First, ensure we have the same preprocessing as training data
            from src.data.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            # Parse dates first to convert strings to numeric epoch days
            processed_df = preprocessor.parse_dates(df)
            processed_df = preprocessor.handle_missing_values(processed_df)

            processed_df = self.convert_tenure_columns(processed_df)
            processed_df = self.create_ratio_features(processed_df)
            processed_df = self.create_interaction_features(processed_df)
            processed_df = self.encode_categorical_features(processed_df, fit=False)
            processed_df = self.handle_outliers(processed_df)
            processed_df = self.scale_features(processed_df, fit=False)

            if self.feature_names:
                # Ensure all expected features are present
                missing_features = set(self.feature_names) - set(processed_df.columns)
                for feature in missing_features:
                    processed_df[feature] = 0
                processed_df = processed_df[self.feature_names]
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Failed to transform new data: {str(e)}")
            raise FeatureEngineeringError(f"Failed to transform new data: {str(e)}")

    def _tenure_to_months(self, val) -> int:
        if pd.isna(val):
            return 0
        s = str(val).lower().strip()
        yrs = 0
        mons = 0
        y = re.search(r'(\d+)\s*yrs?', s)
        m = re.search(r'(\d+)\s*mon', s)
        if y: yrs = int(y.group(1))
        if m: mons = int(m.group(1))
        return int(yrs * 12 + mons)

    def convert_tenure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col in ['AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH']:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(self._tenure_to_months)
        return df_copy
