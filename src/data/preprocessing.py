import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime
from src.utils.logger import setup_logger
from src.utils.exceptions import PreprocessingError

logger = setup_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.logger = logger
        
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date columns"""
        try:
            df_copy = df.copy()
            
            date_columns = ['DATE_OF_BIRTH', 'DISBURSAL_DATE']
            
            for col in date_columns:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_datetime(df_copy[col], format='%d-%m-%Y', errors='coerce')
                    
                    df_copy[col] = (df_copy[col] - pd.Timestamp('1970-01-01')).dt.days
            
            self.logger.info("Date columns parsed successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Failed to parse dates: {str(e)}")
            raise PreprocessingError(f"Failed to parse dates: {str(e)}")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        
        try:
            df_copy = df.copy()
            
            # Fill missing values based on column type
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    # For categorical columns, fill with 'Missing'
                    df_copy[col] = df_copy[col].fillna('Missing')
                else:
                    # For numerical columns, fill with median
                    df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            
            self.logger.info("Missing values handled successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Failed to handle missing values: {str(e)}")
            raise PreprocessingError(f"Failed to handle missing values: {str(e)}")
    
    def clean_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize categorical columns"""
        try:
            df_copy = df.copy()
            
            # Clean employment type
            if 'EMPLOYMENT_TYPE' in df_copy.columns:
                df_copy['EMPLOYMENT_TYPE'] = df_copy['EMPLOYMENT_TYPE'].str.strip().str.title()
                df_copy['EMPLOYMENT_TYPE'] = df_copy['EMPLOYMENT_TYPE'].replace('', 'Missing')
            
            # Clean CNS score description
            if 'PERFORM_CNS_SCORE_DESCRIPTION' in df_copy.columns:
                df_copy['PERFORM_CNS_SCORE_DESCRIPTION'] = df_copy['PERFORM_CNS_SCORE_DESCRIPTION'].str.strip()
                df_copy['PERFORM_CNS_SCORE_DESCRIPTION'] = df_copy['PERFORM_CNS_SCORE_DESCRIPTION'].replace('', 'Not Scored: Sufficient History Not Available')
            
            self.logger.info("Categorical columns cleaned successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Failed to clean categorical columns: {str(e)}")
            raise PreprocessingError(f"Failed to clean categorical columns: {str(e)}")
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        try:
            df_copy = df.copy()
            
            # Get numerical columns
            numerical_columns = df_copy.select_dtypes(include=[np.number]).columns
            numerical_columns = [col for col in numerical_columns if col not in ['LOAN_DEFAULT', 'UNIQUEID']]
            
            for col in numerical_columns:
                if col in df_copy.columns:
                    Q1 = df_copy[col].quantile(0.25)
                    Q3 = df_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Clip outliers
                    df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
            
            self.logger.info("Outliers handled successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Failed to handle outliers: {str(e)}")
            raise PreprocessingError(f"Failed to handle outliers: {str(e)}")
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables to numerical values"""
        try:
            df_copy = df.copy()
            
            # Get categorical columns (object type)
            categorical_columns = df_copy.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                if col not in ['UNIQUEID']:  # Skip ID columns
                    # Use label encoding for categorical variables
                    df_copy[col] = pd.Categorical(df_copy[col]).codes
                    # Handle negative codes (for missing values)
                    df_copy[col] = df_copy[col].replace(-1, 0)
            
            self.logger.info("Categorical variables encoded successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Failed to encode categorical variables: {str(e)}")
            raise PreprocessingError(f"Failed to encode categorical variables: {str(e)}")
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main preprocessing pipeline"""
        try:
            self.logger.info("Starting data preprocessing...")
            
            # Process training data
            train_processed = self.parse_dates(train_df)
            train_processed = self.handle_missing_values(train_processed)
            train_processed = self.clean_categorical_columns(train_processed)
            train_processed = self.handle_outliers(train_processed)
            train_processed = self.encode_categorical_variables(train_processed)  # Add this line
            
            # Process test data if provided
            if test_df is not None:
                test_processed = self.parse_dates(test_df)
                test_processed = self.handle_missing_values(test_processed)
                test_processed = self.clean_categorical_columns(test_processed)
                test_processed = self.handle_outliers(test_processed)
                test_processed = self.encode_categorical_variables(test_processed)  # Add this line
            else:
                test_processed = None
            
            self.logger.info("Data preprocessing completed successfully")
            return train_processed, test_processed
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            raise PreprocessingError(f"Data preprocessing failed: {str(e)}")
    
    def get_preprocessing_summary(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
        """Get summary of preprocessing changes"""
        try:
            summary = {
                'original_shape': original_df.shape,
                'processed_shape': processed_df.shape,
                'missing_values_before': original_df.isnull().sum().sum(),
                'missing_values_after': processed_df.isnull().sum().sum(),
                'columns_processed': list(processed_df.columns)
            }
            
            self.logger.info(f"Preprocessing summary: {summary}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get preprocessing summary: {str(e)}")
            raise PreprocessingError(f"Failed to get preprocessing summary: {str(e)}")
