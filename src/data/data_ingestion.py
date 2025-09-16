import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import kaggle
import shutil

from src.utils.logger import setup_logger
from src.utils.exceptions import DataIngestionError
from src.config import DATA_DIR

logger = setup_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.logger = logger
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        
    def download_data(self) -> None:
        """Download data from Kaggle"""
        try:
            self.logger.info("Downloading data from Kaggle...")
            
            # Download the vehicle loan dataset using kaggle API
            kaggle.api.dataset_download_files(
                'avikpaul4u/vehicle-loan-default-prediction',
                # 'avikpaul/vehicle-loan-default-prediction',
                path=str(self.data_dir),
                unzip=True
            )
            
            self.logger.info("Data downloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to download data: {str(e)}")
            raise DataIngestionError(f"Failed to download data: {str(e)}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data"""
        try:
            self.logger.info("Loading data...")
            
            # Check if data files exist, if not download them
            train_file = self.data_dir / "train.csv"
            test_file = self.data_dir / "test.csv"
            
            if not train_file.exists() or not test_file.exists():
                self.logger.info("Data files not found, downloading...")
                self.download_data()
            
            # Load training data
            train_df = pd.read_csv(train_file)
            self.logger.info(f"Training data loaded: {train_df.shape}")
            
            # Load test data
            test_df = pd.read_csv(test_file)
            self.logger.info(f"Test data loaded: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise DataIngestionError(f"Failed to load data: {str(e)}")
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """Get basic information about the dataset"""
        try:
            info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            self.logger.info(f"Dataset info: {info['shape']} rows, {info['shape'][1]} columns")
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get data info: {str(e)}")
            raise DataIngestionError(f"Failed to get data info: {str(e)}")
