#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.utils.logger import setup_logger

def main():
    logger = setup_logger("training_pipeline")
    
    try:
        logger.info("Starting ML training pipeline...")
        
        data_ingestion = DataIngestion()
        logger.info("Loading data...")
        train_df, test_df = data_ingestion.load_data()
        
        preprocessor = DataPreprocessor()
        logger.info("Preprocessing data...")
        train_processed, test_processed = preprocessor.preprocess_data(train_df, test_df)
        
        feature_engineer = FeatureEngineer()
        logger.info("Engineering features...")
        train_final, test_final, X, y = feature_engineer.prepare_training_data(train_processed, test_processed)
        
        logger.info("Saving feature engineering artifacts...")
        feature_engineer.save_artifacts()
        
        model_trainer = ModelTrainer()
        logger.info("Training models...")
        results = model_trainer.train_models(X, y)
        
        logger.info("Saving models...")
        model_trainer.save_models(results)
        
        best_model_name, best_model = model_trainer.get_best_model(results)
        logger.info(f"Training completed successfully. Best model: {best_model_name}")
        
        for model_name, model_data in results.items():
            metrics = model_data['metrics']
            logger.info(f"{model_name}: ROC AUC = {metrics['roc_auc']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
