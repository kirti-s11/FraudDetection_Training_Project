#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.utils.logger import setup_logger
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

def main():
    logger = setup_logger("quick_training")
    
    try:
        logger.info("Starting quick ML training pipeline...")
        
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
        
        logger.info("Training quick models...")
        
        # Train Decision Tree
        dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42)
        dt_model.fit(X, y)
        joblib.dump(dt_model, "models/decision_tree.joblib")
        logger.info("Decision Tree saved")
        
        # Train Random Forest with fewer estimators for speed
        rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42, n_jobs=-1)
        rf_model.fit(X, y)
        joblib.dump(rf_model, "models/random_forest.joblib")
        logger.info("Random Forest saved")
        
        # Create a simple gradient boosting model
        from sklearn.ensemble import GradientBoostingClassifier
        gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=6, random_state=42)
        gb_model.fit(X, y)
        joblib.dump(gb_model, "models/gradient_boosting.joblib")
        logger.info("Gradient Boosting saved")
        
        logger.info("Quick training completed successfully!")
        
    except Exception as e:
        logger.error(f"Quick training pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
