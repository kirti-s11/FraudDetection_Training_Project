import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any, Tuple
import joblib
import json
from pathlib import Path
from src.utils.logger import setup_logger
from src.utils.exceptions import ModelTrainingError
from src.config import MODEL_CONFIG, MODELS_DIR

logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self):
        self.logger = logger
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)
        self.config = MODEL_CONFIG
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models with their configurations"""
        try:
            models = {}
            
            # Decision Tree
            models['decision_tree'] = DecisionTreeClassifier(
                max_depth=self.config['models']['decision_tree']['max_depth'],
                min_samples_split=self.config['models']['decision_tree']['min_samples_split'],
                min_samples_leaf=self.config['models']['decision_tree']['min_samples_leaf'],
                random_state=self.config['models']['decision_tree']['random_state']
            )
            
            # Random Forest
            models['random_forest'] = RandomForestClassifier(
                n_estimators=self.config['models']['random_forest']['n_estimators'],
                max_depth=self.config['models']['random_forest']['max_depth'],
                min_samples_split=self.config['models']['random_forest']['min_samples_split'],
                min_samples_leaf=self.config['models']['random_forest']['min_samples_leaf'],
                random_state=self.config['models']['random_forest']['random_state'],
                n_jobs=self.config['models']['random_forest']['n_jobs']
            )
            
            # Gradient Boosting
            models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=self.config['models']['gradient_boosting']['n_estimators'],
                learning_rate=self.config['models']['gradient_boosting']['learning_rate'],
                max_depth=self.config['models']['gradient_boosting']['max_depth'],
                min_samples_split=self.config['models']['gradient_boosting']['min_samples_split'],
                min_samples_leaf=self.config['models']['gradient_boosting']['min_samples_leaf'],
                random_state=self.config['models']['gradient_boosting']['random_state']
            )
            
            self.logger.info(f"Initialized {len(models)} models")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
            raise ModelTrainingError(f"Failed to initialize models: {str(e)}")
    
    def train_model(self, model, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """Train a single model and return results"""
        try:
            self.logger.info(f"Training {model_name}...")
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=self.config['test_size'], 
                random_state=self.config['random_state'],
                stratify=y
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted'),
                'recall': recall_score(y_val, y_pred, average='weighted'),
                'f1_score': f1_score(y_val, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else 0.0
            }
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=self.config['cv_folds'], scoring='roc_auc')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            self.logger.info(f"{model_name} training completed - ROC AUC: {metrics['roc_auc']:.4f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train {model_name}: {str(e)}")
            raise ModelTrainingError(f"Failed to train {model_name}: {str(e)}")
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        try:
            self.logger.info("Starting model training...")
            models = self.initialize_models()
            results = {}

            for model_name, model in models.items():
                try:
                    result = self.train_model(model, X, y, model_name)
                    results[model_name] = result
                    # SAVE IMMEDIATELY after training this model
                    self.save_single_model(model_name, result)
                except Exception as e:
                    self.logger.warning(f"Skipping {model_name} due to error: {e}")

            if not results:
                raise ModelTrainingError("All models failed to train.")

            # Optionally write comparison summary for whatever succeeded
            comparison = self.compare_models(results)
            with open(self.models_dir / "model_comparison.json", 'w') as f:
                json.dump(comparison, f, indent=2)

            self.logger.info(f"Trained and saved {len(results)} models successfully")
            return results
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise ModelTrainingError(f"Model training failed: {str(e)}")
    
    # def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    #     """Train all models and return results"""
    #     try:
    #         self.logger.info("Starting model training...")
    #         models = self.initialize_models()
    #         results = {}
    #         for model_name, model in models.items():
    #             try:
    #                 result = self.train_model(model, X, y, model_name)
    #                 results[model_name] = result
    #             except Exception as e:
    #                 self.logger.warning(f"Skipping {model_name} due to error: {e}")
    #         if not results:
    #             raise ModelTrainingError("All models failed to train.")
    #         self.logger.info(f"Trained {len(results)} models successfully")
    #         return results
    #     except Exception as e:
    #         self.logger.error(f"Model training failed: {str(e)}")
    #         raise ModelTrainingError(f"Model training failed: {str(e)}")
    
    def save_models(self, results: Dict[str, Any]) -> None:
        """Save trained models and their metadata"""
        try:
            self.logger.info("Saving models...")
            
            for model_name, result in results.items():
                # Save the model
                model_path = self.models_dir / f"{model_name}.joblib"
                joblib.dump(result['model'], model_path)
                
                # Save metrics
                metrics_path = self.models_dir / f"{model_name}_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(result['metrics'], f, indent=2)
                
                self.logger.info(f"{model_name} saved to {model_path}")
            
            # Save model comparison
            comparison = self.compare_models(results)
            comparison_path = self.models_dir / "model_comparison.json"
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            self.logger.info("All models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {str(e)}")
            raise ModelTrainingError(f"Failed to save models: {str(e)}")
    
    def load_models(self) -> Dict[str, Any]:
        """Load trained models from disk"""
        try:
            self.logger.info("Loading models...")
            
            models = {}
            model_files = list(self.models_dir.glob("*.joblib"))
            
            for model_file in model_files:
                model_name = model_file.stem
                model = joblib.load(model_file)
                models[model_name] = model
                self.logger.info(f"Loaded {model_name}")
            
            self.logger.info(f"Loaded {len(models)} models")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise ModelTrainingError(f"Failed to load models: {str(e)}")
    
    def compare_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare model performance"""
        try:
            comparison = {}
            
            for model_name, result in results.items():
                comparison[model_name] = {
                    'roc_auc': result['metrics']['roc_auc'],
                    'accuracy': result['metrics']['accuracy'],
                    'f1_score': result['metrics']['f1_score'],
                    'cv_mean': result['metrics']['cv_mean'],
                    'cv_std': result['metrics']['cv_std']
                }
            
            # Sort by ROC AUC
            sorted_models = sorted(comparison.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
            
            return {
                'models': dict(sorted_models),
                'best_model': sorted_models[0][0] if sorted_models else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compare models: {str(e)}")
            raise ModelTrainingError(f"Failed to compare models: {str(e)}")
    
    def get_best_model(self, results: Dict[str, Any]) -> Tuple[str, Any]:
        """Get the best performing model"""
        try:
            best_model_name = None
            best_score = -1
            best_model = None
            
            for model_name, result in results.items():
                score = result['metrics']['roc_auc']
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = result['model']
            
            self.logger.info(f"Best model: {best_model_name} (ROC AUC: {best_score:.4f})")
            return best_model_name, best_model
            
        except Exception as e:
            self.logger.error(f"Failed to get best model: {str(e)}")
            raise ModelTrainingError(f"Failed to get best model: {str(e)}")

    def save_single_model(self, model_name: str, result: Dict[str, Any]) -> None:
        model_path = self.models_dir / f"{model_name}.joblib"
        joblib.dump(result['model'], model_path)

        metrics_path = self.models_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(result['metrics'], f, indent=2)

        self.logger.info(f"{model_name} saved to {model_path}")
