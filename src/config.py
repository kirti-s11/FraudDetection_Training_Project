import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'target_column': 'LOAN_DEFAULT',
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'models': {
        'decision_tree': {
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42,
            'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42
        }
    }
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False
}

# Logging configuration
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'app.log'
}
