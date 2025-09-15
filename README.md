# Fraud Detection ML Pipeline

An end-to-end machine learning pipeline for vehicle loan default prediction with Streamlit frontend and FastAPI backend.

## Features

- **Data Ingestion**: Automated data downloading from Kaggle
- **Preprocessing**: Comprehensive data cleaning and transformation
- **Feature Engineering**: Advanced feature creation and scaling
- **Model Training**: Decision Tree, Random Forest, and Gradient Boosting
- **API Backend**: FastAPI with prediction endpoints
- **Web Frontend**: Streamlit interface for predictions
- **Logging**: Comprehensive logging and exception handling
- **Artifact Management**: Model and feature engineering artifact storage

## Project Structure

```
├── src/
│   ├── config.py                 # Configuration settings
│   ├── data/
│   │   ├── data_ingestion.py     # Data loading utilities
│   │   └── preprocessing.py      # Data preprocessing
│   ├── features/
│   │   └── feature_engineering.py # Feature creation and scaling
│   ├── models/
│   │   ├── model_trainer.py      # Model training pipeline
│   │   └── predictor.py          # Prediction service
│   └── utils/
│       ├── logger.py             # Logging configuration
│       └── exceptions.py         # Custom exceptions
├── api/
│   └── main.py                   # FastAPI application
├── frontend/
│   └── app.py                    # Streamlit application
├── scripts/
│   ├── train.py                  # Training script
│   └── predict.py                # Prediction script
├── models/                       # Trained models storage
├── artifacts/                    # Feature engineering artifacts
├── logs/                         # Application logs
└── data/                         # Data storage
```


## Usage

### 1. Training the Models

Run the training pipeline to download data, preprocess, engineer features, and train models:

```bash
python scripts/train.py
```

This will:
- Download the vehicle loan dataset from Kaggle
- Preprocess the data (handle missing values, parse dates, etc.)
- Engineer features (ratios, interactions, scaling)
- Train Decision Tree, Random Forest, and Gradient Boosting models
- Save models and feature engineering artifacts

### 2. Making Predictions

#### Command Line Prediction
```bash
python scripts/predict.py
```

#### API Backend
Start the FastAPI server:
```bash
cd api
python main.py
```

The API will be available at `http://localhost:8000`

API Endpoints:
- `GET /` - Health check
- `GET /health` - API health status
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch prediction
- `POST /predict/all-models` - Predict with all models
- `GET /models` - Available models

#### Web Frontend
Start the Streamlit application:
```bash
streamlit run frontend/app.py
```

The web interface will be available at `http://localhost:8501`

Features:
- Single prediction form
- Batch prediction via CSV upload
- Model information display
- Real-time prediction results

## Model Performance

The pipeline trains three models:
- **Decision Tree**: Fast training, interpretable
- **Random Forest**: Robust ensemble method
- **Gradient Boosting**: High performance, handles complex patterns

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC
- Cross-validation scores

## Configuration

Modify `src/config.py` to adjust:
- Model hyperparameters
- API settings
- Logging configuration
- File paths

## API Usage Examples

### Single Prediction
```python
import requests

data = {
    "UNIQUEID": 12345,
    "DISBURSED_AMOUNT": 50000,
    "ASSET_COST": 60000,
    "LTV": 0.8,
    # ... other fields
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
```

### Batch Prediction
```python
import pandas as pd
import requests

df = pd.read_csv("test_data.csv")
data = {"data": df.to_dict('records')}

response = requests.post("http://localhost:8000/predict/batch", json=data)
results = response.json()
```

## Logging

All components use structured logging with:
- File output to `logs/app.log`
- Console output with timestamps
- Different log levels (INFO, WARNING, ERROR)
- Exception tracking

## Error Handling

The pipeline includes comprehensive error handling:
- Custom exceptions for different components
- Graceful failure handling
- Detailed error messages
- API error responses

## Artifacts

The system saves:
- Trained models in `models/` directory
- Feature engineering artifacts in `artifacts/`
- Model metrics and evaluation results
- Preprocessed data for future use

## Requirements

- Python 3.8+
- Pandas, NumPy, Scikit-learn
- FastAPI, Uvicorn
- Streamlit
- XGBoost, CatBoost, LightGBM
- Kaggle Hub for data access
