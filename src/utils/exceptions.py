class FraudDetectionError(Exception):
    """Base exception for fraud detection pipeline"""
    pass

class DataIngestionError(FraudDetectionError):
    """Raised when data ingestion fails"""
    pass

class PreprocessingError(FraudDetectionError):
    """Raised when data preprocessing fails"""
    pass

class ModelTrainingError(FraudDetectionError):
    """Raised when model training fails"""
    pass

class ModelPredictionError(FraudDetectionError):
    """Raised when model prediction fails"""
    pass

class FeatureEngineeringError(FraudDetectionError):
    """Raised when feature engineering fails"""
    pass
