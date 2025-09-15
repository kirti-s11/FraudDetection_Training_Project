import logging
import sys
from pathlib import Path
from src.config import LOG_CONFIG, LOGS_DIR

def setup_logger(name: str = "fraud_detection", level: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level or LOG_CONFIG['level']))
    
    formatter = logging.Formatter(LOG_CONFIG['format'])
    
    # File handler
    file_handler = logging.FileHandler(LOG_CONFIG['file'])
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
