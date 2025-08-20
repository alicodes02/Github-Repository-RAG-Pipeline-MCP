"""
Centralized Logging Configuration
Provides consistent logging setup across all modules
"""

import logging
import os
from datetime import datetime
from pathlib import Path

# Global variable to track if logging has been configured
_logging_configured = False
_log_file_path = None

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    Setup centralized logging configuration (only once)
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory to store log files
    """
    global _logging_configured, _log_file_path
    
    # Only configure logging once
    if _logging_configured:
        return logging.getLogger("rag_mcp")
    
    # Create logs directory if it doesn't exist
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp (only once)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file_path = log_dir_path / f"rag_mcp_{timestamp}.log"
    
    # Configure logging format
    log_format = (
        "%(asctime)s | %(levelname)8s | %(filename)20s:%(lineno)4d | "
        "%(funcName)20s | %(message)s"
    )
    
    # Clear any existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(_log_file_path, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ],
        force=True  # Force reconfiguration
    )
    
    # Set the configured flag
    _logging_configured = True
    
    # Create logger for the application
    logger = logging.getLogger("rag_mcp")
    logger.info(f"Logging initialized. Log file: {_log_file_path}")
    
    return logger

def get_logger(name=None):
    """
    Get a logger instance for a specific module
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    # Ensure logging is configured
    if not _logging_configured:
        setup_logging()
    
    if name is None:
        name = "rag_mcp"
    
    # Return a child logger of the main logger
    return logging.getLogger("rag_mcp").getChild(name.split('.')[-1] if '.' in name else name)
