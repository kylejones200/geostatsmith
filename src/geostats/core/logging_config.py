"""
Logging configuration for geostats library
"""

import logging
import sys
from typing import Optional

from .constants import LOG_FORMAT, LOG_DATE_FORMAT


def setup_logger(
    name: str,
    level: int = logging.WARNING,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    level : int
        Logging level (default: logging.WARNING)
    format_string : str, optional
        Log message format
    date_format : str, optional
        Date format
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(level)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Formatter
    fmt = format_string or LOG_FORMAT
    date_fmt = date_format or LOG_DATE_FORMAT
    formatter = logging.Formatter(fmt, datefmt=date_fmt)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name)
