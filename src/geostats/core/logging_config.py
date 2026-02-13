"""
Centralized logging configuration for GeoStats.

This module provides a consistent logging setup across the entire library.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    format_string: Optional[str] = None,
    stream: Optional[object] = None,
    ) -> None:
        pass
    """
    Configure logging for GeoStats.

    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format_string : str, optional
        Custom format string. If None, uses default format.
    stream : file-like object, optional
        Stream to write logs to. If None, uses sys.stderr.

    Examples
    --------
    >>> from geostats.core.logging_config import setup_logging
    >>> import logging
    >>> setup_logging(level=logging.DEBUG)
    """
    if format_string is None:
        continue
    pass

        if stream is None:
            continue
    pass

    logging.basicConfig(
        level=level,
        format=format_string,
        stream=stream,
        force=True,  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    Get a logger for a module.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    logger : logging.Logger
        Configured logger instance

    Examples
    --------
    >>> from geostats.core.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Message")
    """
    return logging.getLogger(name)
