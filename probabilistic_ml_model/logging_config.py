"""
Enhanced Logging Configuration Module for PML Finance Project.

This module provides production-ready logging with file rotation:
- File logging with automatic rotation based on size
- Configurable log levels and formatters
- Multiple handlers support (console + file)
- Thread-safe logging operations

Implemented using strict TDD methodology (Test-Driven Development).
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Union

# Global root logger reference for configuration
_root_logger: Optional[logging.Logger] = None
_log_level: int = logging.INFO


def setup_file_logging(
    log_file: Union[str, Path],
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB default
    backup_count: int = 5,
    format_string: Optional[str] = None,
) -> None:
    """
    Setup file logging with automatic rotation.

    Configures the root logger to write to a rotating file handler.
    Files are rotated when they reach max_bytes size.

    Args:
        log_file: Path to log file
        level: Logging level (default: INFO)
        max_bytes: Maximum size in bytes before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        format_string: Custom format string (optional)

    Example:
        >>> setup_file_logging("app.log", level=logging.DEBUG, max_bytes=1024*1024)
        >>> logger = get_logger("my_module")
        >>> logger.info("Application started")
    """
    global _root_logger, _log_level

    # Get or create root logger
    if _root_logger is None:
        _root_logger = logging.getLogger()

    # Set level
    _log_level = level
    _root_logger.setLevel(level)

    # Create log file directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create rotating file handler
    handler = logging.handlers.RotatingFileHandler(
        str(log_file), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setLevel(level)

    # Set formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Add handler to root logger
    _root_logger.addHandler(handler)


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    format_string: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure logging with both console and/or file output.

    More comprehensive configuration than setup_file_logging.
    Allows enabling console logging alongside file logging.

    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (optional)
        console: Whether to log to console (default: True)
        format_string: Custom format string (optional)
        max_bytes: Maximum size for file rotation (default: 10MB)
        backup_count: Number of backup files (default: 5)

    Example:
        >>> configure_logging(level=logging.DEBUG, log_file="app.log", console=True)
    """
    global _root_logger, _log_level

    # Get or create root logger
    if _root_logger is None:
        _root_logger = logging.getLogger()

    # Set level
    _log_level = level
    _root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    _root_logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        _root_logger.addHandler(console_handler)

    # Add file handler if log_file provided
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        _root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Returns a child logger of the configured root logger.
    All loggers share the same configuration.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


def add_file_handler(
    logger: logging.Logger,
    log_file: Union[str, Path],
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    format_string: Optional[str] = None,
) -> None:
    """
    Add a rotating file handler to a specific logger.

    Allows adding file logging to individual loggers without affecting
    the global configuration.

    Args:
        logger: Logger instance to add handler to
        log_file: Path to log file
        level: Logging level for this handler
        max_bytes: Maximum size before rotation
        backup_count: Number of backup files
        format_string: Custom format string (optional)

    Example:
        >>> logger = get_logger("my_module")
        >>> add_file_handler(logger, "module.log")
    """
    # Create log file directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create rotating file handler
    handler = logging.handlers.RotatingFileHandler(
        str(log_file), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setLevel(level)

    # Set formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)


def remove_file_handlers(logger: Optional[logging.Logger] = None) -> None:
    """
    Remove all file handlers from a logger.

    Useful for cleanup in tests or when reconfiguring logging.
    If no logger is provided, removes handlers from root logger and all child loggers.

    Args:
        logger: Logger instance (optional, defaults to root logger)

    Example:
        >>> logger = get_logger("my_module")
        >>> remove_file_handlers(logger)
    """
    import time

    if logger is None:
        # Remove from root logger
        logger = logging.getLogger()

        # Also remove from all child loggers to ensure complete cleanup
        for name in list(logging.Logger.manager.loggerDict.keys()):
            child_logger = logging.getLogger(name)
            if hasattr(child_logger, "handlers"):
                _remove_handlers_from_logger(child_logger)

    _remove_handlers_from_logger(logger)

    # Small delay to allow Windows to release file handles
    time.sleep(0.01)


def _remove_handlers_from_logger(logger: logging.Logger) -> None:
    """Internal helper to remove file handlers from a specific logger."""
    # Find and remove all file handlers
    handlers_to_remove = [
        h
        for h in logger.handlers
        if isinstance(h, (logging.FileHandler, logging.handlers.RotatingFileHandler))
    ]

    for handler in handlers_to_remove:
        try:
            handler.flush()
            handler.close()
        except (OSError, ValueError, AttributeError):
            pass  # Ignore errors during cleanup
        finally:
            logger.removeHandler(handler)


def get_log_level(logger: Optional[logging.Logger] = None) -> int:
    """
    Get the current logging level.

    Args:
        logger: Logger instance (optional, defaults to root logger)

    Returns:
        Current logging level (e.g., logging.INFO, logging.DEBUG)

    Example:
        >>> level = get_log_level()
        >>> print(f"Current level: {logging.getLevelName(level)}")
    """
    global _log_level

    if logger is None:
        return _log_level

    return logger.level


def set_log_level(level: int, logger: Optional[logging.Logger] = None) -> None:
    """
    Set the logging level.

    Changes the logging level for the specified logger and all its handlers.
    If no logger is provided, updates the root logger.

    Args:
        level: New logging level (e.g., logging.DEBUG, logging.INFO)
        logger: Logger instance (optional, defaults to root logger)

    Example:
        >>> set_log_level(logging.DEBUG)
        >>> logger = get_logger("my_module")
        >>> logger.debug("This will now be logged")
    """
    global _log_level, _root_logger

    if logger is None:
        _log_level = level
        if _root_logger is not None:
            logger = _root_logger
        else:
            logger = logging.getLogger()

    logger.setLevel(level)

    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)
