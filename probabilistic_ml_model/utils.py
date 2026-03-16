"""
Common utility functions for Finance ML.

Phase 8 (Restructuring Plan):
This module consolidates utility functions that are used across multiple
modules in the ml_workflow package.

Functions:
- safe_divide: Safe division avoiding division by zero
- to_numeric_safe: Safe numeric conversion with error handling
- flatten_dict: Flatten nested dictionaries
- ensure_list: Ensure input is a list
- clip_outliers: Clip values to specified percentiles
- timer: Context manager for timing code blocks

Usage:
    from finance_ml.ml_workflow.core.utils import safe_divide, to_numeric_safe

    # Safe division
    result = safe_divide(numerator, denominator, default=0.0)

    # Safe numeric conversion
    values = to_numeric_safe(df['column'])
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    default: float = 0.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Perform division with protection against division by zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Value to return when denominator is zero or NaN

    Returns:
        Result of division, with default where denominator is zero/NaN

    Example:
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(np.array([10, 20]), np.array([2, 0]))
        array([5., 0.])
    """
    if isinstance(denominator, (pd.Series, pd.DataFrame)):
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(default)
    elif isinstance(denominator, np.ndarray):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                (denominator == 0) | np.isnan(denominator), default, numerator / denominator
            )
        return result
    else:
        # Scalar
        if denominator == 0 or (isinstance(denominator, float) and np.isnan(denominator)):
            return default
        return numerator / denominator


def to_numeric_safe(
    data: Union[pd.Series, pd.DataFrame, List],
    errors: str = "coerce",
    downcast: Optional[str] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Safely convert data to numeric type.

    Args:
        data: Data to convert
        errors: How to handle errors ('coerce', 'raise', 'ignore')
        downcast: Downcast to smallest numeric type ('integer', 'signed', 'unsigned', 'float')

    Returns:
        Numeric data with non-convertible values as NaN (if errors='coerce')
    """
    if isinstance(data, pd.DataFrame):
        return data.apply(lambda x: pd.to_numeric(x, errors=errors, downcast=downcast))
    elif isinstance(data, list):
        return pd.to_numeric(pd.Series(data), errors=errors, downcast=downcast)
    else:
        return pd.to_numeric(data, errors=errors, downcast=downcast)


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    separator: str = "_",
) -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        separator: Separator between key levels

    Returns:
        Flattened dictionary with concatenated keys

    Example:
        >>> flatten_dict({'a': {'b': 1, 'c': 2}})
        {'a_b': 1, 'a_c': 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


def ensure_list(value: Any) -> List:
    """Ensure input is a list.

    Args:
        value: Value to convert to list

    Returns:
        List containing value, or value if already a list

    Example:
        >>> ensure_list("item")
        ['item']
        >>> ensure_list(["a", "b"])
        ['a', 'b']
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set, np.ndarray)):
        return list(value)
    if isinstance(value, pd.Series):
        return value.tolist()
    return [value]


def clip_outliers(
    data: Union[pd.Series, np.ndarray],
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> Union[pd.Series, np.ndarray]:
    """Clip values to specified percentiles.

    Args:
        data: Data to clip
        lower_percentile: Lower percentile (0-1)
        upper_percentile: Upper percentile (0-1)

    Returns:
        Clipped data

    Example:
        >>> clip_outliers(pd.Series([1, 2, 100]), 0.05, 0.95)
    """
    if isinstance(data, pd.Series):
        lower = data.quantile(lower_percentile)
        upper = data.quantile(upper_percentile)
        return data.clip(lower=lower, upper=upper)
    else:
        arr = np.asarray(data)
        lower = np.nanpercentile(arr, lower_percentile * 100)
        upper = np.nanpercentile(arr, upper_percentile * 100)
        return np.clip(arr, lower, upper)


@contextmanager
def timer(name: str = "Operation", log_level: str = "info"):
    """Context manager for timing code blocks.

    Args:
        name: Name of the operation being timed
        log_level: Logging level ('debug', 'info', 'warning')

    Yields:
        None

    Example:
        >>> with timer("Data loading"):
        ...     df = pd.read_csv("data.csv")
        INFO: Data loading completed in 1.23 seconds
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        message = f"{name} completed in {elapsed:.2f} seconds"

        if log_level == "debug":
            logger.debug(message)
        elif log_level == "warning":
            logger.warning(message)
        else:
            logger.info(message)


def validate_columns_exist(
    df: pd.DataFrame,
    required_columns: List[str],
    raise_error: bool = True,
) -> List[str]:
    """Validate that required columns exist in DataFrame.

    Args:
        df: DataFrame to check
        required_columns: List of required column names
        raise_error: Whether to raise ValueError if columns missing

    Returns:
        List of missing columns

    Raises:
        ValueError: If raise_error=True and columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing and raise_error:
        raise ValueError(f"Missing required columns: {missing}")

    return missing


def get_numeric_columns(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Get list of numeric columns from DataFrame.

    Args:
        df: DataFrame to analyze
        exclude: Column names to exclude

    Returns:
        List of numeric column names
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if exclude:
        numeric_cols = [c for c in numeric_cols if c not in exclude]

    return numeric_cols


def get_categorical_columns(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Get list of categorical/object columns from DataFrame.

    Args:
        df: DataFrame to analyze
        exclude: Column names to exclude

    Returns:
        List of categorical column names
    """
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if exclude:
        cat_cols = [c for c in cat_cols if c not in exclude]

    return cat_cols


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Get memory usage of DataFrame in megabytes.

    Args:
        df: DataFrame to measure

    Returns:
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def reduce_memory_usage(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reduce memory usage by downcasting numeric types.

    Args:
        df: DataFrame to optimize
        verbose: Whether to log memory savings

    Returns:
        DataFrame with optimized dtypes
    """
    start_mem = memory_usage_mb(df)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = memory_usage_mb(df)

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(
            f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)"
        )

    return df
