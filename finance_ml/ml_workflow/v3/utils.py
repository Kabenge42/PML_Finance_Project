from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable, Any, Sequence

import numpy as np
import pandas as pd

# Optional joblib import at module scope — stable names regardless of availability
try:  # pragma: no cover - availability depends on optional deps
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # ImportError or any runtime load issue
    Parallel = None  # type: ignore[assignment]
    delayed = None  # type: ignore[assignment]


def as_float_series(obj: pd.Series | Sequence[float] | np.ndarray) -> pd.Series:
    """Return a numeric pandas Series[float] with NaNs for non-convertible values.

    Ensures downstream operations (mean/std/skew/kurtosis) are executed on a
    well-typed Series to avoid ndarray attribute warnings and mixed-dtype issues.
    """
    if isinstance(obj, pd.Series):
        return pd.to_numeric(obj, errors="coerce").astype(float)
    # Fall back to constructing a pandas Series
    return pd.to_numeric(pd.Series(obj), errors="coerce").astype(float)


@dataclass(frozen=True)
class IdentifierSchema:
    isin: str = "isin"
    ticker: str = "ticker"
    company: str = "name"

    # Common market data columns frequently used in joins/exports
    last_price: str = "last_price"
    industry: str = "industry"
    sector: str = "sector"

    def all(self) -> set[str]:
        return {self.isin, self.ticker, self.company, self.last_price, self.industry, self.sector}


def get_identifier_cols_set(schema: IdentifierSchema | None = None) -> set[str]:
    s = schema or IdentifierSchema()
    return s.all()


def require_identifier_columns(df: pd.DataFrame, schema: IdentifierSchema | None = None) -> pd.DataFrame:
    """Ensure the DataFrame contains required identifier columns.

    - If 'isin' is missing but a plausible alternative exists, copy/rename once.
    - No in-place mutation: returns the input DataFrame if already compliant,
      otherwise returns a shallow copy with normalized identifiers.
    """
    s = schema or IdentifierSchema()
    if df.empty:
        return df

    if s.isin in df.columns:
        return df

    candidates = ["ISIN", "isin_code", "secid", "security_id"]
    for col in candidates:
        if col in df.columns:
            out = df.copy()
            out[s.isin] = out[col]
            return out

    # Nothing we can do safely — return as-is; callers may choose to warn
    return df


def run_parallel_or_sequential(
    tasks: Iterable[Any],
    n_jobs: int,
    worker: Callable[[Any], Any],
) -> list[Any]:
    """Execute tasks in parallel when joblib is available, otherwise sequentially.

    Uses module-scope guarded import to avoid inspection warnings. Falls back to
    sequential execution if n_jobs == 1, joblib is unavailable, or there is zero/no task.
    """
    tasks_list = list(tasks)
    if not tasks_list:
        return []

    # Normalize requested parallelism
    if n_jobs is None or n_jobs == 0:
        n_jobs = 1

    if Parallel is not None and delayed is not None and n_jobs != 1 and len(tasks_list) > 1:
        return Parallel(n_jobs=n_jobs, prefer="processes")(delayed(worker)(t) for t in tasks_list)  # type: ignore[misc]

    # Sequential fallback
    return [worker(t) for t in tasks_list]
