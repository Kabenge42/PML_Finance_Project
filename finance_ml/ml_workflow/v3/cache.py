from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class CategoryAnalyticsCacheKey:
    """Stable key for category probability analytics results."""

    data_checksum: str
    n_categories: int
    use_mcmc: bool
    n_mcmc_samples: int
    burn_in: int
    max_features_per_category: int

    def to_filename(self) -> str:
        return (
            f"category_analytics_{self.data_checksum}_cats{self.n_categories}_"
            f"mcmc{int(self.use_mcmc)}_n{self.n_mcmc_samples}_b{self.burn_in}_"
            f"max{self.max_features_per_category}.json"
        )


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()  # nosec - non-crypto use


def dataframe_stable_checksum(
    df: pd.DataFrame, id_cols: Iterable[str] | None = None, numeric_sample: int = 32
) -> str:
    """Compute a stable checksum based on identifiers and numeric content.

    - Uses all present identifier columns among ``id_cols`` (defaults to common ones).
    - Includes shape metadata to differentiate equivalent heads/tails across shapes.
    - Hashes a deterministic sample of numeric columns (sorted by name) and their head/tail.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "empty"

    if id_cols is None:
        id_cols = ["isin"]

    present_ids = [c for c in id_cols if c in df.columns]
    parts: list[str] = [f"shape={df.shape[0]}x{df.shape[1]}"]

    if present_ids:
        parts.append(df[present_ids].to_csv(index=False))

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    num_cols = sorted(num_cols)
    if numeric_sample > 0 and len(num_cols) > numeric_sample:
        num_cols = num_cols[:numeric_sample]

    if num_cols:
        # Include both head and tail for better variability
        parts.append(df[num_cols].head(50).to_csv(index=False, float_format="%.8g"))
        parts.append(df[num_cols].tail(50).to_csv(index=False, float_format="%.8g"))

    return _sha1("|".join(parts))


def build_cache_path(cache_dir: str | Path, key_filename: str) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / key_filename


def save_json(cache_path: Path, payload: Any) -> None:
    cache_path.write_text(json.dumps(payload, default=str))


def load_json(cache_path: Path, *, ttl_hours: float | None = None) -> Any | None:
    try:
        if not cache_path.exists():
            return None
        if ttl_hours is not None and ttl_hours > 0:
            import time

            age_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
            if age_h > ttl_hours:
                return None
        return json.loads(cache_path.read_text())
    except Exception:
        return None
