from __future__ import annotations

import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategoryAnalyticsCacheKey:
    """Stable key for category probability analytics results."""

    data_checksum: str
    n_categories: int
    use_mcmc: bool
    n_mcmc_samples: int
    burn_in: int
    max_features_per_category: int

    subdir: str = "category_analytics"

    def to_filename(self) -> str:
        return (
            f"category_analytics_{self.data_checksum}_cats{self.n_categories}_"
            f"mcmc{int(self.use_mcmc)}_n{self.n_mcmc_samples}_b{self.burn_in}_"
            f"max{self.max_features_per_category}.json"
        )


@dataclass(frozen=True)
class McmcReturnCacheKey:
    """Stable key for parallel MCMC return analysis results."""

    data_checksum: str
    n_chains: int
    n_samples: int

    subdir: str = "mcmc_results"
    prefix: str = "mcmc_return"

    def to_filename(self) -> str:
        """

        :return:
        """
        return (
            f"{self.prefix}_{self.data_checksum}_" f"chains{self.n_chains}_n{self.n_samples}.json"
        )

    # -- convenience constructors for each analysis type ------------------

    @classmethod
    def for_return(
        cls, data_checksum: str, n_chains: int, n_samples: int
    ) -> "McmcReturnCacheKey":
        """

        :param data_checksum:
        :param n_chains:
        :param n_samples:
        :return:
        """
        return cls(
            data_checksum=data_checksum,
            n_chains=n_chains,
            n_samples=n_samples,
            subdir="mcmc_results/return",
            prefix="mcmc_return",
        )

    @classmethod
    def for_accounting_anomaly(
        cls, data_checksum: str, n_chains: int, n_samples: int
    ) -> "McmcReturnCacheKey":
        """

        :param data_checksum:
        :param n_chains:
        :param n_samples:
        :return:
        """
        return cls(
            data_checksum=data_checksum,
            n_chains=n_chains,
            n_samples=n_samples,
            subdir="mcmc_results/accounting_anomaly",
            prefix="mcmc_accounting_anomaly",
        )

    @classmethod
    def for_credit_risk(
        cls, data_checksum: str, n_chains: int, n_samples: int
    ) -> "McmcReturnCacheKey":
        """

        :param data_checksum:
        :param n_chains:
        :param n_samples:
        :return:
        """
        return cls(
            data_checksum=data_checksum,
            n_chains=n_chains,
            n_samples=n_samples,
            subdir="mcmc_results/credit_risk",
            prefix="mcmc_credit_risk",
        )

    @classmethod
    def for_dividend_safety(
        cls, data_checksum: str, n_chains: int, n_samples: int
    ) -> "McmcReturnCacheKey":
        """

        :param data_checksum:
        :param n_chains:
        :param n_samples:
        :return:
        """
        return cls(
            data_checksum=data_checksum,
            n_chains=n_chains,
            n_samples=n_samples,
            subdir="mcmc_results/dividend_safety",
            prefix="mcmc_dividend_safety",
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
    if 0 < numeric_sample < len(num_cols):
        num_cols = num_cols[:numeric_sample]

    if num_cols:
        # Include both head and tail for better variability
        parts.append(df[num_cols].head(50).to_csv(index=False, float_format="%.8g"))
        parts.append(df[num_cols].tail(50).to_csv(index=False, float_format="%.8g"))

    return _sha1("|".join(parts))


def build_cache_path(
    cache_dir: str | Path, key_filename: str, subdir: str | None = None
) -> Path:
    """Build a cache file path, optionally within a *subdir* under *cache_dir*."""
    cache_root = Path(cache_dir)
    if subdir:
        cache_root /= subdir
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / key_filename


# ---------------------------------------------------------------------------
# Float precision — custom JSON encoder
# ---------------------------------------------------------------------------

_FLOAT_PRECISION = 6


class _RoundingEncoder(json.JSONEncoder):
    """JSON encoder that rounds all float values to *_FLOAT_PRECISION* digits.

    The standard ``json.dumps(default=...)`` hook is only called for objects
    the encoder does *not* know how to serialize.  Native Python ``float``
    values inside lists/dicts therefore bypass ``default`` entirely and are
    written at full 15-digit precision.

    This encoder overrides ``encode`` to walk the object tree and round
    every ``float`` **before** the standard encoder serializes it, ensuring
    consistent precision everywhere.
    """

    def encode(self, o: Any) -> str:
        """

        :param o:
        :return:
        """
        return super().encode(self._round(o))

    def default(self, o: Any) -> Any:
        """

        :param o:
        :return:
        """
        import numpy as np

        if isinstance(o, np.ndarray):
            return [round(float(x), _FLOAT_PRECISION) for x in o.flat]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return round(float(o), _FLOAT_PRECISION)
        return str(o)

    @classmethod
    def _round(cls, obj: Any) -> Any:
        if isinstance(obj, float):
            return round(obj, _FLOAT_PRECISION)
        if isinstance(obj, dict):
            return {k: cls._round(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._round(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(cls._round(v) for v in obj)
        return obj


def _json_default(obj: Any) -> Any:
    """JSON serializer that converts numpy arrays to lists instead of strings."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return [round(float(x), _FLOAT_PRECISION) for x in obj.flat]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), _FLOAT_PRECISION)
    if isinstance(obj, float):
        return round(obj, _FLOAT_PRECISION)
    return str(obj)


# ---------------------------------------------------------------------------
# Stripping large sample arrays before caching
# ---------------------------------------------------------------------------

# Keys that contain large raw sample arrays and should be excluded from
# the on-disk JSON cache.  The in-memory result dict is left untouched
# so that downstream visualisation functions can still access them.
_MCMC_LARGE_KEYS: frozenset[str] = frozenset(
    {
        "chains",
        "combined_samples",
        "inference_data",
    }
)


def _strip_large_mcmc_keys(payload: Any) -> Any:
    """Return a shallow copy of *payload* without bulky sample arrays.

    Only applies when *payload* looks like an MCMC result dict (has the
    ``posterior_mean`` key).  Category-analytics dicts and other payloads
    are returned unchanged.
    """
    if not isinstance(payload, dict):
        return payload
    if "posterior_mean" in payload and any(k in payload for k in _MCMC_LARGE_KEYS):
        return {k: v for k, v in payload.items() if k not in _MCMC_LARGE_KEYS}
    return payload


def _strip_category_simulation_arrays(payload: Any) -> Any:
    """Remove bulky ``simulations`` arrays from category analytics results.

    Each category's ``distribution_fits`` contains per-feature dicts that
    include a ``simulations`` key holding 10 000 Monte Carlo draws.  These
    are regenerable from the distribution parameters already cached, so we
    strip them to save ~99 % of the file size.

    The original in-memory dict is **not** mutated.
    """
    if not isinstance(payload, dict):
        return payload

    # Detect category-analytics shape: top-level keys are category names
    # whose values are dicts containing ``distribution_fits``.
    first_val = next(iter(payload.values()), None)
    if not isinstance(first_val, dict) or "distribution_fits" not in first_val:
        return payload

    result = {}
    for cat_name, cat_data in payload.items():
        if not isinstance(cat_data, dict):
            result[cat_name] = cat_data
            continue
        cat_copy = dict(cat_data)
        if "distribution_fits" in cat_copy and isinstance(cat_copy["distribution_fits"], dict):
            cat_copy["distribution_fits"] = {
                feat: {k: v for k, v in fit.items() if k != "simulations"}
                for feat, fit in cat_copy["distribution_fits"].items()
                if isinstance(fit, dict)
            }
        result[cat_name] = cat_copy
    return result


def _strip_hierarchical_samples(payload: Any) -> Any:
    """Remove bulky ``samples`` arrays from hierarchical MCMC levels.

    Each group under ``hierarchical.levels.<level>.<group>`` contains a
    ``samples`` list of 10 000–25 000 MCMC draws that are regenerable.
    We strip them while preserving all summary statistics.

    The original in-memory dict is **not** mutated.
    """
    if not isinstance(payload, dict) or "hierarchical" not in payload:
        return payload

    hier = payload.get("hierarchical")
    if not isinstance(hier, dict) or "levels" not in hier:
        return payload

    result = dict(payload)
    hier_copy = dict(hier)

    # Strip inference_data from hierarchical if present
    if "inference_data" in hier_copy:
        hier_copy = {k: v for k, v in hier_copy.items() if k != "inference_data"}

    hier_copy["levels"] = {
        level: {
            group: {k: v for k, v in gdata.items() if k != "samples"}
            for group, gdata in groups.items()
            if isinstance(gdata, dict)
        }
        for level, groups in hier_copy["levels"].items()
        if isinstance(groups, dict)
    }
    result["hierarchical"] = hier_copy
    return result


# ---------------------------------------------------------------------------
# Cache eviction — keep only the N most recent files per subdirectory
# ---------------------------------------------------------------------------

_MAX_CACHE_FILES_PER_SUBDIR = 2


def _evict_old_caches(
    cache_dir: str | Path, subdir: str, max_files: int = _MAX_CACHE_FILES_PER_SUBDIR
) -> int:
    """Keep only the *max_files* most recent cache files in *subdir*.

    Returns the number of files removed.
    """
    target = Path(cache_dir) / subdir
    if not target.exists():
        return 0

    # Match both old .json and new .json.gz files
    files = sorted(
        [f for f in target.iterdir() if f.is_file()],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for old in files[max_files:]:
        try:
            old.unlink()
            removed += 1
        except OSError:
            pass
    return removed


# ---------------------------------------------------------------------------
# Save / Load — plain JSON (with legacy gzip read support)
# ---------------------------------------------------------------------------


def save_json(cache_path: Path, payload: Any) -> None:
    """Serialize *payload* to plain JSON, stripping regenerable arrays.

    Applies three layers of stripping:
    1. Top-level MCMC keys (chains, combined_samples, inference_data).
    2. Category-analytics simulation arrays.
    3. Hierarchical MCMC per-group sample arrays.

    After writing, old cache files in the same subdirectory are evicted
    to keep at most ``_MAX_CACHE_FILES_PER_SUBDIR`` files.
    """
    cleaned = _strip_large_mcmc_keys(payload)
    cleaned = _strip_category_simulation_arrays(cleaned)
    cleaned = _strip_hierarchical_samples(cleaned)

    data = json.dumps(cleaned, cls=_RoundingEncoder)

    # Write plain JSON (strip .gz suffix if present from legacy callers)
    out_path = cache_path
    if out_path.suffix == ".gz":
        out_path = out_path.with_suffix("")  # .json.gz → .json
    out_path.write_text(data, encoding="utf-8")

    # Remove stale gzip counterpart left by older code
    gz_path = Path(str(out_path) + ".gz")
    if gz_path.exists():
        try:
            gz_path.unlink()
            logger.debug("Removed stale gzip counterpart: %s", gz_path.name)
        except OSError:
            pass

    # Evict old caches in the same subdirectory
    subdir = out_path.parent
    cache_root = subdir.parent
    if subdir.name and cache_root.exists():
        removed = _evict_old_caches(cache_root, subdir.name)
        if removed > 0:
            logger.debug("Evicted %d old cache file(s) from %s", removed, subdir.name)


def load_json(cache_path: Path, *, ttl_hours: float | None = None) -> Any | None:
    """Load a cached JSON result, supporting both gzip and plain-text formats.

    Returns ``None`` if the file does not exist, has expired, or cannot be
    parsed.
    """
    try:
        # Try the .gz path first (new format), then fall back to plain JSON
        gz_path = (
            cache_path
            if cache_path.suffix == ".gz"
            else cache_path.with_suffix(cache_path.suffix + ".gz")
        )

        # Determine which path to read
        if gz_path.exists():
            read_path = gz_path
        elif cache_path.exists() and cache_path.suffix != ".gz":
            read_path = cache_path
        else:
            return None

        if ttl_hours is not None and ttl_hours > 0:
            import time

            age_h = (time.time() - read_path.stat().st_mtime) / 3600.0
            if age_h > ttl_hours:
                return None

        if read_path.suffix == ".gz":
            with gzip.open(read_path, "rb") as f:
                return json.loads(f.read().decode("utf-8"))
        else:
            # Try plain JSON first; fall back to gzip in case the file was
            # written as compressed data with a .json extension (transitional
            # files from an earlier code version).
            raw = read_path.read_bytes()
            try:
                return json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            # Attempt gzip decompression on the .json file
            try:
                return json.loads(gzip.decompress(raw).decode("utf-8"))
            except Exception:
                return None
    except Exception:
        return None
