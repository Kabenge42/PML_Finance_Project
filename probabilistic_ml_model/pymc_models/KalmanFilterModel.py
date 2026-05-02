"""
Kalman Filter Price Target Model — GaussianRandomWalk state-space.

Uses PyMC GaussianRandomWalk for latent price state with an observation model
for noisy price targets.

Reference: run_kalman_filter() in expected_returns_v3.py (line 1519);
           probability_models.py Kalman references.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal, Optional, TYPE_CHECKING

from arviz import InferenceData
from pymc.backends.base import MultiTrace

try:
    import arviz as az
except ImportError:
    try:
        import arviz_base as az
    except ImportError:
        az = None  # type: ignore[assignment]
import numpy as np
import pandas as pd

try:
    import pymc as pm
except ImportError:
    pm = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import arviz as az_typing  # noqa: F401
    import pymc as pm_typing  # noqa: F401

from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs
from probabilistic_ml_model.pymc_models._feature_alignment import (
    coerce_by_data_type,
    load_feature_metadata_from_db,
)
from probabilistic_ml_model.data_utils.data_utils import load_feature_categories_from_db

logger = logging.getLogger(__name__)

Parameterization = Literal["centered", "non_centered", "marginalized"]

# ---------------------------------------------------------------------------
# Historical "*_ago" cohort -> per-ISIN time-series helpers.
#
# Background. ``mv_all_stock_features`` is a point-in-time snapshot, so any
# event-style date column (e.g. ``next_earnings``) only ever holds one row
# per ISIN -> ``groupby(isin).size() >= 2`` is unsatisfiable. The snapshot,
# however, embeds a historical axis through the ``*_ago`` family
# (``price_target_1w_ago``, ``price_target_high_3m_ago``, ``price_1y_ago``,
# ``price_target_median_ytd_ago`` …). Unpivoting these columns yields up to
# ~16 ordered observations per ISIN with a real ``pd.DatetimeIndex`` derived
# from the suffix — exactly what the GaussianRandomWalk state-space model
# in :class:`KalmanFilterPriceTarget` needs.
#
# This logic lives on the model class so the notebook (Section 7.1 of
# ``pymc_expected_returns_model.ipynb``) and any future batch scorer can
# call a single canonical helper instead of inlining the unpivot.
# ---------------------------------------------------------------------------

# Map textual *_ago suffix -> pandas DateOffset.
_AGO_SUFFIX_PATTERN = r"5d|1w|1m|3m|6m|1y|3y|5y|mtd|qtd|ytd"
_AGO_HISTORY_RE = (
    r"^(price_target(?:_high|_low|_median)?|price)" r"_(?P<suf>" + _AGO_SUFFIX_PATTERN + r")_ago$"
)


def _build_ago_offset_map() -> dict[str, Any]:
    from pandas.tseries.offsets import DateOffset, MonthBegin, QuarterBegin, YearBegin

    return {
        "1w": DateOffset(weeks=1),
        "1m": DateOffset(months=1),
        "3m": DateOffset(months=3),
        "6m": DateOffset(months=6),
        "1y": DateOffset(years=1),
        "3y": DateOffset(years=3),
        "5y": DateOffset(years=5),
        "5d": DateOffset(days=5),
        # Period-to-date — anchor to the start of the current period.
        "mtd": MonthBegin(0),
        "qtd": QuarterBegin(0, startingMonth=1),
        "ytd": YearBegin(0),
    }


# Canonical category names in public.calculated_features_registry whose
# feature_aliases drive the auxiliary "kalman_feature" dim. Aligned with the
# categories backing the Kalman state-space price-target observation model.
_KALMAN_CATEGORY_KEYS: tuple[str, ...] = (
    "Price Target Dynamics",
    "Technical Analysis",
    "Temporal Patterns",
)


class KalmanFilterPriceTarget:
    """Bayesian state-space model for price target filtering.

    Uses GaussianRandomWalk as latent state with HalfNormal priors on
    state and observation noise (scaled to data std).
    """

    def __init__(self) -> None:
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[az_typing.InferenceData] = None

    @staticmethod
    @lru_cache(maxsize=4)
    def _resolve_kalman_feature_aliases(
        connection_string: Optional[str] = None,
    ) -> tuple[str, ...]:
        """Resolve the Kalman feature aliases from calculated_features_registry.

        These serve as labelled coordinates along the ``kalman_feature`` dim
        for the auxiliary ``pm.Data`` container of observed Kalman-feature
        values (e.g. price-target dynamics + technical analysis signals).

        Returns a tuple so the result is hashable / cache-friendly; callers
        should convert to ``list`` if mutation is needed.
        """
        try:
            categories = load_feature_categories_from_db(connection_string)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load feature categories: %s", exc)
            return tuple()
        aliases: list[str] = []
        seen: set[str] = set()
        for key in _KALMAN_CATEGORY_KEYS:
            for alias in categories.get(key, []):
                if alias not in seen:
                    seen.add(alias)
                    aliases.append(alias)
        return tuple(aliases)

    @staticmethod
    def _align_kalman_features(
        kalman_features_df: Optional[pd.DataFrame],
        isin: np.ndarray,
        feature_aliases: list[str],
        *,
        use_typed_coercion: bool = False,
        connection_string: Optional[str] = None,
    ) -> np.ndarray:
        """Align an (isin × kalman_feature) matrix to the model dims.

        Missing columns/rows are filled with ``0.0`` so the container always
        has shape ``(n_isin, n_kalman_feature)``.
        """
        n_isin = len(isin)
        n_feat = len(feature_aliases)
        if kalman_features_df is None or n_feat == 0 or n_isin == 0:
            return np.zeros((n_isin, max(n_feat, 0)), dtype="float64")

        df = kalman_features_df.copy()
        if "isin" in df.columns:
            df = df.drop_duplicates(subset="isin").set_index("isin")
        df = df.reindex(index=isin)

        if use_typed_coercion:
            metadata = load_feature_metadata_from_db(connection_string)
            return coerce_by_data_type(df, list(feature_aliases), metadata)
        return df.reindex(columns=feature_aliases).astype("float64").fillna(0.0).to_numpy()

    # ------------------------------------------------------------------
    # Snapshot -> per-ISIN price-target time-series helpers.
    #
    # These mirror Section 7.1 of ``pymc_expected_returns_model.ipynb``
    # so the unpivot of the embedded ``*_ago`` cohort is owned by the
    # model class rather than inlined per-caller. They turn a
    # point-in-time snapshot (``mv_all_stock_features``) into a real
    # ``(isin, asof_date, price_target)`` long panel suitable for the
    # GaussianRandomWalk likelihood.
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_reference_now(
        df: pd.DataFrame,
        timestamp_col: str = "feature_calculated_at",
    ) -> pd.Timestamp:
        """Return a deterministic "now" anchor for ``*_ago`` offsetting.

        Uses the snapshot's ``feature_calculated_at`` column when present
        (so reruns of the same snapshot land on identical timestamps);
        falls back to ``pd.Timestamp.utcnow()`` otherwise. The result is
        always tz-naive.
        """
        if timestamp_col in df.columns:
            ref = pd.to_datetime(df[timestamp_col], errors="coerce").max()
        else:
            ref = pd.Timestamp.utcnow()
        if pd.isna(ref):
            ref = pd.Timestamp.utcnow()
        return (
            pd.Timestamp(ref).tz_localize(None) if pd.Timestamp(ref).tzinfo else pd.Timestamp(ref)
        )

    @classmethod
    def build_price_target_history(
        cls,
        df: pd.DataFrame,
        *,
        id_col: str = "isin",
        now_cols: tuple[str, ...] = ("price_target", "last_price"),
        min_observations: int = 2,
        timestamp_col: str = "feature_calculated_at",
    ) -> tuple[pd.DataFrame, pd.Series, str]:
        """Build a long ``(isin, asof_date, price_target)`` panel from a snapshot.

        Unpivots the ``*_ago`` cohort
        (``price_target``, ``price_target_high|low|median``, ``price``) using
        :data:`_AGO_HISTORY_RE` and translates each suffix to a real
        ``pd.Timestamp`` via :func:`_build_ago_offset_map`. The
        point-in-time ``price_target`` / ``last_price`` values are seeded
        at the resolved "now" anchor.

        Parameters
        ----------
        df : pandas.DataFrame
            Wide snapshot frame (e.g. ``mv_all_stock_features``).
        id_col : str
            Name of the ISIN column. Defaults to ``"isin"``.
        now_cols : tuple of str
            Columns whose values represent the current price target /
            last price; seeded at the snapshot anchor.
        min_observations : int
            Minimum number of distinct ``(isin, asof_date)`` rows required
            to consider an ISIN eligible for Kalman fitting.
        timestamp_col : str
            Column name carrying the snapshot's calculation timestamp.

        Returns
        -------
        long_df : pandas.DataFrame
            Long-format panel with columns ``[id_col, "asof_date",
            "price_target"]``, sorted by ISIN then ``asof_date``.
        eligible : pandas.Series
            Per-ISIN observation counts for ISINs with
            ``count >= min_observations``.
        date_col : str
            ``"asof_date"`` when the unpivot succeeded; otherwise ``None``.
        """
        import re

        offset_map = _build_ago_offset_map()
        ref_now = cls._resolve_reference_now(df, timestamp_col=timestamp_col)

        def _ago_to_date(suffix: str) -> pd.Timestamp:
            off = offset_map.get(suffix.lower())
            if off is None:
                return pd.NaT
            return ref_now - off

        ago_re = re.compile(_AGO_HISTORY_RE)
        history_cols: list[tuple[str, str]] = []
        for col in df.columns:
            m = ago_re.match(col)
            if m:
                history_cols.append((col, m.group("suf")))

        now_specs: list[tuple[str, pd.Timestamp]] = [
            (c, ref_now) for c in now_cols if c in df.columns
        ]

        frames: list[pd.DataFrame] = []
        for col, suf in history_cols:
            asof = _ago_to_date(suf)
            if pd.isna(asof):
                continue
            piece = (
                df[[id_col, col]]
                .rename(columns={col: "price_target"})
                .dropna(subset=["price_target"])
            )
            piece = piece.assign(asof_date=asof, source_col=col)
            frames.append(piece)
        for col, asof in now_specs:
            piece = (
                df[[id_col, col]]
                .rename(columns={col: "price_target"})
                .dropna(subset=["price_target"])
            )
            piece = piece.assign(asof_date=asof, source_col=col)
            frames.append(piece)

        if not frames:
            empty = pd.DataFrame(columns=[id_col, "asof_date", "price_target"])
            return empty, pd.Series(dtype="int64"), None  # type: ignore[return-value]

        long_df = pd.concat(frames, ignore_index=True)
        long_df = long_df[(long_df["price_target"] > 0) & np.isfinite(long_df["price_target"])]
        # Average duplicate (isin, asof_date) cells across high/low/median/now
        # cohorts so the per-ISIN time axis is strictly monotonic.
        long_df = (
            long_df.groupby([id_col, "asof_date"], as_index=False)["price_target"]
            .mean()
            .sort_values([id_col, "asof_date"])
        )
        counts = long_df.groupby(id_col).size()
        eligible = counts[counts >= min_observations]
        return long_df, eligible, "asof_date"

    @staticmethod
    def select_target_isin(
        eligible: pd.Series,
        cohort: Optional[Any] = None,
    ) -> Optional[str]:
        """Pick the ISIN with the most history, optionally filtered by ``cohort``.

        ``cohort`` is any iterable of ISIN strings (e.g. the index of the
        PriceTarget model's ``pt_df``). When the intersection is empty,
        falls back to the global argmax over ``eligible``.
        """
        if eligible is None or eligible.empty:
            return None
        if cohort is not None:
            cohort_set = {str(x) for x in cohort}
            candidates = eligible[eligible.index.astype(str).isin(cohort_set)]
            if not candidates.empty:
                return candidates.idxmax()
        return eligible.idxmax()

    def fit(
        self,
        price_targets: np.ndarray,
        isin: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        sectors: Optional[np.ndarray] = None,
        categories_df: Optional[pd.DataFrame] = None,
        hierarchy_levels: Optional[list[str]] = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
        parameterization: Parameterization = "non_centered",
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> tuple[InferenceData | MultiTrace, Any]:
        """
        Fit a state-space model to the provided price targets using a Kalman filter
        approach in log-space. The function handles inference via PyMC and returns
        both the inference data object and model instance.

        Parameters
        ----------
        price_targets : numpy.ndarray
            An array of observed price target values, must have at least 2 elements.
            Values must be finite and strictly positive, as the Kalman filter operates
            in log-space.
        isin : str, optional
            International Securities Identification Number (ISIN) associated with the
            price targets. If provided, it is included in model metadata.
        dates : pandas.DatetimeIndex, optional
            Time index corresponding to `price_targets`. If provided, it will be used
            for naming and plotting axes. Non-finite dates will be dropped alongside
            corresponding price targets.
        samples : int
            Number of posterior samples to draw during the MCMC process. Default is
            2000.
        tune : int
            Number of tuning steps before sampling during MCMC inference. The tuned
            values are not included in the returned samples. Default is 1000.
        chains : int
            Number of independent MCMC chains to run. Default is 4.
        target_accept : float
            Target acceptance rate for the NUTS sampler, influencing the step-size
            adaptation during sampling. Default is 0.9.
        random_seed : int
            Seed for random number generation to ensure reproducibility. Default is 42.
        parameterization : Parameterization
            Model parameterization approach used for the latent state. Options are:
            "non_centered", "centered", or "marginalized". Default is "non_centered".
        nuts_sampler : str, optional
            Specific NUTS sampler to use. If None, the default sampler provided by PyMC
            is used.
        **sample_kwargs : Any
            Additional keyword arguments passed to the `pm.sample` method. Overrides
            defaults set internally.

        Returns
        -------
        tuple[InferenceData | MultiTrace, Any]
            A tuple containing:
            - The resulting inference data object (ArviZ InferenceData or PyMC
              MultiTrace depending on the sampling method).
            - The PyMC model object representing the state-space Kalman filter.

        Raises
        ------
        ImportError
            If the PyMC library is not available.
        ValueError
            If the `price_targets` array contains fewer than 2 values, fewer than 2
            finite values, or any non-positive values.

        Notes
        -----
        - The Kalman filter operates in log-space to ensure numerical stability and
          avoid overflow errors when working with strictly positive observed price
          targets.
        - Input price targets are filtered to remove non-finite values (NaN or Inf)
          before building the model. A warning is logged indicating how many
          observations were dropped.
        - Observations with constant or near-constant series will have their scale
          constrained to prevent degenerate prior configurations.
        """
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install pymc + arviz to use KalmanFilterPriceTarget."
            )

        pt_arr = np.asarray(price_targets, dtype="float64")
        if pt_arr.size < 2:
            raise ValueError("price_target must have length ≥ 2.")

        # Drop non-finite observations to avoid propagating NaN/Inf into the
        # GaussianRandomWalk likelihood (a common overflow trigger).
        finite_mask = np.isfinite(pt_arr)
        if finite_mask.sum() < 2:
            raise ValueError(
                "price_target must contain at least 2 finite observations."
            )
        if not finite_mask.all():
            logger.warning(
                "KalmanFilterPriceTarget: dropping %d non-finite price target obs",
                int((~finite_mask).sum()),
            )
            pt_arr = pt_arr[finite_mask]
            if dates is not None:
                dates = pd.DatetimeIndex(np.asarray(dates)[finite_mask])

        # Price targets are strictly positive ⇒ model on log-scale.  This keeps
        # the latent state and noise priors dimensionless and bounded,
        # preventing the float64 overflow observed when raw price levels (often
        # 10²–10³) are combined with a HalfNormal(σ=std(price)) prior on a
        # GaussianRandomWalk: cumulative variance ≈ T·σ² overflows the
        # likelihood for moderate T.  Working in log-space, σ represents
        # log-returns and is naturally O(0.1).
        if np.any(pt_arr <= 0):
            raise ValueError(
                "price_target must be strictly positive for log-space Kalman filter."
            )
        log_pt = np.log(pt_arr)

        T = len(pt_arr)
        # Clamp scale to a sane band so degenerate inputs (constant series, or
        # a single outlier driving std to ∞) cannot blow up the priors.
        raw_scale = float(np.nanstd(log_pt))
        scale = float(np.clip(raw_scale if np.isfinite(raw_scale) else 0.0, 1e-3, 1.0))
        time_coords = dates if dates is not None else np.arange(T, dtype=np.int64)
        coords: dict[str, Any] = {"time": time_coords}
        if isin is not None:
            coords["isin"] = [isin]

        with pm.Model(coords=coords) as model:
            # Store the raw price target series for downstream consumers /
            # `pm.set_data` swaps; the model itself operates on log-prices.
            pm.Data("price_target", pt_arr, dims="time")
            log_obs_data = pm.Data("log_price_target", log_pt, dims="time")

            sigma_state = pm.HalfNormal("sigma_state", sigma=scale)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=scale)

            if parameterization == "centered":
                log_state = pm.GaussianRandomWalk(
                    "log_state",
                    sigma=sigma_state,
                    init_dist=pm.Normal.dist(mu=float(log_pt[0]), sigma=scale),
                    dims="time",
                )
            elif parameterization == "marginalized":
                # Collapse latent state: pin to the observed log-price series.
                # Removes the per-time random-walk latent (no posterior on
                # the smoothed state) but preserves the obs likelihood.
                import pytensor.tensor as _pt

                log_state = pm.Deterministic(
                    "log_state",
                    _pt.as_tensor_variable(log_pt),
                    dims="time",
                )
            else:
                # Non-centred GRW: state = init + cumsum(sigma_state * z).
                import pytensor.tensor as _pt

                z_innov = pm.Normal("z_innov", 0.0, 1.0, dims="time")
                log_state = pm.Deterministic(
                    "log_state",
                    float(log_pt[0]) + _pt.cumsum(sigma_state * z_innov),
                    dims="time",
                )
            # Expose the latent state in the original price space as a
            # Deterministic so downstream code that referenced ``state``
            # continues to work without overflow risk.
            state = pm.Deterministic("state", pm.math.exp(log_state), dims="time")  # noqa: F841

            pm.Normal(
                "obs",
                mu=log_state,
                sigma=sigma_obs,
                observed=log_obs_data,
                dims="time",
            )

            scall: dict[str, Any] = dict(
                draws=samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=True,
                compile_kwargs=get_pytensor_compile_kwargs(),
            )
            if nuts_sampler is not None:
                scall["nuts_sampler"] = nuts_sampler
            scall.setdefault("idata_kwargs", {"log_likelihood": False})
            scall.update(sample_kwargs)

            # nutpie ignores idata_kwargs and emits a UserWarning; strip it
            # to keep logs clean while preserving behaviour for other samplers.
            if scall.get("nuts_sampler") == "nutpie":
                scall.pop("idata_kwargs", None)

            idata = pm.sample(**scall)

        # Store model metadata for downstream consumers
        if isin is not None:
            try:
                model.name = f"KalmanFilter[{isin}]"
            except Exception:
                pass

        self.model_ = model
        self.idata_ = idata if isinstance(idata, InferenceData) else None
        return idata, model

    def fit_from_snapshot(
        self,
        df: pd.DataFrame,
        *,
        id_col: str = "isin",
        cohort: Optional[Any] = None,
        target_isin: Optional[str] = None,
        min_observations: int = 2,
        timestamp_col: str = "feature_calculated_at",
        now_cols: tuple[str, ...] = ("price_target", "last_price"),
        **fit_kwargs: Any,
    ) -> tuple[Optional[InferenceData | MultiTrace], Any]:
        """Build a per-ISIN history from a snapshot frame and fit the model.

        Convenience wrapper around :meth:`build_price_target_history`,
        :meth:`select_target_isin`, and :meth:`fit` that reproduces the
        notebook Section 7.1 / 7.3 flow in a single call.

        Parameters
        ----------
        df : pandas.DataFrame
            Wide snapshot frame containing the ``*_ago`` cohort.
        id_col : str
            ISIN column name.
        cohort : iterable, optional
            Restrict ISIN selection to this cohort (e.g. the
            ``PriceTargetAchievement`` model's ``pt_df`` index).
        target_isin : str, optional
            Skip the auto-selection step and fit on this ISIN directly.
        min_observations : int
            Minimum non-null observations per ISIN required for fitting.
        timestamp_col : str
            Column carrying the snapshot anchor.
        now_cols : tuple of str
            Columns whose values seed the "now" anchor in the long panel.
        **fit_kwargs : Any
            Forwarded verbatim to :meth:`fit` (``samples``, ``tune``,
            ``chains``, ``parameterization``, ``nuts_sampler``, …).

        Returns
        -------
        tuple
            ``(idata, model)`` from :meth:`fit`. When no eligible ISIN
            exists, returns ``(None, None)`` and emits a warning.
        """
        long_df, eligible, date_col = self.build_price_target_history(
            df,
            id_col=id_col,
            now_cols=now_cols,
            min_observations=min_observations,
            timestamp_col=timestamp_col,
        )
        if date_col is None or eligible.empty:
            import warnings

            warnings.warn(
                "No ISIN has >= %d non-null price_target observations across "
                "the *_ago cohort. Skipping Kalman fit (point-in-time snapshot)."
                % min_observations,
                stacklevel=2,
            )
            return None, None

        chosen = target_isin or self.select_target_isin(eligible, cohort=cohort)
        if chosen is None:
            return None, None

        ts = (
            long_df.loc[long_df[id_col] == chosen, ["asof_date", "price_target"]]
            .dropna()
            .sort_values("asof_date")
            .reset_index(drop=True)
        )
        dates = pd.DatetimeIndex(ts["asof_date"])
        return self.fit(
            price_targets=ts["price_target"].to_numpy(),
            isin=chosen,
            dates=dates,
            **fit_kwargs,
        )
