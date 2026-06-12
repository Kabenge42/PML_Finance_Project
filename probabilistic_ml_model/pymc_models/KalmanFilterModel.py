"""
Kalman Filter Price Target Model — GaussianRandomWalk state-space.

Uses PyMC GaussianRandomWalk for latent price state with an observation model
for noisy price targets.

Reference: run_kalman_filter() in expected_returns_v3.py (line 1519);
           probability_models.py Kalman references.
"""

from __future__ import annotations

import logging
import re
import warnings
from functools import lru_cache
from typing import Any, Hashable, Literal, Optional, TYPE_CHECKING, Union

from pandas import Timestamp
from pandas._libs import NaTType
from pandas.tseries.offsets import DateOffset, MonthBegin, QuarterBegin, YearBegin
from pymc.backends.base import MultiTrace

# PyMC 6.0 + ArviZ 1.0: top-level ``arviz`` re-exports the modular API,
# so the legacy ``arviz_base`` fallback (PyMC 5.x transition artefact) is
# no longer needed. The ``ImportError`` guard is kept defensively for
# environments where arviz is absent.
try:
    import arviz as az
except ImportError:
    az = None  # type: ignore[assignment]
import numpy as np
import pandas as pd

try:
    import pymc as pm
    import pytensor.tensor as pt
except ImportError:
    pm = None  # type: ignore[assignment]
    pt = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import arviz as az_typing  # noqa: F401
    import pymc as pm_typing  # noqa: F401

from probabilistic_ml_model._pymc_arviz_compat import InferenceLike
from probabilistic_ml_model.pymc_models._pytensor_compat import get_pytensor_compile_kwargs
from probabilistic_ml_model.pymc_models._feature_alignment import (
    coerce_by_data_type,
    load_feature_metadata_from_db,
)
from probabilistic_ml_model.pymc_models._hierarchy import (
    build_hierarchy_indices,
    coerce_categories,
)

logger = logging.getLogger(__name__)

Parameterization = Literal["centered", "non_centered", "marginalized", "auto"]

# Below this many observations the latent random-walk funnel becomes hard to
# sample; ``parameterization="auto"`` collapses to the funnel-free
# ``marginalized`` form for such short ``*_ago`` cohorts.
_SHORT_SERIES_THRESHOLD = 25

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


class KalmanFilterPriceTarget:
    """Bayesian state-space model for price target filtering.

    Uses GaussianRandomWalk as latent state with HalfNormal priors on
    state and observation noise (scaled to data std).
    """

    def __init__(self) -> None:
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[InferenceLike] = None
        # Fit context retained so :meth:`forecast` can project the fitted
        # local-level process forward to future fiscal events without refitting.
        self._fit_idata_: Optional[InferenceLike] = None
        self._fit_last_price_: Optional[float] = None
        self._fit_has_trend_: bool = False

    @staticmethod
    @lru_cache(maxsize=4)
    def _resolve_kalman_feature_aliases(
        connection_string: Optional[str] = None,
    ) -> tuple[str, ...]:
        """Resolve the Kalman mutable_predictor aliases from the PyMC catalogue.

        Source of truth (pml schema)::

            SELECT feature_alias
            FROM pml.vw_pymc_feature_catalogue
            WHERE model_target = 'kalman_pt' AND pymc_role = 'mutable_predictor'

        These label the ``kalman_feature`` dim of the auxiliary ``pm.Data``
        container of observed Kalman-feature values (price-target dynamics +
        technical-analysis signals).

        Returns a tuple so the result is hashable / cache-friendly; callers
        should convert to ``list`` if mutation is needed. Returns an empty
        tuple on any failure so callers can fall back gracefully.
        """
        try:
            import os

            import pandas as _pd

            from probabilistic_ml_model.data_utils.data_utils import (
                get_analytics_engine,
            )

            try:
                from sqlalchemy import create_engine
            except ImportError:  # pragma: no cover - defensive
                create_engine = None  # type: ignore[assignment]

            sql = (
                "SELECT DISTINCT feature_alias "
                "FROM pml.vw_pymc_feature_catalogue "
                "WHERE model_target = 'kalman_pt' "
                "AND pymc_role = 'mutable_predictor' "
                "ORDER BY feature_alias"
            )
            url = connection_string or os.environ.get("DB_URL")
            if create_engine is not None and url:
                engine = create_engine(url)
            else:
                engine = get_analytics_engine()
            df = _pd.read_sql(sql, engine)
            return tuple(df["feature_alias"].dropna().astype(str).tolist())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load kalman_pt feature catalogue: %s", exc)
            return tuple()

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
    ) -> Union[Timestamp, NaTType]:
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

    @staticmethod
    def _resolve_fiscal_anchor(
            df: pd.DataFrame,
            id_col: str,
            anchor_col: Optional[str],
    ) -> Optional[pd.Series]:
        """Return a per-ISIN fiscal-calendar anchor date for ``*_ago`` offsetting.

        When ``anchor_col`` (e.g. ``income_statement_report_date``) is present,
        each ISIN's ``*_ago`` observations are measured back from that ISIN's own
        fiscal anchor instead of a single global "now". The most recent
        (``max``) date per ISIN is used so the latest snapshot lands on the last
        actual reporting date.

        Parameters
        ----------
        df
            Wide snapshot frame.
        id_col
            ISIN column name.
        anchor_col
            Fiscal-calendar date column to anchor on. ``None`` disables fiscal
            anchoring (callers fall back to :meth:`_resolve_reference_now`).

        Returns
        -------
        pandas.Series or None
            ISIN-indexed tz-naive anchor timestamps, or ``None`` when the column
            is absent or holds no parseable dates.
        """
        if not anchor_col or anchor_col not in df.columns or id_col not in df.columns:
            return None
        anchors = pd.to_datetime(df[anchor_col], errors="coerce")
        tmp = pd.DataFrame({id_col: df[id_col].to_numpy(), "_anchor": anchors})
        tmp = tmp.dropna(subset=["_anchor"])
        if tmp.empty:
            return None
        s = tmp.groupby(id_col)["_anchor"].max()
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
        return s

    @classmethod
    def build_price_target_history(
        cls,
        df: pd.DataFrame,
        *,
        id_col: str = "isin",
        now_cols: tuple[str, ...] = ("price_target", "last_price"),
        min_observations: int = 2,
        timestamp_col: str = "feature_calculated_at",
            fiscal_anchor_col: Optional[str] = None,
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
        fiscal_anchor_col : str, optional
            Fiscal-calendar date column (e.g. ``income_statement_report_date``)
            to anchor the ``*_ago`` axis on. When supplied, each ISIN's
            observations are offset from its own fiscal anchor (resolved via
            :meth:`_resolve_fiscal_anchor`) so the time axis tracks the real
            reporting cadence rather than a single global "now". ISINs without a
            parseable anchor fall back to the global ``ref_now``. ``None`` (the
            default) preserves the prior single-anchor behaviour.

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
        offset_map = _build_ago_offset_map()
        ref_now = cls._resolve_reference_now(df, timestamp_col=timestamp_col)
        anchor_by_isin = cls._resolve_fiscal_anchor(df, id_col, fiscal_anchor_col)

        def _anchor_series(ids: pd.Series) -> pd.Series:
            """Per-row anchor timestamps: the ISIN's fiscal anchor, else ``ref_now``."""
            if anchor_by_isin is None:
                return pd.Series(ref_now, index=ids.index)
            return ids.map(anchor_by_isin).fillna(ref_now)

        ago_re = re.compile(_AGO_HISTORY_RE)
        history_cols: list[tuple[str, str]] = []
        for col in df.columns:
            m = ago_re.match(col)
            if m:
                history_cols.append((col, m.group("suf")))

        frames: list[pd.DataFrame] = []
        # Historical ``*_ago`` snapshots: offset each row from its own anchor so the
        # axis is fiscal-calendar aligned when ``fiscal_anchor_col`` is supplied.
        for col, suf in history_cols:
            off = offset_map.get(suf.lower())
            if off is None:
                continue
            piece = (
                df[[id_col, col]]
                .rename(columns={col: "price_target"})
                .dropna(subset=["price_target"])
            )
            if piece.empty:
                continue
            piece = piece.assign(
                asof_date=_anchor_series(piece[id_col]) - off, source_col=col
            )
            frames.append(piece)
        # "Now" columns seeded at each ISIN's anchor (the latest reporting date when
        # fiscal anchoring is active, else the global ``ref_now``).
        for col in (c for c in now_cols if c in df.columns):
            piece = (
                df[[id_col, col]]
                .rename(columns={col: "price_target"})
                .dropna(subset=["price_target"])
            )
            if piece.empty:
                continue
            piece = piece.assign(asof_date=_anchor_series(piece[id_col]), source_col=col)
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
    ) -> Optional[Hashable]:
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

    @staticmethod
    def _prepare_log_targets(
        price_targets: np.ndarray,
        dates: Optional[pd.DatetimeIndex],
    ) -> tuple[np.ndarray, np.ndarray, float, Optional[pd.DatetimeIndex]]:
        """Validate, clean, and log-transform the observed price targets.

        Drops non-finite observations (and the aligned ``dates`` entries),
        verifies strict positivity, and derives a clamped log-space scale
        for the noise priors.

        Parameters
        ----------
        price_targets : numpy.ndarray
            Raw observed price target series (length >= 2).
        dates : pandas.DatetimeIndex, optional
            Time index aligned to ``price_targets``; filtered alongside it.

        Returns
        -------
        tuple
            ``(pt_arr, log_pt, scale, dates)`` — the cleaned positive series,
            its log transform, the clamped prior scale, and the (possibly
            filtered) ``dates``.

        Raises
        ------
        ValueError
            If fewer than 2 (finite) observations remain, or any value is
            non-positive.
        """
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

        # Clamp scale to a sane band so degenerate inputs (constant series, or
        # a single outlier driving std to ∞) cannot blow up the priors.
        raw_scale = float(np.nanstd(log_pt))
        scale = float(np.clip(raw_scale if np.isfinite(raw_scale) else 0.0, 1e-3, 1.0))
        return pt_arr, log_pt, scale, dates

    @staticmethod
    def implied_upside_from_state(
        state: np.ndarray,
        last_price: float,
    ) -> np.ndarray:
        """Return the implied upside of a (smoothed) price state vs spot.

        Mirrors the SQL ``pml.calc_change_ratio(price_target, last_price)``
        feature ``feat_implied_upside`` and the cross-sectional notebook's
        ``expected_upside`` so the single-ISIN time-series model reports the
        same, directly-comparable quantity:

        .. math:: \\text{implied\\_upside} = \\frac{\\text{state}}{\\text{last\\_price}} - 1.

        Parameters
        ----------
        state : numpy.ndarray
            Price-space latent state (e.g. posterior-mean ``state``).
        last_price : float
            Reference spot price; must be strictly positive.

        Returns
        -------
        numpy.ndarray
            Implied-upside ratio aligned to ``state``. Returns an all-NaN array
            when ``last_price`` is non-finite or non-positive.
        """
        arr = np.asarray(state, dtype="float64")
        if not np.isfinite(last_price) or last_price <= 0:
            return np.full_like(arr, np.nan)
        return arr / float(last_price) - 1.0

    @staticmethod
    def _resolve_coords(
        pt_arr: np.ndarray,
        dates: Optional[pd.DatetimeIndex],
        isin: Optional[str],
        sectors: Optional[np.ndarray],
        categories_df: Optional[pd.DataFrame],
        hierarchy_levels: Optional[list[str]],
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Assemble PyMC model coords and the singleton ``isin`` index.

        Registers a ``time`` coord (from ``dates`` or a positional range), an
        ``isin`` coord (mirroring ``pml.vw_pml_df_coords`` where
        ``column_name = 'isin'``), and any optional category-hierarchy coords
        via :func:`coerce_categories`.

        Returns
        -------
        tuple
            ``(coords, isins_arr)``.
        """
        T = len(pt_arr)
        time_coords = dates if dates is not None else np.arange(T, dtype=np.int64)
        coords: dict[str, Any] = {"time": time_coords}

        # `isin` mirrors pml.vw_pml_df_coords (column_name='isin',
        # pymc_role='coord'). When ``isin`` is not supplied we fall back to a
        # singleton positional index so the optional category-hierarchy coords
        # can still be wired in.
        if isin is not None:
            isins_arr = np.asarray([isin])
        else:
            isins_arr = np.arange(1, dtype="int64")
        coords["isin"] = isins_arr

        # Optional category hierarchy registers coords for downstream pivots.
        cats_df, levels = coerce_categories(
            isins_arr,
            sectors=sectors,
            categories_df=categories_df,
            hierarchy_levels=hierarchy_levels
            or (["sector", "industry"] if categories_df is not None else None),
        )
        if cats_df is not None and levels:
            hierarchy_meta = build_hierarchy_indices(cats_df, isins_arr, levels=levels)
            for lv, meta in hierarchy_meta.items():
                coords[lv] = meta["labels"]
        return coords, isins_arr

    @staticmethod
    def _resolve_time_deltas(
        dates: Optional[pd.DatetimeIndex],
        n_obs: int,
    ) -> np.ndarray:
        """Return a non-decreasing elapsed-time vector (years) anchored at zero.

        The marginalized formulation scales the random-walk innovation variance
        by *real* spacing between observations (the ``*_ago`` cohort is highly
        irregular — 1w, 1m, 3m, 6m, 1y). This converts ``dates`` into cumulative
        elapsed years :math:`\\tau` with :math:`\\tau_0 = 0`, used to build the
        Wiener-process covariance kernel :math:`\\min(\\tau_s, \\tau_t)`.

        Falls back to unit steps (``0, 1, …, n-1``) when ``dates`` are absent,
        contain NaT, or are degenerate (all identical) so the kernel is still
        well defined.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, optional
            Observation timestamps aligned to the cleaned price series.
        n_obs : int
            Number of (finite, positive) observations.

        Returns
        -------
        numpy.ndarray
            Float64 elapsed-time vector of length ``n_obs``.
        """
        if dates is not None and len(dates) == n_obs and n_obs > 1:
            idx = pd.DatetimeIndex(dates)
            if not idx.hasnans:
                # Subtract in datetime space (exact) then convert the small
                # deltas to float days — converting the raw nanosecond epoch to
                # float64 first would lose day-level precision (~1e18 ns).
                days = ((idx - idx[0]) / pd.Timedelta(days=1)).to_numpy(dtype="float64")
                tau = np.maximum.accumulate(days) / 365.25
                if np.isfinite(tau).all() and tau[-1] > 0:
                    return tau
        return np.arange(n_obs, dtype="float64")

    @staticmethod
    def _build_log_state(
        parameterization: Parameterization,
        log_pt: np.ndarray,
        sigma_state: Any,
        scale: float,
            tau: np.ndarray,
        init_mu: Optional[float] = None,
            beta_trend: Any = None,
    ) -> Any:
        """Build the explicit latent log-price state (non-marginalized forms).

        Must be called inside an active ``pm.Model`` context. The
        ``marginalized`` parameterization does *not* go through this helper —
        it integrates the latent path out analytically in
        :meth:`_build_marginalized_likelihood`.

        - ``centered`` — an explicit :class:`pymc.GaussianRandomWalk` (registered
          as ``log_level``).
        - ``non_centered`` (default) — ``init + cumsum(sigma_state * z)``.

        When ``beta_trend`` is supplied, a deterministic linear trend
        ``beta_trend * tau`` is added to the latent mean (the reference notebook's
        structural-trend fix that stops the projection decaying to a flat
        baseline). The combined latent path is always exposed as the ``log_state``
        Deterministic so downstream consumers resolve uniformly.

        Parameters
        ----------
        parameterization : Parameterization
            ``"centered"`` or ``"non_centered"``.
        log_pt : numpy.ndarray
            Cleaned log-price-target series (its first element seeds the
            initial-level prior when ``init_mu`` is not supplied).
        sigma_state : pytensor tensor
            Random-walk (process) noise scale.
        scale : float
            Clamped log-scale for the diffuse initial-level prior.
        tau : numpy.ndarray
            Elapsed-time vector (years) from :meth:`_resolve_time_deltas`; scales
            the optional ``beta_trend`` term.
        init_mu : float, optional
            Initial latent log-level anchor. Defaults to ``log_pt[0]``; pass
            ``log(last_price)`` to anchor the smoother to the current spot
            price instead of the first observed target.
        beta_trend : pytensor tensor, optional
            Per-year log-price trend slope. ``None`` (default) omits the trend.
        """
        anchor = float(log_pt[0]) if init_mu is None else float(init_mu)
        if parameterization == "centered":
            state_mean = pm.GaussianRandomWalk(
                "log_level",
                sigma=sigma_state,
                init_dist=pm.Normal.dist(mu=anchor, sigma=scale),
                dims="time",
            )
        else:
            # Non-centred GRW: state = init + cumsum(sigma_state * z).
            z_innov = pm.Normal("z_innov", 0.0, 1.0, dims="time")
            state_mean = anchor + pt.cumsum(sigma_state * z_innov)
        if beta_trend is not None:
            tau_t = pt.as_tensor_variable(np.asarray(tau, dtype="float64"))
            state_mean = state_mean + beta_trend * tau_t
        return pm.Deterministic("log_state", state_mean, dims="time")

    @staticmethod
    def _build_marginalized_likelihood(
        log_obs_data: Any,
        log_pt: np.ndarray,
        sigma_state: Any,
        sigma_obs: Any,
        scale: float,
        tau: np.ndarray,
        init_mu: Optional[float] = None,
            beta_trend: Any = None,
    ) -> Any:
        """Integrate the latent random walk out into the likelihood covariance.

        Must be called inside an active ``pm.Model`` context. Implements a true
        *marginalized* local-level (integrated Wiener) state-space filter: the
        latent log-price path :math:`x_{1:T}` is collapsed analytically, leaving
        a single multivariate-normal likelihood for the observed log-targets
        whose covariance carries **both** the random-walk (process) variance and
        the observation (measurement) variance.

        For a local-level model with continuous-time process noise,

        .. math::

            x_t = x_1 + W(\\tau_t), \\qquad y_t = x_t + \\varepsilon_t,

        marginalizing :math:`x` yields :math:`y \\sim \\mathcal{N}(\\mu_0\\mathbf{1},\\;\\Sigma)`
        with

        .. math::

            \\Sigma_{st} = P_0 + \\sigma_{\\text{state}}^2 \\min(\\tau_s, \\tau_t)
                          + \\sigma_{\\text{obs}}^2 \\, \\delta_{st}.

        Unlike the previous implementation, ``log_pt`` is used **only** as the
        observed series (never as the mean), so the random-walk and observation
        variances are genuinely identified by the data.

        Parameters
        ----------
        log_obs_data : pytensor tensor
            The :class:`pymc.Data` container holding the observed log-targets.
        log_pt : numpy.ndarray
            Cleaned log-price-target series (used only for static shape / seed).
        sigma_state, sigma_obs : pytensor tensor
            Random-walk (process) and observation (measurement) noise scales.
        scale : float
            Clamped log-scale used for the diffuse initial-level variance/prior.
        tau : numpy.ndarray
            Elapsed-time vector (years) from :meth:`_resolve_time_deltas`.
        init_mu : float, optional
            Initial latent log-level prior mean. Defaults to ``log_pt[0]``; pass
            ``log(last_price)`` to anchor the smoother to the current spot price.

        Returns
        -------
        pytensor tensor
            The analytic smoother mean :math:`\\mathbb{E}[x \\mid y]`, registered
            as the ``log_state`` Deterministic so downstream consumers (the
            ``state`` Deterministic and path plots) continue to resolve.
        """
        from pytensor.tensor import linalg as pt_linalg

        n = int(len(log_pt))
        tau_t = pt.as_tensor_variable(np.asarray(tau, dtype="float64"))
        # Wiener-process kernel: Cov(x_s, x_t) = P0 + sigma_state^2 * min(tau_s, tau_t).
        min_tau = pt.minimum(tau_t[:, None], tau_t[None, :])
        eye = pt.eye(n)
        p0 = float(scale) ** 2  # diffuse initial-level variance
        anchor = float(log_pt[0]) if init_mu is None else float(init_mu)
        mu0 = pm.Normal("log_state_init", mu=anchor, sigma=float(scale))
        state_cov = p0 + pt.sqr(sigma_state) * min_tau
        # Marginal observation covariance = signal covariance + measurement noise
        # on the diagonal (+ tiny jitter for numerical positive-definiteness).
        obs_cov = state_cov + (pt.sqr(sigma_obs) + 1e-6) * eye
        # Local-level mean is the diffuse initial level, plus an optional linear
        # trend in elapsed time (the structural-trend term from the reference).
        mu_vec = mu0 * pt.ones(n)
        if beta_trend is not None:
            mu_vec = mu_vec + beta_trend * tau_t
        pm.MvNormal("obs", mu=mu_vec, cov=obs_cov, observed=log_obs_data, dims="time")

        # Analytic Kalman/RTS smoother mean E[x | y] = mu + K Sigma^{-1} (y - mu),
        # where K (signal covariance) = Cov(x, y) since the measurement noise is
        # independent of the state. Exposed as `log_state` for downstream plots.
        resid = log_obs_data - mu_vec
        smoothed = mu_vec + pt.dot(
            state_cov, pt_linalg.solve(obs_cov, resid, assume_a="pos", b_ndim=1)
        )
        return pm.Deterministic("log_state", smoothed, dims="time")

    @staticmethod
    def _build_sample_kwargs(
        *,
        samples: int,
        tune: int,
        chains: int,
        target_accept: float,
        random_seed: int,
        nuts_sampler: Optional[str],
        sample_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the keyword arguments for :func:`pymc.sample`.

        Applies the project defaults (compile kwargs, no log-likelihood),
        layers in ``nuts_sampler`` and caller overrides, then strips
        ``idata_kwargs`` for nutpie (which ignores it and warns).
        """
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
        return scall

    @staticmethod
    def _log_sample_diagnostics(idata: Any, isin: Optional[str]) -> None:
        """Log divergences and minimum ESS so quality is self-reported.

        Inspects ``idata.sample_stats["diverging"]`` and, when available,
        the bulk effective sample size, emitting warnings rather than
        relying on console scraping of the sampler output.

        Parameters
        ----------
        idata : Any
            The object returned by :func:`pymc.sample` (ArviZ ``InferenceData``
            / ``xarray.DataTree`` or a ``MultiTrace``).
        isin : str, optional
            ISIN tag used to label the log messages.
        """
        tag = isin if isin is not None else "?"
        sample_stats = getattr(idata, "sample_stats", None)
        if sample_stats is None or "diverging" not in getattr(sample_stats, "data_vars", {}):
            return
        try:
            n_div = int(sample_stats["diverging"].sum())
        except Exception:  # pragma: no cover - defensive
            return
        if n_div:
            logger.warning(
                "KalmanFilterPriceTarget[%s]: %d divergences after tuning; "
                "consider parameterization='marginalized' or a higher target_accept.",
                tag,
                n_div,
            )
        if az is not None and hasattr(az, "ess"):
            try:
                min_ess = float(az.ess(idata).to_array().min())
            except Exception:  # pragma: no cover - defensive
                min_ess = float("nan")
            if np.isfinite(min_ess) and min_ess < 100:
                logger.warning(
                    "KalmanFilterPriceTarget[%s]: minimum ESS %.0f < 100; "
                    "increase tune/draws for reliable r-hat / ESS.",
                    tag,
                    min_ess,
                )

    def fit(
        self,
        price_targets: np.ndarray,
        isin: Optional[str] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        last_price: Optional[float] = None,
        sectors: Optional[np.ndarray] = None,
        categories_df: Optional[pd.DataFrame] = None,
        hierarchy_levels: Optional[list[str]] = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
            trend: bool = False,
            trend_sigma: float = 0.5,
        parameterization: Parameterization = "auto",
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> tuple[Union[InferenceLike, MultiTrace], Any]:
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
        last_price : float, optional
            Current spot (last) price. When supplied and strictly positive it
            (a) anchors the latent log-level prior at ``log(last_price)`` instead
            of the first observed target — so the smoother is filtered relative to
            spot — and (b) exposes an ``implied_upside`` Deterministic
            (``state / last_price - 1``) over the ``time`` dim, mirroring the SQL
            ``feat_implied_upside`` feature and the cross-sectional model's
            ``expected_upside``. When ``None`` the model is unchanged.
        sectors : numpy.ndarray, optional
            Optional 1-D array of sector labels aligned to the (singleton) ``isin``
            coord. Forwarded to :func:`coerce_categories` so the model registers a
            ``sector`` coord even when ``categories_df`` is not supplied.
        categories_df : pandas.DataFrame, optional
            Optional ISIN-indexed (or ISIN-columned) frame carrying category columns
            (``sector``, ``industry``, …). When provided, its columns become
            additional model coords keyed by hierarchy level so downstream
            consumers can pivot the latent state by sector / industry.
        hierarchy_levels : list of str, optional
            Subset of column names from ``categories_df`` to register as coords.
            Defaults to ``["sector", "industry"]`` when ``categories_df`` is
            supplied and this argument is omitted.
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
        trend : bool
            When ``True``, add a deterministic linear trend ``beta_trend * tau``
            (per-year log-price slope) to the latent state mean. This is the
            structural-trend component from the *Forecasting with Structural AR
            Timeseries* reference that prevents :meth:`forecast` projections from
            decaying to a flat baseline. Default ``False`` (driftless local level).
        trend_sigma : float
            Prior scale for the ``beta_trend ~ Normal(0, trend_sigma)`` slope (in
            log-price units per year). Only used when ``trend=True``. Default 0.5.
        parameterization : Parameterization
            Model parameterization approach used for the latent state. Options are:
            "auto", "non_centered", "centered", or "marginalized". Default is "auto",
            which selects the funnel-free "marginalized" form for short series
            (fewer than ``_SHORT_SERIES_THRESHOLD`` observations) and
            "non_centered" otherwise.

            - "centered" / "non_centered" sample an explicit latent
              :class:`pymc.GaussianRandomWalk` path with a diagonal-Normal
              observation likelihood.
            - "marginalized" integrates the latent path out analytically into an
              :class:`pymc.MvNormal` likelihood whose covariance carries both the
              random-walk (process) and observation (measurement) variances,
              scaled by the real elapsed time between observations. The smoothed
              latent path is recovered as the analytic Kalman-smoother mean. This
              is funnel-free and exact for short, irregular ``*_ago`` cohorts.
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

        pt_arr, log_pt, scale, dates = self._prepare_log_targets(price_targets, dates)

        # Optional spot-price anchor: when a strictly-positive ``last_price`` is
        # supplied the latent log-level prior is centred at ``log(last_price)``
        # (so the smoother is filtered relative to spot rather than the first
        # observed target), and an ``implied_upside`` Deterministic is exposed.
        has_last_price = last_price is not None and np.isfinite(last_price) and last_price > 0
        init_mu = float(np.log(last_price)) if has_last_price else None

        # Resolve ``"auto"`` to the funnel-free ``marginalized`` parameterization
        # for short ``*_ago`` cohorts (where the explicit latent random walk is
        # poorly identified and funnels) and to ``non_centered`` otherwise. The
        # ``marginalized`` form is now a genuine integrated-out GRW (see
        # :meth:`_build_marginalized_likelihood`) — it analytically collapses the
        # latent path into an MvNormal likelihood, so selecting it for short
        # series is statistically correct rather than the prior no-op.
        if parameterization == "auto":
            parameterization = (
                "marginalized" if len(pt_arr) < _SHORT_SERIES_THRESHOLD else "non_centered"
            )
            logger.info(
                "KalmanFilterPriceTarget: auto-selected %r parameterization for %d obs.",
                parameterization,
                len(pt_arr),
            )

        # Elapsed-time vector for the (optionally marginalized) GRW covariance —
        # scales process variance by real spacing across the irregular cohort.
        tau = self._resolve_time_deltas(dates, len(pt_arr))

        coords, _ = self._resolve_coords(
            pt_arr, dates, isin, sectors, categories_df, hierarchy_levels
        )

        with pm.Model(coords=coords) as model:
            # Store the raw price target series for downstream consumers /
            # `pm.set_data` swaps; the model itself operates on log-prices.
            pm.Data("price_target", pt_arr, dims="time")
            log_obs_data = pm.Data("log_price_target", log_pt, dims="time")

            # Mode-anchored, weakly-informative priors keep both variances away
            # from 0 and break the sigma_state/sigma_obs ridge that produces the
            # funnel (divergences, inflated r-hat, tiny ESS) under sparse data.
            # log-returns are naturally O(0.05-0.2), so anchor sigma_state there;
            # sigma_obs is anchored to the observed log-scale scatter.
            sigma_state = pm.Gamma("sigma_state", mu=0.10, sigma=0.05)
            sigma_obs = pm.Gamma("sigma_obs", mu=scale, sigma=scale)

            # Optional structural trend: a per-year log-price slope shared by the
            # likelihood mean and recovered by :meth:`forecast` so projections
            # carry directional structure (reference: "Specifying a Trend Model").
            beta_trend = pm.Normal("beta_trend", 0.0, float(trend_sigma)) if trend else None

            if parameterization == "marginalized":
                # Integrate the latent path out: the GRW (process) and
                # observation (measurement) variances both enter the MvNormal
                # likelihood covariance. ``log_state`` is the analytic smoother
                # mean, so the per-time latent series is recovered without an
                # explicit (funnel-prone) random walk.
                log_state = self._build_marginalized_likelihood(
                    log_obs_data, log_pt, sigma_state, sigma_obs, scale, tau,
                    init_mu=init_mu, beta_trend=beta_trend,
                )
            else:
                log_state = self._build_log_state(
                    parameterization, log_pt, sigma_state, scale, tau,
                    init_mu=init_mu, beta_trend=beta_trend,
                )
                pm.Normal(
                    "obs",
                    mu=log_state,
                    sigma=sigma_obs,
                    observed=log_obs_data,
                    dims="time",
                )

            # Expose the latent state in the original price space as a
            # Deterministic so downstream code that referenced ``state``
            # continues to work without overflow risk.
            state = pm.Deterministic("state", pm.math.exp(log_state), dims="time")

            # Implied upside vs spot: state / last_price - 1. Mirrors the SQL
            # ``feat_implied_upside`` feature and the cross-sectional model's
            # ``expected_upside`` so both Kalman variants report the same metric.
            if has_last_price:
                pm.Data("last_price", float(last_price))
                pm.Deterministic(
                    "implied_upside", state / float(last_price) - 1.0, dims="time"
                )

            idata = pm.sample(
                **self._build_sample_kwargs(
                    samples=samples,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=random_seed,
                    nuts_sampler=nuts_sampler,
                    sample_kwargs=sample_kwargs,
                )
            )

        # Store model metadata for downstream consumers
        if isin is not None:
            try:
                model.name = f"KalmanFilter[{isin}]"
            except Exception:
                pass

        # Surface sampler-quality diagnostics so the model self-reports funnel
        # problems (divergences / low ESS) instead of relying on console scraping.
        self._log_sample_diagnostics(idata, isin)

        self.model_ = model
        # Detect ArviZ InferenceData (or arviz-base xarray.DataTree per migration
        # guide) at runtime via the shim's typing alias. Falls back to MultiTrace
        # detection so external NUTS samplers that bypass ArviZ still work.
        _az_inference = getattr(az, "InferenceData", None) if az is not None else None
        self.idata_ = idata if (_az_inference is not None and isinstance(idata, _az_inference)) else None

        # Retain fit context so :meth:`forecast` can project forward to future
        # fiscal events. ``_fit_idata_`` keeps whatever has a ``posterior`` group
        # (InferenceData or xarray.DataTree) regardless of the MultiTrace check above.
        self._fit_idata_ = idata if hasattr(idata, "posterior") else None
        self._fit_last_price_ = float(last_price) if has_last_price else None
        self._fit_has_trend_ = bool(trend)
        return idata, model

    def forecast(
            self,
            horizons_days: Any,
            *,
            fiscal_dates: Optional[Any] = None,
            labels: Optional[list[str]] = None,
            last_price: Optional[float] = None,
            random_seed: int = 42,
    ) -> Any:
        """Project the fitted local-level process forward to future fiscal events.

        This is the structural *prediction step* of the *Forecasting with
        Structural AR Timeseries* reference, specialised to the continuous-time
        local-level (Wiener) process fit by :meth:`fit`. It conditions on the
        learned posterior — the terminal smoothed latent log-state
        :math:`x_T = \\text{log\\_state}[-1]`, the process / observation scales
        ``sigma_state`` / ``sigma_obs`` and the optional ``beta_trend`` — and
        propagates it forward to a set of future horizons (typically the
        ``days_to_*`` fiscal-calendar horizons emitted by ``mv_pymc_kalman_pt``).

        For a horizon :math:`\\Delta\\tau` years beyond the last observation the
        latent log-state is

        .. math::

            x_f \\mid x_T \\sim \\mathcal{N}\\bigl(x_T + \\beta_{\\text{trend}}\\,\\Delta\\tau,\\;
                                                  \\sigma_{\\text{state}}^2\\,\\Delta\\tau\\bigr),

        and the predictive (observed) target adds the measurement variance
        :math:`\\sigma_{\\text{obs}}^2` on top of that state variance.
        This is the GRW analogue of the reference's ``ar1_fut`` initialised at
        ``DiracDelta(ar[..., -1])`` and conditioned on the learned coefficients.
        Each horizon is independent of :math:`x_T`, so the predictive variance is
        non-decreasing in the horizon.

        Parameters
        ----------
        horizons_days : array-like
            Strictly-positive, finite day-offsets **beyond the last observation**
            (e.g. ``days_to_next_earnings`` re-based to the last observed date).
        fiscal_dates : array-like, optional
            Absolute future dates aligned 1:1 with ``horizons_days``; used as the
            ``time_future`` coordinate when supplied (else the day-offsets are).
        labels : list of str, optional
            Optional human labels (e.g. ``["next_earnings", "expected_report"]``)
            attached as a ``label`` coord along ``time_future``.
        last_price : float, optional
            Spot price for ``implied_upside_future``. Defaults to the
            ``last_price`` supplied at :meth:`fit` time.
        random_seed : int
            Seed for the predictive draws. Default 42.

        Returns
        -------
        arviz.InferenceData or xarray.Dataset
            A ``predictions`` group (when ArviZ is available, else a raw
            ``xarray.Dataset``) with dims ``(chain, draw, time_future)`` and
            variables ``forecast_state`` (latent, price space), ``forecast_pt``
            (predictive observed target, price space), ``predictive_sd_log``
            (closed-form log-space predictive sd) and — when a positive
            ``last_price`` is known — ``implied_upside_future``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not produced a sampled posterior.
        ValueError
            If ``horizons_days`` is empty / non-positive / non-finite, or
            ``fiscal_dates`` does not align with ``horizons_days``.
        """
        if self._fit_idata_ is None or not hasattr(self._fit_idata_, "posterior"):
            raise RuntimeError(
                "forecast() requires a completed fit() with a sampled posterior; "
                "call fit(...) first (with an ArviZ-returning sampler)."
            )
        import xarray as xr

        h = np.asarray(horizons_days, dtype="float64").ravel()
        if h.size == 0:
            raise ValueError("horizons_days must contain at least one horizon.")
        if not np.isfinite(h).all() or np.any(h <= 0):
            raise ValueError(
                "horizons_days must be strictly positive, finite day-offsets "
                "measured beyond the last observation."
            )
        delta_tau = h / 365.25  # elapsed years from the last observation

        post = self._fit_idata_.posterior
        sigma_state = np.asarray(post["sigma_state"].values, dtype="float64")
        sigma_obs = np.asarray(post["sigma_obs"].values, dtype="float64")
        # Terminal smoothed latent log-state x_T (per posterior draw).
        x_term = np.asarray(post["log_state"].isel(time=-1).values, dtype="float64")
        beta_trend = (
            np.asarray(post["beta_trend"].values, dtype="float64")
            if "beta_trend" in post
            else np.zeros_like(x_term)
        )

        # Broadcast (chain, draw) x (time_future).
        x_term_ = x_term[..., None]
        ss_ = sigma_state[..., None]
        so_ = sigma_obs[..., None]
        bt_ = beta_trend[..., None]
        dt_ = delta_tau[None, None, :]

        mean_log = x_term_ + bt_ * dt_
        proc_sd = ss_ * np.sqrt(dt_)
        # Closed-form predictive (observation) sd in log-space: process + measure.
        pred_sd_log = np.sqrt(ss_ ** 2 * dt_ + so_ ** 2) * np.ones_like(mean_log)

        rng = np.random.default_rng(random_seed)
        state_log = rng.normal(mean_log, proc_sd)  # latent future log-state
        obs_log = rng.normal(state_log, so_ * np.ones_like(dt_))  # predictive log-target

        lp = last_price if last_price is not None else self._fit_last_price_

        if fiscal_dates is not None:
            tf = pd.DatetimeIndex(pd.to_datetime(np.asarray(fiscal_dates)))
            if len(tf) != h.size:
                raise ValueError("fiscal_dates must align 1:1 with horizons_days.")
            time_future: np.ndarray = tf.values
        else:
            time_future = h

        dims = ("chain", "draw", "time_future")
        data_vars: dict[str, Any] = {
            "forecast_state": (dims, np.exp(state_log)),
            "forecast_pt": (dims, np.exp(obs_log)),
            "predictive_sd_log": (dims, pred_sd_log),
        }
        if lp is not None and np.isfinite(lp) and lp > 0:
            data_vars["implied_upside_future"] = (dims, np.exp(state_log) / float(lp) - 1.0)

        ds = xr.Dataset(
            data_vars,
            coords={
                "chain": post.coords["chain"].values,
                "draw": post.coords["draw"].values,
                "time_future": time_future,
            },
        )
        ds = ds.assign_attrs(
            horizons_days=h,
            trend=int(self._fit_has_trend_),
            last_price=float(lp) if (lp is not None and np.isfinite(lp)) else float("nan"),
        )
        if labels is not None:
            if len(labels) != h.size:
                raise ValueError("labels must align 1:1 with horizons_days.")
            ds = ds.assign_coords(label=("time_future", list(labels)))

        # Expose as a ``predictions`` group, mirroring
        # pm.sample_posterior_predictive(predictions=True). ArviZ 1.x aliases
        # InferenceData to xarray.DataTree, so build the group as a DataTree node
        # (accessible as ``result.predictions`` / ``result["predictions"]``).
        # Fall back to the bare Dataset if DataTree is unavailable.
        try:
            return xr.DataTree.from_dict({"predictions": ds})
        except Exception:  # pragma: no cover - older xarray without DataTree
            return ds

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
            fiscal_anchor_col: Optional[str] = None,
        **fit_kwargs: Any,
    ) -> tuple[Optional[Union[InferenceLike, MultiTrace]], Any]:
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
        fiscal_anchor_col : str, optional
            Forwarded to :meth:`build_price_target_history` to anchor the
            ``*_ago`` axis on a per-ISIN fiscal-calendar date (e.g.
            ``income_statement_report_date``). ``None`` keeps the global anchor.
        **fit_kwargs : Any
            Forwarded verbatim to :meth:`fit` (``samples``, ``tune``,
            ``chains``, ``parameterization``, ``trend``, ``nuts_sampler``, …).

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
            fiscal_anchor_col=fiscal_anchor_col,
        )
        if date_col is None or eligible.empty:
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

        # Pull the chosen ISIN's spot price from the snapshot so the smoother can
        # anchor to it and emit ``implied_upside`` (unless the caller already
        # supplied ``last_price`` explicitly).
        if "last_price" not in fit_kwargs and "last_price" in df.columns:
            lp = pd.to_numeric(
                df.loc[df[id_col] == chosen, "last_price"], errors="coerce"
            ).dropna()
            if not lp.empty:
                fit_kwargs["last_price"] = float(lp.iloc[0])

        return self.fit(
            price_targets=ts["price_target"].to_numpy(),
            isin=chosen,
            dates=dates,
            **fit_kwargs,
        )