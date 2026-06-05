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
        offset_map = _build_ago_offset_map()
        ref_now = cls._resolve_reference_now(df, timestamp_col=timestamp_col)

        def _ago_to_date(suffix: str) -> Any:
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
        init_mu: Optional[float] = None,
    ) -> Any:
        """Build the explicit latent log-price state (non-marginalized forms).

        Must be called inside an active ``pm.Model`` context. The
        ``marginalized`` parameterization does *not* go through this helper —
        it integrates the latent path out analytically in
        :meth:`_build_marginalized_likelihood`.

        - ``centered`` — an explicit :class:`pymc.GaussianRandomWalk`.
        - ``non_centered`` (default) — ``init + cumsum(sigma_state * z)``.

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
        init_mu : float, optional
            Initial latent log-level anchor. Defaults to ``log_pt[0]``; pass
            ``log(last_price)`` to anchor the smoother to the current spot
            price instead of the first observed target.
        """
        anchor = float(log_pt[0]) if init_mu is None else float(init_mu)
        if parameterization == "centered":
            return pm.GaussianRandomWalk(
                "log_state",
                sigma=sigma_state,
                init_dist=pm.Normal.dist(mu=anchor, sigma=scale),
                dims="time",
            )
        # Non-centred GRW: state = init + cumsum(sigma_state * z).
        z_innov = pm.Normal("z_innov", 0.0, 1.0, dims="time")
        return pm.Deterministic(
            "log_state",
            anchor + pt.cumsum(sigma_state * z_innov),
            dims="time",
        )

    @staticmethod
    def _build_marginalized_likelihood(
        log_obs_data: Any,
        log_pt: np.ndarray,
        sigma_state: Any,
        sigma_obs: Any,
        scale: float,
        tau: np.ndarray,
        init_mu: Optional[float] = None,
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
        mu_vec = mu0 * pt.ones(n)
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

            if parameterization == "marginalized":
                # Integrate the latent path out: the GRW (process) and
                # observation (measurement) variances both enter the MvNormal
                # likelihood covariance. ``log_state`` is the analytic smoother
                # mean, so the per-time latent series is recovered without an
                # explicit (funnel-prone) random walk.
                log_state = self._build_marginalized_likelihood(
                    log_obs_data, log_pt, sigma_state, sigma_obs, scale, tau,
                    init_mu=init_mu,
                )
            else:
                log_state = self._build_log_state(
                    parameterization, log_pt, sigma_state, scale, init_mu=init_mu
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