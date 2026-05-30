"""
DCF Price Target Model — Discounted Cash Flow with Bayesian priors.

Places priors on FCF growth rate and WACC, computes projected FCFs and
terminal value as PyMC Deterministics, and fits against observed market prices.

Reference: compute_derived_price_target() in expected_returns_v3.py (line 2987).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

# PyMC 6.0 + ArviZ 1.0: top-level ``arviz`` re-exports the modular API,
# so the legacy ``arviz_base`` fallback (PyMC 5.x transition artefact) is
# no longer needed. The ``ImportError`` guard is kept defensively for
# environments where arviz is absent.
try:
    import arviz as az
except ImportError:
    az = None  # type: ignore[assignment]

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
    stamp_feature_provenance,
)
from probabilistic_ml_model.pymc_models._hierarchy import (
    build_hierarchy_indices,
    coerce_categories,
)

logger = logging.getLogger(__name__)

Parameterization = Literal["centered", "non_centered", "marginalized"]

# Canonical category names in public.calculated_features_registry whose
# feature_aliases drive the auxiliary "dcf_feature" dim. Aligned with the
# cash-flow / valuation features feeding the DCF intrinsic-value model.
_DCF_CATEGORY_KEYS: tuple[str, ...] = (
    "Cash Flow",
    "Valuation Ratios",
    "Valuation Timeseries",
    "Revenue Forecasting",
    "Growth Metrics",
    "Profitability",
    "Efficiency Ratios",
    "EPS Trajectory",
    "Temporal Patterns",
    "Leverage & Liquidity",
    "Interest Income",
    "Employement Dynamics",
    "Employement Productivity",
    "Dividend Reliability",
)


class DCFPriceTarget:
    """Bayesian Discounted Cash Flow intrinsic-value model.

    Priors
    ------
    fcf_growth ~ Normal(historical_growth, 0.05)
    wacc ~ TruncatedNormal(0.10, 0.02, lower=terminal_growth+0.005, upper=0.30)
    terminal_growth fixed at 0.02

    The truncated WACC prior guarantees ``wacc > terminal_growth`` so the
    Gordon-growth terminal value remains finite.
    """

    def __init__(self, terminal_growth: float = 0.02) -> None:
        self.terminal_growth = terminal_growth
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[InferenceLike] = None

    # Coords actually emitted by pml.mv_pymc_dcf_pt (DDL order). `feat_next_
    # earnings_status` is intentionally excluded — it is NOT in the DCF MV
    # (it only exists in mv_pymc_earnings_beat).
    _DCF_COORD_COLS: tuple[str, ...] = (
        "region",
        "country",
        "trading_country",
        "exchange",
        "unit",
        "style_class",
        "size_class",
        "sector",
        "industry",
    )
    # Lowest-cardinality / highest-signal coords carrying partial-pooling effects.
    _DCF_GROUP_EFFECTS: tuple[str, ...] = ("sector", "size_class", "style_class")

    @staticmethod
    @lru_cache(maxsize=4)
    def _resolve_dcf_feature_aliases(
        connection_string: Optional[str] = None,
    ) -> tuple[str, ...]:
        """Resolve the DCF mutable_predictor aliases from the PyMC catalogue.

        Source of truth::

            SELECT feature_alias
            FROM pml.vw_pymc_feature_catalogue
            WHERE model_target = 'dcf_pt' AND pymc_role = 'mutable_predictor'

        These label the ``dcf_feature`` dim of the auxiliary ``pm.Data``
        container of observed DCF-feature values (cash-flow + valuation
        signals).
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
                "WHERE model_target = 'dcf_pt' "
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
            logger.warning("Could not load dcf_pt feature catalogue: %s", exc)
            return tuple()

    @staticmethod
    def _align_dcf_features(
        dcf_features_df: Optional[pd.DataFrame],
        isin: np.ndarray,
        feature_aliases: list[str],
        *,
        use_typed_coercion: bool = False,
        connection_string: Optional[str] = None,
    ) -> np.ndarray:
        """Align an (isin × dcf_feature) matrix to the model dims.

        Missing columns/rows are filled with ``0.0`` so the container always
        has shape ``(n_isin, n_dcf_feature)``.
        """
        n_isin = len(isin)
        n_feat = len(feature_aliases)
        if dcf_features_df is None or n_feat == 0 or n_isin == 0:
            return np.zeros((n_isin, max(n_feat, 0)), dtype="float64")

        df = dcf_features_df.copy()
        if "isin" in df.columns:
            df = df.drop_duplicates(subset="isin").set_index("isin")
        df = df.reindex(index=isin)

        if use_typed_coercion:
            metadata = load_feature_metadata_from_db(connection_string)
            return coerce_by_data_type(df, list(feature_aliases), metadata)
        return df.reindex(columns=feature_aliases).astype("float64").fillna(0.0).to_numpy()

    def fit(
        self,
        historical_fcf: np.ndarray,
        price_target: np.ndarray,
        isins: Optional[np.ndarray] = None,
        sectors: Optional[np.ndarray] = None,
        categories_df: Optional[pd.DataFrame] = None,
        hierarchy_levels: Optional[list[str]] = None,
        dcf_features_df: Optional[pd.DataFrame] = None,
        connection_string: Optional[str] = None,
        n_projection_years: int = 5,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 1,
        target_accept: float = 0.9,
        random_seed: int = 42,
        parameterization: Parameterization = "non_centered",
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> tuple[InferenceLike, pm_typing.Model]:
        """Fit DCF model and return ``(InferenceData, Model)``.

        Parameters
        ----------
        hierarchy_levels
        categories_df
        sectors
        nuts_sampler
        random_seed
        target_accept
        cores
        chains
        tune
        samples
        n_projection_years
        connection_string
        dcf_features_df
        isins
        price_target
        historical_fcf
        parameterization : {"centered", "non_centered", "marginalized"}
            Reparameterization strategy for ``fcf_growth`` and ``wacc``.
            ``"centered"`` (legacy) places priors directly on the
            parameters. ``"non_centered"`` (default) parameterizes
            ``fcf_growth = mu + sigma * z`` for better NUTS geometry; the
            truncated WACC prior is preserved (its support is bounded so
            non-centring offers limited benefit). ``"marginalized"``
            collapses both priors to their prior means (deterministic; no
            growth/WACC posterior — fastest path).
        """
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install a compatible version of pymc and arviz "
                "(arviz<1.0 or patch pymc for arviz-base) to use DCFPriceTarget."
            )

        hf = np.asarray(historical_fcf, dtype="float64")
        mp = np.asarray(price_target, dtype="float64")
        if hf.size < 2 or mp.size == 0:
            raise ValueError("Need ≥2 historical_fcf and ≥1 price_target.")

        growth_rates = np.diff(hf) / np.abs(hf[:-1] + 1e-10)
        hist_growth = float(np.mean(growth_rates))
        last_fcf = float(hf[-1])

        t = np.arange(1, n_projection_years + 1, dtype="float64")

        # --- DB-aligned coords --------------------------------------------------
        # `isin` mirrors public.vw_identifier_columns.isin (role='id'). When
        # ``isins`` is not supplied we fall back to a positional index so the
        # auxiliary ``dcf_features`` container can still be wired in.
        if isins is not None:
            isins_arr = np.asarray(isins)
        else:
            isins_arr = np.arange(mp.size).astype("int64")
        coords: dict[str, Any] = {"isin": isins_arr}

        # Register every categorical coord emitted by pml.mv_pymc_dcf_pt.
        # `categories_df` is expected to be the MV frame (indexed by / carrying
        # an `isin` column) so each coord can be factorised into a <coord>_idx.
        coord_idx: dict[str, np.ndarray] = {}
        if categories_df is not None:
            cats = categories_df.copy()
            if cats.index.name != "isin" and "isin" in cats.columns:
                cats = cats.drop_duplicates("isin").set_index("isin")
            cats = cats.reindex(index=isins_arr)
            for col in self._DCF_COORD_COLS:
                if col in cats.columns:
                    labels = cats[col].fillna("Unknown").astype(str).to_numpy()
                    uniques, idx = np.unique(labels, return_inverse=True)
                    coords[col] = uniques
                    coord_idx[col] = idx.astype("int64")

        # Hierarchy metadata (kept for nested-shrinkage downstream use).
        cats_df, levels = coerce_categories(
            isins_arr,
            sectors=sectors,
            categories_df=categories_df,
            hierarchy_levels=hierarchy_levels
                             or [c for c in self._DCF_GROUP_EFFECTS if c in coord_idx],
        )
        if cats_df is not None and levels:
            hierarchy_meta = build_hierarchy_indices(cats_df, isins_arr, levels=levels)
            for lv, meta in hierarchy_meta.items():
                coords.setdefault(lv, meta["labels"])
                coord_idx.setdefault(lv, np.asarray(meta["idx"], dtype="int64"))

        # `dcf_feature` is resolved from pml.vw_pymc_feature_catalogue so the
        # auxiliary pm.Data container carries human-readable feature_alias labels.
        dcf_feature_aliases = list(self._resolve_dcf_feature_aliases(connection_string))
        coords["dcf_feature"] = list(dcf_feature_aliases)

        dcf_features_arr = self._align_dcf_features(dcf_features_df, isins_arr, dcf_feature_aliases)

        with pm.Model(coords=coords) as model:
            price_data = pm.Data("price_target", mp, dims="isin")
            pm.Data(
                "dcf_features",
                dcf_features_arr,
                dims=("isin", "dcf_feature"),
            )

            # One pm.Data per coord-indexed group (sector_idx, size_class_idx, …)
            # so posterior predictive checks can be conditioned via pm.set_data.
            idx_data = {
                col: pm.Data(f"{col}_idx", coord_idx[col], dims="isin")
                for col in coord_idx
            }

            # WACC bounded strictly above terminal_growth → finite terminal value.
            if parameterization == "centered":
                fcf_growth = pm.Normal("fcf_growth", mu=hist_growth, sigma=0.05)
                wacc = pm.TruncatedNormal(
                    "wacc",
                    mu=0.10,
                    sigma=0.02,
                    lower=self.terminal_growth + 0.005,
                    upper=0.30,
                )
            elif parameterization == "marginalized":
                # Collapse latent priors: deterministic point estimates.
                fcf_growth = pm.Deterministic(
                    "fcf_growth", pt.constant(hist_growth, dtype="float64")
                )
                wacc = pm.Deterministic("wacc", pt.constant(0.10, dtype="float64"))
            else:
                # Non-centred Normal for fcf_growth: better NUTS geometry.
                z_growth = pm.Normal("z_growth", 0.0, 1.0)
                fcf_growth = pm.Deterministic("fcf_growth", hist_growth + 0.05 * z_growth)
                # WACC kept truncated (bounded support already well-conditioned).
                wacc = pm.TruncatedNormal(
                    "wacc",
                    mu=0.10,
                    sigma=0.02,
                    lower=self.terminal_growth + 0.005,
                    upper=0.30,
                )

            fcf_projected = last_fcf * (1 + fcf_growth) ** t
            discount_factors = (1 + wacc) ** t
            pv_fcfs = pt.sum(fcf_projected / discount_factors)

            fcf_terminal = last_fcf * (1 + fcf_growth) ** (n_projection_years + 1)
            terminal_value = fcf_terminal / (wacc - self.terminal_growth)
            terminal_pv = terminal_value / (1 + wacc) ** n_projection_years

            base_value = pv_fcfs + terminal_pv

            # Hierarchical partial-pooling multiplicative adjustments on the
            # highest-signal coords. Non-centred Normal on a log-scale so the
            # adjustment is strictly positive and shrinks toward 1.0. ``log_adj``
            # is initialised as an isin-length vector so ``intrinsic_value``
            # always carries the ``isin`` dim, even with no group effects.
            log_adj = pt.zeros(len(isins_arr))
            for col in self._DCF_GROUP_EFFECTS:
                if col not in idx_data:
                    continue
                sigma_g = pm.HalfNormal(f"sigma_{col}", sigma=0.25)
                z_g = pm.Normal(f"z_{col}", 0.0, 1.0, dims=col)
                eff = pm.Deterministic(f"{col}_effect", sigma_g * z_g, dims=col)
                log_adj = log_adj + eff[idx_data[col]]

            intrinsic_value = pm.Deterministic(
                "intrinsic_value", base_value * pt.exp(log_adj), dims="isin"
            )

            sigma = pm.HalfNormal("sigma", sigma=500.0)

            pm.Normal(
                "price_obs",
                mu=intrinsic_value,
                sigma=sigma,
                observed=price_data,
                dims="isin",
            )

            scall: dict[str, Any] = dict(
                draws=samples,
                tune=tune,
                chains=chains,
                cores=cores,
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

        # Recommendation §12.3 #3 — stamp feature_catalogue provenance.
        try:
            stamp_feature_provenance(
                idata,
                "dcf_features",
                dcf_feature_aliases,
                load_feature_metadata_from_db(connection_string),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("DCF provenance stamping failed: %s", exc)

        self.model_ = model
        self.idata_ = idata
        return idata, model