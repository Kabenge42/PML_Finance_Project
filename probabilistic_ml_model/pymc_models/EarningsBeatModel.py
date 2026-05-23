"""
Earnings Beat Bayesian Model — Hierarchical Beta-Binomial.

Replaces scipy Beta with PyMC Beta-Binomial, supports optional sector-level
hierarchical structure and forward-adjustment deterministics.

Reference: EarningsBeatProbabilityModel in probability_models.py (line 1131);
           build_beat_probability_inference_data() in inference_schema.py (line 459).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal, Optional, TYPE_CHECKING

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
except ImportError:
    pm = None  # type: ignore[assignment]

try:
    import nutpie  # noqa: F401
except ImportError:
    nutpie = None  # type: ignore[assignment]

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
from probabilistic_ml_model.data_utils.data_utils import load_feature_categories_from_db
from probabilistic_ml_model.pymc_models._hierarchy import (
    build_hierarchy_indices,
    build_nested_logit_normal_rates,
    coerce_categories,
)

logger = logging.getLogger(__name__)

# Canonical category names in public.calculated_features_registry
# whose feature_aliases drive the auxiliary "earnings_feature" dim.
# Aligned with categories backing public.vw_features_earnings.
_EARNINGS_CATEGORY_KEYS: tuple[str, ...] = ("Earnings Quality", "EPS Trajectory", "Profitability","Efficiency Ratios","Growth Metrics","Revenue Forecasting","Valuation Ratios","Temporal Patterns")

Parameterization = Literal["centered", "non_centered", "marginalized"]


class EarningsBeatBayesian:
    """Hierarchical Beta-Binomial earnings beat probability model.

    Parameters
    ----------
    prior_alpha : float
        Default Beta prior alpha for beat probability.
    prior_beta : float
        Default Beta prior beta for beat probability.
    """

    def __init__(
        self,
        prior_alpha: float = 1.5,
        prior_beta: float = 2.0,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[InferenceLike] = None

    @staticmethod
    @lru_cache(maxsize=4)
    def _resolve_earnings_feature_aliases(
        connection_string: Optional[str] = None,
    ) -> tuple[str, ...]:
        """Resolve the 'Earnings' feature aliases from calculated_features_registry.

        These serve as labelled coordinates along the ``earnings_feature`` dim
        for the auxiliary ``pm.Data`` container of observed earnings-feature
        values (e.g. eps_positive_streak, eps_surprise_pct).

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
        for key in _EARNINGS_CATEGORY_KEYS:
            for alias in categories.get(key, []):
                if alias not in seen:
                    seen.add(alias)
                    aliases.append(alias)
        return tuple(aliases)

    @staticmethod
    def _align_earnings_features(
        earnings_features_df: Optional[pd.DataFrame],
        isin: np.ndarray,
        feature_aliases: list[str],
        *,
        use_typed_coercion: bool = False,
        connection_string: Optional[str] = None,
    ) -> np.ndarray:
        """Align an (isin × earnings_feature) matrix to the model dims.

        Missing columns/rows are filled with ``0.0`` so the container always
        has shape ``(n_isin, n_earnings_feature)``.
        """
        n_isin = len(isin)
        n_feat = len(feature_aliases)
        if earnings_features_df is None or n_feat == 0 or n_isin == 0:
            return np.zeros((n_isin, max(n_feat, 0)), dtype="float64")

        df = earnings_features_df.copy()
        if "isin" in df.columns:
            df = df.drop_duplicates(subset="isin").set_index("isin")
        df = df.reindex(index=isin)

        if use_typed_coercion:
            metadata = load_feature_metadata_from_db(connection_string)
            return coerce_by_data_type(df, list(feature_aliases), metadata)
        return df.reindex(columns=feature_aliases).astype("float64").fillna(0.0).to_numpy()

    def fit(
        self,
        n_beats: np.ndarray,
        n_total: np.ndarray,
        isins: np.ndarray,
        sectors: Optional[np.ndarray] = None,
        categories_df: Optional[pd.DataFrame] = None,
        hierarchy_levels: Optional[list[str]] = None,
        earnings_features_df: Optional[pd.DataFrame] = None,
        connection_string: Optional[str] = None,
        samples: int = 500,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
        parameterization: Parameterization = "non_centered",
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> tuple[InferenceLike, pm_typing.Model]:
        """Fit the Beta-Binomial model and return (InferenceData, Model).

        Parameters
        ----------
        n_beats : array of int
            Number of earnings beats per stock.
        n_total : array of int
            Total number of earnings reports per stock.
        isins : array of str
            ISIN symbols (aligned to ``public.vw_identifier_columns.isin``).
        sectors : array of str, optional
            Sector labels — enables hierarchical sector-rate structure.
        earnings_features_df : pd.DataFrame, optional
            Optional (isin × earnings_feature) matrix of observed feature values
            from ``public.vw_features_earnings``.  Stored as a ``pm.Data``
            container so it can be swapped for out-of-sample prediction.
        connection_string : str, optional
            DB URL used when resolving the 'Earnings' feature aliases from
            ``public.calculated_features_registry`` via
            :func:`load_feature_categories_from_db`.
        samples, tune, chains, target_accept, random_seed
            MCMC sampling parameters.
        parameterization : {"centered", "non_centered", "marginalized"}
            Hierarchical reparameterization strategy. ``"non_centered"``
            (default) uses a logit-Normal hierarchy that NUTS handles
            efficiently. ``"marginalized"`` collapses ``beat_prob`` via
            ``pm.BetaBinomial`` (fastest; no per-stock latent posterior).
            ``"centered"`` preserves the legacy Beta hierarchy for
            reproducibility.
        nuts_sampler : str, optional
            Forwarded to ``pm.sample`` (e.g. ``"nutpie"``, ``"numpyro"``).
        **sample_kwargs
            Additional kwargs forwarded to ``pm.sample`` (e.g.
            ``idata_kwargs={"log_likelihood": False}``).

        Returns
        -------
        tuple[az.InferenceData, pm.Model]
            The fitted ArviZ ``InferenceData`` and the underlying
            ``pm.Model`` (also stored on ``self.idata_`` / ``self.model_``).
        """
        n_beats_arr = np.asarray(n_beats, dtype="int32")
        n_total_arr = np.asarray(n_total, dtype="int32")
        isins_arr = np.asarray(isins)

        # --- Robustness guards -------------------------------------------------
        if n_beats_arr.size == 0:
            raise ValueError("EarningsBeatBayesian.fit received empty n_beats array.")
        if n_beats_arr.shape != n_total_arr.shape:
            raise ValueError("n_beats and n_total must have the same shape.")
        if (n_beats_arr > n_total_arr).any():
            raise ValueError("n_beats cannot exceed n_total for any ISIN.")

        # --- Deduplicate ISINs (keep first occurrence) -------------------------
        # Duplicate ISINs in the coord break xarray alignment downstream
        # (e.g. when computing posterior beat_prob from alpha + n_b).
        _, _first_idx = np.unique(isins_arr, return_index=True)
        if len(_first_idx) != len(isins_arr):
            _first_idx.sort()
            logger.warning(
                "EarningsBeatBayesian.fit: dropping %d duplicate ISIN rows "
                "(kept first occurrence of each).",
                len(isins_arr) - len(_first_idx),
            )
            isins_arr = isins_arr[_first_idx]
            n_beats_arr = n_beats_arr[_first_idx]
            n_total_arr = n_total_arr[_first_idx]
            if sectors is not None:
                sectors = np.asarray(sectors)[_first_idx]

        # --- DB-aligned coords --------------------------------------------------
        # `isin` mirrors public.vw_identifier_columns.isin (role='id').  When
        # ``categories_df`` (or the legacy ``sectors=``) is supplied, every
        # column from ``hierarchy_levels`` becomes a coord and contributes a
        # nested non-centred logit-Normal layer to ``beat_prob``.
        coords: dict[str, Any] = {"isin": isins_arr}
        cats_df, levels = coerce_categories(
            isins_arr,
            sectors=sectors,
            categories_df=categories_df,
            hierarchy_levels=hierarchy_levels
            or (["exchange", "sector", "industry"] if categories_df is not None else None),
        )
        hierarchical = cats_df is not None and levels
        hierarchy_meta: Optional[dict] = None
        if hierarchical:
            hierarchy_meta = build_hierarchy_indices(cats_df, isins_arr, levels=levels)
            for lv, meta in hierarchy_meta.items():
                coords[lv] = meta["labels"]

        # `earnings_feature` is resolved from calculated_features_registry so the
        # auxiliary pm.Data container carries human-readable feature_alias labels.
        earnings_feature_aliases = list(self._resolve_earnings_feature_aliases(connection_string))
        coords["earnings_feature"] = list(earnings_feature_aliases)

        earnings_features_arr = self._align_earnings_features(
            earnings_features_df, isins_arr, earnings_feature_aliases
        )

        with pm.Model(coords=coords) as model:
            # --- pm.Data containers --------------------------------------------
            # Observed & design data live in pm.Data so they appear in the
            # model graph, are stored in idata.constant_data, and can be
            # swapped for forecasting via pm.set_data({...}).
            n_total_data = pm.Data("n_total", n_total_arr, dims="isin")
            n_beats_data = pm.Data("n_beats", n_beats_arr, dims="isin")
            pm.Data(
                "earnings_features",
                earnings_features_arr,
                dims=("isin", "earnings_feature"),
            )

            if hierarchical:
                if parameterization == "centered":
                    # Legacy single-level Beta hierarchy (still supported for
                    # reproducibility; only honoured when a single level was
                    # supplied via the legacy ``sectors=`` argument).
                    leaf_level = list(hierarchy_meta.keys())[-1]
                    leaf_idx_arr = hierarchy_meta[leaf_level]["idx"]
                    leaf_idx_data = pm.Data(f"{leaf_level}_idx", leaf_idx_arr, dims="isin")
                    sector_rate = pm.Beta(
                        "sector_rate",
                        alpha=self.prior_alpha,
                        beta=self.prior_beta,
                        dims=leaf_level,
                    )
                    kappa = pm.HalfNormal("kappa", sigma=10.0)
                    leaf_rate = sector_rate[leaf_idx_data]
                else:
                    # Nested non-centred logit-Normal — single shared helper.
                    nested = build_nested_logit_normal_rates(
                        hierarchy_meta,
                        leaf_dim="isin",
                        name="beat_rate",
                    )
                    leaf_rate = nested["leaf_rate"]
                    kappa = pm.Gamma("kappa", alpha=2.0, beta=0.1)

                # IMPORTANT: do NOT wrap (a, b) as pm.Deterministic with
                # dims="isin" — that would store them in idata.posterior at
                # the trained cohort length and break pm.set_data swaps for
                # OOS prediction. Keep them as plain PyTensor expressions so
                # they are recomputed from parent (non-isin) variables every
                # time pm.sample_posterior_predictive is invoked.
                a = leaf_rate * kappa
                b = (1.0 - leaf_rate) * kappa

                if parameterization == "marginalized":
                    # Collapse per-stock beat_prob latent — fastest path.
                    pm.BetaBinomial(
                        "beats_obs",
                        alpha=a,
                        beta=b,
                        n=n_total_data,
                        observed=n_beats_data,
                        dims="isin",
                    )
                else:
                    beat_prob = pm.Beta("beat_prob", alpha=a, beta=b, dims="isin")
                    pm.Binomial(
                        "beats_obs",
                        n=n_total_data,
                        p=beat_prob,
                        observed=n_beats_data,
                        dims="isin",
                    )
            else:
                if parameterization == "marginalized":
                    pm.BetaBinomial(
                        "beats_obs",
                        alpha=self.prior_alpha,
                        beta=self.prior_beta,
                        n=n_total_data,
                        observed=n_beats_data,
                        dims="isin",
                    )
                else:
                    beat_prob = pm.Beta(
                        "beat_prob",
                        alpha=self.prior_alpha,
                        beta=self.prior_beta,
                        dims="isin",
                    )
                    pm.Binomial(
                        "beats_obs",
                        n=n_total_data,
                        p=beat_prob,
                        observed=n_beats_data,
                        dims="isin",
                    )

            sample_call_kwargs: dict[str, Any] = dict(
                draws=samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=True,
                compile_kwargs=get_pytensor_compile_kwargs(),
            )
            if nuts_sampler is not None:
                sample_call_kwargs["nuts_sampler"] = nuts_sampler
            # Default to skipping log_likelihood unless caller overrides.
            sample_call_kwargs.setdefault("idata_kwargs", {"log_likelihood": False})
            sample_call_kwargs.update(sample_kwargs)

            idata = pm.sample(**sample_call_kwargs)

        # Recommendation §12.3 #3 — stamp feature_catalogue provenance.
        try:
            stamp_feature_provenance(
                idata,
                "earnings_features",
                earnings_feature_aliases,
                load_feature_metadata_from_db(connection_string),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("EarningsBeat provenance stamping failed: %s", exc)

        self.model_ = model
        self.idata_ = idata
        return idata, model