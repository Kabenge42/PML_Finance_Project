"""
Dividend Safety Bayesian Model — Cut probability with FCF coverage.

Models dividend cut probability with Beta priors and conditional risk
adjustments based on payout ratio thresholds.

Reference: DividendCutProbabilityModel in probability_models.py (line 2793).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal, Optional, TYPE_CHECKING

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
from probabilistic_ml_model.data_utils.data_utils import load_feature_categories_from_db
from probabilistic_ml_model.pymc_models._hierarchy import (
    build_hierarchy_indices,
    build_nested_logit_normal_rates,
    coerce_categories,
)

logger = logging.getLogger(__name__)

Parameterization = Literal["centered", "non_centered", "marginalized"]

# Canonical category names in public.calculated_features_registry whose
# feature_aliases drive the auxiliary "dividend_feature" dim. Aligned with
# the dividend / cash-flow features feeding the cut-probability model.
_DIVIDEND_CATEGORY_KEYS: tuple[str, ...] = (
    "Dividend Reliability",
    "Cash Flow",
    "Growth Metrics",
    "Efficiency Ratios",
    "Leverage & Liquidity",
    "Earnings Quality",
    "Profitability",
    "Valuation Ratios",
    "Revenue Forecasting",
    "Interest Income",
)


class DividendSafetyBayesian:
    """Bayesian dividend cut probability model.

    Parameters
    ----------
    prior_alpha : float
        Beta prior alpha for cut probability.
    prior_beta : float
        Beta prior beta for cut probability.
    high_payout_threshold : float
        Payout ratio above which risk adjustment is applied.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 5.0,
        high_payout_threshold: float = 0.9,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.high_payout_threshold = high_payout_threshold
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[InferenceLike] = None

    @staticmethod
    @lru_cache(maxsize=4)
    def _resolve_dividend_feature_aliases(
        connection_string: Optional[str] = None,
    ) -> tuple[str, ...]:
        """Resolve the dividend feature aliases from calculated_features_registry.

        These serve as labelled coordinates along the ``dividend_feature`` dim
        for the auxiliary ``pm.Data`` container of observed dividend-feature
        values (dividend + cash-flow signals).
        """
        try:
            categories = load_feature_categories_from_db(connection_string)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load feature categories: %s", exc)
            return tuple()
        aliases: list[str] = []
        seen: set[str] = set()
        for key in _DIVIDEND_CATEGORY_KEYS:
            for alias in categories.get(key, []):
                if alias not in seen:
                    seen.add(alias)
                    aliases.append(alias)
        return tuple(aliases)

    @staticmethod
    def _align_dividend_features(
        dividend_features_df: Optional[pd.DataFrame],
        isin: np.ndarray,
        feature_aliases: list[str],
        *,
        use_typed_coercion: bool = False,
        connection_string: Optional[str] = None,
    ) -> np.ndarray:
        """Align an (isin × dividend_feature) matrix to the model dims.

        Missing columns/rows are filled with ``0.0`` so the container always
        has shape ``(n_isin, n_dividend_feature)``.
        """
        n_isin = len(isin)
        n_feat = len(feature_aliases)
        if dividend_features_df is None or n_feat == 0 or n_isin == 0:
            return np.zeros((n_isin, max(n_feat, 0)), dtype="float64")

        df = dividend_features_df.copy()
        if "isin" in df.columns:
            df = df.drop_duplicates(subset="isin").set_index("isin")
        df = df.reindex(index=isin)

        if use_typed_coercion:
            metadata = load_feature_metadata_from_db(connection_string)
            return coerce_by_data_type(df, list(feature_aliases), metadata)
        return df.reindex(columns=feature_aliases).astype("float64").fillna(0.0).to_numpy()

    def fit(
        self,
        payout_ratios: np.ndarray,
        fcf_coverage: np.ndarray,
        isins: np.ndarray,
        sectors: Optional[np.ndarray] = None,
        categories_df: Optional[pd.DataFrame] = None,
        hierarchy_levels: Optional[list[str]] = None,
        dividend_features_df: Optional[pd.DataFrame] = None,
        connection_string: Optional[str] = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
        parameterization: Parameterization = "non_centered",
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> tuple[InferenceLike, pm_typing.Model]:
        """Fit dividend safety model and return ``(InferenceData, Model)``.

        Parameters
        ----------
        parameterization : {"centered", "non_centered", "marginalized"}
            Hierarchical reparameterization strategy for ``cut_prob``.
            ``"non_centered"`` (default) uses a logit-Normal latent that
            NUTS handles efficiently. ``"marginalized"`` collapses the
            per-ISIN latent to the Beta prior mean (fastest; no per-stock
            latent posterior). ``"centered"`` preserves the legacy Beta
            hierarchy for reproducibility.
        """
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install pymc + arviz to use DividendSafetyBayesian."
            )

        payout_ratios = np.asarray(payout_ratios, dtype="float64")
        fcf_coverage = np.asarray(fcf_coverage, dtype="float64")
        isins_arr = np.asarray(isins)

        if payout_ratios.size == 0:
            raise ValueError("DividendSafetyBayesian.fit received empty payout_ratios.")
        if payout_ratios.shape != fcf_coverage.shape:
            raise ValueError("payout_ratios and fcf_coverage must share shape.")

        # --- DB-aligned coords --------------------------------------------------
        # `isin` mirrors public.vw_identifier_columns.isin (role='id').
        coords: dict[str, Any] = {"isin": isins_arr}

        cats_df, levels = coerce_categories(
            isins_arr,
            sectors=sectors,
            categories_df=categories_df,
            hierarchy_levels=hierarchy_levels
            or (["region", "country", "sector", "industry"] if categories_df is not None else None),
        )
        hierarchical = cats_df is not None and levels
        hierarchy_meta: Optional[dict] = None
        if hierarchical:
            hierarchy_meta = build_hierarchy_indices(cats_df, isins_arr, levels=levels)
            for lv, meta in hierarchy_meta.items():
                coords[lv] = meta["labels"]

        # `dividend_feature` is resolved from calculated_features_registry so the
        # auxiliary pm.Data container carries human-readable feature_alias labels.
        dividend_feature_aliases = list(self._resolve_dividend_feature_aliases(connection_string))
        coords["dividend_feature"] = list(dividend_feature_aliases)

        dividend_features_arr = self._align_dividend_features(
            dividend_features_df, isins_arr, dividend_feature_aliases
        )

        with pm.Model(coords=coords) as model:
            payout_data = pm.Data("payout_data", payout_ratios, dims="isin")
            fcf_data = pm.Data("fcf_coverage_data", fcf_coverage, dims="isin")
            pm.Data(
                "dividend_features",
                dividend_features_arr,
                dims=("isin", "dividend_feature"),
            )

            if parameterization == "centered":
                cut_prob = pm.Beta(
                    "cut_prob",
                    alpha=self.prior_alpha,
                    beta=self.prior_beta,
                    dims="isin",
                )
            elif parameterization == "marginalized":
                # Collapse per-ISIN latent: use Beta prior mean directly.
                prior_mean = float(self.prior_alpha / (self.prior_alpha + self.prior_beta))
                cut_prob = pm.Deterministic(
                    "cut_prob",
                    pt.ones_like(payout_data) * prior_mean,
                    dims="isin",
                )
            else:
                if hierarchical:
                    nested = build_nested_logit_normal_rates(
                        hierarchy_meta,
                        leaf_dim="isin",
                        name="cut_rate",
                    )
                    cut_prob = pm.Deterministic(
                        "cut_prob",
                        nested["leaf_rate"],
                        dims="isin",
                    )
                else:
                    mu_logit = pm.Normal("mu_logit", 0.0, 1.5)
                    sigma_logit = pm.HalfNormal("sigma_logit", 1.0)
                    z_isin = pm.Normal("z_isin", 0.0, 1.0, dims="isin")
                    cut_prob = pm.Deterministic(
                        "cut_prob",
                        pm.math.sigmoid(mu_logit + sigma_logit * z_isin),
                        dims="isin",
                    )

            risk_adj = pm.Deterministic(
                "risk_adj",
                pt.clip(
                    pt.switch(
                        pt.gt(payout_data, self.high_payout_threshold),
                        cut_prob * 1.3,
                        cut_prob,
                    ),
                    0.0,
                    1.0,
                ),
                dims="isin",
            )

            expected_coverage = pm.Deterministic(
                "expected_coverage",
                pt.clip(1.0 / (risk_adj + 0.01), 0.0, 20.0),
                dims="isin",
            )

            sigma = pm.HalfNormal("sigma", sigma=1.0)

            pm.Normal(
                "fcf_coverage_obs",
                mu=expected_coverage,
                sigma=sigma,
                observed=fcf_data,
                dims="isin",
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

            idata = pm.sample(**scall)

        # Recommendation §12.3 #3 — stamp feature_catalogue provenance.
        try:
            stamp_feature_provenance(
                idata,
                "dividend_features",
                dividend_feature_aliases,
                load_feature_metadata_from_db(connection_string),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("DividendSafety provenance stamping failed: %s", exc)

        self.model_ = model
        self.idata_ = idata
        return idata, model
