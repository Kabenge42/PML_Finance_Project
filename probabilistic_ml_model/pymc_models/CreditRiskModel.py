"""
Credit Risk Bayesian Model — Multi-signal distress estimation.

Uses PyMC with Altman Z-score zones, debt-to-equity signals, and optional
sector-level hierarchical shrinkage (non-centred logit-Normal by default).

Reference: CreditRiskProbabilityModel in probability_models.py (line 2456);
           build_credit_risk_inference_data() in inference_schema.py (line 591).
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

# Canonical category names in public.calculated_features_registry whose
# feature_aliases drive the auxiliary "credit_feature" dim. Aligned with
# the credit-risk / quality / leverage features feeding the distress model.
_CREDIT_CATEGORY_KEYS: tuple[str, ...] = (
    "Financial Distress",
    "Quality & Risk",
    "Balance Sheet",
    "Cash Flow",
    "Leverage & Liquidity",
)

Parameterization = Literal["centered", "non_centered", "marginalized"]


class CreditRiskBayesian:
    """Bayesian credit risk / distress probability model.

    Parameters
    ----------
    prior_alpha : float
        Beta prior alpha for distress probability.
    prior_beta : float
        Beta prior beta for distress probability.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 3.0,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[az_typing.InferenceData] = None

    @staticmethod
    @lru_cache(maxsize=4)
    def _resolve_credit_feature_aliases(
        connection_string: Optional[str] = None,
    ) -> tuple[str, ...]:
        """Resolve the credit-risk feature aliases from calculated_features_registry.

        These serve as labelled coordinates along the ``credit_feature`` dim
        for the auxiliary ``pm.Data`` container of observed credit-feature
        values (credit-risk + quality + leverage signals).
        """
        try:
            categories = load_feature_categories_from_db(connection_string)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load feature categories: %s", exc)
            return tuple()
        aliases: list[str] = []
        seen: set[str] = set()
        for key in _CREDIT_CATEGORY_KEYS:
            for alias in categories.get(key, []):
                if alias not in seen:
                    seen.add(alias)
                    aliases.append(alias)
        return tuple(aliases)

    @staticmethod
    def _align_credit_features(
        credit_features_df: Optional[pd.DataFrame],
        isin: np.ndarray,
        feature_aliases: list[str],
        *,
        use_typed_coercion: bool = False,
        connection_string: Optional[str] = None,
    ) -> np.ndarray:
        """Align an (isin × credit_feature) matrix to the model dims.

        Missing columns/rows are filled with ``0.0`` so the container always
        has shape ``(n_isin, n_credit_feature)``.
        """
        n_isin = len(isin)
        n_feat = len(feature_aliases)
        if credit_features_df is None or n_feat == 0 or n_isin == 0:
            return np.zeros((n_isin, max(n_feat, 0)), dtype="float64")

        df = credit_features_df.copy()
        if "isin" in df.columns:
            df = df.drop_duplicates(subset="isin").set_index("isin")
        df = df.reindex(index=isin)

        if use_typed_coercion:
            metadata = load_feature_metadata_from_db(connection_string)
            return coerce_by_data_type(df, list(feature_aliases), metadata)
        return df.reindex(columns=feature_aliases).astype("float64").fillna(0.0).to_numpy()

    def fit(
        self,
        z_scores: np.ndarray,
        debt_to_equity: np.ndarray,
        isins: np.ndarray,
        sectors: Optional[np.ndarray] = None,
        categories_df: Optional[pd.DataFrame] = None,
        hierarchy_levels: Optional[list[str]] = None,
        distress_observed: Optional[np.ndarray] = None,
        credit_features_df: Optional[pd.DataFrame] = None,
        connection_string: Optional[str] = None,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
        parameterization: Parameterization = "non_centered",
        nuts_sampler: Optional[str] = None,
        **sample_kwargs: Any,
    ) -> tuple[az_typing.InferenceData, pm_typing.Model]:
        """Fit credit-risk model and return ``(InferenceData, Model)``."""
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install pymc + arviz to use CreditRiskBayesian."
            )

        z_scores = np.asarray(z_scores, dtype="float64")
        debt_to_equity = np.asarray(debt_to_equity, dtype="float64")
        isins_arr = np.asarray(isins)

        if z_scores.size == 0:
            raise ValueError("CreditRiskBayesian.fit received empty z_scores.")
        if z_scores.shape != debt_to_equity.shape:
            raise ValueError("z_scores and debt_to_equity must share shape.")

        # --- DB-aligned coords --------------------------------------------------
        # `isin` mirrors public.vw_identifier_columns.isin (role='id')
        # `sector` mirrors public.vw_identifier_columns.sector (role='categorical')
        coords: dict[str, Any] = {"isin": isins_arr}
        cats_df, levels = coerce_categories(
            isins_arr,
            sectors=sectors,
            categories_df=categories_df,
            hierarchy_levels=hierarchy_levels
            or (
                ["region", "country", "exchange", "sector", "industry"]
                if categories_df is not None
                else None
            ),
        )
        hierarchical = cats_df is not None and levels
        hierarchy_meta: Optional[dict] = None
        if hierarchical:
            hierarchy_meta = build_hierarchy_indices(cats_df, isins_arr, levels=levels)
            for lv, meta in hierarchy_meta.items():
                coords[lv] = meta["labels"]

        # `credit_feature` is resolved from calculated_features_registry so the
        # auxiliary pm.Data container carries human-readable feature_alias labels.
        credit_feature_aliases = list(self._resolve_credit_feature_aliases(connection_string))
        coords["credit_feature"] = list(credit_feature_aliases)

        credit_features_arr = self._align_credit_features(
            credit_features_df, isins_arr, credit_feature_aliases
        )

        # Precompute the deterministic Altman-zone adjustment in NumPy so
        # PyTensor sees it as a constant data vector instead of a nested
        # `pt.switch` Composite.  The nested switch was the main blocker
        # of Elemwise loop-fusion and triggered:
        #   "Loop fusion failed because the resulting node would exceed
        #    the kernel argument limit."
        zone_adj_np = np.where(
            z_scores < 1.81,
            0.75,
            np.where(z_scores < 2.67, 0.35, 0.15),
        ).astype("float64")

        with pm.Model(coords=coords) as model:
            z_data = pm.Data("z_score_data", z_scores, dims="isin")
            de_data = pm.Data("de_data", debt_to_equity, dims="isin")
            pm.Data(
                "credit_features",
                credit_features_arr,
                dims=("isin", "credit_feature"),
            )
            # `zone_adj` is fully determined by z_scores (data, not an RV),
            # so we expose it as a pm.Data container — keeps it visible in
            # idata.constant_data and lets pm.set_data() swap it together
            # with z_score_data for out-of-sample prediction.
            zone_adj_data = pm.Data("zone_adj", zone_adj_np, dims="isin")

            if hierarchical:
                leaf_level = list(hierarchy_meta.keys())[-1]
                if parameterization == "centered":
                    leaf_idx_arr = hierarchy_meta[leaf_level]["idx"]
                    leaf_idx_data = pm.Data(f"{leaf_level}_idx", leaf_idx_arr, dims="isin")
                    sector_rate = pm.Beta(
                        "sector_rate",
                        alpha=self.prior_alpha,
                        beta=self.prior_beta,
                        dims=leaf_level,
                    )
                    kappa = pm.HalfNormal("kappa", sigma=5.0)
                    a = sector_rate[leaf_idx_data] * kappa
                    b = (1.0 - sector_rate[leaf_idx_data]) * kappa
                    distress_prob = pm.Beta("distress_prob", alpha=a, beta=b, dims="isin")
                else:
                    # Nested non-centred logit-Normal hierarchy across all
                    # supplied levels (region → country → exchange → sector
                    # → industry by default for CreditRisk).
                    nested = build_nested_logit_normal_rates(
                        hierarchy_meta,
                        leaf_dim="isin",
                        name="distress_rate",
                    )
                    leaf_rate = nested["leaf_rate"]
                    if parameterization == "marginalized":
                        distress_prob = pm.Deterministic(
                            "distress_prob",
                            leaf_rate,
                            dims="isin",
                        )
                    else:
                        kappa = pm.Gamma("kappa", alpha=2.0, beta=0.1)
                        a = leaf_rate * kappa
                        b = (1.0 - leaf_rate) * kappa
                        distress_prob = pm.Beta("distress_prob", alpha=a, beta=b, dims="isin")
            else:
                if parameterization == "centered":
                    distress_prob = pm.Beta(
                        "distress_prob",
                        alpha=self.prior_alpha,
                        beta=self.prior_beta,
                        dims="isin",
                    )
                elif parameterization == "marginalized":
                    # Collapse latent: use prior mean directly.
                    prior_mean = float(self.prior_alpha / (self.prior_alpha + self.prior_beta))
                    distress_prob = pm.Deterministic(
                        "distress_prob",
                        pt.ones_like(z_data) * prior_mean,
                        dims="isin",
                    )
                else:
                    # Non-centred logit-Normal: better NUTS geometry than Beta.
                    mu_logit = pm.Normal("mu_logit", 0.0, 1.5)
                    sigma_logit = pm.HalfNormal("sigma_logit", 1.0)
                    z_isin = pm.Normal("z_isin", 0.0, 1.0, dims="isin")
                    distress_prob = pm.Deterministic(
                        "distress_prob",
                        pm.math.sigmoid(mu_logit + sigma_logit * z_isin),
                        dims="isin",
                    )

            # `zone_adj` previously used a nested `pt.switch` — replaced by
            # the precomputed `zone_adj_data` container above.
            zone_adj = pm.Deterministic("zone_adj_det", zone_adj_data, dims="isin")

            # Replace the per-ISIN `debt_trend` latent with a single
            # shared multiplicative noise scalar.  The original
            # `pm.Normal("debt_trend", mu=de_data, sigma=0.1, dims="isin")`
            # added n_isin extra latents whose only role was to smear
            # `de_data` by sigma=0.1 — that smearing now lives in the
            # observation noise and a small global slope, which keeps
            # the model identifiable but stops blowing up the fused
            # Elemwise kernel.
            debt_slope = pm.Normal("debt_slope", mu=0.05, sigma=0.01)
            debt_trend = pm.Deterministic("debt_trend", debt_slope * de_data, dims="isin")

            expected_distress = pm.Deterministic(
                "expected_distress",
                pt.clip(distress_prob * zone_adj + debt_trend, 0.0, 1.0),
                dims="isin",
            )

            if distress_observed is not None:
                obs_target = np.asarray(distress_observed, dtype="float64")
            else:
                obs_target = np.clip(1.0 - (z_scores - 1.0) / 3.0, 0.0, 1.0)
            obs_data = pm.Data("distress_target", obs_target, dims="isin")

            pm.Normal(
                "distress_obs",
                mu=expected_distress,
                sigma=0.1,
                observed=obs_data,
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

            # nutpie ignores idata_kwargs and emits a UserWarning; strip it
            # to keep logs clean while preserving behaviour for other samplers.
            if scall.get("nuts_sampler") == "nutpie":
                scall.pop("idata_kwargs", None)

            idata = pm.sample(**scall)

        # Recommendation §12.3 #3 — stamp feature_catalogue provenance.
        try:
            stamp_feature_provenance(
                idata,
                "credit_features",
                credit_feature_aliases,
                load_feature_metadata_from_db(connection_string),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("CreditRisk provenance stamping failed: %s", exc)

        self.model_ = model
        self.idata_ = idata
        return idata, model
