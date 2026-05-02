"""
Price Target Achievement Model — Beta-Normal hybrid.

Models the probability of achieving analyst price targets using a Beta prior
on achievement probability and a Normal model for expected returns with
risk adjustment.

Reference: PriceTargetAchievementModel in probability_models.py (line 3057);
           _resolve_price_target_inputs() in inference_schema.py (line 746).
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
# feature_aliases drive the auxiliary "pt_feature" dim. Aligned with the
# categories backing public.vw_features_analyst_sentiment and the
# price-target dynamics calculator.
_PT_CATEGORY_KEYS: tuple[str, ...] = (
    "Price Target Dynamics",
    "Analyst Sentiment",
    "Technical Analysis",
    "Composite Scores",
)


class PriceTargetAchievement:
    """Bayesian price target achievement model.

    Parameters
    ----------
    prior_alpha : float
        Beta prior alpha for achievement probability.
    prior_beta : float
        Beta prior beta for achievement probability.
    risk_penalty : float
        Exponential risk penalty factor applied to expected return.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        risk_penalty: float = 0.1,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.risk_penalty = risk_penalty
        self.model_: Optional[pm_typing.Model] = None
        self.idata_: Optional[InferenceLike] = None

    @staticmethod
    @lru_cache(maxsize=4)
    def _resolve_pt_feature_aliases(
        connection_string: Optional[str] = None,
    ) -> tuple[str, ...]:
        """Resolve the 'Price Target' feature aliases from calculated_features_registry.

        These serve as labelled coordinates along the ``pt_feature`` dim
        for the auxiliary ``pm.Data`` container of observed pt-feature
        values (e.g. 'upside_potential', 'price_target_spread_pct',
        'price_target_revision_1m', 'pe_forward_discount').

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
        for key in _PT_CATEGORY_KEYS:
            for alias in categories.get(key, []):
                if alias not in seen:
                    seen.add(alias)
                    aliases.append(alias)
        return tuple(aliases)

    @staticmethod
    def _align_pt_features(
        pt_features_df: Optional[pd.DataFrame],
        isin: np.ndarray,
        feature_aliases: list[str],
        *,
        use_typed_coercion: bool = False,
        connection_string: Optional[str] = None,
    ) -> np.ndarray:
        """Align an (isin × pt_feature) matrix to the model dims.

        Missing columns/rows are filled with ``0.0`` so the container always
        has shape ``(n_isin, n_pt_feature)``. When ``use_typed_coercion=True``
        the per-column dtype/clipping is driven by ``feature_catalogue.data_type``
        via :func:`coerce_by_data_type` (recommendation §12.3 #1).
        """
        n_isin = len(isin)
        n_feat = len(feature_aliases)
        if pt_features_df is None or n_feat == 0 or n_isin == 0:
            return np.zeros((n_isin, max(n_feat, 0)), dtype="float64")

        df = pt_features_df.copy()
        if "isin" in df.columns:
            df = df.drop_duplicates(subset="isin").set_index("isin")
        df = df.reindex(index=isin)

        if use_typed_coercion:
            metadata = load_feature_metadata_from_db(connection_string)
            return coerce_by_data_type(df, list(feature_aliases), metadata)
        return df.reindex(columns=feature_aliases).astype("float64").fillna(0.0).to_numpy()

    def fit(
        self,
        upside_potential: np.ndarray,
        analyst_conviction: np.ndarray,
        isins: np.ndarray,
        sectors: Optional[np.ndarray] = None,
        categories_df: Optional[pd.DataFrame] = None,
        hierarchy_levels: Optional[list[str]] = None,
        pt_features_df: Optional[pd.DataFrame] = None,
        connection_string: Optional[str] = None,
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
        """Fit price target achievement model and return ``(InferenceData, Model)``.

        Notes
        -----
        ``upside_potential`` is observed *once* via ``upside_obs``; the
        latent ``expected_return`` has a prior centred at 0 with a
        dispersion-aware sigma to avoid the previous degenerate likelihood
        where the same series acted as both prior mean and observation.
        """
        if pm is None:
            raise ImportError(
                "PyMC is not available. Install pymc + arviz to use PriceTargetAchievement."
            )

        upside_potential = np.asarray(upside_potential, dtype="float64")
        analyst_conviction = np.asarray(analyst_conviction, dtype="float64")
        isins_arr = np.asarray(isins)
        if upside_potential.size == 0:
            raise ValueError("PriceTargetAchievement.fit received empty upside_potential.")
        if upside_potential.shape != analyst_conviction.shape:
            raise ValueError("upside_potential and analyst_conviction must share shape.")

        # --- DB-aligned coords --------------------------------------------------
        # `isin` mirrors public.vw_identifier_columns.isin (role='id').
        coords: dict[str, Any] = {"isin": isins_arr}

        cats_df, levels = coerce_categories(
            isins_arr,
            sectors=sectors,
            categories_df=categories_df,
            hierarchy_levels=hierarchy_levels
            or (
                ["exchange", "sector", "industry", "size_class"]
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

        # `pt_feature` is resolved from calculated_features_registry so the
        # auxiliary pm.Data container carries human-readable feature_alias labels.
        pt_feature_aliases = list(self._resolve_pt_feature_aliases(connection_string))
        coords["pt_feature"] = list(pt_feature_aliases)

        pt_features_arr = self._align_pt_features(pt_features_df, isins_arr, pt_feature_aliases)

        with pm.Model(coords=coords) as model:
            cu_data = pm.Data("upside_potential", upside_potential, dims="isin")
            ad_data = pm.Data("analyst_conviction", analyst_conviction, dims="isin")
            pm.Data(
                "pt_features",
                pt_features_arr,
                dims=("isin", "pt_feature"),
            )

            if parameterization == "centered":
                pm.Beta(
                    "achieve_prob",
                    alpha=self.prior_alpha,
                    beta=self.prior_beta,
                    dims="isin",
                )
            elif parameterization == "marginalized":
                # Collapse per-ISIN latent: use Beta prior mean directly.
                prior_mean = float(self.prior_alpha / (self.prior_alpha + self.prior_beta))
                pm.Deterministic(
                    "achieve_prob",
                    pt.ones_like(cu_data) * prior_mean,
                    dims="isin",
                )
            else:
                if hierarchical:
                    nested = build_nested_logit_normal_rates(
                        hierarchy_meta,
                        leaf_dim="isin",
                        name="achieve_rate",
                    )
                    pm.Deterministic(
                        "achieve_prob",
                        nested["leaf_rate"],
                        dims="isin",
                    )
                else:
                    mu_logit = pm.Normal("mu_logit", 0.0, 1.5)
                    sigma_logit = pm.HalfNormal("sigma_logit", 1.0)
                    z_isin = pm.Normal("z_isin", 0.0, 1.0, dims="isin")
                    pm.Deterministic(
                        "achieve_prob",
                        pm.math.sigmoid(mu_logit + sigma_logit * z_isin),
                        dims="isin",
                    )

            expected_return = pm.Normal(
                "expected_return",
                mu=0.0,
                sigma=pt.maximum(ad_data, 1e-3),
                dims="isin",
            )

            risk_adj_return = pm.Deterministic(
                "risk_adj_return",
                expected_return * pt.exp(-self.risk_penalty * ad_data),
                dims="isin",
            )

            sigma = pm.HalfNormal("sigma", sigma=0.5)

            pm.Normal(
                "upside_obs",
                mu=risk_adj_return,
                sigma=sigma,
                observed=cu_data,
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

            idata = pm.sample(**scall)

        # Recommendation §12.3 #3 — stamp feature_catalogue provenance.
        try:
            stamp_feature_provenance(
                idata,
                "pt_features",
                pt_feature_aliases,
                load_feature_metadata_from_db(connection_string),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("PriceTarget provenance stamping failed: %s", exc)

        self.model_ = model
        self.idata_ = idata
        return idata, model
