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
from probabilistic_ml_model.data_utils.data_utils import load_feature_categories_from_db

logger = logging.getLogger(__name__)

# Canonical category names in public.calculated_features_registry whose
# feature_aliases drive the auxiliary "credit_feature" dim. Aligned with
# the credit-risk / quality / leverage features feeding the distress model.
_CREDIT_CATEGORY_KEYS: tuple[str, ...] = (
    "Credit Risk",
    "Quality & Risk",
    "Leverage & Liquidity",
)

Parameterization = Literal["centered", "non_centered"]


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

        aligned = df.reindex(index=isin, columns=feature_aliases)
        return aligned.astype("float64").fillna(0.0).to_numpy()

    def fit(
        self,
        z_scores: np.ndarray,
        debt_to_equity: np.ndarray,
        tickers: np.ndarray,
        sectors: Optional[np.ndarray] = None,
        distress_observed: Optional[np.ndarray] = None,
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
        tickers_arr = np.asarray(tickers)

        if z_scores.size == 0:
            raise ValueError("CreditRiskBayesian.fit received empty z_scores.")
        if z_scores.shape != debt_to_equity.shape:
            raise ValueError("z_scores and debt_to_equity must share shape.")

        coords: dict[str, Any] = {"ticker": tickers_arr}
        hierarchical = sectors is not None
        sector_idx_arr: Optional[np.ndarray] = None
        if hierarchical:
            sectors_arr = np.asarray(sectors)
            unique_sectors, sector_idx_arr = np.unique(sectors_arr, return_inverse=True)
            sector_idx_arr = sector_idx_arr.astype("int32")
            coords["sector"] = unique_sectors

        with pm.Model(coords=coords) as model:
            z_data = pm.Data("z_score_data", z_scores, dims="ticker")
            de_data = pm.Data("de_data", debt_to_equity, dims="ticker")

            if hierarchical:
                sector_idx_data = pm.Data("sector_idx", sector_idx_arr, dims="ticker")
                if parameterization == "centered":
                    sector_rate = pm.Beta(
                        "sector_rate",
                        alpha=self.prior_alpha,
                        beta=self.prior_beta,
                        dims="sector",
                    )
                    kappa = pm.HalfNormal("kappa", sigma=5.0)
                else:
                    mu_logit = pm.Normal("mu_logit", 0.0, 1.5)
                    sigma_sector = pm.HalfNormal("sigma_sector", 1.0)
                    z_sector = pm.Normal("z_sector", 0.0, 1.0, dims="sector")
                    sector_rate = pm.Deterministic(
                        "sector_rate",
                        pm.math.sigmoid(mu_logit + sigma_sector * z_sector),
                        dims="sector",
                    )
                    kappa = pm.Gamma("kappa", alpha=2.0, beta=0.1)
                a = sector_rate[sector_idx_data] * kappa
                b = (1.0 - sector_rate[sector_idx_data]) * kappa
                distress_prob = pm.Beta("distress_prob", alpha=a, beta=b, dims="ticker")
            else:
                distress_prob = pm.Beta(
                    "distress_prob",
                    alpha=self.prior_alpha,
                    beta=self.prior_beta,
                    dims="ticker",
                )

            zone_adj = pm.Deterministic(
                "zone_adj",
                pt.switch(
                    pt.lt(z_data, 1.81),
                    0.75,
                    pt.switch(pt.lt(z_data, 2.67), 0.35, 0.15),
                ),
                dims="ticker",
            )
            debt_trend = pm.Normal("debt_trend", mu=de_data, sigma=0.1, dims="ticker")
            expected_distress = pm.Deterministic(
                "expected_distress",
                pt.clip(distress_prob * zone_adj + 0.05 * debt_trend, 0.0, 1.0),
                dims="ticker",
            )

            if distress_observed is not None:
                obs_target = np.asarray(distress_observed, dtype="float64")
            else:
                obs_target = np.clip(1.0 - (z_scores - 1.0) / 3.0, 0.0, 1.0)
            obs_data = pm.Data("distress_target", obs_target, dims="ticker")

            pm.Normal(
                "distress_obs",
                mu=expected_distress,
                sigma=0.1,
                observed=obs_data,
                dims="ticker",
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

        self.model_ = model
        self.idata_ = idata
        return idata, model
