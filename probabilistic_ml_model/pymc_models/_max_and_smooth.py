"""Max-and-Smooth: a fast screening backend for v2 model-comparison arms.

Hrafnkelsson, Siegert, Huser, Bakka and Jóhannesson, *Max-and-Smooth: A Two-Step
Approach for Approximate Bayesian Inference in Latent Gaussian Models*, Bayesian
Analysis 16(2), 611-638 (2021). See ``docs/``.

Why this exists, and what it is NOT
-----------------------------------
The exact comparison harness (``pymc_kalman_filter_pt_v2.run_model_comparison``)
refits the production model once per arm and attaches a pointwise
``log_likelihood``, so an N-arm contrast costs roughly N production runs. That is
why ``level_off`` and ``hierarchy_fine`` were built and then sat unrun. This
module makes a *screening* pass cheap enough that arms can be explored, and
leaves the deciding to the exact harness.

**It does not replace the production fit.** v2 already does the thing that makes
Max-and-Smooth fast: ``build_kalman_pt_model_v2`` integrates the per-ISIN level
and OU state out in closed form (32,503 latents -> ~60 free parameters, ~1.6 s ->
2.3 ms per gradient). That IS the Gaussian-Gaussian conjugacy the paper exploits,
already applied. A full two-step restructure would also fight the model: the
paper's Step 1 assumes each group's likelihood is free of hyperparameters, while
here the hyperparameters being learned (``ell``, the variance simplex, ``tau``)
live *inside* the within-name covariance.

What is genuinely cheap is the other direction. An arm that changes only the
*latent* structure is a pure Step-2 change, so it can be scored against per-ISIN
pseudo-observations computed once.

The partition, and why ``level_off`` is screenable
--------------------------------------------------
The model's generative form for name ``i`` at lookback ``t`` is::

    y[i, t] = m_i + alpha_time[t] + tau[t] * L_i + tau[t] * (s[i, t] + e[i, t])

with ``m_i = signal_scale_i * mu_reg_i``, ``Var(L_i) = sigma_i^2 * w_L`` (the
permanent per-name level) and the state/noise pair contributing
``sigma_i^2 * (w_S * K(ell) + w_O * I)``. Splitting there::

    data level   (Max)     A_obs = (w_S*K + w_O*I) scaled by tau   -- sampling noise
    latent level (Smooth)  w_L, the drift betas, the group effects

Putting ``w_L`` on the **latent** side is what makes ``level_off`` a
one-thing-different arm: the Smooth step represents it as a free scale that the
arm pins to zero, rather than as a different data-level covariance that would
invalidate the Max step. ``hierarchy_fine`` is latent-side by construction.

Step 1 (Max) then gives, per name, a GLS estimate of ``m_i`` using ``A_obs``
only. Because the level is left out of the weights it leaks into the estimate
with a *known* loading::

    eta_hat_i = (1' Ao^-1 (y_i - alpha)) / (1' Ao^-1 1)
    E[eta_hat_i] = m_i + c * L_i,        c = (1' Ao^-1 tau) / (1' Ao^-1 1)
    Var(eta_hat_i | m_i) = sigma_i^2 / (1' Ao^-1 1)  +  c^2 * sigma_i^2 * w_L

so ``eta_hat_i`` is a pseudo-observation of ``m_i`` whose variance separates
cleanly into a sampling part (fixed, carried into Step 2 as known, exactly as the
paper prescribes) and a level part the Smooth step re-estimates.

Two properties worth being explicit about, because a T=4 panel is inside the
regime the paper warns about (it recommends >= 10-20 replicates per group):

* For the **Normal** likelihood the data level is linear-Gaussian, so this Max
  step is **exact**, not approximate. The whole approximation is the freezing of
  the covariance hyperparameters at their baseline posterior means.
* For **Student-t** the paper's *second* Gaussian approximation -- the mean and
  covariance of the normalised likelihood rather than the ML estimate and
  observed information -- has a closed form for a multivariate t: the same GLS
  mean, with the variance inflated by ``nu / (nu - 2)``. So the variant the paper
  reserves for skewed likelihoods costs nothing here and needs no integration.

Cost
----
``sigma_i^2`` factors out of the GLS weights, so ``1' Ao^-1 1`` and ``c`` depend
only on a name's **missingness pattern**. The Max step is a handful of ``T x T``
solves, not 6,500 of them, and the Smooth step is a linear-Gaussian regression
with ~20 free parameters (~175 under ``GROUP_EFFECTS_FINE``) over ~6.5k rows.

Reading a result
----------------
The contrast is on the *pseudo-observations*, not on the exact ELPD. Use it to
pick the candidate; confirm with ``run_model_comparison`` before promoting
anything. ``compare_arms_fast`` refuses arms whose config delta touches a
covariance field, because the Max step conditions on exactly those.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

try:
    import pymc as pm
except ImportError:  # pragma: no cover - optional dependency
    pm = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    import pymc as pm_typing

from probabilistic_ml_model._pymc_arviz_compat import InferenceLike
from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
    GROUP_EFFECT_SCALE,
    KalmanModelConfig,
    KalmanPanelV2,
    build_group_effect_terms,
)
from probabilistic_ml_model.pymc_models._hierarchy import (
    build_hierarchy_indices as _build_hierarchy_indices,
    order_levels as _order_levels,
)

logger = logging.getLogger(__name__)

#: Config fields the Max step CONDITIONS ON. An arm that changes any of them
#: changes the data-level covariance the pseudo-observations were built from, so
#: screening it here would contrast two arms against one arm's noise model.
#:
#: This is a guard, not a formality: ``level_off`` looks like a covariance change
#: (it drops a variance leg) and is admissible only because the partition above
#: deliberately assigns ``w_L`` to the latent side. Nothing else gets that
#: treatment by accident.
COVARIANCE_FIELDS: frozenset[str] = frozenset(
    {
        "lookbacks",
        "lookback_days",
        "time_scale_applies_to",
        "enable_time_scale",
        "time_scale_prior_sigma",
        "rho_scale_buckets",
        "rho_scale_slope_prior",
        "enable_ou_state",
        "coverage_scale_per_cell",
        "coverage_profile_buckets",
        "likelihood",
        "nu_floor",
        "sigma_n_exponent_prior",
        "enable_sector_scale_offset",
        "variance_split_alpha",
        "ou_length_scale_days_mu",
        "ou_length_scale_days_sigma",
        "log_sigma_clip",
        "log_sigma_total_mu",
        "log_sigma_total_sigma",
        "enable_signal_scaling",
        "signal_exponent_prior",
    }
)

__all__ = [
    "COVARIANCE_FIELDS",
    "PseudoObservations",
    "assert_arm_is_screenable",
    "build_pseudo_model",
    "gaussian_likelihood_approximation",
]


@dataclass(frozen=True)
class PseudoObservations:
    """Step-1 output: one Gaussian pseudo-observation of ``mu_reg`` per name.

    Attributes
    ----------
    isins
        Identifiers, shape ``(n,)``.
    m_hat
        Pseudo-observation of ``mu_reg``, shape ``(n,)``.
    var_obs
        Its SAMPLING variance, shape ``(n,)``. Treated as known in Step 2 —
        this is the paper's "estimated in the first step, exact in the second".
    level_scale
        Per-name multiplier of the latent level scale, shape ``(n,)``. The Smooth
        step's residual sd is ``sqrt(var_obs + (kappa * level_scale) ** 2)``, so
        ``kappa`` plays the role of ``sqrt(w_L)`` and ``kappa = 0`` is
        ``enable_isin_level = False``.
    n_obs
        Observed cells per name, shape ``(n,)``. A name with one observation is
        carried but its ``var_obs`` is large; a name with none is dropped.
    X_drift, drift_names, coord_idx, coord_uniques, coord_parent_of
        Carried from the panel so Step 2 needs nothing else.
    diagnostics
        What the Max step conditioned on, for the record.
    """

    isins: np.ndarray
    m_hat: np.ndarray
    var_obs: np.ndarray
    level_scale: np.ndarray
    n_obs: np.ndarray
    X_drift: np.ndarray
    drift_names: list[str]
    coord_idx: dict[str, np.ndarray]
    coord_uniques: dict[str, np.ndarray]
    #: ``level -> (n_level,) int`` child-index to parent-index map, for the
    #: nested arms. Indexed by LEVEL, so unlike ``coord_idx`` it is NOT sliced
    #: to the kept names -- ``coord_uniques`` is not sliced either, and the two
    #: index spaces have to agree.
    coord_parent_of: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.isins)


def _posterior_mean(idata: InferenceLike, name: str) -> Optional[np.ndarray]:
    """Posterior mean of ``name``, averaged over chain and draw, or ``None``."""
    post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
    if name not in post:
        return None
    arr = np.asarray(post[name])
    return arr.reshape(-1, *arr.shape[2:]).mean(axis=0)


def assert_arm_is_screenable(base: KalmanModelConfig, arm: KalmanModelConfig,
                             arm_name: str = "arm") -> None:
    """Raise if ``arm`` differs from ``base`` in a field the Max step fixed.

    Raises
    ------
    ValueError
        Naming the offending field, so the caller learns which arm needs the
        exact harness rather than being told the screen "failed".
    """
    changed = [
        f for f in COVARIANCE_FIELDS
        if getattr(base, f, None) != getattr(arm, f, None)
    ]
    if changed:
        raise ValueError(
            f"Arm {arm_name!r} changes {sorted(changed)!r}, which the "
            "Max-and-Smooth pseudo-observations were conditioned on. Screening it "
            "here would contrast two arms against one arm's noise model. Use the "
            "exact harness (run_model_comparison / --compare) for this arm."
        )


def gaussian_likelihood_approximation(
    panel: KalmanPanelV2,
    idata: InferenceLike,
    config: Optional[KalmanModelConfig] = None,
    *,
    extra_group_cols: Sequence[str] = (),
) -> PseudoObservations:
    """Step 1 (Max): per-name Gaussian approximation of the likelihood.

    Conditions on the baseline fit's posterior-mean covariance hyperparameters
    and returns one pseudo-observation of ``mu_reg`` per name, with its sampling
    variance and the loading of the latent level.

    Parameters
    ----------
    panel
        The prepared panel the baseline was fitted to.
    idata
        The baseline posterior. Needs ``within_name_cov``, ``sigma_isin``,
        ``alpha_time``, ``sigma_time``, ``sigma_level`` and ``sigma_total``;
        ``signal_scale`` and ``nu`` are used when present.
    config
        Parameterisation of the baseline fit. Defaults to
        :class:`KalmanModelConfig`.
    extra_group_cols
        Categorical columns to factorise from ``panel.frame`` in ADDITION to the
        ones ``prepare_panel`` already indexed.

        **Not optional in practice for a multi-arm screen.** ``prepare_panel``
        builds ``coord_idx`` from ``model_cfg.group_effects`` alone, so a panel
        prepared for the shipped four-level hierarchy carries no ``country`` or
        ``industry`` index — and the ``hierarchy_fine`` arm would then quietly
        reduce to the baseline and screen as "no difference". A false negative
        dressed as a result is worse than an error. Pass the UNION of every arm's
        ``group_effects``.

    Returns
    -------
    PseudoObservations

    Raises
    ------
    ValueError
        If a required posterior variable is missing, ``T < 2``, or a name in
        ``extra_group_cols`` is not a column of ``panel.frame``.
    """
    cfg = config or KalmanModelConfig()
    Y = np.asarray(panel.Y, dtype="float64")
    n, T = Y.shape
    if T < 2:
        raise ValueError(
            f"Max-and-Smooth needs T >= 2 to separate a level from noise, got T={T}."
        )

    required = ("within_name_cov", "sigma_isin", "alpha_time")
    missing = [v for v in required if _posterior_mean(idata, v) is None]
    if missing:
        raise ValueError(
            f"posterior lacks {missing!r}; it is not a v2 baseline fit. Available "
            "variables come from build_kalman_pt_model_v2."
        )

    A_full = np.asarray(_posterior_mean(idata, "within_name_cov"), dtype="float64")
    sigma_isin = np.asarray(_posterior_mean(idata, "sigma_isin"), dtype="float64")
    alpha_time = np.asarray(_posterior_mean(idata, "alpha_time"), dtype="float64")

    tau = _posterior_mean(idata, "sigma_time")
    tau = np.ones(T) if tau is None else np.asarray(tau, dtype="float64")

    # w_L as a SHARE, recovered from the two scales the model already exports
    # rather than re-derived from the simplex — one fewer thing to keep in sync.
    s_level = _posterior_mean(idata, "sigma_level")
    s_total = _posterior_mean(idata, "sigma_total")
    if s_level is None or s_total is None or float(s_total) <= 0:
        w_level = 0.0
        logger.warning(
            "sigma_level / sigma_total absent: treating the permanent level as "
            "zero, so the pseudo-observations carry no level loading."
        )
    else:
        w_level = float(np.square(float(s_level) / float(s_total)))

    # Strip the level leg back out of the shared shape. Where tau sits differs by
    # parameterisation, and it has to match _cov() in the builder exactly.
    if cfg.time_scale_applies_to == "covariance":
        A_obs = A_full - w_level * np.outer(tau, tau)
    else:
        A_obs = A_full - w_level * np.ones((T, T))

    signal_scale = _posterior_mean(idata, "signal_scale")
    signal_scale = (np.ones(n) if signal_scale is None
                    else np.asarray(signal_scale, dtype="float64"))
    signal_scale = np.where(np.isfinite(signal_scale) & (signal_scale > 1e-8),
                            signal_scale, 1.0)

    nu = _posterior_mean(idata, "nu")
    # The paper's SECOND Gaussian approximation, in closed form: for a
    # multivariate t the normalised likelihood has the same GLS mean and a
    # variance inflated by nu/(nu-2). It is the variant the paper reserves for
    # skewed likelihoods, and here it is free.
    t_inflation = 1.0
    if nu is not None and cfg.likelihood == "student_t":
        nu_f = float(nu)
        if nu_f > 2.0:
            t_inflation = nu_f / (nu_f - 2.0)
        else:  # pragma: no cover - nu_floor forbids it
            logger.warning("nu = %.3f <= 2: variance undefined, not inflating.", nu_f)

    resid = Y - alpha_time[None, :]
    mask = np.isfinite(resid)

    m_hat = np.full(n, np.nan)
    var_unit = np.full(n, np.nan)   # (1' Ao^-1 1)^-1, sigma-free
    c_load = np.full(n, np.nan)     # level loading, sigma-free

    # sigma_i^2 factors out of the weights, so everything here depends only on
    # WHICH cells a name observes. Group by pattern: a handful of small solves
    # instead of one per name.
    codes = (mask * (1 << np.arange(T))[None, :]).sum(axis=1)
    n_singleton = 0
    for code in np.unique(codes):
        rows = np.flatnonzero(codes == code)
        obs = mask[rows[0]]
        k = int(obs.sum())
        if k == 0:
            continue
        Ao = A_obs[np.ix_(obs, obs)]
        one = np.ones(k)
        try:
            Ainv_one = np.linalg.solve(Ao, one)
            Ainv_tau = np.linalg.solve(Ao, tau[obs])
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            logger.warning("pattern %d has a singular A_obs; skipping %d names.",
                           int(code), len(rows))
            continue
        denom = float(one @ Ainv_one)
        if not np.isfinite(denom) or denom <= 0:  # pragma: no cover - defensive
            continue
        w = Ainv_one / denom
        m_hat[rows] = resid[np.ix_(rows, np.flatnonzero(obs))] @ w
        var_unit[rows] = 1.0 / denom
        c_load[rows] = float(one @ Ainv_tau) / denom
        if k == 1:
            n_singleton += len(rows)

    keep = np.isfinite(m_hat) & np.isfinite(sigma_isin) & (sigma_isin > 0)
    dropped = int((~keep).sum())
    if dropped:
        logger.info("Max step drops %d of %d names with no usable trail.", dropped, n)

    # Onto mu_reg's own scale: eta_hat estimates signal_scale * mu_reg.
    s = signal_scale[keep]
    m_out = m_hat[keep] / s
    var_out = (t_inflation * var_unit[keep] * sigma_isin[keep] ** 2) / s ** 2
    level_out = np.abs(c_load[keep]) * sigma_isin[keep] / s

    diagnostics = {
        "w_level": w_level,
        "t_inflation": t_inflation,
        "nu": None if nu is None else float(nu),
        "tau": tau.tolist(),
        "n_input": int(n),
        "n_kept": int(keep.sum()),
        "n_dropped": dropped,
        "n_singleton_trail": int(n_singleton),
        "n_missingness_patterns": int(len(np.unique(codes))),
        "median_var_obs": float(np.median(var_out)) if keep.any() else float("nan"),
        "median_level_scale": float(np.median(level_out)) if keep.any() else float("nan"),
    }
    logger.info(
        "Max step: %d names, %d missingness patterns, w_level %.4f, t-inflation "
        "%.3f, median sampling sd %.4f vs median level scale %.4f",
        diagnostics["n_kept"], diagnostics["n_missingness_patterns"], w_level,
        t_inflation, float(np.sqrt(diagnostics["median_var_obs"])),
        diagnostics["median_level_scale"],
    )

    # Factorise any group column prepare_panel did not index, so an arm that
    # names a finer hierarchy actually gets one.
    coord_idx = {k: np.asarray(v) for k, v in panel.coord_idx.items()}
    coord_uniques = {k: np.asarray(v) for k, v in panel.coord_uniques.items()}
    wanted = [c for c in extra_group_cols if c not in coord_idx]
    if wanted:
        import pandas as pd

        frame = panel.frame
        absent = [c for c in wanted if c not in frame.columns]
        if absent:
            raise ValueError(
                f"extra_group_cols {absent!r} are not columns of panel.frame; the "
                f"arm naming them cannot be screened. Available: "
                f"{sorted(frame.columns)[:20]}..."
            )
        for col in wanted:
            codes, uniques = pd.factorize(frame[col].astype("string"), sort=True)
            if (codes < 0).any():
                raise ValueError(
                    f"{col!r} has {int((codes < 0).sum())} null values; a "
                    "ZeroSumNormal over it would silently pool them into one level."
                )
            coord_idx[col] = codes.astype("int32")
            coord_uniques[col] = np.asarray(uniques)
        logger.info("Max step factorised %d extra group column(s): %s",
                    len(wanted), ", ".join(wanted))

    # Parent maps for every indexed level whose nearest indexed ancestor is also
    # present. Built here rather than copied from the panel because the Max step
    # may have factorised levels the panel never indexed -- and a nested arm
    # screened without them would be screened as its CROSSED equivalent, which
    # is the same class of false negative `extra_group_cols` exists to prevent.
    coord_parent_of = {
        k: np.asarray(v) for k, v in getattr(panel, "coord_parent_of", {}).items()
    }
    ordered_levels = _order_levels(list(coord_idx))
    if len(ordered_levels) > 1:
        import pandas as pd

        ident = panel.frame["isin"].astype(str)
        meta = _build_hierarchy_indices(
            panel.frame.set_index(ident)[
                [c for c in ordered_levels if c in panel.frame.columns]
            ],
            ident.to_numpy(),
            levels=[c for c in ordered_levels if c in panel.frame.columns],
        )
        for col, entry in meta.items():
            if entry.get("parent_of") is None:
                continue
            if not np.array_equal(
                np.asarray(entry["labels"]).astype(str),
                np.asarray(coord_uniques[col]).astype(str),
            ):
                logger.warning(
                    "label order for %r disagrees between the hierarchy helper "
                    "and the Max step's factorisation; not carrying its parent "
                    "map, so any arm nesting %r will refuse rather than "
                    "mis-attribute.", col, col,
                )
                continue
            coord_parent_of[col] = np.asarray(entry["parent_of"], dtype="int32")

    return PseudoObservations(
        isins=np.asarray(panel.isins)[keep],
        m_hat=m_out,
        var_obs=var_out,
        level_scale=level_out,
        n_obs=mask.sum(axis=1)[keep],
        X_drift=np.asarray(panel.X_drift, dtype="float64")[keep],
        drift_names=list(panel.drift_names),
        coord_idx={k: v[keep] for k, v in coord_idx.items()},
        coord_uniques=coord_uniques,
        coord_parent_of=coord_parent_of,
        diagnostics=diagnostics,
    )


def build_pseudo_model(
    pseudo: PseudoObservations,
    config: Optional[KalmanModelConfig] = None,
) -> "pm_typing.Model":
    """Step 2 (Smooth): the Gaussian-Gaussian pseudo model.

    ``m_hat_i ~ Normal(X_i @ beta + sum_g u_g, sqrt(var_obs_i + (kappa *
    level_scale_i)^2))`` with ``beta ~ Normal(0, beta_prior_scale)`` and crossed
    ``ZeroSumNormal`` group effects at the fixed :data:`GROUP_EFFECT_SCALE`.

    ``kappa`` is the permanent-level scale. ``enable_isin_level = False`` pins it
    to zero, which is what makes ``level_off`` a one-parameter contrast here.

    Parameters
    ----------
    pseudo
        Step-1 output.
    config
        The ARM's config — this is where ``group_effects`` and
        ``enable_isin_level`` take effect.

    Returns
    -------
    pymc.Model

    Raises
    ------
    ImportError
        If PyMC is not installed.
    """
    if pm is None:
        raise ImportError("PyMC is not available. Install pymc to use Max-and-Smooth.")
    cfg = config or KalmanModelConfig()

    # Hard error, not a warning. Skipping a missing level would reduce a finer
    # hierarchy arm to the baseline and make the contrast report "no difference"
    # -- a false negative indistinguishable from a real one. The Max step takes
    # `extra_group_cols` precisely so this cannot happen; if it does, the caller
    # forgot to pass the union of the arms' group_effects.
    missing = [g for g in cfg.group_effects if g not in pseudo.coord_idx]
    if missing:
        raise ValueError(
            f"group effects {missing!r} have no index in the pseudo-observations, "
            f"so this arm would silently BE the baseline. Pass them as "
            f"extra_group_cols to gaussian_likelihood_approximation. Indexed: "
            f"{sorted(pseudo.coord_idx)}"
        )
    # Parents before children: `build_group_effect_terms` topologically
    # sorts anyway, but the coords must be registered in an order the
    # hierarchy helper agrees with.
    groups = _order_levels(list(cfg.group_effects))

    coords: dict[str, Any] = {
        "isin": pseudo.isins,
        "drift_feature": pseudo.drift_names,
    }
    for g in groups:
        coords[g] = pseudo.coord_uniques[g]

    with pm.Model(coords=coords) as model:
        X = pm.Data("X_drift", pseudo.X_drift, dims=("isin", "drift_feature"))
        var_obs = pm.Data("var_obs", pseudo.var_obs, dims="isin")
        level_scale = pm.Data("level_scale", pseudo.level_scale, dims="isin")

        beta = pm.Normal("beta", mu=0.0, sigma=cfg.beta_prior_scale, dims="drift_feature")
        eta = X @ beta

        # Shared with the full model, so a nested arm is SCREENED as the model
        # it would be FITTED as. Building crossed effects here regardless of
        # `cfg.group_parents` would make `hierarchy_nested` screen identically to
        # its crossed control and the contrast between them meaningless.
        idx_data = {
            g: pm.Data(f"{g}_idx", pseudo.coord_idx[g].astype("int64"), dims="isin")
            for g in groups
        }
        level_effect, leaves = build_group_effect_terms(
            cfg, idx_data, pseudo.coord_parent_of
        )
        for g in leaves:
            eta = eta + level_effect[g][idx_data[g]]

        mu = pm.Deterministic("mu_reg", eta, dims="isin")

        if cfg.enable_isin_level:
            kappa = pm.HalfNormal("kappa_level", sigma=1.0)
            sd = pm.math.sqrt(var_obs + (kappa * level_scale) ** 2)
        else:
            # Pinned, not merely small: this is the arm's whole content.
            pm.Deterministic("kappa_level", pm.math.constant(0.0))
            sd = pm.math.sqrt(var_obs)

        pm.Normal("m_hat", mu=mu, sigma=sd, observed=pseudo.m_hat, dims="isin")

    return model
