#!/usr/bin/env python
"""Profile the Kalman v2 signal-scaling exponent offline, before spending a fit.

Why this exists
---------------
Run ``fa532b925732`` failed ``ppc_decay`` (``rho_inf`` 0.406 observed against a
replicated ``[0.319, 0.389]``) with every convergence gate clear. Reconstructing
the posterior against the exported panel frame localises it to the **mean**, not
the covariance:

* slope of ``y_now`` on ``mu_reg`` = **1.230** (1.0 if calibrated),
* ``Var(mu_reg)`` = 0.292 against 0.471 for unweighted OLS on the same design,
* ``2 Cov(mu_reg, resid_now)`` = 0.134 — **15.1 % of Var(y_now)**, and exactly
  **zero** under the generative model, which redraws the residual independently
  of the mean.

``mu_reg`` is constant in ``t``, so that 15.1 % is permanent variance the
replicates cannot carry. The cause is that the likelihood weights each name by
``1 / sigma_i^2`` while ``sigma_i`` spans 0.26-0.92, so ``beta`` is fitted to the
low-scale names — but the *signal* scales with ``sigma_i`` too, which the
weighting is blind to.

The proposed fix gives the mean its own scaling exponent::

    mean[i, t] = (sigma_i / geomean(sigma)) ** lam * mu_reg[i] + alpha_time[t]

``lam = 0`` is today's model. This script decides whether ``lam`` is worth a
40-minute run, and where to centre its prior, **without sampling**.

What it does
------------
For each ``lam`` on a grid it runs an alternating GLS to convergence — the same
three blocks the PyMC model fits jointly:

1. ``beta`` and ``alpha_time`` by generalised least squares, weighted by the
   current ``sigma_i`` and the current within-name correlation ``A``;
2. ``sigma_i`` by the model's own log-linear scale regression on the residual;
3. ``A = diag(tau) (w_L J + w_S K(ell) + w_O I) diag(tau)`` by least squares
   against the standardised residual's second moments.

Then it reports, per ``lam``, the three quantities the decision turns on:
``Var(mu)``, the calibration slope, and the **predicted replicated rho_inf** —
the last computed with :func:`fit_trail_correlation_kernel`, so the offline
statistic *is* the gate statistic rather than a proxy for it.

Accept/reject
-------------
Proceed only if some ``lam`` puts the predicted ``rho_inf`` at the observed
value **and** improves the profile log-likelihood over ``lam = 0``. If the two
disagree, the parameter is buying the gate rather than the fit and the plan
needs revisiting.

Usage
-----
Run on the PyCharm SDK interpreter — the repo ``.venv`` and the pipenv env are
known-broken for this stack::

    . .\\set_env.ps1
    ~\\AppData\\Local\\Python\\bin\\python.exe scripts\\profile_signal_exponent.py

Options::

    --grid A,B,C     lambda values to profile (default 0,.1,...,1.0)
    --iters N        alternating-GLS sweeps per lambda (default 6)
    --posterior PATH 09_diagnostics CSV used to seed sigma and nu
                     (default: the v2 results directory)
    --csv PATH       also write the profile table
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymc_kalman_filter_pt_v2 as wf  # noqa: E402
from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (  # noqa: E402
    fit_trail_correlation_kernel,
)

logger = logging.getLogger("profile_signal_exponent")

_EPS = 1e-12
_DEFAULT_POSTERIOR = Path("pymc_kalman_filter_pt_v2_results") / "09_diagnostics_v2.csv"


# --------------------------------------------------------------------------- #
# Posterior seed                                                               #
# --------------------------------------------------------------------------- #


def load_posterior_means(path: Path) -> dict[str, float]:
    """Read an exported ``09_diagnostics`` table into ``{var: posterior mean}``.

    Parameters
    ----------
    path
        CSV written by the §9 export.

    Returns
    -------
    dict[str, float]
    """
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = [c.strip() for c in frame.columns]
    frame["index"] = frame["index"].astype(str).str.strip()
    return dict(zip(frame["index"], frame["mean"]))


def seed_sigma(panel: Any, post: dict[str, float], model_cfg: Any) -> np.ndarray:
    """Reproduce the fitted ``sigma_isin`` from the exported posterior means.

    The scale model is the log-linear one from 0.9.9.16; this is the same
    arithmetic ``build_kalman_pt_model_v2`` performs, evaluated at the posterior
    mean rather than per draw.
    """
    log_sigma = (
        post["log_sigma_total"]
        + np.log1p(panel.dispersion_cv)
        + post["sigma_delta_vol_level"] * panel.vol_level
        + post["sigma_delta_log_mcap"] * panel.log_mcap
        + post["sigma_delta_range"] * panel.range_norm
        - post["sigma_n_exponent"] * np.log(panel.precision_weight)
    )
    if "sector" in panel.coord_idx:
        offsets = np.array(
            [post.get(f"sigma_sector_offset[{u}]", 0.0) for u in panel.coord_uniques["sector"]]
        )
        log_sigma = log_sigma + offsets[panel.coord_idx["sector"]]
    return np.exp(np.clip(log_sigma, *model_cfg.log_sigma_clip))


# --------------------------------------------------------------------------- #
# Design                                                                       #
# --------------------------------------------------------------------------- #


def build_design(panel: Any, model_cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean design matrix and the scale-model covariate matrix.

    The mean design is the drift matrix plus dummy-coded crossed group effects —
    the model's ``ZeroSumNormal`` effects at their unshrunk limit, which is the
    right reference for a profile that is asking how much mean the data supports.
    """
    blocks: list[np.ndarray] = [np.ones((panel.n_isin, 1)), panel.X_drift]
    for col in model_cfg.group_effects:
        if col in panel.coord_idx:
            idx = panel.coord_idx[col]
            n_lev = len(panel.coord_uniques[col])
            if n_lev < 2:
                continue
            dummies = np.zeros((len(idx), n_lev))
            dummies[np.arange(len(idx)), idx] = 1.0
            blocks.append(dummies[:, 1:])
    design = np.hstack(blocks)

    scale_cov = np.column_stack(
        [
            np.ones(panel.n_isin),
            np.log1p(panel.dispersion_cv),
            panel.vol_level,
            panel.log_mcap,
            panel.range_norm,
            -np.log(panel.precision_weight),
        ]
    )
    return design, scale_cov


# --------------------------------------------------------------------------- #
# The three refit blocks                                                       #
# --------------------------------------------------------------------------- #


def _pattern_groups(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition names by missingness pattern — one solve per pattern, not per name."""
    keys = np.packbits(mask.astype(np.uint8), axis=1).tobytes()
    width = mask.shape[1] // 8 + (1 if mask.shape[1] % 8 else 0)
    out: dict[bytes, list[int]] = {}
    for i in range(mask.shape[0]):
        out.setdefault(keys[i * width : (i + 1) * width], []).append(i)
    groups = []
    for rows in out.values():
        rows_arr = np.asarray(rows)
        cols = np.flatnonzero(mask[rows_arr[0]])
        if cols.size:
            groups.append((rows_arr, cols))
    return groups


def refit_mean(
    Y: np.ndarray,
    mask: np.ndarray,
    design: np.ndarray,
    signal: np.ndarray,
    sigma: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """GLS for ``beta`` and ``alpha_time`` given the scale and the correlation.

    Minimises

    ``sum_i (y_i - s_i X_i b - alpha)' (sigma_i^2 A_i)^-1 (y_i - s_i X_i b - alpha)``

    accumulating over missingness patterns, so ``A`` is inverted once per pattern
    rather than once per name. ``alpha`` is anchored at the snapshot column
    (the last one), matching ``alpha_time`` in the model.

    Parameters
    ----------
    Y, mask
        Response matrix and its observed mask, shape ``(n, T)``.
    design
        Mean design matrix, shape ``(n, p)``.
    signal
        Per-name multiplier on the linear predictor — ``(sigma_i / geo) ** lam``.
    sigma, A
        Current per-name scale and within-name covariance shape.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``beta`` of length ``p`` and ``alpha`` of length ``T`` (last entry 0).
    """
    T = Y.shape[1]
    p = design.shape[1]
    dim = p + (T - 1)
    XtX = np.zeros((dim, dim))
    Xty = np.zeros(dim)

    for rows, cols in _pattern_groups(mask):
        Wg = np.linalg.inv(A[np.ix_(cols, cols)] + 1e-10 * np.eye(len(cols)))
        w = 1.0 / sigma[rows] ** 2
        s = signal[rows]
        Dg = design[rows]
        Yg = Y[np.ix_(rows, cols)]

        col_w = Wg.sum(axis=0)          # sum_t W[t, u], one entry per column in `cols`
        total_w = float(Wg.sum())       # 1' W 1

        # ---- beta block ----------------------------------------------------
        Dw = Dg * (s * s * w)[:, None]
        XtX[:p, :p] += total_w * (Dw.T @ Dg)
        Dsw = Dg * (s * w)[:, None]     # sum_i s_i w_i x_i, weighted below
        Xty[:p] += Dsw.T @ (Yg @ col_w)

        # ---- alpha block and the beta/alpha cross terms ---------------------
        # Global free-time indices: column c maps to slot p + c for c < T - 1.
        free = [(k, c) for k, c in enumerate(cols) if c < T - 1]
        for k, c in free:
            slot = p + int(c)
            cross = col_w[k] * Dsw.sum(axis=0)
            XtX[:p, slot] += cross
            XtX[slot, :p] += cross
            Xty[slot] += float(np.dot(Wg[k], Yg.T @ w))
            for k2, c2 in free:
                XtX[slot, p + int(c2)] += Wg[k, k2] * float(w.sum())

    sol = np.linalg.solve(XtX + 1e-8 * np.eye(dim), Xty)
    return sol[:p], np.concatenate([sol[p:], [0.0]])


def refit_sigma(
    resid: np.ndarray,
    mask: np.ndarray,
    A: np.ndarray,
    scale_cov: np.ndarray,
    clip: tuple[float, float],
) -> np.ndarray:
    """Per-name ML scale, smoothed through the model's log-linear covariates.

    ``sigma_hat_i^2 = r_i' A_i^-1 r_i / T_i`` is the exact ML scale for one name;
    regressing its log on the scale covariates is what the model does with them,
    so the profile keeps the same degrees of freedom the fit has.
    """
    n = resid.shape[0]
    q = np.zeros(n)
    cnt = np.zeros(n)
    for rows, cols in _pattern_groups(mask):
        Ag = A[np.ix_(cols, cols)]
        Wg = np.linalg.inv(Ag + 1e-10 * np.eye(len(cols)))
        Rg = resid[np.ix_(rows, cols)]
        q[rows] = np.einsum("ij,jk,ik->i", Rg, Wg, Rg)
        cnt[rows] = len(cols)
    sigma_hat = np.sqrt(np.maximum(q / np.maximum(cnt, 1.0), _EPS))
    coef, *_ = np.linalg.lstsq(scale_cov, np.log(sigma_hat), rcond=None)
    return np.exp(np.clip(scale_cov @ coef, *clip))


def refit_correlation(
    resid: np.ndarray,
    mask: np.ndarray,
    sigma: np.ndarray,
    time_days: np.ndarray,
) -> np.ndarray:
    """Least-squares fit of ``diag(tau)(w_L J + w_S K + w_O I)diag(tau)``.

    Fitted against the standardised residual's second-moment matrix, computed
    pairwise-complete. ``tau`` is anchored at 1 on the snapshot column, matching
    the model's ``sigma_time``.
    """
    from scipy.optimize import least_squares

    Z = np.where(mask, resid / sigma[:, None], np.nan)
    T = Z.shape[1]
    S = np.full((T, T), np.nan)
    for a in range(T):
        for b in range(T):
            ok = np.isfinite(Z[:, a]) & np.isfinite(Z[:, b])
            if ok.sum() > 2:
                S[a, b] = float(np.mean(Z[ok, a] * Z[ok, b]))
    gaps = np.abs(time_days[:, None] - time_days[None, :])

    def build(par: np.ndarray) -> np.ndarray:
        wl, ws, ell = par[0], par[1], par[2]
        tau = np.concatenate([np.exp(par[3:]), [1.0]])
        wo = max(1.0 - wl - ws, 0.0)
        shape = wl + ws * np.exp(-gaps / max(ell, 1.0)) + wo * np.eye(T)
        return shape * np.outer(tau, tau)

    def resid_fn(par: np.ndarray) -> np.ndarray:
        return (build(par) - S)[np.isfinite(S)]

    x0 = np.concatenate([[0.05, 0.9, 90.0], np.zeros(T - 1)])
    lo = np.concatenate([[0.0, 0.0, 5.0], np.full(T - 1, -2.0)])
    hi = np.concatenate([[1.0, 1.0, 3000.0], np.full(T - 1, 2.0)])
    fit = least_squares(resid_fn, x0, bounds=(lo, hi))
    return build(fit.x)


def profile_loglik(
    resid: np.ndarray, mask: np.ndarray, sigma: np.ndarray, A: np.ndarray
) -> float:
    """Gaussian profile log-likelihood, summed over names."""
    total = 0.0
    for rows, cols in _pattern_groups(mask):
        Ag = A[np.ix_(cols, cols)] + 1e-10 * np.eye(len(cols))
        sign, logdet = np.linalg.slogdet(Ag)
        Wg = np.linalg.inv(Ag)
        Rg = resid[np.ix_(rows, cols)]
        q = np.einsum("ij,jk,ik->i", Rg, Wg, Rg) / sigma[rows] ** 2
        Tg = len(cols)
        total += float(
            -0.5 * np.sum(q)
            - 0.5 * Tg * np.sum(np.log(sigma[rows] ** 2))
            - 0.5 * len(rows) * (logdet + Tg * math.log(2 * math.pi))
        )
    return total


def predicted_rho_inf(
    var_mu: float, sigma: np.ndarray, A: np.ndarray, nu: float, time_days: np.ndarray
) -> tuple[float, float]:
    """Replicated ``rho_inf`` the generative model would produce.

    The mean is fixed across replicates and constant in ``t``, so it contributes
    ``Var(mu)`` at every gap; the Student-t scale mixture inflates the residual
    covariance by ``nu / (nu - 2)``. The kernel is then fitted with the same
    function the ``ppc_decay`` gate uses, so this number is directly comparable
    to the gate's replicated interval.
    """
    inflate = nu / (nu - 2.0) if nu > 2.0 else 1.0
    S = var_mu + float(np.mean(sigma**2)) * inflate * A
    C = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    n = len(time_days)
    synthetic = np.linalg.cholesky(C + 1e-10 * np.eye(n))
    draws = np.random.default_rng(0).standard_normal((200_000, n)) @ synthetic.T
    kern = fit_trail_correlation_kernel(draws, time_days)
    return float(kern["rho_inf"]), float(kern["ell_days"])


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def profile(
    lam_grid: list[float], iters: int, posterior_path: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run the alternating GLS at every ``lam`` and return the profile table."""
    run_cfg = wf.KalmanRunConfigV2.from_env()
    model_cfg = wf.KalmanModelConfig()
    panel = wf.prepare_panel(wf.load_kalman_frame(run_cfg), model_cfg, run_cfg)

    post = load_posterior_means(posterior_path)
    sigma0 = seed_sigma(panel, post, model_cfg)
    nu = float(post.get("nu", 1e6))
    design, scale_cov = build_design(panel, model_cfg)

    Y = panel.Y
    mask = panel.observed_mask
    T = Y.shape[1]
    y_now = Y[:, T - 1]
    finite_now = np.isfinite(y_now)
    time_days = panel.time_days

    observed = fit_trail_correlation_kernel(Y, time_days)
    baseline = {
        "observed_rho_inf": float(observed["rho_inf"]),
        "observed_ell": float(observed["ell_days"]),
        "nu": nu,
    }

    geo = float(np.exp(np.mean(np.log(sigma0))))
    rows: list[dict[str, Any]] = []

    for lam in lam_grid:
        sigma = sigma0.copy()
        A = np.array(
            [[post[f"within_name_cov[t{a}, t{b}]"] for b in range(T)] for a in range(T)]
        )
        beta = alpha = None
        for _ in range(iters):
            signal = (sigma / geo) ** lam
            beta, alpha = refit_mean(Y, mask, design, signal, sigma, A)
            mu = signal * (design @ beta)
            resid = np.where(mask, Y - (mu[:, None] + alpha[None, :]), 0.0)
            sigma = refit_sigma(resid, mask, A, scale_cov, model_cfg.log_sigma_clip)
            A = refit_correlation(resid, mask, sigma, time_days)

        signal = (sigma / geo) ** lam
        mu = signal * (design @ beta)
        resid = np.where(mask, Y - (mu[:, None] + alpha[None, :]), 0.0)
        var_mu = float(np.var(mu))
        slope = float(np.polyfit(mu[finite_now], y_now[finite_now], 1)[0])
        corr = float(np.corrcoef(mu[finite_now], y_now[finite_now])[0, 1])
        cov2 = 2.0 * float(np.cov(mu[finite_now], resid[finite_now, T - 1])[0, 1])
        rho, ell = predicted_rho_inf(var_mu, sigma, A, nu, time_days)
        rows.append(
            {
                "lambda": lam,
                "var_mu": var_mu,
                "slope": slope,
                "corr_mu_y": corr,
                "cov2_share": cov2 / float(np.nanvar(y_now)),
                "loglik": profile_loglik(resid, mask, sigma, A),
                "pred_rho_inf": rho,
                "pred_ell": ell,
            }
        )
        logger.info(
            "lam %.2f  Var(mu) %.4f  slope %.4f  corr %.4f  rho_inf %.4f",
            lam,
            var_mu,
            slope,
            corr,
            rho,
        )

    return pd.DataFrame(rows), baseline


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0")
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--posterior", type=Path, default=_DEFAULT_POSTERIOR)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    grid = [float(x) for x in args.grid.split(",")]
    table, base = profile(grid, args.iters, args.posterior)

    obs = base["observed_rho_inf"]
    table["rho_inf_miss"] = table["pred_rho_inf"] - obs
    table["dloglik"] = table["loglik"] - float(table.loc[table["lambda"] == 0.0, "loglik"].iloc[0])

    pd.set_option("display.width", 200)
    print()
    print("=" * 96)
    print("SIGNAL-SCALING EXPONENT PROFILE")
    print("=" * 96)
    print(f"observed rho_inf {obs:.4f}  ell {base['observed_ell']:.1f}d   nu {base['nu']:.2f}")
    print()
    print(
        table[
            [
                "lambda",
                "var_mu",
                "slope",
                "corr_mu_y",
                "cov2_share",
                "dloglik",
                "pred_rho_inf",
                "rho_inf_miss",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:9.4f}")
    )

    by_rho = table.iloc[table["rho_inf_miss"].abs().argmin()]
    by_fit = table.iloc[table["dloglik"].argmax()]
    by_slope = table.iloc[(table["slope"] - 1.0).abs().argmin()]
    print()
    print(f"  closest on rho_inf : lambda {by_rho['lambda']:.2f}  (miss {by_rho['rho_inf_miss']:+.4f})")
    print(f"  best profile loglik: lambda {by_fit['lambda']:.2f}  (dloglik {by_fit['dloglik']:+.1f})")
    print(f"  slope nearest 1.0  : lambda {by_slope['lambda']:.2f}  (slope {by_slope['slope']:.4f})")
    print()
    agree = by_fit["dloglik"] > 0 and abs(by_rho["rho_inf_miss"]) < 0.02
    if agree:
        print("  VERDICT: proceed. A positive lambda both improves the fit and reaches the")
        print("           observed decay. Centre the Beta prior between the rho_inf and")
        print("           loglik optima above.")
    else:
        print("  VERDICT: STOP. lambda does not both improve the fit and reach the observed")
        print("           decay on this panel — it would be buying the gate, not the model.")
    print("=" * 96)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
