#!/usr/bin/env python
"""Profile the Kalman v2 forecast-error multiplier offline, before a fit.

Why this exists
---------------
``KalmanPortfolioConfig.forecast_error_multiplier`` is a **prior, not an identified
parameter**. The panel's own autocorrelation cannot separate forecast error (how
far consensus sits from fair value) from reporting noise (how noisily a target is
republished over a week), which is the whole reason the term is supplied rather
than fitted — see ``apply_forecast_error_shrinkage`` for the full argument.

A number nobody can estimate is a number that must at least be *auditable*. This
script shows what each candidate multiplier does to the quantities the gates
grade, using an exported panel frame and, optionally, an exported screen — no
sampling, so it runs in seconds instead of the ~11 minutes a fit costs.

It answers three questions:

1. **How much shrinkage?** Median gain ``g``, and the implied median revision
   from consensus. The ``shrinkage_slope`` gate wants a median revision of at
   least ``gate_shrinkage_revision_min_pp`` and a Spearman rho against consensus
   below ``gate_shrinkage_rho_max``.
2. **Does the coverage gradient steepen?** ``g`` falls with dispersion and rises
   with analyst count, so the ratio between the thinnest and best-covered buckets
   is the lever ``coverage_gradient`` reads.
3. **Does ``prob_pos`` un-pin?** The added posterior width is
   ``sqrt(g * fe_var)``, which is what stops P(upside > 0) saturating.

Measured reference (run ``49e84d7e9d59``, before the term existed): median
revision 0.03pp, Spearman 0.999995, ``prob_pos`` 87.4 % pinned, coverage spread
1.53x. v1, which did shrink, sat at a 14 % median revision.

Usage
-----
Run on the PyCharm SDK interpreter — the repo ``.venv`` and the pipenv env are
known-broken for this stack::

    . .\\set_env.ps1
    $env:PYTHONIOENCODING = "utf-8"
    ~\\AppData\\Local\\Python\\bin\\python.exe scripts\\profile_forecast_error.py

Options::

    --run-id ID       panel/screen vintage to profile (default: the latest)
    --kappa "a,b,c"   multiplier grid (default: 0,0.5,1,1.5,2,2.5,3,4)
    --n-exponent X    exponent on the analyst count (default: 0.5)
    --csv PATH        also write the grid as CSV
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PANEL_TABLE = "04_panel_frame_v2"
_SCREEN_TABLE = "10_screen_results_v2"

#: Lookback grid ``KalmanModelConfig`` ships. Only used to recover the response
#: standardisation, which is a property of the panel and not of any fit.
_RESPONSE_COLS = (
    "feat_log_uplift_1y",
    "feat_log_uplift_3m",
    "feat_log_uplift_1w",
    "feat_log_uplift_now",
)


def _engine():
    from sqlalchemy import create_engine

    url = os.environ.get("DB_URL")
    if not url:
        raise SystemExit(
            "DB_URL is not set. Run '. .\\set_env.ps1' first — this script reads "
            "the exported panel frame, it does not refit anything."
        )
    return create_engine(url)


def load_vintage(run_id: str | None) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Return ``(panel_frame, screen_or_None, run_id)`` for one exported run."""
    from sqlalchemy import text

    eng = _engine()
    with eng.connect() as conn:
        if run_id is None:
            run_id = conn.execute(
                text(
                    f'SELECT run_id FROM analytics."{_PANEL_TABLE}" '
                    "ORDER BY exported_at DESC LIMIT 1"
                )
            ).scalar()
            if run_id is None:
                raise SystemExit(f'analytics."{_PANEL_TABLE}" is empty.')
        panel = pd.read_sql(
            text(f'SELECT * FROM analytics."{_PANEL_TABLE}" WHERE run_id = :r'),
            conn,
            params={"r": run_id},
        )
        try:
            screen = pd.read_sql(
                text(f'SELECT * FROM analytics."{_SCREEN_TABLE}" WHERE run_id = :r'),
                conn,
                params={"r": run_id},
            )
        except Exception:  # pragma: no cover - the screen is optional here
            screen = None
    if panel.empty:
        raise SystemExit(f"no panel rows for run_id {run_id!r}")
    return panel, (screen if screen is not None and not screen.empty else None), run_id


def response_moments(panel: pd.DataFrame) -> tuple[float, float]:
    """Pooled mean and sd of the log-uplift response, as ``prepare_panel`` does.

    Pooled across every response column, not the snapshot alone — the snapshot
    moments differ, and using them is a historical bug this project has already
    paid for once (see the v1 de-standardisation note in CLAUDE.md).
    """
    cols = [c for c in _RESPONSE_COLS if c in panel.columns]
    if not cols:
        raise SystemExit(
            f"panel frame carries none of {_RESPONSE_COLS}; it is not a v2 export."
        )
    y = panel[cols].to_numpy(dtype="float64")
    return float(np.nanmean(y)), float(np.nanstd(y))


def profile(
    panel: pd.DataFrame,
    kappas: list[float],
    n_exponent: float,
    struct_sd_std: float,
) -> pd.DataFrame:
    """Grid the multiplier and report what each value does to the gates.

    ``struct_sd_std`` is the structured within-name sd on the STANDARDISED
    response scale — ``sigma_state`` from a previous fit is the right value, and
    it is stable across runs because it is a property of the panel's own
    dispersion rather than of the decision layer.
    """
    cv = (
        pd.to_numeric(panel["feat_pt_noise_sigma"], errors="coerce")
        / pd.to_numeric(panel["observed_pt"], errors="coerce").abs()
    )
    cv = cv.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 5.0).to_numpy()
    n = (
        pd.to_numeric(panel["n_analysts"], errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0)
        .to_numpy()
    )
    mu, sd = response_moments(panel)
    struct_var = struct_sd_std ** 2

    thin, thick = n <= 3, n >= 21
    rows = []
    for k in kappas:
        fe = np.square(k * (cv / np.power(n, n_exponent)) / (sd or 1.0))
        g = struct_var / (struct_var + fe)
        # Implied revision: the shrinkage moves the latent by (1 - g) times its
        # distance from the fitted mean. That distance is not known without a
        # fit, so use the structured sd as the scale — it is exactly the sd of
        # that distance under the model.
        revision_pp = float(np.median((1.0 - g) * struct_sd_std * sd) * 100.0)
        added_sd_pp = float(np.median(np.sqrt(g * fe) * sd) * 100.0)
        rows.append(
            {
                "kappa": k,
                "median_g": float(np.median(g)),
                "g_p05": float(np.quantile(g, 0.05)),
                "g_p95": float(np.quantile(g, 0.95)),
                "g_thin_cov": float(np.median(g[thin])) if thin.any() else np.nan,
                "g_thick_cov": float(np.median(g[thick])) if thick.any() else np.nan,
                "coverage_ratio": (
                    float(np.median(g[thick]) / np.median(g[thin]))
                    if thin.any() and thick.any()
                    else np.nan
                ),
                "median_revision_pp": revision_pp,
                "added_post_sd_pp": added_sd_pp,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default=None, help="default: latest export")
    ap.add_argument("--kappa", default="0,0.5,1,1.5,2,2.5,3,4")
    ap.add_argument("--n-exponent", type=float, default=0.5)
    ap.add_argument(
        "--struct-sd",
        type=float,
        default=0.5559,
        help="structured within-name sd on the standardised scale; the "
        "sigma_state of the reference fit (default: %(default)s, from "
        "run 49e84d7e9d59)",
    )
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    kappas = [float(x) for x in args.kappa.split(",") if x.strip()]
    panel, screen, run_id = load_vintage(args.run_id)
    mu, sd = response_moments(panel)

    print(f"run_id            {run_id}")
    print(f"names             {len(panel):,}")
    print(f"response mean/sd  {mu:.5f} / {sd:.5f}")
    print(f"structured sd     {args.struct_sd:.4f} standardised "
          f"= {args.struct_sd * sd:.5f} in log-uplift units")
    print(f"n_exponent        {args.n_exponent}")

    if screen is not None and {"expected_upside", "implied_upside"} <= set(screen.columns):
        v = screen[["expected_upside", "implied_upside"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        rho = v["expected_upside"].corr(v["implied_upside"], method="spearman")
        rev = (v["expected_upside"] - v["implied_upside"]).abs().median() * 100
        print(f"\nAS EXPORTED       Spearman {rho:.6f}, median revision {rev:.3f}pp")

    grid = profile(panel, kappas, args.n_exponent, args.struct_sd)
    print("\n" + grid.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(
        "\ncoverage_ratio is g(21+ analysts) / g(2-3); the coverage_gradient "
        "gate wants the resulting sd spread >= 2x.\n"
        "median_revision_pp is an IMPLIED scale, not a prediction: it uses the "
        "structured sd as the typical distance from the fitted mean, which is "
        "what that sd measures under the model. The realised revision after a "
        "fit will differ."
    )
    if args.csv:
        grid.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
