#!/usr/bin/env python
"""Score two Kalman v2 vintages against the returns that happened between them.

Why
---
Every gate in the workflow grades the model against the analyst price-target
trail it was fitted to. That is a real check of the likelihood and a useless
check of the decision layer: a model that simply republishes consensus fits its
own input perfectly, which is exactly what run ``49e84d7e9d59`` did while
clearing 19 of 21 gates.

This is the missing half. Given an earlier vintage and a later one from
``analytics.panel_vintage_v2`` it computes, per name::

    realised = last_price(later) / last_price(earlier) - 1

and grades the earlier vintage's predictions against it on three axes:

**Discrimination** — Spearman rank IC of ``expected_return_kalman`` against
realised, plus the decile spread. This is the number that decides whether the
screen is worth anything. Indicative prior evidence, from a single cross-section
of the surviving universe (NOT a backtest — see the caveats below): analyst
implied upside had a 12-month rank IC of **0.026** and a 3-month IC of 0.112.

**Bias** — median predicted minus median realised, and the OLS slope of realised
on predicted. A slope well below 1 means the predictions need scaling down, and
its reciprocal is the first honest estimate of how much shrinkage the decision
layer should apply. Indicative: slope 0.450 at 12 months, 0.120 at 3 months.

**Calibration** — the share of realised returns falling inside the earlier
vintage's ``[er_p05, er_p95]`` band. Nominal is 90%. This is the only coverage
statistic in the project computed against outcomes rather than against
replicated price targets.

The bias slope is what eventually replaces
``KalmanRunConfigV2.forecast_error_multiplier`` — a prior today, because the
panel's own autocorrelation cannot identify it — with an estimate.

CAVEATS, which apply to any run of this script and must be quoted with its output
--------------------------------------------------------------------------------
* **Survivorship.** Only names present in BOTH vintages are scored. A name that
  delisted, was acquired or fell out of the universe is dropped, and those are
  disproportionately the bad outcomes. Expect the realised numbers to be
  optimistic.
* **No FX or total return.** ``last_price`` is a local-currency price level.
  Dividends are not added back and currency moves are not stripped out.
* **Horizon mismatch.** The price target is a ~12-month quantity. Scoring it
  over a shorter gap measures partial progress toward it, not achievement.
* **One sample.** Two vintages is one overlapping period, not a distribution
  over periods. Do not read a single IC as evidence of skill either way.

Usage
-----
::

    . .\\set_env.ps1
    $env:PYTHONIOENCODING = "utf-8"
    ~\\AppData\\Local\\Python\\bin\\python.exe scripts\\score_panel_vintages.py
    ~\\AppData\\Local\\Python\\bin\\python.exe scripts\\score_panel_vintages.py --list

Options::

    --from YYYY-MM-DD   earlier vintage (default: the oldest available)
    --to YYYY-MM-DD     later vintage   (default: the newest available)
    --min-gap-days N    refuse a gap shorter than this (default: 60)
    --by sector|trading_region|size_class   also break the scores down
    --csv PATH          write the per-name scored frame
    --list              list available vintages and exit
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

_VINTAGE_TABLE = "panel_vintage_v2"

#: Winsorisation applied to realised returns before any statistic. A single
#: 40x survivor on a thin listing otherwise dominates every mean and slope on a
#: 6,500-name cross-section. Rank statistics are unaffected; the mean and OLS
#: slope are, which is the point.
_REALISED_CLIP = (-0.95, 5.0)


def _engine():
    from sqlalchemy import create_engine

    url = os.environ.get("DB_URL")
    if not url:
        raise SystemExit("DB_URL is not set. Run '. .\\set_env.ps1' first.")
    return create_engine(url)


def list_vintages(conn) -> pd.DataFrame:
    from sqlalchemy import text

    return pd.read_sql(
        text(
            f'SELECT asof_date, count(*) AS names, min(run_id) AS run_id, '
            f'max(captured_at) AS captured_at '
            f'FROM analytics."{_VINTAGE_TABLE}" GROUP BY asof_date ORDER BY asof_date'
        ),
        conn,
    )


def load(conn, asof) -> pd.DataFrame:
    from sqlalchemy import text

    return pd.read_sql(
        text(f'SELECT * FROM analytics."{_VINTAGE_TABLE}" WHERE asof_date = :d'),
        conn, params={"d": asof},
    )


def score(early: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    """Join two vintages on ISIN and attach the realised return."""
    a = early.set_index("isin")
    b = late.set_index("isin")[["last_price"]].rename(
        columns={"last_price": "last_price_later"}
    )
    j = a.join(b, how="inner").reset_index()

    p0 = pd.to_numeric(j["last_price"], errors="coerce")
    p1 = pd.to_numeric(j["last_price_later"], errors="coerce")
    realised = (p1 / p0.where(p0 > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
    j["realised"] = realised.clip(*_REALISED_CLIP)
    # out_of_support rows have NULL ranking metrics by design; keep them in the
    # frame but out of every statistic, so the dropped count is visible.
    if "out_of_support" in j.columns:
        j["scored"] = j["realised"].notna() & ~j["out_of_support"].fillna(False)
    else:
        j["scored"] = j["realised"].notna()
    return j


def _stats(df: pd.DataFrame, pred_col: str) -> dict:
    d = df[df["scored"]][[pred_col, "realised", "er_p05", "er_p95"]].copy()
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce")
    d = d.dropna(subset=[pred_col, "realised"])
    if len(d) < 100:
        return {"n": len(d)}
    slope, intercept = np.polyfit(d[pred_col], d["realised"], 1)
    q = pd.qcut(d[pred_col], 10, labels=False, duplicates="drop")
    lo = d.loc[q == q.min(), "realised"].median()
    hi = d.loc[q == q.max(), "realised"].median()
    cov = np.nan
    band = d.dropna(subset=["er_p05", "er_p95"])
    if len(band) >= 100:
        cov = float(
            ((band["realised"] >= band["er_p05"])
             & (band["realised"] <= band["er_p95"])).mean()
        )
    return {
        "n": len(d),
        "spearman_ic": float(d[pred_col].corr(d["realised"], method="spearman")),
        "pearson": float(d[pred_col].corr(d["realised"])),
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "median_pred": float(d[pred_col].median()),
        "median_realised": float(d["realised"].median()),
        "bias_pp": float((d[pred_col].median() - d["realised"].median()) * 100),
        "share_pred_pos": float((d[pred_col] > 0).mean()),
        "share_realised_pos": float((d["realised"] > 0).mean()),
        "decile_spread_pp": float((hi - lo) * 100),
        "band_coverage": cov,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="d_from", default=None)
    ap.add_argument("--to", dest="d_to", default=None)
    ap.add_argument("--min-gap-days", type=int, default=60)
    ap.add_argument("--by", default=None,
                    choices=["sector", "trading_region", "size_class", "style_class"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    eng = _engine()
    with eng.connect() as conn:
        try:
            vintages = list_vintages(conn)
        except Exception as exc:
            raise SystemExit(
                f"cannot read analytics.{_VINTAGE_TABLE}: {exc}\n"
                "Capture a vintage first: python scripts/capture_panel_vintage.py"
            )
        if vintages.empty:
            raise SystemExit(
                f"analytics.{_VINTAGE_TABLE} is empty. Run "
                "scripts/capture_panel_vintage.py after your next export."
            )
        if args.list:
            print(vintages.to_string(index=False))
            return 0
        if len(vintages) < 2:
            print(vintages.to_string(index=False))
            raise SystemExit(
                "\nOnly one vintage exists. Scoring needs two separated in time — "
                "this is the expected state right after the harness is built. "
                "Capture again after a quarter."
            )

        d_from = pd.Timestamp(args.d_from).date() if args.d_from else vintages["asof_date"].iloc[0]
        d_to = pd.Timestamp(args.d_to).date() if args.d_to else vintages["asof_date"].iloc[-1]
        gap = (pd.Timestamp(d_to) - pd.Timestamp(d_from)).days
        if gap <= 0:
            raise SystemExit(f"--to ({d_to}) must be after --from ({d_from})")
        if gap < args.min_gap_days:
            raise SystemExit(
                f"gap is {gap} days, below --min-gap-days {args.min_gap_days}. "
                "A price target is a ~12-month quantity; scoring it over a few "
                "weeks measures noise. Pass a lower --min-gap-days to override."
            )

        early, late = load(conn, d_from), load(conn, d_to)

    j = score(early, late)
    print(f"from {d_from}  ->  to {d_to}   ({gap} days, {gap / 365.25:.2f}y)")
    print(f"names: {len(early):,} early, {len(late):,} late, {len(j):,} matched, "
          f"{int(j['scored'].sum()):,} scored")
    dropped = len(early) - len(j)
    print(f"survivorship: {dropped:,} name(s) in the earlier vintage did not "
          f"survive to the later one and are NOT scored "
          f"({dropped / max(len(early), 1):.1%} of the universe)")

    rows = []
    for col, label in (
        ("expected_return_kalman", "model expected upside"),
        ("implied_upside", "analyst consensus"),
        ("er_mean", "MC forward mean"),
    ):
        if col in j.columns:
            st = _stats(j, col)
            st["predictor"] = label
            rows.append(st)
    if not rows:
        raise SystemExit(
            "the earlier vintage carries no prediction columns — it was captured "
            "from a panel-only export. Nothing to score."
        )
    out = pd.DataFrame(rows).set_index("predictor")
    print("\n" + out.to_string(float_format=lambda x: f"{x:,.4f}"))

    if args.by and args.by in j.columns and "expected_return_kalman" in j.columns:
        print(f"\nby {args.by}:")
        grp = []
        for key, sub in j.groupby(args.by):
            st = _stats(sub, "expected_return_kalman")
            if st.get("n", 0) >= 100:
                st[args.by] = key
                grp.append(st)
        if grp:
            print(pd.DataFrame(grp).set_index(args.by)[
                ["n", "spearman_ic", "ols_slope", "bias_pp", "band_coverage"]
            ].to_string(float_format=lambda x: f"{x:,.4f}"))

    print(
        "\nReading this: ols_slope is the factor the predictions should be "
        "scaled by, so 1/slope is roughly the shrinkage the decision layer owes "
        "-- the first evidence-based estimate of "
        "KalmanRunConfigV2.forecast_error_multiplier, which is a prior today. "
        "band_coverage is the only coverage number in this project computed "
        "against outcomes rather than replicated price targets; nominal is 0.90."
        "\nSurvivorship, local-currency prices, horizon mismatch and a single "
        "overlapping period all apply -- see the module docstring before quoting "
        "any of these figures."
    )
    if args.csv:
        j.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
