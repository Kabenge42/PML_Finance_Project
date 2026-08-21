#!/usr/bin/env python
"""Extract one Kalman v2 run's headline numbers and diff them against the last.

Read-only against the database. The only thing it writes is the append-only run
history, and only when ``--append`` is given.

Why a script rather than ad-hoc queries
---------------------------------------
Two runs compared on subtly different definitions is worse than no comparison,
and that is the failure mode of re-deriving the statistics by hand each time.
This file owns *extraction*; the skill that calls it owns *judgement*. In
particular the gate reconstructions below reproduce the pipeline's own
definitions exactly — the coverage gradient is the MEAN of ``er_sd`` over the
buckets ``[0,3,8,20,inf]``, not a median over quintiles, because that is what
``run_screen`` grades.

The database holds exactly ONE run: the v2 analytics tables are DROP-and-RECREATE
and the previous run is gone the moment a new one exports. That is why the
history file exists and why it is append-only.

Usage
-----
::

    . .\\set_env.ps1
    python .claude/skills/kalman-v2-postrun/analyze.py            # report only
    python .claude/skills/kalman-v2-postrun/analyze.py --append   # + record it

Options::

    --history PATH   history file (default: analysis/kalman_v2_run_history.json)
    --append         append this run to the history (idempotent on run_id)
    --json           emit only the JSON summary, no human-readable report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

SCHEMA = os.environ.get("DB_ANALYTICS_SCHEMA", "analytics")

#: Tables one run must have exported. A `run_id` that is not identical across
#: all of them means a partial export, which is a stop condition rather than
#: something to average over.
CORE_TABLES: tuple[str, ...] = (
    "04_panel_frame_v2",
    "09_diagnostics_v2",
    "10_screen_results_v2",
    "10_screen_mc_summary_v2",
    "10b_risk_analytics_v2",
    "10b_risk_book_v2",
    "kalman_filtered_price_targets_v2",
)
GATE_TABLE = "09_gate_report_v2"

#: Gate thresholds, mirroring ``KalmanRunConfigV2``. Duplicated deliberately:
#: this script must run without importing the PyMC stack, which costs ~20s and
#: pulls in a compiler backend to read seven tables. Keep in sync by hand — the
#: reconstruction is a fallback, and the exported gate table supersedes it the
#: moment the pipeline writes one.
GATES = {
    "r_hat_max": 1.01,
    "ess_min": 400,
    "shrinkage_slope_lo": 0.80,
    "shrinkage_slope_hi": 0.98,
    "shrinkage_center_shift_max": 0.02,
    "shrinkage_rho_max": 0.995,
    "shrinkage_revision_min_pp": 0.25,
    "coverage_gradient_min_x": 2.0,
    "prob_pos_pinned_max": 0.60,
}

#: Gates that only exist inside a live run. Naming them individually is the
#: point: a reconstructed subset that does not say what is missing reads as a
#: full report, which is exactly how a pass-through once cleared 19 of 21.
UNAVAILABLE_WITHOUT_GATE_TABLE: tuple[str, ...] = (
    "divergences",
    "ppc_coverage",
    "ppc_t_spread",
    "ppc_decay",
    "ppc_decay_residual",
    "mean_spread",
    "mean_calibration",
    "panel_t_eff",
    "panel_kernel_fit",
    "drift_contrast_leakage",
    "prior_scale",
    "runtime_estimate",
    "export_rowcount",
    "export_finite",
    "export_ranking_range",
    "export_vintage",
)

#: The same quantity under two names. The intermediate frames and the analytics
#: table disagree, which is recommendation 06's complaint; until that is fixed,
#: the mapping lives here so a comparison never silently comes from the wrong
#: column.
COLUMN_ALIASES = {
    "expected_upside": "expected_return_kalman",
    "cvar05": "cvar_5pct_kalman",
    "exp_vol": "expected_vol_kalman",
    "starr": "reward_to_cvar",
}

#: Ladder reported in the artifact's return/risk comparison, in the order a
#: reader walks it: consensus, then the model's point estimate, then the forward
#: simulation, then the risk-normalised columns.
#:
#: **The unit tag is load-bearing, not decoration.** Only the first four are raw
#: decimal returns that may be rendered as percentages. ``risk_adj_return`` is a
#: standardised log-uplift, and ``expected_sharpe_ratio`` / ``reward_to_cvar``
#: are dimensionless ratios -- printing any of them x100 turns a Sharpe of 1.02
#: into "102%", which is the exact class of units confusion the analytics DDL's
#: ``COMMENT ON COLUMN`` convention exists to prevent.
LADDER: tuple[tuple[str, str], ...] = (
    ("implied_upside", "pct"),
    ("expected_return_kalman", "pct"),
    ("er_mean", "pct"),
    ("er_p50", "pct"),
    ("risk_adj_return", "ratio"),
    ("expected_sharpe_ratio", "ratio"),
    ("reward_to_cvar", "ratio"),
    ("p_upside_pos_cond", "prob"),
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _engine():
    url = os.environ.get("DB_URL")
    if not url:
        raise SystemExit("DB_URL is not set; run `. .\\set_env.ps1` first.")
    from sqlalchemy import create_engine

    return create_engine(url)


def _f(x: Any) -> Optional[float]:
    """Coerce to a JSON-safe float, mapping non-finite to None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _table_exists(eng, name: str) -> bool:
    from sqlalchemy import text

    with eng.connect() as c:
        return bool(
            list(
                c.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema=:s AND table_name=:t"
                    ),
                    {"s": SCHEMA, "t": name},
                )
            )
        )


def _read(eng, name: str) -> pd.DataFrame:
    return pd.read_sql(f'SELECT * FROM {SCHEMA}."{name}"', eng)


def _spearman(a: pd.Series, b: pd.Series) -> Optional[float]:
    d = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 100:
        return None
    return _f(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))


# --------------------------------------------------------------------------- #
# Run resolution                                                               #
# --------------------------------------------------------------------------- #


def resolve_run(eng) -> tuple[str, str, list[str]]:
    """Return ``(run_id, exported_at, missing_tables)`` for the exported run.

    Raises
    ------
    SystemExit
        If the core tables disagree on ``run_id``. A split vintage means a
        partial export; analysing it would silently mix two fits.
    """
    from sqlalchemy import text

    seen: dict[str, set[str]] = {}
    stamps: set[str] = set()
    missing: list[str] = []
    with eng.connect() as c:
        for t in CORE_TABLES:
            try:
                rows = list(
                    c.execute(
                        text(f'SELECT DISTINCT run_id, exported_at FROM {SCHEMA}."{t}"')
                    )
                )
            except Exception:
                missing.append(t)
                continue
            for r in rows:
                seen.setdefault(str(r.run_id), set()).add(t)
                stamps.add(str(r.exported_at))
    if not seen:
        raise SystemExit("No v2 analytics tables carry a run_id; has a run exported?")
    if len(seen) > 1:
        detail = {k: sorted(v) for k, v in seen.items()}
        raise SystemExit(
            "Split vintage -- the v2 tables disagree on run_id, so this is a "
            f"partial export and must not be analysed as one run: {detail}"
        )
    run_id = next(iter(seen))
    return run_id, sorted(stamps)[-1] if stamps else "", missing


# --------------------------------------------------------------------------- #
# Sections                                                                     #
# --------------------------------------------------------------------------- #


def read_gates(eng, screen: pd.DataFrame, diag: pd.DataFrame) -> dict[str, Any]:
    """The run's gates: the exported table when present, else a reconstruction."""
    if _table_exists(eng, GATE_TABLE):
        g = _read(eng, GATE_TABLE)
        counts = g["status"].value_counts().to_dict()
        return {
            "source": "table",
            "n_pass": int(counts.get("PASS", 0)),
            "n_warn": int(counts.get("WARN", 0)),
            "n_fail": int(counts.get("FAIL", 0)),
            "unavailable": [],
            "results": g[["gate", "status", "value", "threshold"]].to_dict("records"),
        }

    # ---- reconstruction --------------------------------------------------
    out: list[dict[str, Any]] = []

    free = diag[diag["sd"] > 1e-12]
    max_rhat = _f(free["r_hat"].max())
    min_ess = _f(free["ess_bulk"].min())
    out.append(
        {
            "gate": "r_hat",
            "status": "PASS" if (max_rhat or 9) < GATES["r_hat_max"] else "FAIL",
            "value": f"{max_rhat:.4f} ({free.loc[free['r_hat'].idxmax(), 'index']})",
            "threshold": f"< {GATES['r_hat_max']}",
        }
    )
    out.append(
        {
            "gate": "ess_bulk",
            "status": "PASS" if (min_ess or 0) >= GATES["ess_min"] else "FAIL",
            "value": f"{min_ess:.0f} ({free.loc[free['ess_bulk'].idxmin(), 'index']})",
            "threshold": f">= {GATES['ess_min']}",
        }
    )

    # shrinkage_slope -- the four-part test, exactly as run_screen grades it
    v = (
        screen[["expected_return_kalman", "implied_upside"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(v) >= 100:
        slope, intercept = np.polyfit(v["implied_upside"], v["expected_return_kalman"], 1)
        rho = float(
            v["expected_return_kalman"].corr(v["implied_upside"], method="spearman")
        )
        rev_pp = float(
            (v["expected_return_kalman"] - v["implied_upside"]).abs().median() * 100
        )
        shift = float(v["expected_return_kalman"].mean() - v["implied_upside"].mean())
        ok = (
            GATES["shrinkage_slope_lo"] <= slope <= GATES["shrinkage_slope_hi"]
            and abs(shift) <= GATES["shrinkage_center_shift_max"]
            and not rho > GATES["shrinkage_rho_max"]
            and rev_pp >= GATES["shrinkage_revision_min_pp"]
        )
        out.append(
            {
                "gate": "shrinkage_slope",
                "status": "PASS" if ok else "FAIL",
                "value": (
                    f"slope {slope:.3f}, shift {shift:+.4f}, rho {rho:.5f}, "
                    f"median revision {rev_pp:.2f}pp"
                ),
                "threshold": (
                    f"slope in [{GATES['shrinkage_slope_lo']}, "
                    f"{GATES['shrinkage_slope_hi']}], |shift| <= "
                    f"{GATES['shrinkage_center_shift_max']}, rho <= "
                    f"{GATES['shrinkage_rho_max']}, revision >= "
                    f"{GATES['shrinkage_revision_min_pp']}pp"
                ),
            }
        )

    # coverage_gradient -- MEAN of er_sd over the pipeline's own buckets
    cov = screen.dropna(subset=["n_analysts"]).copy()
    if len(cov) >= 500:
        cov["bucket"] = pd.cut(
            cov["n_analysts"], [0, 3, 8, 20, np.inf], labels=["1-3", "4-8", "9-20", "21+"]
        )
        col = "er_sd" if "er_sd" in cov.columns else "expected_upside_sd"
        grad = cov.groupby("bucket", observed=True)[col].mean()
        monotone = bool(grad.is_monotonic_decreasing)
        spread = _f(grad.max() / max(grad.min(), 1e-12))
        out.append(
            {
                "gate": "coverage_gradient",
                "status": "PASS"
                if monotone and (spread or 0) >= GATES["coverage_gradient_min_x"]
                else "WARN",
                "value": f"{'monotone' if monotone else 'NOT monotone'}, spread {spread:.2f}x",
                "threshold": f"monotone decreasing, spread >= {GATES['coverage_gradient_min_x']}x",
            }
        )

    pinned = _f((screen["prob_pos"] >= 0.99995).mean())
    out.append(
        {
            "gate": "prob_pos_degenerate",
            "status": "PASS" if (pinned or 1) <= GATES["prob_pos_pinned_max"] else "WARN",
            "value": f"{pinned:.1%} pinned at 1.0",
            "threshold": f"<= {GATES['prob_pos_pinned_max']:.0%}",
        }
    )

    counts: dict[str, int] = {}
    for r in out:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "source": "reconstructed",
        "n_pass": counts.get("PASS", 0),
        "n_warn": counts.get("WARN", 0),
        "n_fail": counts.get("FAIL", 0),
        "unavailable": list(UNAVAILABLE_WITHOUT_GATE_TABLE),
        "results": out,
    }


def read_convergence(diag: pd.DataFrame) -> dict[str, Any]:
    free = diag[diag["sd"] > 1e-12]
    return {
        "n_monitored": int(len(diag)),
        "n_pinned": int(len(diag) - len(free)),
        "max_rhat": _f(free["r_hat"].max()),
        "worst_rhat_param": str(free.loc[free["r_hat"].idxmax(), "index"]),
        "min_ess_bulk": _f(free["ess_bulk"].min()),
        "worst_ess_param": str(free.loc[free["ess_bulk"].idxmin(), "index"]),
        "min_ess_tail": _f(free["ess_tail"].min()),
        # Only a live run knows this. Never defaulted to 0 -- an absent
        # divergence count and a zero divergence count are different claims.
        "divergences": None,
    }


def read_structure(diag: pd.DataFrame) -> dict[str, Any]:
    def scalar(name: str) -> Optional[float]:
        row = diag[diag["index"] == name]
        return _f(row["mean"].iloc[0]) if len(row) else None

    betas = diag[diag["index"].str.startswith("beta[")]
    straddle = betas[(betas["eti89_lb"] < 0) & (betas["eti89_ub"] > 0)]
    groups = sorted({i.split("_effect[")[0] for i in diag["index"] if "_effect[" in i})
    return {
        "n_drift": int(len(betas)),
        "drift_names": [i[5:-1] for i in betas["index"]],
        "n_drift_straddling_zero": int(len(straddle)),
        "drift_straddling_zero": [i[5:-1] for i in straddle["index"]],
        "group_effects": groups,
        "variance": {
            "w_level": scalar("variance_weights[level]"),
            "w_state": scalar("variance_weights[state]"),
            "w_obs": scalar("variance_weights[observation]"),
            "rho_inf": scalar("rho_inf"),
            "ell_days": scalar("ou_length_scale_days"),
            "obs_share": scalar("obs_share"),
            "sigma_total": scalar("sigma_total"),
            "nu": scalar("nu"),
            "signal_exponent": scalar("signal_exponent"),
        },
    }


def read_screen(a: pd.DataFrame) -> dict[str, Any]:
    """The return ladder and the shrinkage triple."""
    ladder = {}
    for c, unit in LADDER:
        if c not in a.columns:
            continue
        s = pd.to_numeric(a[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        ladder[c] = {
            "unit": unit,
            "mean": _f(s.mean()),
            "median": _f(s.median()),
            "sd": _f(s.std()),
            "p05": _f(s.quantile(0.05)),
            "p95": _f(s.quantile(0.95)),
            "spearman_vs_implied": _spearman(a[c], a["implied_upside"]),
        }

    v = (
        a[["expected_return_kalman", "implied_upside"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    slope, intercept = np.polyfit(v["implied_upside"], v["expected_return_kalman"], 1)
    rev = (v["expected_return_kalman"] - v["implied_upside"]) * 100

    cov = a.dropna(subset=["n_analysts"]).copy()
    cov["bucket"] = pd.cut(
        cov["n_analysts"], [0, 3, 8, 20, np.inf], labels=["1-3", "4-8", "9-20", "21+"]
    )
    grad = cov.groupby("bucket", observed=True)["er_sd"].mean()

    return {
        "n_names": int(len(a)),
        "ladder": ladder,
        "spearman_vs_consensus": _spearman(a["expected_return_kalman"], a["implied_upside"]),
        "revision_pp_med": _f(rev.abs().median()),
        "revision_pp_p95": _f(rev.abs().quantile(0.95)),
        "ols_slope": _f(slope),
        "ols_intercept": _f(intercept),
        "sd_ratio": _f(a["expected_return_kalman"].std() / a["implied_upside"].std()),
        "post_sd_pp_med": _f(a["expected_upside_sd"].median() * 100),
        "er_sd_pp_med": _f(a["er_sd"].median() * 100),
        "dispersion_ratio": _f(a["er_sd"].median() / max(a["expected_upside_sd"].median(), 1e-12)),
        "shrink_gain_mean": _f(a["shrink_gain"].mean()),
        "prob_pos_pinned_pct": _f((a["prob_pos"] >= 0.99995).mean() * 100),
        "kalman_gain_boundary_pct": _f(
            ((a["kalman_gain"] <= 1e-9) | (a["kalman_gain"] >= 1 - 1e-9)).mean() * 100
        ),
        "p_upside_interior_pct": _f(a["p_upside_pos_cond"].between(0.02, 0.98).mean() * 100),
        "coverage_gradient_x": _f(grad.max() / max(grad.min(), 1e-12)),
        "coverage_gradient_monotone": bool(grad.is_monotonic_decreasing),
    }


def read_risk(a: pd.DataFrame, book: pd.DataFrame) -> dict[str, Any]:
    w = book["weight"] if "weight" in book.columns else book["book_weight"]
    hhi = float((w ** 2).sum())
    sect = book.groupby("sector")[w.name].sum().sort_values(ascending=False)
    return {
        "cvar_med": _f(a["cvar_5pct_kalman"].median()),
        "cvar_pos_pct": _f((a["cvar_5pct_kalman"] > 0).mean() * 100),
        "er_p05_neg_pct": _f((a["er_p05"] < 0).mean() * 100),
        "tail_floor_binds": int(np.isclose(a["tail_risk"], 0.25 * a["er_sd"], rtol=1e-6).sum()),
        "book_n": int(len(book)),
        "book_weight_sum": _f(w.sum()),
        "book_hhi": _f(hhi),
        "book_eff_n": _f(1.0 / hhi) if hhi else None,
        "book_cvar_pos_n": int((book["cvar05"] > 0).sum()),
        "book_w_cvar": _f((w * book["cvar05"]).sum()),
        "book_w_er_p05": _f((w * book["er_p05"]).sum()),
        "book_w_er_sd": _f((w * book["er_sd"]).sum()),
        "book_top_sector": str(sect.index[0]) if len(sect) else None,
        "book_top_sector_w": _f(sect.iloc[0]) if len(sect) else None,
        "book_regions": {str(k): _f(v) for k, v in book.groupby("trading_region")[w.name].sum().items()},
    }


# --------------------------------------------------------------------------- #
# Diff                                                                         #
# --------------------------------------------------------------------------- #


def diff_against(prev: Optional[dict[str, Any]], cur: dict[str, Any]) -> list[dict[str, Any]]:
    """Flag what moved materially since the previous run.

    The thresholds encode which changes are worth writing up. They are one-sided
    where the failure is one-sided: a Spearman ABOVE the gate ceiling is the
    pass-through failure and is always material, whatever it was last run.
    """
    if prev is None:
        return [{"metric": "history", "verdict": "material", "note": "first recorded run; nothing to diff against"}]

    out: list[dict[str, Any]] = []

    def cmp(path: str, label: str, *, abs_move: Optional[float] = None,
            rel_move: Optional[float] = None, crossing: Optional[float] = None,
            fmt: str = "{:.4f}") -> None:
        a, b = prev, cur
        for k in path.split("."):
            a = (a or {}).get(k) if isinstance(a, dict) else None
            b = (b or {}).get(k) if isinstance(b, dict) else None
        if a is None or b is None:
            return
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            if a != b:
                out.append({"metric": label, "verdict": "material", "from": a, "to": b})
            return
        material = False
        if crossing is not None and ((a <= crossing) != (b <= crossing)):
            material = True
        if abs_move is not None and abs(b - a) > abs_move:
            material = True
        if rel_move is not None and a not in (0,) and abs(b - a) / abs(a) > rel_move:
            material = True
        moved = abs(b - a) > 1e-9
        out.append(
            {
                "metric": label,
                "verdict": "material" if material else ("moved" if moved else "flat"),
                "from": fmt.format(a),
                "to": fmt.format(b),
            }
        )

    # one-sided: at or above the gate ceiling the screen is a consensus sort
    rho = cur["screen"].get("spearman_vs_consensus")
    if rho is not None and rho >= GATES["shrinkage_rho_max"]:
        out.append({
            "metric": "PASS-THROUGH ALARM", "verdict": "material",
            "from": "-", "to": f"{rho:.6f}",
            "note": f"Spearman vs consensus at/above the {GATES['shrinkage_rho_max']} gate ceiling",
        })

    cmp("screen.spearman_vs_consensus", "Spearman vs consensus", abs_move=0.002, fmt="{:.5f}")
    cmp("screen.revision_pp_med", "median revision (pp)", abs_move=0.25, crossing=0.25, fmt="{:.3f}")
    cmp("screen.post_sd_pp_med", "posterior sd (pp)", rel_move=0.15, fmt="{:.3f}")
    cmp("screen.prob_pos_pinned_pct", "prob_pos pinned (%)", crossing=60.0, fmt="{:.1f}")
    cmp("screen.coverage_gradient_x", "coverage gradient (x)", crossing=2.0, fmt="{:.2f}")
    cmp("screen.shrink_gain_mean", "shrink_gain mean", abs_move=0.02, fmt="{:.4f}")
    cmp("convergence.max_rhat", "max R-hat", crossing=GATES["r_hat_max"], abs_move=0.005, fmt="{:.4f}")
    cmp("convergence.min_ess_bulk", "min bulk ESS", crossing=float(GATES["ess_min"]), rel_move=0.25, fmt="{:.0f}")
    cmp("structure.variance.w_level", "variance weight: level", crossing=0.01, rel_move=1.0, fmt="{:.5f}")
    cmp("structure.variance.ell_days", "OU length scale (days)", rel_move=0.10, fmt="{:.1f}")
    cmp("structure.n_drift", "drift columns", abs_move=0.5, fmt="{:.0f}")
    cmp("risk.book_w_cvar", "book weighted cvar05", abs_move=0.03, fmt="{:+.4f}")
    cmp("risk.book_cvar_pos_n", "book names with cvar05 > 0", abs_move=3.0, fmt="{:.0f}")
    cmp("risk.book_eff_n", "book effective N", abs_move=3.0, fmt="{:.1f}")

    pg, cg = prev.get("structure", {}).get("group_effects"), cur["structure"]["group_effects"]
    if pg is not None and set(pg) != set(cg):
        out.append({"metric": "group effects", "verdict": "material",
                    "from": ", ".join(pg), "to": ", ".join(cg)})

    # gate status changes
    pr = {r["gate"]: r["status"] for r in prev.get("gates", {}).get("results", [])}
    for r in cur["gates"]["results"]:
        was = pr.get(r["gate"])
        if was and was != r["status"]:
            out.append({"metric": f"gate: {r['gate']}", "verdict": "material",
                        "from": was, "to": r["status"]})
    return out


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def build(eng) -> dict[str, Any]:
    run_id, exported_at, missing = resolve_run(eng)
    diag = _read(eng, "09_diagnostics_v2")
    a = _read(eng, "kalman_filtered_price_targets_v2")
    book = _read(eng, "10b_risk_book_v2")
    return {
        "run_id": run_id,
        "exported_at": exported_at,
        "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "missing_tables": missing,
        "gates": read_gates(eng, a, diag),
        "convergence": read_convergence(diag),
        "structure": read_structure(diag),
        "screen": read_screen(a),
        "risk": read_risk(a, book),
    }


def render(cur: dict[str, Any], deltas: list[dict[str, Any]]) -> str:
    L: list[str] = []
    g, cv, st, sc, rk = cur["gates"], cur["convergence"], cur["structure"], cur["screen"], cur["risk"]
    L += ["=" * 78, f"KALMAN v2 POST-RUN  --  {cur['run_id']}  (exported {cur['exported_at'][:19]})", "=" * 78]

    L += ["", f"(a) GATES  [source: {g['source']}]  {g['n_pass']} pass / {g['n_warn']} warn / {g['n_fail']} fail"]
    for r in g["results"]:
        L.append(f"    [{r['status']:4s}] {r['gate']:24s} {r['value']}")
    if g["unavailable"]:
        L += ["", f"    NOT VISIBLE from the exports ({len(g['unavailable'])} gates) -- these exist only",
              "    during a live run and were not persisted for this one:",
              "      " + ", ".join(g["unavailable"])]
    L += ["", f"    convergence: max R-hat {cv['max_rhat']:.4f} ({cv['worst_rhat_param']}), "
              f"min bulk ESS {cv['min_ess_bulk']:.0f} ({cv['worst_ess_param']})",
          f"    divergences: {'not persisted' if cv['divergences'] is None else cv['divergences']}"]

    L += ["", "(b) RETURN / RISK LADDER",
          "    unit: pct = raw decimal return rendered as %; ratio = dimensionless; prob = probability as %",
          f"    {'column':24s} {'unit':>5s} {'mean':>8s} {'median':>8s} {'sd':>8s} {'p05':>9s} "
          f"{'p95':>9s}  rho(consensus)"]
    for c, d in sc["ladder"].items():
        # A ratio scaled by 100 reads as a percentage and is then compared with
        # one. Scale only what is actually a decimal fraction.
        k = 100.0 if d["unit"] in ("pct", "prob") else 1.0
        rho = "" if d["spearman_vs_implied"] is None else f"{d['spearman_vs_implied']:+.4f}"
        L.append(f"    {c:24s} {d['unit']:>5s} {d['mean']*k:8.2f} {d['median']*k:8.2f} "
                 f"{d['sd']*k:8.2f} {d['p05']*k:9.2f} {d['p95']*k:9.2f}  {rho:>8s}")
    L += ["", f"    shrinkage : slope {sc['ols_slope']:.4f}  intercept {sc['ols_intercept']:+.4f}  "
              f"sd ratio {sc['sd_ratio']:.4f}  median revision {sc['revision_pp_med']:.3f}pp",
          f"    dispersion: posterior sd {sc['post_sd_pp_med']:.3f}pp vs er_sd {sc['er_sd_pp_med']:.3f}pp "
          f"({sc['dispersion_ratio']:.2f}x)",
          f"    degeneracy: prob_pos pinned {sc['prob_pos_pinned_pct']:.1f}%  "
          f"kalman_gain at 0/1 {sc['kalman_gain_boundary_pct']:.1f}%  "
          f"p_upside_pos_cond interior {sc['p_upside_interior_pct']:.1f}%",
          f"    structure : {st['n_drift']} drift columns "
          f"({st['n_drift_straddling_zero']} straddle 0), groups: {', '.join(st['group_effects'])}",
          f"    variance  : level {st['variance']['w_level']:.5f}  state {st['variance']['w_state']:.5f}  "
          f"obs {st['variance']['w_obs']:.5f}  ell {st['variance']['ell_days']:.1f}d",
          f"    book      : n={rk['book_n']} effN {rk['book_eff_n']:.1f}  "
          f"cvar05>0 {rk['book_cvar_pos_n']}/{rk['book_n']}  "
          f"w-avg cvar05 {rk['book_w_cvar']*100:+.2f}%  w-avg er_p05 {rk['book_w_er_p05']*100:+.2f}%"]

    L += ["", "(c) MOVED SINCE THE PREVIOUS RECORDED RUN"]
    mat = [d for d in deltas if d["verdict"] == "material"]
    if not mat:
        L.append("    nothing material.")
    for d in mat:
        note = f"  -- {d['note']}" if d.get("note") else ""
        L.append(f"    [MATERIAL] {d['metric']:32s} {d.get('from','')} -> {d.get('to','')}{note}")
    other = [d for d in deltas if d["verdict"] == "moved"]
    if other:
        L.append(f"    ({len(other)} other metric(s) moved within tolerance)")
    L += ["", "=" * 78]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history", default="analysis/kalman_v2_run_history.json")
    ap.add_argument("--append", action="store_true", help="record this run in the history")
    ap.add_argument("--json", action="store_true", help="emit only the JSON summary")
    args = ap.parse_args()

    eng = _engine()
    cur = build(eng)

    hist_path = Path(args.history)
    history: list[dict[str, Any]] = []
    if hist_path.exists():
        history = json.loads(hist_path.read_text(encoding="utf-8"))
    prev = next((h for h in reversed(history) if h["run_id"] != cur["run_id"]), None)
    deltas = diff_against(prev, cur)
    cur["deltas"] = deltas

    if args.append:
        # Idempotent on run_id: re-running the skill must not double-record.
        history = [h for h in history if h["run_id"] != cur["run_id"]] + [cur]
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(cur, indent=2))
    else:
        print(render(cur, deltas))
        print(f"\nhistory: {len(history)} run(s) in {hist_path}"
              f"{'  [appended]' if args.append else '  [not appended; pass --append]'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
