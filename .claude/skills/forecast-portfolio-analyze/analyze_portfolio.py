"""Extract the investment-facing numbers from a Kalman v2 forecast + decision replay.

The replay (``kalman_portfolio.py``) writes seven frames and seven gates. This script
turns them into the statistics the published artifact
(https://claude.ai/code/artifact/1dde4885-697d-4d0f-8292-ed30d76ec2a2, "The Decision
Layer") is written from, so that two editions of that page are never comparing
different definitions of the same word.

Definitions live HERE, not in the prose. If a number the artifact needs is missing,
add it below rather than computing it by hand in an analysis session.

Usage
-----
    python .claude/skills/forecast-portfolio-analyze/analyze_portfolio.py
    python .claude/skills/forecast-portfolio-analyze/analyze_portfolio.py --json
    python .claude/skills/forecast-portfolio-analyze/analyze_portfolio.py --results <dir>

Notes
-----
Reads CSV only -- no database, no PyMC import, so it runs in seconds and cannot
disturb a fit. ``KALMAN_V2_RESULTS_DIR`` is honoured; the default is
``pymc_kalman_filter_pt_v2_results``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

#: Frame stem -> section directory, mirroring `export_layout.EXPORT_SECTION_DIRS`.
FRAMES: dict[str, str] = {
    "15c_forecast_engines": "15c_forecast",
    "15c_forecast_summary": "15c_forecast",
    "15e_decision_books": "15e_books",
    "14b_group_signals": "14b_recommendations",
    "14b_name_actions": "14b_recommendations",
    "14b_size_down_watch": "14b_recommendations",
    "09_gate_report_portfolio": "09_gates",
}

#: The fit's own CVaR risk book, for the two-books contrast in artifact section 2.
FIT_RISK_BOOK = ("10b_risk", "10b_risk_book_v2")

#: The action ladder, duplicated here DELIBERATELY: this script is a standalone
#: analysis tool that reads exported CSVs and must not import the PyMC stack to
#: learn five strings. `_recommendations.ACTIONS` is the SSOT -- change both.
_BUY_ACTIONS = frozenset({"STRONG BUY", "BUY"})
_SELL_ACTIONS = frozenset({"SELL", "STRONG SELL"})

#: Ranking arms, default first. The exported `rank_by` column names which produced a row.
RANKING_ARMS = ("reward_to_downside", "reward_to_cvar", "p_upside_pos_cond")


def _load(root: Path, stem: str, sub: str) -> Optional[pd.DataFrame]:
    path = root / sub / f"{stem}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _effective_n(w: np.ndarray) -> float:
    w = np.asarray(w, dtype="float64")
    w = w[w > 0]
    if not w.size:
        return float("nan")
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def provenance(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Which fit, which replay, which source revision. All four columns or none."""
    for name in ("15e_decision_books", "09_gate_report_portfolio"):
        df = frames.get(name)
        if df is None or not len(df):
            continue
        row = df.iloc[0]
        out = {c: (None if pd.isna(row.get(c)) else row.get(c))
               for c in ("run_id", "exported_at", "source_sha", "source_dirty")}
        out["source_frame"] = name
        # The FIT's run_id is carried in the handoff-provenance gate's value string,
        # not in the replay's own run_id -- they are different runs by design.
        gates = frames.get("09_gate_report_portfolio")
        if gates is not None and "gate" in gates.columns:
            hit = gates[gates["gate"] == "portfolio_handoff_provenance"]
            if len(hit):
                out["handoff"] = str(hit.iloc[0]["value"])
        return out
    return {}


def book_shape(books: pd.DataFrame, arm: Optional[str] = None) -> dict[str, Any]:
    """Concentration, tilts and the cap plateau -- the 'what am I holding' block.

    `effective_n` is the Herfindahl reciprocal, NOT the nominal `k_book`. The
    distinction is the whole point: cap-and-spill on a plateaued ranking fills the
    cap and stops, so a 50-name book routinely carries the risk of a dozen.
    """
    if "rank_by" in books.columns:
        arm = arm or next((a for a in RANKING_ARMS if a in set(books["rank_by"])),
                          books["rank_by"].iloc[0])
        books = books[books["rank_by"] == arm]
    held = books[books["weight"] > 0].sort_values("weight", ascending=False)
    if not len(held):
        return {"arm": arm, "n_held": 0}
    w = held["weight"].to_numpy()
    cap = float(np.nanmax(w))
    at_cap = int((w >= cap - 1e-9).sum())
    out: dict[str, Any] = {
        "arm": arm,
        "n_held": int(len(held)),
        "effective_n": round(_effective_n(w), 2),
        "cap": round(cap, 4),
        "n_at_cap": at_cap,
        "weight_at_cap": round(float(w[w >= cap - 1e-9].sum()), 4),
        "cum_weight": {k: round(float(np.sort(w)[::-1][:k].sum()), 4)
                       for k in (1, 5, 8, 10, 20) if k <= len(w)},
        "smallest_weight": float(w.min()),
        "weighted_expected_return": round(float((held["weight"] * held["expected_return"]).sum()), 4),
        "weighted_er_sd": round(float((held["weight"] * held["er_sd"]).sum()), 4),
    }
    for col, key in (("sector", "sector_mix"), ("trading_region", "region_mix"),
                     ("country_name", "country_mix"), ("size_class", "size_mix"),
                     ("style_class", "style_mix")):
        if col in held.columns:
            mix = held.groupby(col)["weight"].sum().sort_values(ascending=False)
            out[key] = {str(k): round(float(v), 4) for k, v in mix.head(8).items()}
    # Active weight needs the universe share of the SAME frame, not an external index.
    if "sector" in books.columns:
        univ = books.groupby("sector").size() / len(books)
        act = (held.groupby("sector")["weight"].sum() - univ).dropna()
        out["sector_active"] = {str(k): round(float(v), 4)
                                for k, v in act.sort_values(ascending=False).items()}
    return out


def risk_ladder(books: pd.DataFrame, arm: Optional[str] = None) -> dict[str, Any]:
    """GVaR >= GES >= GTR per name, and whether any of them is on the loss side.

    A book whose 95%-worst modelled outcome is a GAIN is not a diversified book, it
    is a forward simulation with no left tail. That is the claim the artifact makes,
    so it is measured here as a count against zero rather than described.
    """
    if "rank_by" in books.columns:
        arm = arm or books["rank_by"].iloc[0]
        books = books[books["rank_by"] == arm]
    held = books[books["weight"] > 0]
    out: dict[str, Any] = {"arm": arm}
    for k in ("gvar", "ges", "gtr"):
        if k not in held.columns:
            continue
        s = pd.to_numeric(held[k], errors="coerce").dropna()
        out[k] = {
            "min": round(float(s.min()), 4), "median": round(float(s.median()), 4),
            "max": round(float(s.max()), 4),
            "weighted": round(float((held["weight"] * held[k]).sum()), 4),
            "n_positive": int((s > 0).sum()), "n": int(len(s)),
        }
    if "gvar" in books.columns:
        u = pd.to_numeric(books["gvar"], errors="coerce").dropna()
        out["universe_share_gvar_positive"] = round(float((u > 0).mean()), 4)
    return out


def ranking_mechanics(books: pd.DataFrame, arm: Optional[str] = None) -> dict[str, Any]:
    """Why these names: the denominator's position, and the numerator/denominator sign.

    `corr_denominator_vs_expected_return` is the load-bearing one. A reward-to-risk
    ratio whose two halves are NEGATIVELY correlated does not trade reward against
    risk -- it multiplies a high numerator by a vanishing denominator.
    """
    full = books
    if "rank_by" in books.columns:
        arm = arm or books["rank_by"].iloc[0]
        full = books[books["rank_by"] == arm]
    held = full[full["weight"] > 0]
    denom = next((c for c in ("downside_dev", "tail_risk") if c in full.columns), None)
    out: dict[str, Any] = {"arm": arm, "denominator": denom}
    if denom is None:
        return out
    u = pd.to_numeric(full[denom], errors="coerce").dropna()
    b = pd.to_numeric(held[denom], errors="coerce").dropna()
    if len(u) and len(b):
        med = float(u.median())
        out.update({
            "universe_median": med,
            "book_min": float(b.min()), "book_max": float(b.max()),
            "ratio_median_to_book_max": round(med / max(float(b.max()), 1e-300), 2),
        })
    if "rank_denominator_pctile" in held.columns:
        out["book_max_pctile"] = round(float(held["rank_denominator_pctile"].max()), 4)
    if "expected_return" in full.columns:
        out["corr_denominator_vs_expected_return"] = round(
            float(full[denom].corr(full["expected_return"], method="spearman")), 4)
    for flag in ("tail_risk_on_floor", "downside_dev_floored"):
        if flag in full.columns:
            out[f"universe_share_{flag}"] = round(float(full[flag].astype(bool).mean()), 4)
    if "kelly_fraction" in full.columns:
        kf = pd.to_numeric(full["kelly_fraction"], errors="coerce")
        out["kelly_pinned_universe"] = round(float((kf >= 1.0 - 1e-9).mean()), 4)
        out["kelly_pinned_book"] = round(
            float((pd.to_numeric(held["kelly_fraction"], errors="coerce") >= 1.0 - 1e-9).mean()), 4)
    if "kelly_interior" in full.columns:
        out["kelly_interior_universe"] = round(float(full["kelly_interior"].astype(bool).mean()), 4)
    if "kelly_unbounded" in full.columns:
        out["kelly_unbounded_universe"] = round(float(full["kelly_unbounded"].astype(bool).mean()), 4)
    return out


def consensus_position(books: pd.DataFrame, actions: Optional[pd.DataFrame],
                       arm: Optional[str] = None) -> dict[str, Any]:
    """How far the book sits from the median analyst view, and whether it is just a
    consensus sort.

    Spearman of the model's own upside against raw `implied_upside` at or above
    ~0.995 means the screen is reproducing consensus; the artifact's whole reading of
    "why these names" depends on this number being reported, not assumed.
    """
    if actions is None or "implied_upside" not in actions.columns:
        return {}
    keep = [c for c in ("isin", "implied_upside", "expected_upside", "n_analysts",
                        "p_upside_pos_cond", "expected_upside_sd", "action")
            if c in actions.columns]
    m = books.merge(actions[keep], on="isin", how="left")
    if "rank_by" in m.columns:
        arm = arm or m["rank_by"].iloc[0]
        m = m[m["rank_by"] == arm]
    held = m[m["weight"] > 0]
    iu = pd.to_numeric(m["implied_upside"], errors="coerce")
    out: dict[str, Any] = {
        "arm": arm,
        "universe_median_implied_upside": round(float(iu.median()), 4),
        "book_median_implied_upside": round(float(pd.to_numeric(
            held["implied_upside"], errors="coerce").median()), 4),
        "book_median_consensus_pctile": round(float(
            iu.rank(pct=True)[held.index.intersection(m.index)].median()
            if len(held) else float("nan")), 4),
    }
    if "expected_upside" in m.columns:
        out["spearman_model_vs_consensus"] = round(
            float(m["expected_upside"].corr(m["implied_upside"], method="spearman")), 5)
    if "expected_return" in m.columns:
        out["spearman_forward_vs_consensus"] = round(
            float(m["expected_return"].corr(m["implied_upside"], method="spearman")), 5)
    top = set(m.nlargest(50, "implied_upside")["isin"])
    out["book_names_in_consensus_top50"] = int(len(top & set(held["isin"])))
    if "n_analysts" in held.columns:
        na = pd.to_numeric(held["n_analysts"], errors="coerce")
        out["book_analysts_median"] = _f(na.median())
        out["book_names_thin_coverage"] = int((na <= 3).sum())
    if actions is not None and "action" in actions.columns:
        out["actions"] = actions["action"].value_counts().to_dict()
        # The ladder went from three values to five on 2026-08-28. Testing
        # `== "BUY"` still runs, and silently EXCLUDES the strongest names --
        # exactly the reading error the wider vocabulary was meant to prevent.
        act = actions["action"]
        out["share_buy"] = round(float(act.isin(_BUY_ACTIONS).mean()), 4)
        out["share_strong_buy"] = round(float((act == "STRONG BUY").mean()), 4)
        out["share_sell"] = round(float(act.isin(_SELL_ACTIONS).mean()), 4)
        if "consensus_gap" in actions.columns:
            gap = pd.to_numeric(actions["consensus_gap"], errors="coerce")
            if gap.notna().any():
                # Where the model DIFFERS from the analyst panel, on the panel's
                # own 1-5 scale. Positive = more bullish than consensus.
                out["consensus_gap_median"] = _f(gap.median())
                out["share_above_consensus"] = round(float((gap > 0).mean()), 4)
    return out


def posture(signals: pd.DataFrame, books: Optional[pd.DataFrame] = None,
            arm: Optional[str] = None) -> dict[str, Any]:
    """Group over/underweights, and -- the part nothing else measures -- whether the
    book actually expresses them.

    The posture layer and the ranking layer never consult each other, so their
    agreement is an empirical question every run. `sector_alignment_spearman` is that
    question in one number.
    """
    out: dict[str, Any] = {"n_groups": int(len(signals)), "levels": {}}
    if "lambda_g" in signals.columns:
        out["lambda_range"] = [round(float(signals["lambda_g"].min()), 3),
                               round(float(signals["lambda_g"].max()), 3)]
    for level, blk in signals.groupby("level"):
        blk = blk.sort_values("excess_shrunk", ascending=False)
        out["levels"][str(level)] = {
            "band": round(float(blk["band"].iloc[0]), 5),
            "verdicts": blk["verdict"].value_counts().to_dict(),
            "top": [{"group": str(r["group"]), "n": int(r["n"]),
                     "raw": round(float(r["excess_raw"]), 4),
                     "lambda": round(float(r["lambda_g"]), 3),
                     "shrunk": round(float(r["excess_shrunk"]), 4),
                     "verdict": str(r["verdict"])}
                    for _, r in pd.concat([blk.head(3), blk.tail(3)]).drop_duplicates(
                        "group").iterrows()],
        }
    if books is None or "sector" not in books.columns:
        return out
    full = books
    if "rank_by" in books.columns:
        arm = arm or books["rank_by"].iloc[0]
        full = books[books["rank_by"] == arm]
    held = full[full["weight"] > 0]
    sec = signals[signals["level"] == "sector"].set_index("group")
    if not len(sec):
        return out
    univ = full.groupby("sector").size() / len(full)
    active = (held.groupby("sector")["weight"].sum().reindex(sec.index).fillna(0)
              - univ.reindex(sec.index).fillna(0))
    out["sector_alignment_spearman"] = round(
        float(active.corr(sec["excess_shrunk"], method="spearman")), 3)
    # The two cases worth naming every run: the book's biggest bet, and whether it
    # holds anything the model formally dislikes.
    out["largest_book_sector"] = str(active.idxmax())
    out["largest_book_sector_verdict"] = str(sec.loc[active.idxmax(), "verdict"])
    uw = sec[sec["verdict"] == "UNDERWEIGHT"].index
    bw = held.groupby("sector")["weight"].sum()
    out["book_weight_in_underweight_sectors"] = round(
        float(bw.reindex(uw).fillna(0).sum()), 4)
    ow = sec[sec["verdict"] == "OVERWEIGHT"].index
    out["book_weight_in_overweight_sectors"] = round(
        float(bw.reindex(ow).fillna(0).sum()), 4)
    return out


def two_books(decision: pd.DataFrame, fit_book: Optional[pd.DataFrame],
              actions: Optional[pd.DataFrame], arm: Optional[str] = None) -> dict[str, Any]:
    """The fit's CVaR risk book against the replay's decision book.

    Both are 'the book' from one posterior and nothing declares precedence. Overlap is
    reported alongside the ELIGIBILITY difference, because a zero overlap driven by a
    market-cap screen is a different finding from a zero overlap driven by ranking.
    """
    if fit_book is None or not len(fit_book):
        return {}
    wcol = next((c for c in ("book_weight", "weight") if c in fit_book.columns), None)
    if wcol is None:
        return {}
    full = decision
    if "rank_by" in decision.columns:
        arm = arm or decision["rank_by"].iloc[0]
        full = decision[decision["rank_by"] == arm]
    held = full[full["weight"] > 0]
    w = fit_book[wcol].to_numpy()
    out = {
        "fit_book_n": int(len(fit_book)),
        "fit_book_effective_n": round(_effective_n(w), 2),
        "fit_book_max_weight": round(float(np.nanmax(w)), 4),
        "decision_book_n": int(len(held)),
        "decision_book_effective_n": round(_effective_n(held["weight"].to_numpy()), 2),
        "shared_names": int(len(set(fit_book["isin"]) & set(held["isin"]))),
    }
    for label, frame, col in (("fit", fit_book, wcol), ("decision", held, "weight")):
        if "sector" in frame.columns:
            mix = frame.groupby("sector")[col].sum().sort_values(ascending=False)
            out[f"{label}_largest_sector"] = str(mix.index[0])
            out[f"{label}_largest_sector_weight"] = round(float(mix.iloc[0] / mix.sum()), 4)
        if "size_class" in frame.columns:
            mix = frame.groupby("size_class")[col].sum()
            out[f"{label}_size_mix"] = {str(k): round(float(v / mix.sum()), 4)
                                        for k, v in mix.items()}
    # How much of the disjointness is the risk book's own eligibility screen.
    if actions is not None and "mcap_country_r" in actions.columns:
        mc = actions.set_index("isin")["mcap_country_r"]
        elig = mc.reindex(held["isin"]).dropna()
        out["decision_names_passing_mcap_screen_0p03"] = int((elig <= 0.03).sum())
        out["universe_passing_mcap_screen_0p03"] = int((mc <= 0.03).sum())
    return out


def engine_contrast(engines: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Forward dispersion, this engine against the shipped AR simulator.

    The two decay differently -- a fitted OU kernel against a hand-set rho -- so the
    ratio is the size of that modelling choice, and it is reported, never gated.
    """
    if engines is None or "sd_ratio" not in engines.columns:
        return {}
    e = engines.dropna(subset=["er_sd_fc", "er_sd_ar"])
    return {
        "n_matched": int(len(e)),
        "median_sd_ratio": round(float(e["sd_ratio"].median()), 4),
        "spearman": round(float(e["er_sd_fc"].corr(e["er_sd_ar"], method="spearman")), 4),
    }


def watch(watch_df: Optional[pd.DataFrame], books: Optional[pd.DataFrame],
          arm: Optional[str] = None) -> dict[str, Any]:
    """The size-down veto: how many names, why, and how many reached the book.

    Reported rather than applied, so the count that matters is the intersection with
    the SIZED book -- and the weight it carries, since one flagged name at the cap is
    a different fact from three at a basis point.
    """
    if watch_df is None or not len(watch_df):
        return {}
    out: dict[str, Any] = {"n_flagged": int(len(watch_df))}
    for flag in ("flag_wide_band", "flag_thin_coverage"):
        if flag in watch_df.columns:
            out[flag] = int(watch_df[flag].astype(bool).sum())
    if books is None:
        return out
    full = books
    if "rank_by" in books.columns:
        arm = arm or books["rank_by"].iloc[0]
        full = books[books["rank_by"] == arm]
    held = full[full["weight"] > 0]
    hit = held[held["isin"].astype(str).isin(set(watch_df["isin"].astype(str)))]
    out["in_book"] = int(len(hit))
    out["weight_in_book"] = round(float(hit["weight"].sum()), 4)
    out["max_flagged_weight"] = round(float(hit["weight"].max()), 4) if len(hit) else 0.0
    return out


def collect(root: Path) -> dict[str, Any]:
    frames = {stem: df for stem, sub in FRAMES.items()
              if (df := _load(root, stem, sub)) is not None}
    missing = sorted(set(FRAMES) - set(frames))
    books = frames.get("15e_decision_books")
    if books is None:
        raise SystemExit(
            f"no 15e_decision_books.csv under {root} -- run kalman_portfolio.py first")
    actions = frames.get("14b_name_actions")
    signals = frames.get("14b_group_signals")
    fit_book = _load(root, FIT_RISK_BOOK[1], FIT_RISK_BOOK[0])

    arms = list(dict.fromkeys(books["rank_by"])) if "rank_by" in books.columns else []
    arm = next((a for a in RANKING_ARMS if a in arms), arms[0] if arms else None)

    report: dict[str, Any] = {
        "results_root": str(root),
        "missing_frames": missing,
        "arms_present": arms,
        "default_arm": arm,
        "universe_n": int(len(books) // max(len(arms), 1)),
        "provenance": provenance(frames),
        "book": book_shape(books, arm),
        "risk": risk_ladder(books, arm),
        "ranking": ranking_mechanics(books, arm),
        "consensus": consensus_position(books, actions, arm),
        "two_books": two_books(books, fit_book, actions, arm),
        "engines": engine_contrast(frames.get("15c_forecast_engines")),
        "watch": watch(frames.get("14b_size_down_watch"), books, arm),
    }
    if signals is not None:
        report["posture"] = posture(signals, books, arm)
    gates = frames.get("09_gate_report_portfolio")
    if gates is not None:
        report["gates"] = [
            {"gate": r["gate"], "status": r["status"], "value": r["value"],
             "blocking": bool(r["blocking"])}
            for _, r in gates.iterrows()
        ]
        report["gates_emitted"] = int(len(gates))
    if frames.get("15c_forecast_summary") is not None:
        s = frames["15c_forecast_summary"]
        report["forecast"] = {
            "horizon_days": _f(s["horizon_days"].iloc[0]) if "horizon_days" in s else None,
            "factor_share": _f(s["factor_share"].iloc[0]) if "factor_share" in s else None,
            "backend": str(s["backend"].iloc[0]) if "backend" in s else None,
            "median_terminal_return": _f(s["er_mean_terminal"].median())
            if "er_mean_terminal" in s else None,
        }
    return report


def render(r: dict[str, Any]) -> str:
    L: list[str] = []
    p = r.get("provenance", {})
    L.append("=" * 78)
    L.append("KALMAN v2 -- FORECAST + DECISION REPLAY")
    L.append("=" * 78)
    L.append(f"  replay {p.get('run_id')}  src {str(p.get('source_sha'))[:7]}"
             f"{' DIRTY' if p.get('source_dirty') else ' clean'}  {p.get('exported_at')}")
    if p.get("handoff"):
        L.append(f"  handoff: {p['handoff']}")
    L.append(f"  arms present: {r['arms_present']}  ->  reporting {r['default_arm']!r}")
    if r["missing_frames"]:
        L.append(f"  MISSING FRAMES: {r['missing_frames']}")
    fc = r.get("forecast") or {}
    if fc:
        L.append(f"  horizon {fc.get('horizon_days')}d  factor_share {fc.get('factor_share')}"
                 f"  backend {fc.get('backend')}")

    b = r["book"]
    L.append("\n-- BOOK SHAPE " + "-" * 63)
    L.append(f"  {b['n_held']} names, effective N {b['effective_n']} "
             f"({b['n_at_cap']} at the {b['cap']:.0%} cap = {b['weight_at_cap']:.1%})")
    L.append(f"  cumulative weight: {b['cum_weight']}")
    L.append(f"  weighted E[r] {b['weighted_expected_return']:+.1%}, "
             f"weighted er_sd {b['weighted_er_sd']:.1%}")
    for k in ("sector_mix", "size_mix", "style_mix", "country_mix"):
        if k in b:
            L.append(f"  {k}: " + ", ".join(f"{n} {v:.1%}" for n, v in
                                            list(b[k].items())[:5]))

    rk = r["risk"]
    L.append("\n-- RISK LADDER " + "-" * 62)
    for k, lab in (("gvar", "GVaR 95%"), ("ges", "GES 95%"), ("gtr", "GTR worst")):
        if k in rk:
            s = rk[k]
            L.append(f"  {lab:10s} {s['min']:+.2%} .. {s['max']:+.2%}  "
                     f"median {s['median']:+.2%}  weighted {s['weighted']:+.2%}  "
                     f"positive for {s['n_positive']}/{s['n']}")
    if "universe_share_gvar_positive" in rk:
        L.append(f"  universe share with positive GVaR: "
                 f"{rk['universe_share_gvar_positive']:.1%}")

    rm = r["ranking"]
    L.append("\n-- RANKING MECHANICS " + "-" * 56)
    L.append(f"  denominator {rm.get('denominator')!r}: universe median "
             f"{rm.get('universe_median')}, book max {rm.get('book_max')} "
             f"({rm.get('ratio_median_to_book_max')}x below)")
    if "book_max_pctile" in rm:
        L.append(f"  every book name at or below the "
                 f"{rm['book_max_pctile'] * 100:.1f}th percentile")
    if "corr_denominator_vs_expected_return" in rm:
        L.append(f"  corr(denominator, expected_return) = "
                 f"{rm['corr_denominator_vs_expected_return']:+.3f}  "
                 "<- negative means the ratio compounds rather than trades off")
    for k in ("kelly_pinned_universe", "kelly_pinned_book", "kelly_interior_universe",
              "kelly_unbounded_universe"):
        if k in rm:
            L.append(f"  {k}: {rm[k]:.1%}")

    c = r.get("consensus") or {}
    if c:
        L.append("\n-- CONSENSUS POSITION " + "-" * 55)
        L.append(f"  implied upside: universe median "
                 f"{c['universe_median_implied_upside']:+.1%}, book median "
                 f"{c['book_median_implied_upside']:+.1%} "
                 f"(pctile {c.get('book_median_consensus_pctile')})")
        if "spearman_model_vs_consensus" in c:
            L.append(f"  spearman(model upside, consensus) = "
                     f"{c['spearman_model_vs_consensus']:.5f}"
                     + ("   *** PASS-THROUGH ALARM ***"
                        if c["spearman_model_vs_consensus"] >= 0.995 else ""))
        L.append(f"  book names inside the consensus top 50: "
                 f"{c.get('book_names_in_consensus_top50')}")
        if "actions" in c:
            L.append(
                f"  actions: {c['actions']}  "
                f"({c['share_buy']:.1%} buy, {c.get('share_strong_buy', 0):.1%} strong)"
            )
        if "consensus_gap_median" in c:
            L.append(
                f"  vs consensus: median gap {c['consensus_gap_median']:+.2f} "
                f"on the 1-5 analyst scale, "
                f"{c['share_above_consensus']:.0%} more bullish than the panel"
            )

    po = r.get("posture") or {}
    if po:
        L.append("\n-- GROUP POSTURE " + "-" * 60)
        for lvl, blk in po["levels"].items():
            L.append(f"  {lvl:15s} band {blk['band'] * 100:+.2f}pp  {blk['verdicts']}")
        if "sector_alignment_spearman" in po:
            L.append(f"  book/posture sector agreement (spearman): "
                     f"{po['sector_alignment_spearman']:+.2f}")
            L.append(f"  largest book sector {po['largest_book_sector']} -> "
                     f"model says {po['largest_book_sector_verdict']}")
            L.append(f"  book weight in UNDERWEIGHT sectors: "
                     f"{po['book_weight_in_underweight_sectors']:.1%}; "
                     f"in OVERWEIGHT: {po['book_weight_in_overweight_sectors']:.1%}")

    tb = r.get("two_books") or {}
    if tb:
        L.append("\n-- TWO BOOKS, ONE POSTERIOR " + "-" * 49)
        L.append(f"  fit risk book: n {tb['fit_book_n']}, effN {tb['fit_book_effective_n']}, "
                 f"largest sector {tb.get('fit_largest_sector')} "
                 f"{tb.get('fit_largest_sector_weight', 0):.1%}")
        L.append(f"  decision book: n {tb['decision_book_n']}, "
                 f"effN {tb['decision_book_effective_n']}, largest sector "
                 f"{tb.get('decision_largest_sector')} "
                 f"{tb.get('decision_largest_sector_weight', 0):.1%}")
        L.append(f"  SHARED NAMES: {tb['shared_names']}")
        if "decision_names_passing_mcap_screen_0p03" in tb:
            L.append(f"  of the decision book, "
                     f"{tb['decision_names_passing_mcap_screen_0p03']} pass the risk book's "
                     f"mcap_country_r <= 0.03 screen "
                     f"({tb['universe_passing_mcap_screen_0p03']} names qualify universe-wide)")

    wt = r.get("watch") or {}
    if wt:
        L.append("\n-- SIZE-DOWN WATCH " + "-" * 58)
        L.append(f"  {wt['n_flagged']} flagged (wide band "
                 f"{wt.get('flag_wide_band')}, thin coverage {wt.get('flag_thin_coverage')})"
                 f"; {wt.get('in_book')} in the book carrying "
                 f"{wt.get('weight_in_book', 0):.1%}, largest at "
                 f"{wt.get('max_flagged_weight', 0):.1%}")

    en = r.get("engines") or {}
    if en:
        L.append("\n-- ENGINE CONTRAST " + "-" * 58)
        L.append(f"  {en['n_matched']} matched by ISIN; median sd ratio "
                 f"{en['median_sd_ratio']}x, rank agreement {en['spearman']}")

    if "gates" in r:
        L.append(f"\n-- REPLAY GATES ({r['gates_emitted']} of 9 emitted) " + "-" * 40)
        for g in r["gates"]:
            L.append(f"  {g['status']:5s} {g['gate']:34s} {g['value']}")
        emitted = {g["gate"] for g in r["gates"]}
        absent = {"portfolio_book_agreement", "portfolio_factor_sensitivity"} - emitted
        if absent:
            L.append(f"  NOT EMITTED: {sorted(absent)} -- needs --rank-arms / --sweep")
    L.append("")
    return "\n".join(L)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None,
                    help="results root (default: $KALMAN_V2_RESULTS_DIR or "
                         "pymc_kalman_filter_pt_v2_results)")
    ap.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    ap.add_argument("--arm", default=None, help="ranking arm to report (default: shipped)")
    args = ap.parse_args(argv)

    root = Path(args.results or os.environ.get("KALMAN_V2_RESULTS_DIR")
                or "pymc_kalman_filter_pt_v2_results")
    if not root.exists():
        raise SystemExit(f"results root not found: {root}")
    report = collect(root)
    if args.json:
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
