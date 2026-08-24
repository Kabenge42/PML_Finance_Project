"""Regression tests for return-draw alignment in :mod:`RiskBookModel`.

The bug these pin down (fixed 2026-08-22): ``compute_cvar_aware_book`` assigned
``_tail_stats(return_draws)`` to ``cvar05`` / ``exp_vol`` POSITIONALLY, guarded
only by a length check. ``run_screen`` returns the screen sorted by
``expected_upside`` while ``ScreenDraws.pooled_returns`` stays in ``panel.isins``
order, so every risk column was attributed to the wrong name.

The invariant that catches it is an identity rather than a plausibility check:
``exp_vol`` and ``er_sd`` are the pooled standard deviation of the SAME draws, so
they must agree to floating point. On the corrupted export they correlated
-0.007 while their sorted values matched to 1e-9 for 100 % of names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from probabilistic_ml_model.pymc_models.RiskBookModel import compute_cvar_aware_book

N_NAMES = 40
N_SAMPLES = 400


def _make_case(seed: int = 0):
    """Return ``(screen_sorted, draws, draw_isins)`` with a deliberate reorder.

    ``draws`` stays in the original ("panel") order; the screen frame is sorted
    by ``expected_upside`` exactly as ``run_screen`` leaves it.
    """
    rng = np.random.default_rng(seed)
    isins = np.array([f"ISIN{i:04d}" for i in range(N_NAMES)])

    # Per-name scale spread widely enough that a permutation cannot go unnoticed.
    scale = rng.uniform(0.05, 0.80, size=N_NAMES)
    loc = rng.uniform(-0.10, 0.40, size=N_NAMES)
    draws = rng.normal(loc[:, None], scale[:, None], size=(N_NAMES, N_SAMPLES))

    # er_* are joined ON isin in the real workflow, so build them keyed.
    screen = pd.DataFrame(
        {
            "isin": isins,
            "expected_upside": loc,
            "expected_pt": 100.0 * (1.0 + loc),
            "expected_pt_hdi_lo": 90.0,
            "expected_pt_hdi_hi": 110.0,
            "prob_pos": 0.7,
            "p_upside_pos": 0.7,
            "p_upside_pos_cond": 0.7,
            "kalman_gain": 1.0,
            "mc_prob_pos": 0.7,
            "mcap_global_r": 0.005,
            "er_mean": draws.mean(axis=1),
            "er_sd": draws.std(axis=1),
            "er_p05": np.quantile(draws, 0.05, axis=1),
            "er_p50": np.quantile(draws, 0.50, axis=1),
            "er_p95": np.quantile(draws, 0.95, axis=1),
        }
    )
    screen_sorted = screen.sort_values("expected_upside", ascending=False).reset_index(drop=True)
    assert not screen_sorted["isin"].equals(pd.Series(isins)), "fixture must reorder"

    # Posterior upside draws, keyed by an ``isin`` coord exactly as the workflow
    # supplies them. This path was always correctly aligned (``_row_of``); it is
    # here so the function under test runs, not because it is what is being pinned.
    eu = xr.DataArray(
        rng.normal(loc[None, None, :], 0.05, size=(2, 50, N_NAMES)),
        dims=("chain", "draw", "isin"),
        coords={"chain": [0, 1], "draw": np.arange(50), "isin": isins},
    )
    return screen_sorted, draws, isins, eu


def _book(screen, draws, isins, eu, **kw):
    return compute_cvar_aware_book(
        idata=None, eu=eu, results=screen, return_draws=draws, k_book=10, **kw
    )


def test_exp_vol_matches_er_sd_when_labels_are_passed():
    """The identity that the corrupted export violated."""
    screen, draws, isins, eu = _make_case()
    rb = _book(screen, draws, isins, eu, return_draws_isins=isins)
    got = rb.analytics.set_index("isin")["exp_vol"]
    want = screen.set_index("isin")["er_sd"]
    np.testing.assert_allclose(got.reindex(want.index), want, rtol=1e-9)


def test_cvar05_never_exceeds_er_p05():
    """CVaR at 5 % is the mean below the 5 % quantile, so it cannot exceed it."""
    screen, draws, isins, eu = _make_case(seed=3)
    rb = _book(screen, draws, isins, eu, return_draws_isins=isins)
    got = rb.analytics.set_index("isin")
    assert (got["cvar05"] <= got["er_p05"] + 1e-12).all()


def test_positional_fallback_is_detectably_wrong():
    """Without labels the old behaviour returns — and must be visible, not silent."""
    screen, draws, isins, eu = _make_case(seed=7)
    rb = _book(screen, draws, isins, eu)  # no labels -> positional
    got = rb.analytics.set_index("isin")
    ref = screen.set_index("isin")["er_sd"].reindex(got.index)
    rel = (got["exp_vol"] - ref).abs() / ref.abs()
    # The fixture reorders, so the positional path must NOT reproduce er_sd.
    assert (rel > 1e-6).mean() > 0.5, "fixture failed to expose positional misalignment"


def test_missing_draw_rows_become_nan_not_another_name():
    """A screen name absent from the draws yields NaN, never a neighbour's risk."""
    screen, draws, isins, eu = _make_case(seed=11)
    keep = np.arange(N_NAMES) != 5
    rb = _book(screen, draws[keep], isins[keep], eu, return_draws_isins=isins[keep])
    got = rb.analytics.set_index("isin")
    dropped = isins[5]
    assert np.isnan(got.loc[dropped, "exp_vol"])
    assert np.isnan(got.loc[dropped, "cvar05"])
    others = got.drop(index=dropped)
    want = screen.set_index("isin")["er_sd"].reindex(others.index)
    np.testing.assert_allclose(others["exp_vol"], want, rtol=1e-9)


def test_label_length_mismatch_raises():
    screen, draws, isins, eu = _make_case(seed=13)
    with pytest.raises(ValueError, match="return_draws_isins"):
        _book(screen, draws, isins, eu, return_draws_isins=isins[:-1])


#: Spellings retired on 2026-08-24. Each was a byte-identical duplicate of the
#: canonical name beside it, and neither frame may carry one again.
RETIRED_ALIASES: tuple[tuple[str, str], ...] = (
    ("weight", "book_weight"),
    ("expected_sharpe", "expected_sharpe_ratio"),
)


def test_risk_frames_carry_one_column_per_quantity():
    """One run must not export the same book under two schemas -- or twice under one.

    Two defects, one test. The first was the frames DISAGREEING: ``rb.book``
    carried ``weight`` while ``rb.analytics`` carried ``book_weight``, so which
    spelling an export had depended on which path produced it. Stamping both
    names on both frames fixed that and created the second: every quantity
    shipped TWICE, byte-identical, and ``export_unique_columns`` could not see it
    because that gate checks duplicate NAMES rather than duplicate CONTENT.

    The aliases are now gone at the source. This test pins their ABSENCE, not
    merely the presence of the canonical names -- asserting only the latter is
    what let the duplication survive a passing suite.
    """
    screen, draws, isins, eu = _make_case(seed=19)
    rb = _book(screen, draws, isins, eu, return_draws_isins=isins)

    for label, frame in (("analytics", rb.analytics), ("book", rb.book)):
        cols = list(frame.columns)
        assert len(cols) == len(set(cols)), f"{label} has duplicate columns"
        assert "book_weight" in cols, f"{label} lacks book_weight"
        assert "expected_sharpe_ratio" in cols, f"{label} lacks expected_sharpe_ratio"
        for alias, canonical in RETIRED_ALIASES:
            assert alias not in cols, (
                f"{label} still carries the retired alias {alias!r}; "
                f"{canonical!r} is the only name for that quantity"
            )

    # The book's sizing must actually be there, not a column of zeros -- which is
    # the half of the original defect that the de-duplication must not undo.
    if len(rb.book):
        assert rb.book["book_weight"].sum() > 0.99


def test_no_exported_frame_carries_a_duplicated_column_pair(tmp_path, monkeypatch):
    """Byte-identical column pairs must be caught by a gate, not by a reader.

    ``export_unique_columns`` passes on a frame carrying the same numbers under
    two names, correctly -- it checks names. ``export_duplicate_content`` is the
    sibling gate that checks values, and it WARNS rather than blocks: two
    genuinely all-zero columns are legitimate and must not abort an export that
    has already paid for a fit.
    """
    import pymc_kalman_filter_pt_v2 as v2

    report = v2.GateReport()
    frames = {
        "clean_v2": pd.DataFrame(
            {"isin": ["a", "b"], "starr": [1.0, 2.0], "exp_vol": [0.3, 0.4]}
        ),
        "dupe_v2": pd.DataFrame(
            {
                "isin": ["a", "b"],
                "book_weight": [0.5, 0.5],
                "weight": [0.5, 0.5],  # the retired alias, reintroduced
            }
        ),
    }
    # Two side effects to contain, both of which this test would otherwise leave
    # in the repository:
    #
    #   * `export_analytics` writes its CSVs UNCONDITIONALLY -- only the DATABASE
    #     write is gated -- so it needs a tmp results root.
    #   * `write_analytics_ddl_v2` renders into `sql_scripts/analytics/`, which is
    #     deliberate (that DDL is the reviewable, committed schema) and therefore
    #     must be stubbed rather than redirected.
    monkeypatch.setattr(v2, "write_analytics_ddl_v2", lambda *a, **k: tmp_path)
    v2.export_analytics(
        frames,
        v2.KalmanRunConfigV2(write_analytics=False, results_dir=str(tmp_path)),
        report,
        run_id="t0",
    )
    gate = next(r for r in report.results if r.name == "export_duplicate_content")
    assert not gate.passed
    assert not gate.blocking, "a duplicate-content warning must never abort an export"
    assert "dupe_v2" in gate.detail
    assert "book_weight" in gate.detail and "weight" in gate.detail
    assert "clean_v2" not in gate.detail


def test_export_blocks_frames_with_duplicate_columns():
    """A duplicated column name must fail a gate, not crash the export.

    This bug shipped twice from the same cause: `compute_cvar_aware_book` used
    to emit both `expected_sharpe` and `expected_sharpe_ratio`, and BOTH places
    that build an export frame renamed one onto the other. pandas allows it
    silently; the ranking-range gate then hands `pd.to_numeric` a DataFrame and
    the export dies with "arg must be a list, tuple, 1-d array, or Series" --
    after the fit has already been paid for.

    The alias is gone as of 2026-08-24, so this now guards the class of defect
    rather than one instance of it: any future rename that collides must still
    be stopped BEFORE the loops that would raise.
    """
    import pymc_kalman_filter_pt_v2 as v2

    report = v2.GateReport()
    frames = {
        "good_v2": pd.DataFrame([[1, 2.0]], columns=["isin", "expected_sharpe_ratio"]),
        "bad_v2": pd.DataFrame(
            [[1, 2.0, 3.0]],
            columns=["isin", "expected_sharpe_ratio", "expected_sharpe_ratio"],
        ),
    }
    counts = v2.export_analytics(
        frames, v2.KalmanRunConfigV2(write_analytics=False), report, run_id="t0"
    )
    gate = next(r for r in report.results if r.name == "export_unique_columns")
    assert not gate.passed
    assert gate.blocking
    assert "bad_v2" in gate.detail and "expected_sharpe_ratio" in gate.detail
    # It must stop BEFORE the loops that would raise, not write a partial export.
    assert counts == {}


def test_tail_risk_has_no_expected_upside_leg():
    """tail_risk must not move when only expected_upside changes.

    The removed dispersion leg made it do exactly that, which is how a favourable
    tail was counted in both the numerator and the denominator of STARR.
    """
    screen, draws, isins, eu = _make_case(seed=17)
    base = _book(screen, draws, isins, eu, return_draws_isins=isins)
    bumped = screen.copy()
    bumped["expected_upside"] = bumped["expected_upside"] + 0.25
    shifted = _book(bumped, draws, isins, eu, return_draws_isins=isins)
    a = base.analytics.set_index("isin")["tail_risk"]
    b = shifted.analytics.set_index("isin")["tail_risk"]
    np.testing.assert_allclose(b.reindex(a.index), a, rtol=1e-12)
