"""The identity block every per-ISIN v2 export frame carries.

What each test pins:

* The block is joined **by ISIN, never by position**. `run_screen` returns its
  frame sorted by `expected_upside` while `panel.frame` stays in universe order,
  so a positional attach hands every name someone else's country and sector while
  every row count still matches. That is the same class of defect the risk
  columns already shipped once.
* A frame's own column is **kept, not overwritten**, so two stages cannot end up
  publishing different values for one name under one name.
* Declared SQL types beat inferred ones. Inference cannot recover a DATE from a
  `datetime64[ns]` (it says TIMESTAMP) or a text column that happens to be all
  NULL (it says DOUBLE PRECISION).
* Frames with no ISIN axis are skipped **deliberately** and stay untouched.
* The Python SSOT and the SQL registration in `pml_df_metadata_populate.sql` §7m
  list the same 42 columns. Nothing enforces this at runtime — the catalogue
  coverage check scans only `feat_`/`observed_`/`n_` prefixes — so it is checked
  here or not at all.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd
import pytest

import pymc_kalman_filter_pt_v2 as v2

SEED = 20260826
_REPO = pathlib.Path(__file__).resolve().parent.parent


def _source(n: int = 12) -> pd.DataFrame:
    """A stand-in for ``panel.frame`` carrying the whole identity block."""
    isins = np.array([f"TEST{i:08d}" for i in range(n)])
    cols: dict[str, object] = {}
    for name, sql_type in v2.EXPORT_IDENTITY_COLUMNS:
        if name == "isin":
            cols[name] = isins
        elif sql_type == "TEXT":
            cols[name] = [f"{name}#{i}" for i in range(n)]
        elif sql_type == "DATE":
            cols[name] = pd.to_datetime(
                [f"2026-0{(i % 9) + 1}-1{i % 9}" for i in range(n)]
            )
        else:
            cols[name] = np.arange(1, n + 1, dtype="int64")
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# The SSOT itself
# ---------------------------------------------------------------------------


def test_ssot_has_42_unique_columns_with_known_types():
    names = [c for c, _ in v2.EXPORT_IDENTITY_COLUMNS]
    assert len(names) == 42
    assert len(set(names)) == 42, "duplicate column in EXPORT_IDENTITY_COLUMNS"
    assert names[0] == "isin", "isin must lead: it is the join key"
    assert set(t for _, t in v2.EXPORT_IDENTITY_COLUMNS) == {
        "TEXT",
        "DATE",
        "INTEGER",
        "DOUBLE PRECISION",
    }


def test_no_identity_column_is_scanned_by_the_catalogue_coverage_check():
    """None matches feat_/observed_/n_, which is WHY §7m has to be deliberate.

    If one ever did, tagging it in the catalogue would start being enforced —
    and, more to the point, forgetting to tag it would start raising
    MISSING_FROM_CATALOGUE instead of passing in silence.
    """
    for name, _ in v2.EXPORT_IDENTITY_COLUMNS:
        assert not name.startswith(("feat_", "observed_", "n_")), name


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------


def test_identity_is_joined_by_isin_not_by_position():
    """The defect this whole helper exists to make impossible."""
    src = _source()
    # Reversed relative to the source, exactly as a sorted screen would be.
    frame = pd.DataFrame(
        {"isin": src["isin"].to_numpy()[::-1], "expected_upside": np.arange(12) / 10}
    )
    out = v2.attach_identity_columns(frame, src, label="screen")
    want = src.set_index("isin")["ticker"]
    got = out.set_index("isin")["ticker"]
    pd.testing.assert_series_equal(got.reindex(want.index), want, check_names=False)


def test_existing_columns_are_kept_not_overwritten():
    src = _source()
    frame = pd.DataFrame(
        {
            "isin": src["isin"],
            "sector": ["MINE"] * len(src),
            "market_cap": np.full(len(src), 999.0),
        }
    )
    out = v2.attach_identity_columns(frame, src, label="own")
    assert out["sector"].eq("MINE").all()
    assert out["market_cap"].eq(999.0).all()


def test_identity_columns_lead_the_frame():
    src = _source()
    frame = pd.DataFrame({"zzz": np.arange(12), "isin": src["isin"]})
    out = v2.attach_identity_columns(frame, src, label="ordered")
    lead = [c for c, _ in v2.EXPORT_IDENTITY_COLUMNS]
    assert list(out.columns)[: len(lead)] == lead
    assert out.columns[-1] == "zzz"


def test_missing_source_columns_degrade_rather_than_raise():
    """A catalogue gap is something to report, not a reason to lose a fit."""
    src = _source().drop(columns=["unit_name", "exchange_name"])
    frame = pd.DataFrame({"isin": src["isin"], "x": 1.0})
    out = v2.attach_identity_columns(frame, src, label="partial")
    assert "unit_name" not in out.columns
    assert "ticker" in out.columns


def test_frame_without_isin_is_returned_unchanged():
    src = _source()
    diag = pd.DataFrame({"parameter": ["nu", "sigma_total"], "r_hat": [1.0, 1.002]})
    assert v2.attach_identity_columns(diag, src, label="diag").equals(diag)


# ---------------------------------------------------------------------------
# Dtypes and the DDL
# ---------------------------------------------------------------------------


def test_declared_types_survive_into_the_ddl(tmp_path):
    src = _source()
    frame = v2.attach_identity_columns(
        pd.DataFrame({"isin": src["isin"], "expected_upside": 0.1}), src
    )
    out = tmp_path / "probe.sql"
    v2.write_analytics_ddl_v2(frame, table="probe", out_path=out)
    declared = dict(
        re.findall(
            r'"([a-z_0-9]+)"\s+(TEXT|DATE|INTEGER|BIGINT|DOUBLE PRECISION|REAL|'
            r"BOOLEAN|TIMESTAMP)",
            out.read_text(encoding="utf-8"),
        )
    )
    for name, sql_type in v2.EXPORT_IDENTITY_COLUMNS:
        assert declared[name] == sql_type, f"{name}: {declared[name]} != {sql_type}"


def test_all_null_text_column_still_declares_text(tmp_path):
    """Inference would call this DOUBLE PRECISION. The declared type must win."""
    src = _source()
    src["country_name"] = np.nan
    frame = v2.attach_identity_columns(pd.DataFrame({"isin": src["isin"]}), src)
    out = tmp_path / "probe.sql"
    v2.write_analytics_ddl_v2(frame, table="probe", out_path=out)
    assert '"country_name" TEXT' in out.read_text(encoding="utf-8")


def test_integer_ranks_are_nullable_not_float():
    """``Int64``, so a name outside a ranking universe can be NULL.

    numpy ``int64`` cannot hold that, and routing it through ``float64`` is how
    an integer rank ends up declared DOUBLE PRECISION.
    """
    src = _source()
    src.loc[0, "market_cap_global_r"] = np.nan
    frame = v2.attach_identity_columns(pd.DataFrame({"isin": src["isin"]}), src)
    assert str(frame["market_cap_global_r"].dtype) == "Int64"
    assert frame["market_cap_global_r"].isna().sum() == 1


# ---------------------------------------------------------------------------
# Frame-level application
# ---------------------------------------------------------------------------


def test_non_isin_frames_are_skipped_by_name():
    src = _source()
    frames = {
        "10_screen_results_v2": pd.DataFrame({"isin": src["isin"], "a": 1.0}),
        "09_diagnostics_v2": pd.DataFrame({"parameter": ["nu"], "r_hat": [1.0]}),
        "09b_comparison_v2": pd.DataFrame({"arm": ["baseline"], "elpd": [-1.0]}),
        v2._GATE_REPORT_KEY: pd.DataFrame({"gate": ["r_hat"], "status": ["PASS"]}),
    }
    out = v2._attach_identity_frames(frames, src)
    assert "ticker" in out["10_screen_results_v2"].columns
    for skipped in ("09_diagnostics_v2", "09b_comparison_v2", v2._GATE_REPORT_KEY):
        assert out[skipped].equals(frames[skipped])


def test_redundant_duplicate_column_is_dropped():
    """`p_upside_pos` is byte-identical to `prob_pos` and one suffix from the
    primary ranking column `p_upside_pos_cond`."""
    frame = pd.DataFrame(
        {
            "isin": ["A", "B"],
            "prob_pos": [0.9, 1.0],
            "p_upside_pos": [0.9, 1.0],
            "p_upside_pos_cond": [0.6, 0.7],
        }
    )
    out = v2.drop_redundant_export_columns(frame, label="t")
    assert "p_upside_pos" not in out.columns
    assert "prob_pos" in out.columns and "p_upside_pos_cond" in out.columns


def test_dropping_a_column_that_diverged_warns(caplog):
    """The warning is the point: if the twin assumption breaks, look."""
    frame = pd.DataFrame(
        {"isin": ["A"], "prob_pos": [0.9], "p_upside_pos": [0.1]}
    )
    with caplog.at_level("WARNING"):
        v2.drop_redundant_export_columns(frame, label="t")
    assert any("NOT identical" in r.message for r in caplog.records)


def test_exp_vol_and_its_rename_are_dropped_in_favour_of_er_sd():
    """One quantity, one name — and the drop verifies the equality on the way out.

    ``exp_vol`` and ``er_sd`` are the pooled sd of the same Monte-Carlo draws;
    that identity IS ``compute_cvar_aware_book``'s ISIN-alignment self-check, so a
    warning here is the same finding it raises, caught at the export boundary.
    """
    frame = pd.DataFrame(
        {
            "isin": ["A", "B"],
            "er_sd": [0.20, 0.31],
            "exp_vol": [0.20, 0.31],
            "expected_vol_kalman": [0.20, 0.31],
            "expected_upside_sd": [0.01, 0.02],
        }
    )
    out = v2.drop_redundant_export_columns(frame, label="t")
    assert "exp_vol" not in out.columns
    assert "expected_vol_kalman" not in out.columns
    assert "er_sd" in out.columns
    # The estimation-uncertainty view is a DIFFERENT quantity and must survive.
    assert "expected_upside_sd" in out.columns


def test_every_redundant_column_declares_the_twin_it_is_checked_against():
    """The mapping is the SSOT, so a name cannot be dropped unverified.

    This used to be a tuple of names beside a hard-coded ``twin`` dict inside the
    function, so a name added to the tuple silently got no verification at all.
    """
    assert all(v2.EXPORT_REDUNDANT_COLUMNS.values())
    # A canonical name must never itself be on the drop list.
    assert not set(v2.EXPORT_REDUNDANT_COLUMNS.values()) & set(
        v2.EXPORT_REDUNDANT_COLUMNS
    )


# ---------------------------------------------------------------------------
# Export gates: finiteness and duplicated content
# ---------------------------------------------------------------------------


def _gate(frames, tmp_path, monkeypatch):
    """Run the export gates over ``frames`` with every write redirected.

    ``export_analytics`` writes CSVs under ``results_dir`` and regenerates
    ``sql_scripts/analytics/<frame>.sql`` at a path relative to the CWD, so the
    chdir is not tidiness -- without it a toy two-column fixture overwrites the
    repository's real generated DDL.
    """
    import dataclasses

    monkeypatch.chdir(tmp_path)
    (tmp_path / "sql_scripts" / "analytics").mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(
        v2.KalmanRunConfigV2.from_env(),
        results_dir=str(tmp_path / "results"),
        write_analytics=False,
    )
    report = v2.GateReport()
    v2.export_analytics(frames, cfg, report, run_id="test00000000")
    return {r.name: r for r in report.results}


def test_finiteness_verdict_names_the_column_not_only_the_frame(tmp_path, monkeypatch):
    """Run 6efb530d5881 reported ``offending: ['15b_decision_analytics_v2']``.

    One column at +inf on 9.6 % of rows, and finding it meant a forensic pass over
    a 5 MB CSV. A gate that can block an export has to say what blocked it.
    """
    frames = {
        "10_screen_results_v2": pd.DataFrame(
            {"isin": ["A", "B"], "kelly_max_feasible": [np.inf, 2.0],
             "other": [1.0, 2.0]}
        )
    }
    res = _gate(frames, tmp_path, monkeypatch)["export_finite"]
    assert not res.passed
    assert "kelly_max_feasible" in res.detail
    assert "1 x +inf" in res.detail


def test_nan_is_not_a_finiteness_failure(tmp_path, monkeypatch):
    """A NaN is a SQL NULL — "not applicable" — and always passed this gate.

    That is the whole reason NULL + a boolean is the right encoding for an
    unbounded Kelly fraction, and an infinity is not.
    """
    frames = {
        "10_screen_results_v2": pd.DataFrame(
            {"isin": ["A", "B"], "kelly_max_feasible": [np.nan, 2.0],
             "kelly_unbounded": [True, False]}
        )
    }
    assert _gate(frames, tmp_path, monkeypatch)["export_finite"].passed


def test_declared_alias_is_reported_as_declared_and_does_not_warn(tmp_path, monkeypatch):
    """A warning that always fires is not a warning.

    Run 6efb530d5881 reported five frames and thirteen pairs, eleven of which had
    a settled reason recorded elsewhere in the tree — and a genuinely new
    duplicate would have arrived in the middle of that list.
    """
    frames = {
        "04_panel_frame_v2": pd.DataFrame(
            {"isin": ["A", "B"],
             "n_analysts_1w": [3.0, 5.0],
             "price_target_num_1w_ago": [3.0, 5.0],
             "unrelated": [1.0, 9.0]}
        )
    }
    res = _gate(frames, tmp_path, monkeypatch)["export_duplicate_content"]
    assert res.passed
    assert "declared and re-verified" in res.detail


def test_undeclared_duplicate_still_warns(tmp_path, monkeypatch):
    """The point of declaring is that the undeclared list stays actionable."""
    frames = {
        "04_panel_frame_v2": pd.DataFrame(
            {"isin": ["A", "B"], "a_col": [1.0, 2.0], "b_col": [1.0, 2.0]}
        )
    }
    res = _gate(frames, tmp_path, monkeypatch)["export_duplicate_content"]
    assert not res.passed
    assert "a_col == b_col" in res.detail


def test_declared_pair_that_stops_being_equal_is_reported(tmp_path, monkeypatch, caplog):
    """Declaring is not suppressing.

    Three declared pairs are equal *empirically* — distinct ``pml_df`` vendor
    columns carrying identical data — so they are re-verified every run. This is
    what says so on the day the vendor separates them.
    """
    frames = {
        "04_panel_frame_v2": pd.DataFrame(
            {"isin": ["A", "B"],
             "price_1w_ago": [10.0, 20.0],
             "price_5d_ago": [10.0, 20.5]}
        )
    }
    with caplog.at_level("WARNING"):
        res = _gate(frames, tmp_path, monkeypatch)["export_duplicate_content"]
    assert not res.passed
    assert "no longer equal" in " ".join(r.message for r in caplog.records)


def test_every_declared_pair_names_a_reason():
    """A declaration without a reason is a suppression."""
    for key, entries in v2.EXPORT_DECLARED_ALIASES.items():
        for col_a, col_b, why in entries:
            assert col_a != col_b
            assert why.strip(), f"{key}: {col_a} == {col_b} has no reason"


# ---------------------------------------------------------------------------
# Python <-> SQL agreement
# ---------------------------------------------------------------------------


def test_sql_section_7m_tags_exactly_the_python_ssot():
    """Nothing enforces this at runtime, so it is enforced here.

    §7m adds `kalman_pt_v2` to `model_targets` for the identity block. Because no
    identity column matches the coverage check's `feat_`/`observed_`/`n_` scan, a
    column present in one list and absent from the other raises nothing, ever.
    """
    sql = (_REPO / "pml_df_metadata_populate.sql").read_text(encoding="utf-8")
    start = sql.index("-- 7m.2 Tag all 42")
    end = sql.index("AND NOT ('kalman_pt_v2' = ANY (model_targets));", start)
    tagged = set(re.findall(r"'([a-z_0-9]+)'", sql[start:end])) - {"kalman_pt_v2", "TEXT"}
    assert tagged == set(v2.EXPORT_IDENTITY_NAMES)


def test_generated_analytics_ddls_carry_the_identity_block():
    """Every generated per-ISIN DDL on disk agrees with the SSOT.

    These files are regenerated on every export, so a stale one means the
    committed schema and the code disagree about what the table looks like.
    """
    d = _REPO / "sql_scripts" / "analytics"
    checked = 0
    for path in sorted(d.glob("*_v2.sql")):
        text = path.read_text(encoding="utf-8")
        if "Generated by pymc_kalman_filter_pt_v2.py" not in text:
            continue  # hand-written (09_gate_report_v2)
        m = re.search(r'CREATE TABLE analytics\."[^"]+"\s*\(\s*(.*?)\n\);', text, re.S)
        assert m, path.name
        cols = re.findall(r'"([^"]+)"\s+([A-Z ]+?)(?:,\s*\n|\s*$)', m.group(1))
        names = [c for c, _ in cols]
        if "isin" not in names:
            continue  # no ISIN axis
        declared = dict(cols)
        for col, sql_type in v2.EXPORT_IDENTITY_COLUMNS:
            assert col in declared, f"{path.name} is missing {col}"
            assert declared[col] == sql_type, (
                f"{path.name}: {col} declared {declared[col]}, expected {sql_type}"
            )
        assert names[:42] == list(v2.EXPORT_IDENTITY_NAMES), (
            f"{path.name}: identity block must lead, in SSOT order"
        )
        for col in v2.EXPORT_REDUNDANT_COLUMNS:
            assert col not in names, f"{path.name} still declares {col}"
        checked += 1
    assert checked >= 6, f"expected at least six per-ISIN DDLs, checked {checked}"


# ---------------------------------------------------------------------------
# Provenance scoping (post-run analysis item 02)
# ---------------------------------------------------------------------------


def test_source_revision_scope_excludes_generated_analytics():
    """The export regenerates `sql_scripts/analytics/*.sql` on every run.

    Including that directory in the dirty check let a run dirty its own tree and
    then report itself unpinned, which is how `source_dirty` came to be TRUE on
    essentially every run and therefore to carry no information.
    """
    paths = v2._SOURCE_REVISION_PATHS
    assert not any(p.startswith("sql_scripts/analytics") for p in paths)
    assert "probabilistic_ml_model/" in paths
    assert "pymc_kalman_filter_pt_v2.py" in paths
    # The dashboard hand-mirrors RiskBookModel's tail-risk constants, so a change
    # there can make the card and the book disagree about a name's downside.
    assert any(p.startswith("dashboards/geib") for p in paths)


@pytest.mark.parametrize("path", list(v2._SOURCE_REVISION_PATHS))
def test_every_scoped_source_path_exists(path):
    """A typo'd path silently narrows the check to nothing and reads as clean."""
    assert (_REPO / path).exists(), f"{path} does not exist; the scope is a no-op"
