"""The results tree: one SSOT, two roots, every stem routed.

What each test pins:

* **The data paths can import it.** This module resolves artifact paths for
  ``pymc_kalman_filter_pt_v2.py`` and ``kalman_portfolio.py``, neither of which
  may pay for matplotlib. The rule used to live in ``kalman_shared``, which does,
  and the consequence was that every figure went to a section directory and every
  CSV went to the results root.
* **v1 and v2 do not share a root.** Both workflows resolved
  ``KALMAN_PT_RESULTS_DIR``, so a v2 run scattered its frames through v1's tree
  under names one suffix from v1's own.
* **Every stem either routes or is deliberately miscellaneous.** A stem that
  silently lands in ``00_misc`` is an artifact nobody will find again.
* **``kalman_shared`` still answers identically**, since ~200 call sites read its
  private aliases rather than this module.
"""
from __future__ import annotations

import pandas as pd
import pytest

from probabilistic_ml_model.export_layout import (
    DEFAULT_RESULTS_DIRNAME_V1,
    DEFAULT_RESULTS_DIRNAME_V2,
    EXPORT_DIR_ALIASES,
    EXPORT_MISC_DIR,
    EXPORT_SECTION_DIRS,
    RESULTS_DIR_ENV_V1,
    RESULTS_DIR_ENV_V2,
    V2_ONLY_SECTION_DIRS,
    export_dir_for,
    resolve_results_root,
    section_path,
)

#: Every stem the two workflows and the replay actually write, and where it goes.
#: Written out rather than derived, so a change to the routing rule has to change
#: this table too and cannot pass by agreeing with itself.
ROUTES = {
    # ---- v2 workflow -------------------------------------------------------
    "04_panel_frame_v2": "04_panel",
    "09_diagnostics_v2": "09_diagnostics",
    "09b_comparison_v2": "09b_comparison",
    "09_gate_report_v2": "09_gates",
    "10_screen_results_v2": "10_screen",
    "10_screen_mc_summary_v2": "10_screen",
    "10b_risk_analytics_v2": "10b_risk",
    "10b_risk_book_v2": "10b_risk",
    "kalman_filtered_price_targets_v2": "10c_analytics",
    "07_forecast_handoff_v2": "07_posterior",
    "15_forecast_summary_v2": "15_forecast",
    "15b_decision_analytics_v2": "15b_decision",
    # ---- the replay --------------------------------------------------------
    "15c_forecast_summary": "15c_forecast",
    "15c_forecast_engines": "15c_forecast",
    "15d_factor_share_sweep": "15d_sweeps",
    "15d_multiplier_sweep": "15d_sweeps",
    "15e_decision_books": "15e_books",
    "15e_book_agreement": "15e_books",
    "14b_group_signals": "14b_recommendations",
    "14b_name_actions": "14b_recommendations",
    "14b_size_down_watch": "14b_recommendations",
    "09_gate_report_portfolio": "09_gates",
    # ---- v1, which must not move -------------------------------------------
    "10c_kalman_results": "10c_analytics",
    "14b_recommendations_01_table": "14b_recommendations",
    "02_eda_07_something": "02_eda",
    "04b_audit_03_decay_ladder": "04b_audit",
}


@pytest.mark.parametrize("stem,expected", sorted(ROUTES.items()))
def test_every_shipped_stem_routes_to_its_section(stem, expected):
    assert export_dir_for(stem) == expected


def test_an_unknown_stem_is_miscellaneous_rather_than_an_error():
    """Burying a one-off frame beats refusing to write it."""
    assert export_dir_for("something_nobody_declared") == EXPORT_MISC_DIR


def test_the_longer_prefix_wins():
    """`04b_audit` sits beside `04_panel`, and `04b` is not `04`.

    Without the longest-match rule every §4b artifact lands in `00_misc`, which is
    how the decay ladder would have gone missing.
    """
    assert export_dir_for("04b_audit_01") == "04b_audit"
    assert export_dir_for("04_panel_frame") == "04_panel"


def test_the_two_workflows_do_not_share_a_root_or_a_variable():
    """The whole reason this module names both."""
    assert DEFAULT_RESULTS_DIRNAME_V1 != DEFAULT_RESULTS_DIRNAME_V2
    assert RESULTS_DIR_ENV_V1 != RESULTS_DIR_ENV_V2


def test_v2_only_sections_are_real_sections():
    """The migration decides ownership from this set; a typo would silently
    orphan a whole stage in v1's tree."""
    assert V2_ONLY_SECTION_DIRS <= set(EXPORT_SECTION_DIRS)


def test_every_alias_target_is_a_real_section():
    for key, target in EXPORT_DIR_ALIASES.items():
        assert target in EXPORT_SECTION_DIRS, f"{key} -> {target} is not a section"


def test_section_path_creates_the_directory_and_never_the_root_file(tmp_path):
    path = section_path(tmp_path, "10b_risk_book_v2")
    assert path.parent.name == "10b_risk"
    assert path.parent.is_dir()
    assert path == tmp_path / "10b_risk" / "10b_risk_book_v2.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    assert not [p for p in tmp_path.iterdir() if p.is_file()]


def test_a_relative_root_is_anchored_at_the_project_not_the_cwd(tmp_path, monkeypatch):
    """A run launched from a notebook, an IDE and a shell must write to one tree."""
    monkeypatch.chdir(tmp_path)
    root = resolve_results_root(None, env_value=None,
                                default_dirname=DEFAULT_RESULTS_DIRNAME_V2)
    assert root.is_absolute()
    assert root.name == DEFAULT_RESULTS_DIRNAME_V2
    assert tmp_path not in root.parents


def test_an_explicit_value_beats_the_environment():
    """This is what makes `main(config=...)` redirect artifacts."""
    root = resolve_results_root("/tmp/explicit", env_value="/tmp/from_env")
    assert root.name == "explicit"


def test_kalman_shared_answers_identically():
    """~200 call sites read the private aliases; they must not have drifted."""
    shared = pytest.importorskip(
        "probabilistic_ml_model.visualizations.kalman_shared"
    )
    assert shared._EXPORT_SECTION_DIRS is EXPORT_SECTION_DIRS
    assert shared._EXPORT_MISC_DIR == EXPORT_MISC_DIR
    for stem in ROUTES:
        assert shared._export_dir_for(stem) == export_dir_for(stem)


# ---------------------------------------------------------------------------
# The import direction
#
# `kalman_portfolio` imports `pymc_kalman_filter_pt_v2`, never the reverse. The
# whole 2026-08-31 split rests on that: the fit script ends at the posterior and
# the decision layer replays off the handoff, so a back-import would both close a
# cycle and mean the fit had started deciding again.
# ---------------------------------------------------------------------------

from dataclasses import fields as _dc_fields  # noqa: E402
import pathlib as _pathlib  # noqa: E402
import re as _re  # noqa: E402

_REPO_ROOT = _pathlib.Path(__file__).resolve().parent.parent


def test_the_fit_script_never_imports_the_decision_layer():
    src = (_REPO_ROOT / "pymc_kalman_filter_pt_v2.py").read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in src.splitlines()
        if _re.match(r"\s*(from|import)\s+kalman_portfolio\b", line)
    ]
    assert not offenders, (
        "pymc_kalman_filter_pt_v2 imports kalman_portfolio: "
        f"{offenders}. The dependency runs the other way."
    )


def test_the_decision_layer_does_import_the_fit_script():
    """The positive half. A split where neither side imports the other means the
    shared SSOTs (gates, identity block, provenance, out-of-support) have been
    copied rather than shared."""
    src = (_REPO_ROOT / "kalman_portfolio.py").read_text(encoding="utf-8")
    assert _re.search(r"^from pymc_kalman_filter_pt_v2 import", src, _re.M)


def test_the_fit_script_no_longer_defines_the_decision_stages():
    src = (_REPO_ROOT / "pymc_kalman_filter_pt_v2.py").read_text(encoding="utf-8")
    for name in ("run_screen", "run_risk_book", "run_forecast_layer",
                 "_book_group_labels", "class ScreenDraws"):
        pattern = rf"^(def {name}\(|{name})" if name.startswith("class") else \
            rf"^def {name}\("
        assert not _re.search(pattern, src, _re.M), f"{name} is still defined here"


def test_the_decision_layer_defines_them_instead():
    src = (_REPO_ROOT / "kalman_portfolio.py").read_text(encoding="utf-8")
    for name in ("run_screen", "run_risk_book"):
        assert _re.search(rf"^def {name}\(", src, _re.M), f"{name} is missing"
    assert _re.search(r"^class ScreenDraws", src, _re.M)


# ---------------------------------------------------------------------------
# The config split
#
# 25 decision-layer fields left `KalmanRunConfigV2` with the stages that read
# them. They are DELETED rather than deprecated: a knob that can still be set on
# the fit's config and no longer reaches anything is `--write is accepted but
# does nothing` in dataclass form, which is the pattern this whole change retires.
# ---------------------------------------------------------------------------

_MOVED_KNOBS = (
    "enable_forecast_error_shrinkage", "forecast_error_multiplier",
    "forecast_error_n_exponent", "mc_horizon", "mc_rho", "cvar_alpha", "k_book",
    "book_min_weight", "p_long", "mcap_global_r_max", "tail_risk_vol_floor_k",
    "weight_cap", "group_caps",
    "gate_shrinkage_slope_lo", "gate_shrinkage_slope_hi",
    "gate_shrinkage_center_shift_max", "gate_shrinkage_rho_max",
    "gate_shrinkage_revision_min_pp",
)


def test_the_fit_config_no_longer_carries_the_decision_knobs():
    import pymc_kalman_filter_pt_v2 as v2

    fields = {f.name for f in _dc_fields(v2.KalmanRunConfigV2)}
    still_there = sorted(set(_MOVED_KNOBS) & fields)
    assert not still_there, (
        f"{still_there} can still be set on KalmanRunConfigV2 and reaches nothing"
    )


def test_the_decision_config_carries_them_instead():
    import kalman_portfolio as kp

    fields = {f.name for f in _dc_fields(kp.KalmanPortfolioConfig)}
    missing = sorted(set(_MOVED_KNOBS) - fields)
    assert not missing, f"{missing} moved off one config and onto neither"


def test_the_handoffs_thinning_budget_stayed_with_the_writer():
    """`forecast_scenarios` is the one forecast knob the fit still owns.

    It sizes the file this script writes, which is its decision; everything else
    is a decision about what a replay does with that file."""
    import pymc_kalman_filter_pt_v2 as v2

    assert "forecast_scenarios" in {f.name for f in _dc_fields(v2.KalmanRunConfigV2)}


def test_the_validation_moved_with_the_fields():
    """A negative multiplier gives a negative variance and a gain above 1, i.e.
    ANTI-shrinkage — a plausible-looking number several stages later."""
    import kalman_portfolio as kp

    with pytest.raises(ValueError, match="forecast_error_multiplier"):
        kp.KalmanPortfolioConfig(forecast_error_multiplier=-0.5)
    with pytest.raises(ValueError, match="must be increasing"):
        kp.KalmanPortfolioConfig(gate_shrinkage_slope_lo=0.99)


def test_the_postrun_skill_still_finds_the_eligibility_threshold():
    """It reads the default by REGEX out of a source file, so a field that moves
    file returns the fallback silently — the exact staleness that function was
    written to prevent."""
    import re as _re2

    src = (_REPO_ROOT / "kalman_portfolio.py").read_text(encoding="utf-8")
    m = _re2.search(r"^\s*mcap_global_r_max:\s*float\s*=\s*([0-9.eE+-]+)", src, _re2.M)
    assert m, "the postrun skill's regex no longer matches; update analyze.py"

    import kalman_portfolio as kp

    assert float(m.group(1)) == kp.KalmanPortfolioConfig().mcap_global_r_max


def test_the_fit_config_has_no_field_that_nothing_reads():
    """The property the 25-field removal bought, stated so it stays true.

    A knob that can be set and reaches nothing is the same defect as a flag that
    is accepted and ignored, and it is harder to notice: there is no message, and
    the value looks applied to anyone reading the config back.
    """
    import re as _re3
    from dataclasses import fields as _f

    import pymc_kalman_filter_pt_v2 as v2

    src = (_REPO_ROOT / "pymc_kalman_filter_pt_v2.py").read_text(encoding="utf-8")
    start = src.index("class KalmanRunConfigV2:")
    end = src.index("# §1  Data")
    body, rest = src[start:end], src[:start] + src[end:]
    from_env = body[body.index("def from_env"):]

    dead = [
        f.name for f in _f(v2.KalmanRunConfigV2)
        # read anywhere outside the dataclass body...
        if not _re3.search(rf"\b(?:run_cfg|cfg|self|config)\.{f.name}\b", rest)
        # ...or set by from_env, or reached through a property inside the body
        and not _re3.search(rf"\b{f.name}\s*=", from_env)
        and not _re3.search(rf"self\.{f.name}\b", body)
    ]
    # `fig_width_px` is read by `visualizations.kalman_shared` through a
    # duck-typed config protocol, so no attribute access appears in this file.
    dead = [d for d in dead if d != "fig_width_px"]
    assert not dead, f"{dead} can be set on KalmanRunConfigV2 and reach nothing"
