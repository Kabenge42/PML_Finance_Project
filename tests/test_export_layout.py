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
