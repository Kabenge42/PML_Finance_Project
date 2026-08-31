"""Where a Kalman artifact lands: the results tree, as one dependency-free SSOT.

Why this module exists at all
-----------------------------
The artifact tree **is** the workflow — a stage means a section directory, and
``export_dir_for`` is the only thing allowed to turn an artifact stem into a
path. That rule was already written down, but it was enforced in
:mod:`probabilistic_ml_model.visualizations.kalman_shared`, which imports
matplotlib, seaborn, arviz-plots and xarray at module level.

The data paths cannot pay for that. ``pymc_kalman_filter_pt_v2.py`` imports its
figure layer inside a ``try`` precisely so a missing plotly never costs a run its
analytics write, and ``kalman_portfolio.py`` is meant to be cheap enough to run
many times over one fit. So both wrote their CSVs **flat into the results root**
while every figure went into a section directory — one tree, two conventions,
and no single place that knew both.

This module is that place: pathlib and typing only, importable from a data path
without a figure stack behind it. ``kalman_shared`` imports it and keeps its
private names as aliases, so its call sites are untouched.

The two trees
-------------
v1 (``pymc_kalman_filter_pt.py``) and v2 (``pymc_kalman_filter_pt_v2.py``) are
different models producing different numbers under overlapping stem names, and
they must not share a results root. They used to: both resolved
``KALMAN_PT_RESULTS_DIR``, which ``set_env.ps1`` points at v1's tree, so a v2 run
scattered nine flat ``*_v2.csv`` files through v1's sectioned directories. v2 and
its replay now resolve :data:`RESULTS_DIR_ENV_V2` and default to
:data:`DEFAULT_RESULTS_DIRNAME_V2`; v1 is untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = [
    "DEFAULT_RESULTS_DIRNAME_V1",
    "DEFAULT_RESULTS_DIRNAME_V2",
    "EXPORT_DIR_ALIASES",
    "EXPORT_MISC_DIR",
    "EXPORT_SECTION_DIRS",
    "RESULTS_DIR_ENV_V1",
    "RESULTS_DIR_ENV_V2",
    "V2_ONLY_SECTION_DIRS",
    "export_dir_for",
    "resolve_results_root",
    "section_path",
]

#: Artifact roots. Separate on purpose — see the module docstring.
DEFAULT_RESULTS_DIRNAME_V1 = "pymc_kalman_filter_pt_results"
DEFAULT_RESULTS_DIRNAME_V2 = "pymc_kalman_filter_pt_v2_results"
DEFAULT_RESULTS_DIRNAME_PORTFOLIO = "kalman_portfolio_results"

#: The environment variable each workflow resolves its root from.
RESULTS_DIR_ENV_V1 = "KALMAN_PT_RESULTS_DIR"
RESULTS_DIR_ENV_V2 = "KALMAN_V2_RESULTS_DIR"
RESULTS_DIR_ENV_PORTFOLIO = "KALMAN_PORTFOLIO_RESULTS_DIR"

#: Where an artifact goes when no prefix matches. Not an error: a one-off frame
#: is a legitimate thing to write, and burying it is better than refusing it.
EXPORT_MISC_DIR = "00_misc"

#: Results-tree subdirectories, matched against an artifact stem by LONGEST
#: prefix. Both workflows, both figure suites and every bulk frame key off this
#: one tuple.
#:
#: Ordering is the workflow's, not the alphabet's, because the tree is meant to
#: be read top to bottom as the stages ran. Two entries deliberately differ from
#: their section label so the bulk stems land correctly: ``04_panel`` catches
#: ``04_panel_frame`` (exported outside any section) and ``10b_risk`` catches
#: ``10b_risk_analytics`` / ``_book`` / ``_summary`` alongside the section stems.
EXPORT_SECTION_DIRS: tuple[str, ...] = (
    # ---- shared by both workflows -----------------------------------------
    "01_data", "02_eda", "03_features", "04_panel",
    # `04b_audit` is v2's panel-information audit (`run_panel_diagnostics`), the
    # stage v1 has no equivalent of. Safe beside `04_panel` because
    # `export_dir_for` resolves ties with `max(matches, key=len)`, so the longer
    # prefix wins; without the entry every §4b artifact silently lands in
    # `00_misc`, which is how the decay ladder would have gone missing.
    "04b_audit",
    "06_prior", "07_posterior", "08_ppc",
    "09_diagnostics", "09b_comparison",
    # NEW: the gate report is a verdict on the run, not a convergence table, and
    # it is the first artifact anybody opens after a failed export. It had been
    # landing in the results ROOT, which is exactly why a stray flat file was
    # indistinguishable from a stage's output.
    "09_gates",
    "10_screen", "10b_risk", "10c_analytics",
    # ---- v1 only -----------------------------------------------------------
    "10k_universe", "11_single_isin", "11b_single_sv", "12_mingled",
    "12b_mingled_sv", "13_forest", "13b_further_views", "14_summary",
    # ---- the recommendation layer, shared by v1 §14b and the replay --------
    "14b_recommendations",
    # ---- v2 workflow: the forecast and decision layers ---------------------
    # After §14 rather than beside §10b because they are a separate stage, not a
    # variant of the risk book: §10b sizes on posterior upside draws, §15
    # simulates a forward horizon first and §15b decides on THOSE draws.
    "15_forecast", "15b_decision",
    # ---- kalman_portfolio.py: the replay ----------------------------------
    # Continues the v2 numbering so a reader who knows that tree knows where
    # these land, and stays SEPARATE from §15/§15b so a replay can never be
    # mistaken for the fit's own export. The replay runs many times over one
    # fit; §15 is written once by the fit itself.
    "15c_forecast", "15d_sweeps", "15e_books",
    EXPORT_MISC_DIR,
)

#: Stems whose name does not begin with the directory that owns them.
#:
#: Each entry is a quantity whose published name is older than the tree, and
#: renaming the artifact to fit the directory would break a consumer. The alias
#: is the cheaper half of that trade, and it is checked first so it beats the
#: prefix scan.
EXPORT_DIR_ALIASES: dict[str, str] = {
    # v1's analytics frame, exported as `10c_kalman_results`.
    "10c_kalman": "10c_analytics",
    # v2's canonical analytics table. Its name is the DB table's, which cannot
    # carry a section number.
    "kalman_filtered_price_targets": "10c_analytics",
    # The forecast handoff is a persisted POSTERIOR — the four quantities the
    # forward simulation reads — so it belongs with §7 rather than with the §15
    # stage that consumes it.
    "07_forecast_handoff": "07_posterior",
    # Gate reports from both the workflow and the replay.
    "09_gate_report": "09_gates",
    # ---- families whose stems name the QUANTITY, not the stage -------------
    # `15d_factor_share_sweep` and `15d_multiplier_sweep` share no prefix beyond
    # the section number, and `14b_group_signals` / `14b_name_actions` /
    # `14b_size_down_watch` share none with `14b_recommendations`. Renaming the
    # stems to fit would be the wrong half of the trade: they are the published
    # CSV names, referenced from the post-run analysis and read back by consumers,
    # while a directory name is read by nobody but a person browsing the tree.
    #
    # A bare section number as the key is deliberate and safe: the match is
    # `stem == key or stem.startswith(key + '_')`, so `"15c"` catches every
    # `15c_*` stem and cannot reach `15b_*`. v1's own `14b_recommendations_NN_*`
    # figures resolve through the `"14b"` entry to the same directory they always
    # used, so nothing moves for v1.
    "14b": "14b_recommendations",
    "15c": "15c_forecast",
    "15d": "15d_sweeps",
    "15e": "15e_books",
}


#: Sections only the v2 workflow or its replay ever writes.
#:
#: Used by the layout migration to decide ownership without a hand-maintained
#: stem list: a file found in v1's tree belongs to v2 when its name carries the
#: ``_v2`` suffix, or when its stem resolves to one of these. Derived from the
#: SSOT above rather than restated, so adding a section cannot leave the
#: migration behind.
V2_ONLY_SECTION_DIRS: frozenset[str] = frozenset({
    "04b_audit", "09b_comparison", "09_gates",
    "15_forecast", "15b_decision",
    "15c_forecast", "15d_sweeps", "15e_books",
})

#: Sections the DECISION LAYER owns, i.e. `kalman_portfolio.py` and nothing else.
#:
#: Its migration moves a file out of v2's tree when the stem resolves to one of
#: these, or when the stem ends `_portfolio`. Ownership is provable rather than
#: guessed, and it had to be checked rather than assumed: neither v2 nor v1
#: writes any `14b_*` stem into the v2 tree (v1 has its own `14b_recommendations`
#: under its own root), and `09_gate_report_portfolio` already carries a suffix
#: that distinguishes it from `09_gate_report_v2` in the shared `09_gates`
#: directory -- which is why `09_gates` is NOT listed here.
#:
#: `10_screen`, `10b_risk` and `10c_analytics` joined on 2026-08-31, when the
#: screen and the risk book moved. They are not in `V2_ONLY_SECTION_DIRS`
#: because v1 writes them too, under its own root.
PORTFOLIO_ONLY_SECTION_DIRS: frozenset[str] = frozenset({
    "10_screen", "10b_risk", "10c_analytics", "14b_recommendations",
    "15c_forecast", "15d_sweeps", "15e_books",
})

#: Sections the v2 FIT still writes, after the decision layer moved out.
#:
#: Stated rather than derived by subtraction, because the two sets overlap:
#: `09_gates` holds one report from each side and `04_panel` is read by the
#: replay and written by the fit.
V2_FIT_SECTION_DIRS: frozenset[str] = frozenset({
    "01_data", "02_eda", "03_features", "04_panel", "04b_audit",
    "06_prior", "07_posterior", "08_ppc", "09_diagnostics", "09b_comparison",
    "09_gates",
})


def export_dir_for(stem: str) -> str:
    """Return the results subdirectory owning ``stem`` (longest-prefix match).

    Matching is on the **stem**, not on any active section context, because bulk
    frames are written after every section has exited.

    Parameters
    ----------
    stem
        Artifact filename stem, e.g. ``'10b_risk_book_v2'`` or ``'02_eda_07'``.

    Returns
    -------
    str
        A member of :data:`EXPORT_SECTION_DIRS`; :data:`EXPORT_MISC_DIR` when no
        prefix matches.
    """
    for prefix, directory in EXPORT_DIR_ALIASES.items():
        if stem == prefix or stem.startswith(f"{prefix}_"):
            return directory
    matches = [d for d in EXPORT_SECTION_DIRS if stem == d or stem.startswith(f"{d}_")]
    if not matches:
        return EXPORT_MISC_DIR
    return max(matches, key=len)


def section_path(root: Path, stem: str, *, suffix: str = ".csv") -> Path:
    """Resolve ``root/<section>/<stem><suffix>``, creating the directory.

    The one way a workflow turns an artifact stem into a path. Building one by
    hand is how a frame ends up in the root beside the section tree, which is the
    state this module was written to end.
    """
    directory = Path(root) / export_dir_for(stem)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}{suffix}"


def resolve_results_root(
    explicit: Optional[str],
    *,
    env_value: Optional[str] = None,
    default_dirname: str = DEFAULT_RESULTS_DIRNAME_V2,
    project_root: Optional[Path] = None,
) -> Path:
    """Resolve an artifact root: explicit → environment → default.

    A relative value is anchored at the PROJECT root rather than the working
    directory, so a run launched from anywhere writes to the same tree.

    Parameters
    ----------
    explicit
        A configured value — ``KalmanRunConfigV2.results_dir`` or the replay's.
        Wins over everything, which is what makes ``main(config=...)`` redirect.
    env_value
        Already-read environment value, or ``None``.
    default_dirname
        Used when neither is set.
    project_root
        Anchor for a relative value. Defaults to this file's parent's parent.
    """
    raw = explicit or env_value or default_dirname
    root = Path(raw)
    if not root.is_absolute():
        root = (project_root or Path(__file__).resolve().parents[1]) / root
    return root
