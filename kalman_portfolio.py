"""Forecast and decision layers, replayed off one fit.

What this is
------------
``pymc_kalman_filter_pt_v2.py`` fits the panel, screens it, sizes a risk book and --
since the forecast handoff landed -- writes the four posterior quantities the forward
simulation actually reads to ``07_forecast_handoff_v2.nc``. This script picks that file
up and runs everything downstream of it: the forward simulation, the two prior sweeps,
the three ranking arms, the mean-model contrast and the recommendation layer. **No
sampling, no database write by default.**

Why it is a separate script
---------------------------
The layers it runs are governed by two priors the panel cannot identify --
``forecast_error_multiplier`` and ``forecast_factor_share`` -- and by a choice of
ranking rule that two shipped candidates have now failed. Every one of those questions
is answered by running the same fit many times with one thing changed. Inside the
workflow that costs a NUTS run each; here it costs seconds, which is the difference
between a sensitivity that gets reported and one that gets asserted.

What it deliberately does not do
--------------------------------
It does not move a default. All three ranking arms are exported side by side with the
shipped one labelled, because which of them is *better* is a question about realised
returns and no gate here can answer it. Every gate below scores the model against the
analyst trail it was fitted to; that is why none of THOSE blocks.

Two gates do block, and both grade something other than the model.
``portfolio_input_vintage`` grades LINEAGE -- whether the files this replay read
came from the fit it is replaying, which is a fact about provenance and is
checkable now. ``portfolio_sector_concentration`` grades whether a stated
CONSTRAINT was actually applied to the weight it claims to cap. Neither is an
opinion about whether the model is any good, which is the thing no gate here can
have.

Usage
-----
.. code-block:: powershell

    python kalman_portfolio.py --handoff pymc_kalman_filter_pt_v2_results/07_forecast_handoff_v2.nc
    python kalman_portfolio.py --fit                        # fit first, then replay
    python kalman_portfolio.py --rank-arms all              # three books, one posterior
    python kalman_portfolio.py --sweep factor_share,multiplier
    python kalman_portfolio.py --arms level_off,hierarchy_fine   # mean-model contrast
    python kalman_portfolio.py --sector-cap 0.30 --size-down-veto
    python kalman_portfolio.py --write                      # opt in to the analytics write

Set ``PYTHONIOENCODING=utf-8`` in the SHELL before any redirected run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("kalman_portfolio")

# The v2 workflow owns the gate vocabulary, the identity block, the provenance SSOT
# and the artifact layout. This script is a consumer of all four, never a second copy.
from pymc_kalman_filter_pt_v2 import (  # noqa: E402
    EXPORT_IDENTITY_COLUMNS,
    GATE_CATALOGUE,
    LOG_UPLIFT_CLIP_HI,
    LOG_UPLIFT_CLIP_LO,
    GateReport,
    GateResult,
    KalmanRunConfigV2,
    PROVENANCE_COLUMNS,
    _ANALYTICS_TABLE_V2,
    _EPS,
    _HANDOFF_STEM,
    apply_out_of_support,
    write_analytics_ddl_v2,
    attach_identity_columns,
    resolve_source_revision,
    stamp_export_provenance,
)
from probabilistic_ml_model.export_layout import (  # noqa: E402
    DEFAULT_RESULTS_DIRNAME_PORTFOLIO,
    DEFAULT_RESULTS_DIRNAME_V2,
    EXPORT_SECTION_DIRS,
    PORTFOLIO_ONLY_SECTION_DIRS,
    RESULTS_DIR_ENV_PORTFOLIO,
    RESULTS_DIR_ENV_V2,
    export_dir_for,
    resolve_results_root,
    section_path,
)
from probabilistic_ml_model.pymc_models import _recommendations as recs  # noqa: E402
from probabilistic_ml_model.pymc_models.KalmanForecast import (  # noqa: E402
    ForecastConfig,
    ForecastHandoff,
    compare_forecast_engines,
    forecast_from_posterior,
    load_forecast_handoff,
    summarize_forecast,
    sweep_factor_share,
)
from probabilistic_ml_model.pymc_models.PortfolioOptimizationModel import (  # noqa: E402
    DEFAULT_OBJECTIVE,
    DEFAULT_RANKING_RULE,
    PORTFOLIO_OBJECTIVES,
    RANKING_RULES,
    RANKING_RULES_EXTERNAL,
    efficient_frontier,
    ergodicity_report,
    optimize_portfolio,
    tangency_portfolio,
    terminal_wealth_curve,
)

__all__ = [
    "KalmanPortfolioConfig",
    "PORTFOLIO_GATES",
    "VintageMismatch",
    "check_input_vintage",
    "migrate_portfolio_layout",
    "publish_canonical_table",
    "load_handoff",
    "ScreenDraws",
    "run_screen",
    "run_risk_book",
    "run_forecast",
    "run_prior_sweeps",
    "run_decision_books",
    "run_mean_model_arms",
    "run_recommendations_v2",
    "export_frames",
    "main",
]


# =========================================================================== #
# §P0  Gate vocabulary                                                        #
# =========================================================================== #
#
# Every gate here is NON-BLOCKING, and that is a position rather than caution. Each
# one measures either the consequence of a prior nothing identifies, or a property of
# a forward tail nothing has validated. A threshold on the first tests only that the
# prior was applied; a threshold on the second would assert exactly the thing that is
# unknown. They are recorded so a reader can see the number and argue with it.

PORTFOLIO_GATES: dict[str, str] = {
    "portfolio_handoff_provenance": (
        "The handoff names the run and the revision that produced it, and says how "
        "far its posterior was thinned. A replay whose source cannot be attributed "
        "cannot be contrasted against a later one, which is the same argument the "
        "vintage table rests on. `source_dirty` is a FACT, not an error."
    ),
    "portfolio_kelly_interior": (
        "Share of the universe whose Kelly fraction is an interior solution rather "
        "than a pin at the cap. A pinned fraction is the bisection reporting that "
        "E[log(1+f*r)] never turned over -- i.e. that no draw loses money -- not a "
        "sizing recommendation. Reported because it is a property of the forward "
        "simulation's left tail, which nothing internal can validate."
    ),
    "portfolio_rank_denominator": (
        "Where the sized book's names sit in the eligible universe's distribution of "
        "the ranking denominator. A book whose every name sits in the bottom few per "
        "cent has been selected on the ABSENCE of modelled downside. Both shipped "
        "ratio candidates failed exactly here."
    ),
    "portfolio_rank_tie_span": (
        "How many tied names the top-k cut fell among. `p_upside_pos_cond` is bounded "
        "-- its virtue -- but saturates, so a cut can land entirely inside a tie and "
        "the sort order becomes the selection rule. The explicit tie-break is what "
        "makes the answer reproducible; this reports how much work it did."
    ),
    "portfolio_book_agreement": (
        "Top-k membership overlap between the ranking arms run on one posterior. Two "
        "books from one fit that share under a third of their names is a statement "
        "about how underdetermined the ranking choice is, not about either book."
    ),
    "portfolio_solver_breadth": (
        "How many names the optimiser actually wanted, against how many were "
        "eligible -- and how far that is from the nominal position count. Run "
        "807df55e7158 published FIFTY names of which thirty-eight held 1.17% "
        "between them and the smallest held 0.0002%, an effective N of 11.7. "
        "Nothing reported that; it took a human reading a bar chart. Breadth is "
        "now solved rather than chosen, and this is the gate that says so."
    ),
    "portfolio_action_ladder": (
        "How the universe distributes over STRONG BUY / BUY / HOLD / SELL / "
        "STRONG SELL, and where the top gate sits in the probability "
        "distribution. A ladder whose top rung holds most of the universe has "
        "not expressed conviction, it has relabelled the old single BUY bucket "
        "-- and since the gates are scaled by the universe-mean confidence, a "
        "low mean confidence pulls the STRONG gate down onto the ordinary one. "
        "Reported because the three-valued version returned 83.5% BUY for a "
        "release without anything saying so."
    ),
    "portfolio_sector_concentration": (
        "Largest single-group weight per book. Reported whether or not a sector cap "
        "is set, because the absence of a cap is also a decision and the only way it "
        "should never be taken is by omission."
    ),
    "portfolio_factor_effect": (
        "How much of a book's forward dispersion the SHARED factors carry, "
        "measured as the ratio of an equal-weight book's sd at the shipped "
        "`factor_share` to its sd with independent shocks. Reported, never gated: "
        "`factor_share` is a prior, so a threshold on it would test only that the "
        "prior was applied. Independent shocks make diversification free, which "
        "is how a LONG book comes to report a positive expected shortfall."
    ),
    "portfolio_factor_sensitivity": (
        "Spread of book membership and dispersion across the `factor_share` grid. "
        "The split is variance-preserving, so per-name marginals are invariant and "
        "only the JOINT distribution moves -- harmless for the screen, decisive for "
        "every portfolio statistic."
    ),
    "portfolio_input_vintage": (
        "Whether every v2 artifact this replay reads was stamped by the same fit "
        "as the handoff. BLOCKING. A replay joins a posterior to frames sitting "
        "beside it in a results directory, and nothing about the file system says "
        "they came from one run -- so on run `b00f8d8ca093` a 2026-08-30 "
        "posterior of 6,507 names met a 2026-08-27 screen of 6,513 from a "
        "different fit, and the 145 names the two universes did not share became "
        "a phantom sector that evaded a 30% cap."
    ),
    "portfolio_size_down_coverage": (
        "How many eligible names the size-down watch never saw. A name absent "
        "from the frame the watch was scored on comes back SIZEABLE, which is "
        "indistinguishable from one the watch examined and cleared -- so a veto "
        "can report itself applied while being blind to part of the universe."
    ),
    "portfolio_size_down_overlap": (
        "How many of the sized book's names the size-down watch would veto -- a wide "
        "posterior or an analyst panel of two. This is the one gate that can say "
        "'most of the book', and no existing gate can produce that finding."
    ),
    "portfolio_mean_model_arms": (
        "Whether book membership survives a change of MEAN model, screened by "
        "Max-and-Smooth against a frozen covariance. It RANKS arms and never decides "
        "one, and it needs a live panel: a handoff carries the four quantities the "
        "simulator reads, not the model graph."
    ),
    "portfolio_engine_contrast": (
        "Median ratio of the forecast layer's per-name dispersion to the shipped AR "
        "simulator's, joined by ISIN. The two decay differently -- a fitted OU kernel "
        "against a hand-set rho=0.85 -- so this is the size of that choice."
    ),
}


# =========================================================================== #
# §P0b  Run configuration                                                     #
# =========================================================================== #


@dataclass(frozen=True)
class KalmanPortfolioConfig:
    """Everything this workflow can be told to do. Frozen; override with ``replace``.

    Attributes
    ----------
    handoff_path
        The ``07_forecast_handoff_v2.nc`` to replay. ``None`` resolves it under
        ``results_dir``.
    results_dir
        Artifact root. Defaults to the v2 workflow's, so a replay lands beside the run
        it replays.
    horizon_days, step_days, scenarios, backend, factor_share, factor_levels
        Forward-simulation knobs, mirroring :class:`ForecastConfig`.
    factor_share_grid, multiplier_grid
        The two prior sweeps. Both include the shipped value so a row of the table is
        always the run itself.
    rank_arms
        Ranking arms to size a book on. The first is the *recommendation*; the rest are
        contrasts, and the export says which is which.
    max_names, min_weight, objective, weight_cap, sector_cap, group_caps,
    kelly_multiplier
        Sizing knobs. Breadth is an OUTPUT: ``max_names`` only bounds the
        book and ``min_weight`` decides it. ``sector_cap`` now defaults to
        0.30; ``None`` reproduces the pre-2026-08-28 unconstrained book.
    apply_size_down_veto
        Feed the size-down watch into ``optimize_portfolio``'s ``eligible`` mask.
        **Off by default**: applying it changes which names a book holds, and whether
        that is an improvement is a question about realised returns. Measuring it is
        not optional, which is what ``portfolio_size_down_overlap`` is for.
    relative_denominator_q
        Relative floor on the ranking denominator; ``0.0`` reproduces the shipped
        behaviour exactly.
    write_analytics
        Opt in to the database write. Default off: this script's whole point is to be
        cheap to run repeatedly, and a workflow that writes on every run is not.
    """

    handoff_path: Optional[str] = None
    #: Where this replay WRITES. ``None`` resolves `KALMAN_PORTFOLIO_RESULTS_DIR`.
    results_dir: Optional[str] = None
    #: Where it READS the fit's artifacts. ``None`` resolves `KALMAN_V2_RESULTS_DIR`.
    v2_results_dir: Optional[str] = None
    random_seed: int = 42
    log_level: str = "INFO"
    #: Target figure width in px. Required by the figure layer's resolver
    #: contract, not decoration: ``set_viz_config_resolver`` promises an object
    #: carrying this attribute and ``_display_width_px`` reads it BARE, so a
    #: config without it turns every arviz-plots / matplotlib panel into an
    #: ``AttributeError`` the moment ``kalman_portfolio_viz.install`` re-points
    #: the resolver. Matches ``KalmanRunConfigV2.fig_width_px`` so one
    #: ``PML_FIG_WIDTH_PX`` sizes the fit and its replay identically.
    fig_width_px: int = 1150

    horizon_days: int = 365
    step_days: int = 91
    scenarios: int = 2000
    backend: str = "native"
    factor_share: float = 0.35
    factor_levels: tuple[str, ...] = ("trading_region", "sector")

    factor_share_grid: tuple[float, ...] = (0.0, 0.15, 0.35, 0.55, 0.75)
    multiplier_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)

    rank_arms: tuple[str, ...] = (DEFAULT_RANKING_RULE,)
    #: Ceiling on the number of positions, or ``None`` for no ceiling. **Not the
    #: book size** — see :data:`min_weight`.
    max_names: Optional[int] = None
    #: Weights below this are dropped and the problem re-solved on the survivors.
    #: This is what makes breadth an output.
    min_weight: float = 0.005
    #: Weighting objective; a key of ``PORTFOLIO_OBJECTIVES``.
    objective: str = DEFAULT_OBJECTIVE
    weight_cap: float = 0.10
    #: Maximum weight in any one sector. **0.30 by default since 2026-08-28.**
    #: This MOVES the shipped book, deliberately: run ``807df55e7158`` put 33.2 %
    #: in Health Care — 30 % of it in three small-cap therapeutics with binary
    #: clinical readouts nothing in the pipeline models — endorsed by a group
    #: signal clearing its band by 0.04pp. The gate that reports it exists so no
    #: cap is a choice someone made rather than a line nobody wrote; leaving it
    #: at ``None`` was taking that choice by omission. ``None`` restores the
    #: uncapped book, and the uncapped arm is worth exporting as a contrast.
    sector_cap: Optional[float] = 0.30
    sector_col: str = "sector"
    #: Further ``dimension -> cap`` limits, merged over ``sector_cap``.
    #: Concentration is not one-dimensional — the same run was also 19.2 % in one
    #: country and 85.9 % small cap — but only the sector cap is defaulted on,
    #: because each additional cap moves the book again.
    group_caps: dict[str, float] = field(default_factory=dict)
    kelly_multiplier: float = 0.5
    apply_size_down_veto: bool = True
    relative_denominator_q: float = 0.0

    mean_model_arms: tuple[str, ...] = ()
    #: Append the frames to the `kalman_portfolio` schema.
    #:
    #: **Default False since 2026-08-31**, matching what the docstring above has
    #: always said. It was `True` while the write was a no-op, so `--write` set a
    #: flag that was already set and the documented "opt in" was not one. Now that
    #: the write appends real rows to an append-only store, the difference is a
    #: row in a history rather than a log line.
    write_analytics: bool = False
    #: Run against v2 artifacts stamped with a DIFFERENT fit's ``run_id``.
    #:
    #: Default ``False``, and the refusal is the point: a replay of the wrong fit
    #: is not a degraded replay, it is a different one. With this on, a
    #: mismatched frame is DROPPED rather than joined, so the stages that needed
    #: it skip and say so -- degrading to empty, never to wrong.
    allow_stale_inputs: bool = False

    # ---- screen and risk book, moved here from KalmanRunConfigV2 -----------
    #
    # The DEFAULTS ARE THE FIT'S, value for value, so a replay of a handoff
    # reproduces the screen the fit used to publish. They live here because the
    # stages that read them live here now -- and because the first two are a
    # PRIOR the panel cannot identify, which belongs beside `--sweep multiplier`
    # rather than inside the artifact the sweep reads.
    enable_forecast_error_shrinkage: bool = True
    forecast_error_multiplier: float = 1.0
    forecast_error_n_exponent: float = 0.5
    #: Forward-return Monte Carlo: unitless periods, and the AR decay across them.
    mc_horizon: int = 4
    mc_rho: float = 0.85
    #: CVaR-aware risk book (§10b). `k_book` is a CEILING since 2026-08-28, not
    #: the book size; `book_min_weight` is what decides breadth.
    #:
    #: `book_min_weight` is NOT a duplicate of :attr:`min_weight` despite sharing
    #: its value. They floor two different books built by two different sizers:
    #: `min_weight` is the DECISION book's, applied by `optimize_portfolio`;
    #: this one is the RISK book's, applied by `compute_cvar_aware_book`. The two
    #: books share no name on a typical run -- the risk book screens to
    #: `mcap_global_r_max` and the decision book does not screen at all -- so
    #: collapsing them into one field would tie two independent decisions
    #: together on the strength of a coincidence of defaults.
    cvar_alpha: float = 0.05
    k_book: int = 50
    book_min_weight: float = 0.005
    p_long: float = 0.67
    mcap_global_r_max: float = 0.03
    tail_risk_vol_floor_k: float = 0.25
    #: `shrinkage_slope`, the four-part calibration/disagreement test. Slope and
    #: intercept grade CALIBRATION; rho and the median revision grade
    #: DISAGREEMENT, and both halves are needed -- an exact copy of the input
    #: scores a perfect 1.0/0.0 on the first pair, which is how run
    #: `49e84d7e9d59` passed while reproducing consensus at rho 0.999995.
    gate_shrinkage_slope_lo: float = 0.80
    gate_shrinkage_slope_hi: float = 0.98
    gate_shrinkage_center_shift_max: float = 0.02
    gate_shrinkage_rho_max: float = 0.995
    gate_shrinkage_revision_min_pp: float = 0.25

    @property
    def results_path(self) -> Path:
        """Where this replay WRITES: ``kalman_portfolio_results`` by default.

        Separate from :attr:`v2_results_path`, which is where it reads. They were
        one root until 2026-08-31, so a fit and every replay of it interleaved in
        one tree and a reader browsing it could not tell which run had produced
        what. A fit happens once; a replay happens many times over that fit.
        """
        return resolve_results_root(
            self.results_dir,
            env_value=os.environ.get(RESULTS_DIR_ENV_PORTFOLIO),
            default_dirname=DEFAULT_RESULTS_DIRNAME_PORTFOLIO,
        )

    @property
    def v2_results_path(self) -> Path:
        """Where this replay READS: the fit's tree. **Never written to.**"""
        return resolve_results_root(
            self.v2_results_dir,
            env_value=os.environ.get(RESULTS_DIR_ENV_V2),
            default_dirname=DEFAULT_RESULTS_DIRNAME_V2,
        )

    @property
    def handoff(self) -> Path:
        """The handoff to replay, resolved through the section tree.

        Falls back to the results ROOT when the sectioned path is absent, so a
        handoff written before the 2026-08-27 layout migration is still found
        rather than reported missing.
        """
        if self.handoff_path:
            return Path(self.handoff_path)
        stem, suffix = Path(_HANDOFF_STEM).stem, Path(_HANDOFF_STEM).suffix
        sectioned = section_path(self.v2_results_path, stem, suffix=suffix)
        if sectioned.exists():
            return sectioned
        legacy = self.v2_results_path / _HANDOFF_STEM
        if legacy.exists():
            logger.info(
                "reading the handoff from the pre-migration flat path %s; "
                "`python pymc_kalman_filter_pt_v2.py --migrate-layout --apply` "
                "moves it into %s", legacy, sectioned.parent,
            )
            return legacy
        return sectioned

    def __post_init__(self) -> None:
        """Reject the knobs that would otherwise fail silently several stages on.

        Moved here with the fields on 2026-08-31. Each of these produces a
        plausible-looking number downstream rather than an error at the point of
        the mistake: a negative multiplier gives a negative variance and a gain
        above 1, i.e. ANTI-shrinkage; a slope band that does not contain a
        shrinkage estimator makes `shrinkage_slope` unsatisfiable, so it fails on
        every run and stops being read.
        """
        if self.forecast_error_multiplier < 0:
            raise ValueError(
                "forecast_error_multiplier must be non-negative, got "
                f"{self.forecast_error_multiplier!r}"
            )
        if self.forecast_error_n_exponent < 0:
            raise ValueError(
                "forecast_error_n_exponent must be non-negative, got "
                f"{self.forecast_error_n_exponent!r}"
            )
        if self.tail_risk_vol_floor_k < 0:
            raise ValueError(
                f"tail_risk_vol_floor_k must be non-negative, got "
                f"{self.tail_risk_vol_floor_k!r}"
            )
        if not self.gate_shrinkage_slope_lo < self.gate_shrinkage_slope_hi:
            raise ValueError(
                "gate_shrinkage_slope band must be increasing, got "
                f"[{self.gate_shrinkage_slope_lo}, {self.gate_shrinkage_slope_hi}]"
            )
        if not 0.0 < self.gate_shrinkage_rho_max <= 1.0:
            raise ValueError(
                "gate_shrinkage_rho_max must be in (0, 1], got "
                f"{self.gate_shrinkage_rho_max!r}"
            )

    @property
    def mcap_country_r_max(self) -> float:
        """Deprecated alias of :attr:`mcap_global_r_max`.

        The old name claimed a country-relative rank; the comparison has always
        been against ``mcap_global_r``. Read-only on purpose -- a writable alias
        on a frozen dataclass would let ``replace()`` appear to work while
        setting nothing.
        """
        warnings.warn(
            "mcap_country_r_max is deprecated; it was renamed to "
            "mcap_global_r_max because the threshold is compared against the "
            "mcap_global_r column, not a country rank.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mcap_global_r_max

    def forecast_config(self, **overrides: Any) -> ForecastConfig:
        """A :class:`ForecastConfig` from these knobs, with optional overrides."""
        cfg = ForecastConfig(
            horizon_days=self.horizon_days,
            step_days=self.step_days,
            n_scenarios=self.scenarios,
            backend=self.backend,
            factor_levels=self.factor_levels,
            factor_share=self.factor_share,
            random_seed=self.random_seed,
        )
        return replace(cfg, **overrides) if overrides else cfg

    @classmethod
    def from_env(cls) -> "KalmanPortfolioConfig":
        """Read only what the v2 config reads from the environment, and no more.

        Deliberately narrow, for the reason ``KalmanRunConfigV2.from_env`` is: a knob
        that changes a book should be visible at the call site, not inherited from a
        shell.
        """
        return cls(
            # Two roots, and neither is v1's: this replay writes its own tree and
            # reads the fit's. `set_env.ps1` points KALMAN_PT_RESULTS_DIR at v1,
            # which is why neither of these is that variable.
            results_dir=os.environ.get(RESULTS_DIR_ENV_PORTFOLIO) or None,
            v2_results_dir=os.environ.get(RESULTS_DIR_ENV_V2) or None,
            random_seed=int(os.environ.get("RANDOM_SEED", 42)),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


# =========================================================================== #
# §P0c  Load                                                                  #
# =========================================================================== #


def load_handoff(cfg: KalmanPortfolioConfig, report: GateReport) -> ForecastHandoff:
    """Read the handoff and record what it says about its own provenance."""
    handoff = load_forecast_handoff(cfg.handoff)
    attrs = handoff.attrs
    dirty = bool(attrs.get("source_dirty", False))
    thin = float(attrs.get("thin_factor", 1.0))
    report.add(GateResult(
        name="portfolio_handoff_provenance",
        passed=True,
        value=f"run {attrs.get('run_id', '?')} @ {attrs.get('source_sha', '?')[:7]}"
              f"{' dirty' if dirty else ''}, thinned {thin:.1f}x",
        threshold="reported, not gated",
        blocking=False,
        detail=(
            f"{handoff.n_isin} names x {handoff.n_samples} samples from "
            f"{attrs.get('n_samples_original', '?')}. source_dirty is a fact, not an "
            f"error: it says the SHA does not fully determine what ran."
        ),
    ))
    return handoff


# =========================================================================== #
# §P0c-b  Input vintage                                                       #
# =========================================================================== #

#: Provenance column the vintage check compares on. One spelling, shared with
#: `dashboards/geib/data.py`, which has asserted the same relation since it was
#: written.
VINTAGE_COLUMN: str = "run_id"


class VintageMismatch(RuntimeError):
    """A v2 artifact this replay read was stamped by a different fit.

    Raised before any book is sized, because the point is not to report the
    mismatch afterwards -- it is to not produce the book. Caught by ``_cli``,
    which exits 2.
    """


def check_input_vintage(
    handoff: ForecastHandoff,
    frames: dict[str, Optional[pd.DataFrame]],
    cfg: KalmanPortfolioConfig,
    report: GateReport,
) -> dict[str, Optional[pd.DataFrame]]:
    """Refuse, or drop, any read frame that did not come from the handoff's fit.

    The gate `dashboards/geib/data.py::_join_panel` has had since it was written,
    which the replay did not. A results directory is a directory: the file system
    says nothing about whether the CSV beside a handoff came from the run that
    wrote the handoff. On run ``b00f8d8ca093`` it did not, and the join still
    succeeded -- 6,362 names matched, 145 did not, and the 145 were filled to
    ``"Unknown"`` and sized.

    Three verdicts, and the difference between the last two is the whole design:

    ``clean``
        The frame carries exactly the handoff's ``run_id``.
    ``MISMATCH``
        It carries a **different** ``run_id``. That is a positive assertion of the
        wrong lineage, so it BLOCKS -- or, under ``allow_stale_inputs``, the frame
        is dropped and the stages needing it skip with their own warnings.
    ``unstamped``
        It carries no ``run_id`` at all: an export predating the provenance SSOT.
        WARNED and KEPT. "No provenance" is not "wrong provenance", and every
        frame that can still reach this path has no ISIN axis -- it is one row per
        model PARAMETER -- so it cannot mis-attribute a company to a name. If a
        per-ISIN frame is ever read here again, this branch must become a refusal.

    Parameters
    ----------
    frames
        Stem -> frame, as read. ``None`` entries pass through untouched: a file
        that is absent was already handled by whoever tried to read it.

    Returns
    -------
    dict[str, pandas.DataFrame or None]
        The same mapping with mismatched frames replaced by ``None``.

    Raises
    ------
    VintageMismatch
        On a mismatch when ``allow_stale_inputs`` is False.
    """
    fit_run = str(handoff.attrs.get("run_id") or "").strip()
    present = {k: v for k, v in frames.items() if v is not None and len(v)}

    mismatched: dict[str, list[str]] = {}
    unstamped: list[str] = []
    for stem, frame in present.items():
        if VINTAGE_COLUMN not in frame.columns:
            unstamped.append(stem)
            continue
        ids = sorted(
            {str(v) for v in frame[VINTAGE_COLUMN].dropna().unique() if str(v).strip()}
        )
        if not ids:
            unstamped.append(stem)
        elif not fit_run or ids != [fit_run]:
            mismatched[stem] = ids

    if mismatched:
        detail = "; ".join(
            f"{stem} at run_id {', '.join(ids)}" for stem, ids in mismatched.items()
        )
        # Name BOTH sides. A verdict that reports only the offending frame costs
        # its reader a forensic pass to find what it was supposed to match.
        message = (
            f"the handoff is run_id {fit_run or '<unstamped>'} but {detail}. "
            f"These are different fits. Re-run "
            f"`python pymc_kalman_filter_pt_v2.py --write` so the handoff and the "
            f"frames beside it carry one run_id, or pass --allow-stale-inputs to "
            f"drop the mismatched frames and replay without them."
        )
        report.add(GateResult(
            name="portfolio_input_vintage",
            passed=False,
            value=f"{len(mismatched)} frame(s) from another fit",
            threshold=f"every read frame at run_id {fit_run or '<unstamped>'}",
            blocking=not cfg.allow_stale_inputs,
            detail=message,
        ))
        if not cfg.allow_stale_inputs:
            raise VintageMismatch(message)
        logger.error(
            "VINTAGE MISMATCH (--allow-stale-inputs): %s Dropping them rather "
            "than joining them -- the stages that needed them will degrade to "
            "empty, which is recoverable, instead of to wrong, which is not.",
            message,
        )
        return {k: (None if k in mismatched else v) for k, v in frames.items()}

    report.add(GateResult(
        name="portfolio_input_vintage",
        passed=True,
        value=(
            f"{len(present)} frame(s) at run_id {fit_run or '<unstamped>'}"
            + (f", {len(unstamped)} unstamped" if unstamped else "")
        ),
        threshold=f"every read frame at run_id {fit_run or '<unstamped>'}",
        blocking=True,
        detail=(
            "The screen, the risk book and the identity block are built from the "
            "handoff itself since 2026-08-31, so this gate now guards only what "
            "is still read from disk beside it."
            + (
                f" {', '.join(unstamped)} carries no run_id -- an export predating "
                "the provenance SSOT. Kept, because 'no provenance' is not 'wrong "
                "provenance' and it has no ISIN axis to mis-attribute; a per-ISIN "
                "frame in this position would be refused."
                if unstamped else ""
            )
        ),
    ))
    return frames


# =========================================================================== #
# §P0d  Screen                                                                #
# =========================================================================== #
#
# Moved here from `pymc_kalman_filter_pt_v2.run_screen` on 2026-08-31 and
# re-expressed over the handoff. The fit no longer decides: it fits, checks and
# writes `07_forecast_handoff_v2.nc`, and everything from here on is this file's.
#
# The arithmetic is unchanged, value for value. What moved with it is the
# forecast-error shrinkage -- a PRIOR the panel cannot identify, gridded by
# `--sweep multiplier`. Baking one reading of it into the artifact the sweep
# reads would make the sweep measure only that the prior had been applied, so the
# handoff carries the UNSHRUNK latent and the update happens below.


@dataclass(frozen=True)
class ScreenDraws:
    """The draw arrays the screen builds and the risk book needs, carried explicitly.

    Handed over rather than re-derived. Re-resolving the latent in the risk book
    would size the book on the UNSHRUNK latent while the screen reported the
    shrunk one, and every gate would still pass.

    Attributes
    ----------
    eu
        Expected-upside draws in RETURN space, dims ``(chain, draw, isin)`` with a
        single pseudo-chain. The handoff's sample axis is a seeded subsample drawn
        ACROSS chains, so chain identity is genuinely gone; one pseudo-chain says
        that, where reshaping into a plausible-looking grid would claim a
        between-chain structure the file does not carry.
    mc_returns
        Monte-Carlo forward returns, ``(n_isin, n_samples, horizon)``.
    isins
        Identifiers, aligned to the ``isin`` axis of both.
    shrink_gain
        Per-name weight the update put on the name's own observation.
    """

    eu: Any
    mc_returns: np.ndarray
    isins: np.ndarray
    shrink_gain: np.ndarray

    @property
    def pooled_returns(self) -> np.ndarray:
        """``mc_returns`` flattened to ``(n_isin, n_samples * horizon)``.

        The pooling ``summarize_mc_returns`` uses, so a CVaR taken from this and
        the exported ``er_p05`` describe the same distribution.
        """
        return np.asarray(self.mc_returns).reshape(len(self.isins), -1)


def _handoff_vector(handoff: ForecastHandoff, name: str) -> Optional[np.ndarray]:
    """One numeric per-ISIN column, from the panel vectors or the identity block."""
    vec = (handoff.panel_vectors or {}).get(name)
    if vec is not None:
        return np.asarray(vec, dtype="float64")
    ident = handoff.identity
    if ident is not None and name in getattr(ident, "columns", []):
        keyed = ident.drop_duplicates("isin").set_index("isin")[name]
        return pd.to_numeric(keyed.reindex(handoff.isins), errors="coerce").to_numpy()
    return None


def _handoff_labels(handoff: ForecastHandoff, name: str) -> Optional[np.ndarray]:
    """One per-ISIN label column from the identity block. Never filled."""
    ident = handoff.identity
    if ident is None or name not in getattr(ident, "columns", []):
        return None
    keyed = ident.drop_duplicates("isin").set_index("isin")[name]
    return keyed.reindex(handoff.isins).to_numpy()


def _risk_adjusted_prob_positive(
    mc_log: np.ndarray,
    latent: np.ndarray,
    rar: np.ndarray,
    response_std: float,
    *,
    chunk: int = 512,
) -> Optional[np.ndarray]:
    """P(risk-adjusted forward return > 0) per name, from log-space MC draws.

    Replaces ``mc_prob_pos * kalman_gain``, which multiplied a probability by a
    sigmoid of a location parameter: usable as an ordering, meaningless as a
    level. This is one probability of one stated event -- that the forward return,
    net of the same risk / size / volume penalties the model applies to
    ``risk_adj_return``, is positive.

    Two implementation notes that are not incidental:

    * The penalty is recovered as ``latent - rar`` rather than rebuilt from the
      three loadings and their data columns. That is the penalty the model
      actually applied, so it cannot drift out of step with the builder the way a
      reimplementation would.
    * The test is on the LOG draws. ``expm1`` is monotone and the clip bounds
      straddle zero, so ``expm1(x) > 0`` exactly when ``x > 0`` -- converting
      first would allocate a second multi-gigabyte array to learn nothing.

    Parameters
    ----------
    mc_log
        Simulated log-uplift, ``(n_isin, n_samples, horizon)``, already clipped.
        Not modified.
    latent, rar
        ``(n_isin, n_samples)`` on the standardised scale. Both come off one
        handoff and are therefore aligned draw for draw.
    chunk
        Names per block. The comparison materialises
        ``chunk * n_samples * horizon`` floats, so this bounds peak memory at a
        few hundred MB against a ~1.7 GB source array.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(n_isin,)``, or ``None`` when the arrays do not align, which
        signals the caller to fall back to the legacy product.
    """
    penalty = (np.asarray(latent) - np.asarray(rar)) * response_std
    n_isin = mc_log.shape[0]
    if penalty.shape[0] != n_isin or penalty.shape[1] != mc_log.shape[1]:
        logger.warning(
            "penalty draws %s do not align with the MC array %s; "
            "p_upside_pos_cond falls back to the legacy product.",
            penalty.shape, mc_log.shape,
        )
        return None

    out = np.empty(n_isin, dtype="float64")
    for i0 in range(0, n_isin, chunk):
        sl = slice(i0, min(i0 + chunk, n_isin))
        adj = mc_log[sl] - penalty[sl, :, None]
        out[sl] = (adj > 0.0).mean(axis=(1, 2))
    return out


def run_screen(
    handoff: ForecastHandoff,
    cfg: KalmanPortfolioConfig,
    report: GateReport,
) -> tuple[pd.DataFrame, ScreenDraws]:
    """Build the per-ISIN screen off the handoff, and gate the decision layer.

    Raises
    ------
    ValueError
        If the handoff cannot be screened from. That is a refusal, not a
        degradation: a screen assembled from part of a handoff would publish a
        decision quantity with no definition.
    """
    import xarray as xr

    from probabilistic_ml_model.pymc_models._price_target_mc import (
        simulate_lagged_risk_adjusted_returns,
        summarize_mc_returns,
    )
    from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
        forecast_error_variance_from_arrays,
        shrink_latent_from_arrays,
    )

    if not handoff.screen_ready:
        raise ValueError(
            "this handoff carries the forward simulation's inputs but not the "
            "screen's (latent_mean / latent_sd / fitted_mean / rar / "
            "variance_weights). Re-run `python pymc_kalman_filter_pt_v2.py` to "
            "write a current one."
        )

    isins = np.asarray(handoff.isins)
    n_isin = len(isins)
    rstd, rmean = handoff.response_std, handoff.response_mean

    # ---- the decision latent, shrunk or not --------------------------------
    if cfg.enable_forecast_error_shrinkage:
        cv = handoff.panel_vectors.get("dispersion_cv")
        n_an = handoff.panel_vectors.get("n_analysts")
        if cv is None or n_an is None:
            raise ValueError(
                "forecast-error shrinkage is on but the handoff carries no "
                "dispersion_cv / n_analysts. Screening without it is possible "
                "(enable_forecast_error_shrinkage=False) and reproduces analyst "
                "consensus at Spearman 0.999995 -- see run 49e84d7e9d59."
            )
        fe_var = forecast_error_variance_from_arrays(
            cv, n_an, rstd,
            multiplier=cfg.forecast_error_multiplier,
            n_exponent=cfg.forecast_error_n_exponent,
        )
        latent_std, shrink_gain = shrink_latent_from_arrays(
            latent_mean=handoff.latent_mean,
            latent_sd=handoff.latent_sd,
            fitted_mean=handoff.fitted_mean,
            sigma=handoff.sigma_std,
            variance_weights=handoff.variance_weights,
            fe_var=fe_var,
            random_seed=cfg.random_seed,
        )
        logger.info(
            "forecast-error shrinkage: multiplier %.2f, median gain %.3f "
            "(p05 %.3f, p95 %.3f)",
            cfg.forecast_error_multiplier, float(np.median(shrink_gain)),
            float(np.quantile(shrink_gain, 0.05)),
            float(np.quantile(shrink_gain, 0.95)),
        )
    else:
        # `resolve_screen_latent_v2(include_latent_noise=True)`, on arrays. The
        # smoother stores a conditional mean and sd rather than a sampled path, so
        # the residual uncertainty about where a name's latent SITS has to be drawn
        # back in or per-name spread is understated.
        rng = np.random.default_rng(cfg.random_seed)
        sd = handoff.latent_sd if handoff.latent_sd is not None else 0.0
        latent_std = handoff.latent_mean + sd * rng.standard_normal(
            size=(handoff.n_samples, n_isin)
        ).T
        shrink_gain = np.ones(n_isin)

    draws = latent_std.T                                    # (sample, isin)

    # De-standardise back to log-uplift, then to a return. The clip is applied in
    # LOG space so it is sign-preserving; converting first and clipping after
    # would distort prob_pos.
    log_uplift = np.clip(draws * rstd + rmean, LOG_UPLIFT_CLIP_LO, LOG_UPLIFT_CLIP_HI)
    upside = np.expm1(log_uplift)
    eu_draws = xr.DataArray(
        upside[None, :, :], dims=("chain", "draw", "isin"), coords={"isin": isins},
    )

    screen = pd.DataFrame({"isin": isins})
    for col in ("ticker", "name", "sector", "industry", "trading_region",
                "country", "country_name", "style_class", "size_class"):
        labels = _handoff_labels(handoff, col)
        if labels is not None:
            screen[col] = labels
    for src, dest in (
        ("n_analysts", "n_analysts"),
        # The analyst panel's own 1-5 consensus (5 = Strong Buy), carried so
        # `name_action_list` can emit `consensus_gap` -- the model's action score
        # against the panel's rating, on one scale.
        ("feat_analyst_rating", "feat_analyst_rating"),
        ("market_cap", "market_cap"),
        ("feat_mcap_global_r", "mcap_global_r"),
        ("feat_mcap_country_r", "mcap_country_r"),
        ("last_price", "last_price"),
        ("observed_pt", "observed_pt"),
    ):
        vec = _handoff_vector(handoff, src)
        if vec is not None:
            screen[dest] = vec
        else:
            logger.warning(
                "%s absent from the handoff; the screen omits %r", src, dest
            )

    screen["expected_upside"] = upside.mean(axis=0)
    screen["expected_upside_sd"] = upside.std(axis=0)
    screen["prob_pos"] = (upside > 0).mean(axis=0)
    screen["implied_upside"] = screen["observed_pt"] / screen["last_price"] - 1.0
    screen["expected_pt"] = screen["last_price"] * (1.0 + screen["expected_upside"])
    # v1's column names, not v2's first draft. `compute_cvar_aware_book` requires
    # `expected_pt_hdi_lo` / `_hi` BY NAME; supplying `expected_upside_p05` /
    # `_p95` instead silently degrades three risk columns rather than raising.
    screen["expected_pt_hdi_lo"] = screen["last_price"] * (
        1.0 + np.percentile(upside, 3, axis=0)
    )
    screen["expected_pt_hdi_hi"] = screen["last_price"] * (
        1.0 + np.percentile(upside, 97, axis=0)
    )
    screen["shrink_gain"] = shrink_gain
    screen["risk_adj_return"] = np.asarray(handoff.rar).mean(axis=1)
    # `kalman_gain` was `sigmoid(risk_adj_return)` -- a sigmoid of a STANDARDISED
    # LOG-UPLIFT, which is not the probability of any defined event: it has no
    # calibration target and pins at 0.5 wherever the risk-adjusted latent is near
    # zero. It is now the posterior probability that that latent is positive -- a
    # real tail probability of a stated event, and a reduction over draws rather
    # than a per-draw Deterministic, which is why it lives here and not in the graph.
    screen["kalman_gain"] = (np.asarray(handoff.rar) > 0.0).mean(axis=1)

    # ---- Monte-Carlo forward returns ---------------------------------------
    # mu and sigma must be de-standardised onto the response scale BEFORE the
    # simulation, and the draws clipped in log space afterwards. Skipping the
    # de-standardisation yields z-scores rather than returns -- a real historical
    # bug that reached the exported table.
    sigma_draws = np.asarray(handoff.sigma_std) * rstd
    mu_draws = latent_std * rstd + rmean                     # (isin, sample)
    nu_draws = np.asarray(handoff.nu).ravel()
    if sigma_draws.shape != mu_draws.shape:
        raise ValueError(
            f"MC input shape mismatch: sigma {sigma_draws.shape} vs mu "
            f"{mu_draws.shape}. Both must be (n_isin, n_samples)."
        )
    mc = simulate_lagged_risk_adjusted_returns(
        mu_draws, sigma_draws, nu_draws,
        horizon=cfg.mc_horizon, rho=cfg.mc_rho, random_seed=cfg.random_seed,
    )
    # In place, and in this order, because the array is large: at the production
    # budget it is (6500, samples, horizon) ~ 1.7 GB. Clip -> read the log array
    # -> expm1 over the SAME buffer keeps exactly one copy alive.
    np.clip(mc, LOG_UPLIFT_CLIP_LO, LOG_UPLIFT_CLIP_HI, out=mc)
    p_cond = _risk_adjusted_prob_positive(
        mc, latent_std, np.asarray(handoff.rar), rstd
    )
    np.expm1(mc, out=mc)
    mc_summary = summarize_mc_returns(mc, isins)
    screen = screen.merge(
        mc_summary.rename(columns={"prob_pos": "mc_prob_pos"}), on="isin", how="left"
    )

    if p_cond is not None:
        screen["p_upside_pos_cond"] = p_cond
    else:
        # Documented fallback, matching `compute_cvar_aware_book`'s
        # degrade-with-a-warning pattern: the product form still orders names, it
        # just cannot be read as a probability of anything.
        screen["p_upside_pos_cond"] = (
            screen["mc_prob_pos"].fillna(screen["prob_pos"])
            * screen["kalman_gain"].fillna(1.0)
        )

    screen = screen.sort_values(
        "expected_upside", ascending=False
    ).reset_index(drop=True)
    _report_screen_gates(screen, cfg, report)
    return screen, ScreenDraws(
        eu=eu_draws, mc_returns=mc, isins=isins, shrink_gain=shrink_gain
    )


def _report_screen_gates(
    screen: pd.DataFrame, cfg: KalmanPortfolioConfig, report: GateReport
) -> None:
    """The three gates that grade the screen. Moved with it, unchanged."""
    # ---- shrinkage slope ---------------------------------------------------
    valid = screen[["expected_upside", "implied_upside"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(valid) >= 100:
        slope, intercept = np.polyfit(
            valid["implied_upside"], valid["expected_upside"], 1
        )
        above = float((screen["expected_upside"] > screen["implied_upside"]).mean())
        # Slope and intercept grade CALIBRATION; rho and the median revision grade
        # DISAGREEMENT. Both halves are needed: an exact copy of the input scores a
        # perfect 1.0 / 0.0 on the first pair, which is how run `49e84d7e9d59`
        # passed while reproducing consensus at rho 0.999995.
        rho = float(
            valid["expected_upside"].corr(valid["implied_upside"], method="spearman")
        )
        revision_pp = float(
            (valid["expected_upside"] - valid["implied_upside"]).abs().median() * 100.0
        )
        # The universe-wide shift, which is what the old |intercept| threshold was
        # reaching for. The intercept itself is (1 - slope) * centre by
        # construction and so cannot separate an offset from shrinkage.
        center_shift = float(
            valid["expected_upside"].mean() - valid["implied_upside"].mean()
        )
        slope_ok = cfg.gate_shrinkage_slope_lo <= slope <= cfg.gate_shrinkage_slope_hi
        shift_ok = abs(center_shift) <= cfg.gate_shrinkage_center_shift_max
        rho_ok = not (np.isfinite(rho) and rho > cfg.gate_shrinkage_rho_max)
        rev_ok = revision_pp >= cfg.gate_shrinkage_revision_min_pp
        report.add(GateResult(
            name="shrinkage_slope",
            passed=bool(slope_ok and shift_ok and rho_ok and rev_ok),
            value=(
                f"slope {slope:.3f}, shift {center_shift:+.4f}, above {above:.1%}, "
                f"rho {rho:.5f}, median revision {revision_pp:.2f}pp "
                f"(intercept {intercept:+.4f})"
            ),
            threshold=(
                f"slope in [{cfg.gate_shrinkage_slope_lo}, "
                f"{cfg.gate_shrinkage_slope_hi}], |shift| <= "
                f"{cfg.gate_shrinkage_center_shift_max}, rho <= "
                f"{cfg.gate_shrinkage_rho_max}, revision >= "
                f"{cfg.gate_shrinkage_revision_min_pp}pp"
            ),
            blocking=False,
            detail=(
                "A shift of the centre is a universe-wide offset, not a signal. "
                "Expect ~50% of names above consensus, not 80%. A rho at the "
                "ceiling or a revision at the floor means the opposite failure: "
                "the screen is a consensus sort, and the drift betas and the "
                "hierarchy are being estimated but are not reaching the exported "
                "number. The intercept is reported for continuity but is not "
                "graded -- it equals (1 - slope) * centre."
            ),
        ))

    # ---- coverage gradient -------------------------------------------------
    cov = screen.dropna(subset=["n_analysts"]).copy() if "n_analysts" in screen else None
    if cov is not None and len(cov) >= 500:
        cov["bucket"] = pd.cut(
            cov["n_analysts"], [0, 3, 8, 20, np.inf],
            labels=["1-3", "4-8", "9-20", "21+"],
        )
        # GRADE THE POSTERIOR SD. This gate asks whether the hierarchy prices
        # information -- whether a name covered by 30 analysts gets a tighter
        # ESTIMATE than one covered by 2 -- so it has to be measured on the
        # estimate's own uncertainty.
        #
        # It used to read `er_sd if "er_sd" in cov.columns else expected_upside_sd`,
        # and the fallback silently became the primary when the 2026-08-20 change
        # made `er_sd` the pooled sd of the FORWARD-RETURN Monte Carlo. Named
        # explicitly rather than resolved by membership, because a
        # column-availability fallback is precisely how the measured quantity
        # changed underneath the threshold without anything failing.
        col = "expected_upside_sd"
        if col not in cov.columns:
            logger.warning(
                "%s absent from the screen; the coverage gradient cannot be "
                "measured on the posterior sd this run", col,
            )
        else:
            grad = cov.groupby("bucket", observed=True)[col].mean()
            monotone = bool(grad.is_monotonic_decreasing)
            spread = float(grad.max() / max(grad.min(), _EPS))
            # The forward-return gradient, REPORTED and never graded. `er_sd`
            # carries the forecast-error term, whose steepness is set by
            # `forecast_error_n_exponent` -- a prior the panel cannot identify, so
            # a threshold on it would test only that the prior was applied.
            fwd = ""
            if "er_sd" in cov.columns:
                g2 = cov.groupby("bucket", observed=True)["er_sd"].mean()
                fwd = (
                    f"; forward-return er_sd {g2.round(4).to_dict()} "
                    f"(spread {float(g2.max() / max(g2.min(), _EPS)):.2f}x, "
                    "reported not gated: its steepness is set by "
                    "forecast_error_n_exponent, a prior)"
                )
            report.add(GateResult(
                name="coverage_gradient",
                passed=monotone and spread >= 2.0,
                value=(
                    f"{'monotone' if monotone else 'NOT monotone'}, "
                    f"spread {spread:.2f}x"
                ),
                threshold="monotone decreasing, spread >= 2x (posterior sd)",
                blocking=False,
                detail=f"mean {col} by bucket: {grad.round(4).to_dict()}{fwd}",
            ))

    # ---- prob_pos degeneracy -----------------------------------------------
    pinned = float((screen["prob_pos"] >= 0.99995).mean())
    # The span check is the half that was missing. Grading only `prob_pos` meant
    # its collapse was reported while the column that INHERITED the ranking went
    # unmeasured; a future collapse of the promoted column would then be silent.
    # p95 - p05 is the range the ranking actually has to work in.
    _cond = pd.to_numeric(screen.get("p_upside_pos_cond"), errors="coerce")
    span = (
        float(_cond.quantile(0.95) - _cond.quantile(0.05))
        if _cond is not None and _cond.notna().any() else float("nan")
    )
    span_ok = bool(np.isfinite(span) and span >= 0.30)
    report.add(GateResult(
        name="prob_pos_degenerate",
        passed=(pinned <= 0.60) and span_ok,
        value=f"{pinned:.1%} pinned at 1.0, p_upside_pos_cond span {span:.3f}",
        threshold="pinned <= 60%, p_upside_pos_cond p95-p05 >= 0.30",
        blocking=False,
        detail=(
            "prob_pos is a REPORTED diagnostic and is never ranked on. "
            "p_upside_pos_cond is the primary probability column -- since "
            "2026-08-20 it is P(risk-adjusted forward return > 0) computed "
            "directly from the MC draws, not the old mc_prob_pos * sigmoid "
            "product. The export ranks on it and suppresses it alongside the "
            "other out-of-support metrics."
        ),
    ))


# =========================================================================== #
# §P0e  CVaR risk book                                                        #
# =========================================================================== #


def run_risk_book(
    screen: pd.DataFrame,
    cfg: KalmanPortfolioConfig,
    draws: ScreenDraws,
) -> Any:
    """Size a CVaR-aware long book, reusing :mod:`RiskBookModel` unchanged.

    ``compute_cvar_aware_book`` needs the screen frame to already carry
    ``er_mean`` / ``er_sd`` / ``er_p05`` and ``mc_prob_pos``. Without them
    ``expected_sharpe_ratio`` silently becomes NaN, ``tail_risk`` loses its
    Monte-Carlo loss leg and ``p_upside_pos_cond`` degrades to
    ``p_upside_pos * kalman_gain`` -- three quiet degradations rather than one
    loud failure, which is why :func:`run_screen` builds those columns first.

    ``idata`` is passed as ``None``: the only posterior read left in
    ``compute_cvar_aware_book`` is the legacy ``achieve_prob`` fallback for
    ``kalman_gain``, and v2 posteriors have never carried that variable -- the
    screen supplies the column.

    Returns
    -------
    RiskBook or None
        ``None`` when the book cannot be computed; the caller keeps going with
        the screen alone rather than losing the whole replay.
    """
    from probabilistic_ml_model.pymc_models.RiskBookModel import compute_cvar_aware_book

    try:
        # Labels for the capped dimensions, aligned BY ISIN and never filled.
        # `run_screen` returns the screen sorted by `expected_upside` while the
        # draws stay in handoff order, so a positional attach would give every
        # name someone else's sector with the row count still matching.
        group_labels: dict[str, np.ndarray] = {}
        keyed = screen.drop_duplicates("isin").set_index("isin")
        for dim in (cfg.group_caps or {}):
            if dim not in keyed.columns:
                logger.warning(
                    "group cap set for %r but the screen has no such column, so "
                    "the cap CANNOT be applied.", dim,
                )
                continue
            group_labels[dim] = keyed[dim].reindex(screen["isin"]).to_numpy()

        book = compute_cvar_aware_book(
            None,
            draws.eu,
            screen,
            alpha=cfg.cvar_alpha,
            cap=cfg.weight_cap,
            # A CEILING since 2026-08-28, not the book size. `book_min_weight`
            # decides breadth; this only bounds it.
            max_names=cfg.k_book,
            min_weight=cfg.book_min_weight,
            group_labels=group_labels,
            group_caps=cfg.group_caps,
            p_long=cfg.p_long,
            mcap_r_max=cfg.mcap_global_r_max,
            return_draws=draws.pooled_returns,
            return_draws_isins=draws.isins,
            tail_risk_vol_floor_k=cfg.tail_risk_vol_floor_k,
        )
        logger.info(
            "Risk book: %d names, port_up %.3f, STARR %.3f, div %.3f",
            int(book.summary.get("n_book", 0)),
            book.summary.get("port_up", float("nan")),
            book.summary.get("starr_book", float("nan")),
            book.summary.get("div", float("nan")),
        )
        return book
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Risk book failed: %s", exc, exc_info=True)
        return None


# =========================================================================== #
# §P1  Forecast                                                               #
# =========================================================================== #


def run_forecast(
    handoff: ForecastHandoff,
    cfg: KalmanPortfolioConfig,
    report: GateReport,
    *,
    mc_summary: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Simulate forward returns and, where possible, contrast the two engines."""
    fc = forecast_from_posterior(handoff, config=cfg.forecast_config())
    out: dict[str, Any] = {"draws": fc}

    summary = summarize_forecast(fc, terminal=False)
    terminal = summarize_forecast(fc, terminal=True).rename(
        columns=lambda c: c if c == "isin" else f"{c}_terminal"
    )
    summary = summary.merge(terminal, on="isin", how="left")
    summary["backend"] = fc.backend
    summary["factor_share"] = fc.factor_share
    summary["horizon_days"] = fc.horizon_days
    out["summary"] = summary

    if mc_summary is not None and len(mc_summary):
        contrast = compare_forecast_engines(fc, mc_summary)
        out["engines"] = contrast
        if "sd_ratio" in contrast.columns and contrast["sd_ratio"].notna().any():
            median = float(contrast["sd_ratio"].median())
            report.add(GateResult(
                name="portfolio_engine_contrast",
                passed=True,
                value=f"{median:.3f}x",
                threshold="reported, not gated",
                blocking=False,
                detail=(
                    f"{len(contrast)} names matched by ISIN; forward dispersion "
                    f"against the AR simulator's, whose decay is a hand-set rho "
                    f"rather than the fitted OU kernel"
                ),
            ))

    # `forecast_factor_effect`, carried over from the v2 workflow's §15 when that
    # stage was retired. How much diversification the shared factors removed.
    # ALWAYS PASSES: `factor_share` is a prior, so this measures a prior's
    # consequence, and a gate on an assumption would only test that the
    # assumption was applied. At 0.0 the shocks are cross-sectionally
    # independent, which is what the AR simulator assumes -- and that assumption
    # is why a LONG book can report a positive expected shortfall, because
    # pooling names averages their idiosyncratic risk to nearly nothing.
    try:
        from dataclasses import replace as _replace

        independent = forecast_from_posterior(
            handoff, config=_replace(cfg.forecast_config(), factor_share=0.0)
        )
        k = min(cfg.max_names or 50, fc.n_isin)
        w = np.full(k, 1.0 / k)
        rows = slice(0, k)
        sd_joint = float((w @ fc.terminal[rows]).std())
        sd_indep = float((w @ independent.terminal[rows]).std())
        ratio = sd_joint / sd_indep if sd_indep > 0 else float("nan")
        report.add(GateResult(
            name="portfolio_factor_effect",
            passed=True,
            value=f"{ratio:.2f}x wider",
            threshold="reported, not gated",
            blocking=False,
            detail=(
                f"equal-weight {k}-name book sd {sd_joint:.4f} with shared "
                f"factors vs {sd_indep:.4f} with independent shocks, at "
                f"factor_share {cfg.factor_share}"
            ),
        ))
        logger.info(
            "factor structure widens an equal-weight %d-name book's sd by %.2fx "
            "(%.4f -> %.4f); independent shocks make diversification free",
            k, ratio, sd_indep, sd_joint,
        )
    except Exception as exc:  # pragma: no cover - additive, never fatal
        logger.info("factor-effect contrast skipped: %s", exc)
    return out


# =========================================================================== #
# §P2  The two prior sweeps                                                   #
# =========================================================================== #


def run_prior_sweeps(
    handoff: ForecastHandoff,
    cfg: KalmanPortfolioConfig,
    report: GateReport,
    *,
    which: Sequence[str] = ("factor_share",),
    rank_values: Optional[np.ndarray] = None,
) -> dict[str, pd.DataFrame]:
    """Grid the two priors the whole departure from consensus rests on.

    ``factor_share`` is swept by re-simulating; ``forecast_error_multiplier`` is swept
    by rescaling the latent's dispersion about its cross-sectional mean, which is what
    the shrinkage does to it. Neither reports the parameter's own posterior -- that
    would say only that the prior was applied.
    """
    out: dict[str, pd.DataFrame] = {}

    if "factor_share" in which:
        frame = sweep_factor_share(
            handoff,
            cfg.factor_share_grid,
            config=cfg.forecast_config(),
            baseline_share=cfg.factor_share,
            k_book=cfg.max_names or 50,
            rank_values=rank_values,
        )
        out["factor_share"] = frame
        span = float(frame["book_sd_ratio"].max() - frame["book_sd_ratio"].min())
        overlap = frame["top_k_overlap"].min()
        report.add(GateResult(
            name="portfolio_factor_sensitivity",
            passed=True,
            value=f"book sd spans {span:.2f}x, min top-{cfg.max_names or 50} overlap {overlap}",
            threshold="reported, not gated",
            blocking=False,
            detail=(
                f"grid {list(cfg.factor_share_grid)} around the shipped "
                f"{cfg.factor_share}; per-name er_sd moves at most "
                f"{frame['er_sd_max_abs_diff'].max():.2e}, which is the "
                f"variance-preservation property measured rather than assumed"
            ),
        ))

    if "multiplier" in which:
        out["multiplier"] = _sweep_multiplier(handoff, cfg)

    return out


def _sweep_multiplier(
    handoff: ForecastHandoff, cfg: KalmanPortfolioConfig
) -> pd.DataFrame:
    """Grid ``forecast_error_multiplier`` on the handoff's latent.

    The shrinkage pulls each name's latent toward the pooled cross-sectional mean by a
    factor set by this multiplier. Sweeping it here shows what the SCREEN's departure
    from consensus is worth: at 0 the latent is unshrunk and the screen reproduces
    analyst consensus almost exactly -- run ``49e84d7e9d59`` measured Spearman
    0.999995 -- and every result downstream is a function of a value chosen from a
    feasible band rather than estimated.

    This is an approximation of the shipped shrinkage, not a re-implementation of it:
    it applies the same *shape* (a pull toward the pooled mean) to the persisted
    latent. Use ``scripts/profile_forecast_error.py`` for the exact form. What it is
    for is the sensitivity, and the sensitivity is a property of the shape.
    """
    from scipy import stats as _stats

    mu = handoff.mu_std
    pooled = mu.mean(axis=0, keepdims=True)
    base_rank = None
    rows: list[dict[str, Any]] = []
    for kappa in cfg.multiplier_grid:
        # kappa=0 -> unshrunk (the pass-through); larger -> pulled harder to pooled.
        weight = 1.0 / (1.0 + float(kappa))
        shrunk = pooled + weight * (mu - pooled)
        name_mean = shrunk.mean(axis=1)
        if base_rank is None or float(kappa) == 1.0:
            base_rank = name_mean.copy()
        rows.append({
            "forecast_error_multiplier": float(kappa),
            "shrink_weight": weight,
            "cross_sectional_sd": float(name_mean.std()),
            "spearman_vs_unshrunk": float(
                _stats.spearmanr(name_mean, mu.mean(axis=1)).statistic
            ),
            "mean_abs_revision": float(np.mean(np.abs(name_mean - mu.mean(axis=1)))),
        })
    frame = pd.DataFrame(rows)
    logger.info(
        "multiplier sweep: cross-sectional sd falls %.4f -> %.4f across %s; the "
        "screen's whole departure from consensus is a function of this knob",
        frame["cross_sectional_sd"].iloc[0], frame["cross_sectional_sd"].iloc[-1],
        list(cfg.multiplier_grid),
    )
    return frame


# =========================================================================== #
# §P3  Decision books                                                         #
# =========================================================================== #


def run_decision_books(
    forecast: dict[str, Any],
    handoff: ForecastHandoff,
    cfg: KalmanPortfolioConfig,
    report: GateReport,
    *,
    screen: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Size one book per ranking arm, on one posterior, and measure their disagreement.

    The arms are contrasts. ``cfg.rank_arms[0]`` is labelled the recommendation and
    the rest are labelled contrast, because a reader handed two books from one
    posterior with no statement of precedence will use whichever they find first.
    """
    fc = forecast["draws"]
    isins = np.asarray(fc.isins)

    groups = _group_labels(handoff, screen, cfg.sector_col, isins)
    # Labels for every FURTHER capped dimension. `groups` / `sector_cap` stay the
    # sector pair so the existing gate and the figure layer keep reading what
    # they always read; anything else the caller caps is resolved here.
    label_map: dict[str, np.ndarray] = {}
    for dim in cfg.group_caps:
        if dim == cfg.sector_col:
            continue
        labels = _group_labels(handoff, screen, dim, isins)
        if labels is None:
            logger.warning(
                "group cap set for %r but no labels are resolvable from the "
                "screen or the handoff, so the cap CANNOT be applied.", dim,
            )
            continue
        label_map[dim] = labels
    eligible = None
    veto_names: set[str] = set()
    if screen is not None:
        # Computed whether or not it is APPLIED. With the veto on, the book cannot
        # contain a flagged name, so an overlap of zero would be true by construction
        # and would read as reassurance; what is worth reporting then is how many
        # names it removed from eligibility. Measuring is not optional; applying is.
        sizeable, veto_unseen = recs.size_down_mask(
            screen, isins, return_unseen=True
        )
        veto_names = set(isins[~sizeable].tolist())
        if cfg.apply_size_down_veto:
            eligible = sizeable
        # A name the watch never saw comes back sizeable, which is
        # indistinguishable from one it examined and cleared. That silence sized
        # MaaT Pharma at 9.13 % on two analysts -- the watch's own thin-coverage
        # condition -- because the screen came from a different fit and did not
        # contain it. Recorded as a finding rather than swallowed; the vintage
        # gate is what will stop it happening.
        if veto_unseen:
            report.add(GateResult(
                name="portfolio_size_down_coverage",
                passed=False,
                value=f"{veto_unseen} of {len(isins)} names unseen by the watch",
                threshold="the watch must see every eligible name",
                blocking=False,
                detail=(
                    "These names are sizeable by silence, not by assessment: the "
                    "frame the watch was scored on does not contain them, so no "
                    "leg of the veto could be evaluated. Check that the screen "
                    "and the handoff come from the same fit."
                ),
            ))

    rank_source = screen if screen is not None else handoff.identity
    books: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []
    for position, arm in enumerate(cfg.rank_arms):
        kwargs: dict[str, Any] = {}
        if arm in RANKING_RULES_EXTERNAL:
            col = RANKING_RULES[arm]
            if rank_source is None or col not in getattr(rank_source, "columns", []):
                logger.warning(
                    "skipping arm %r: %r is a screen column and no screen carrying it "
                    "was supplied. It is not something the forward draws can produce.",
                    arm, col,
                )
                continue
            kwargs = dict(
                rank_values=pd.to_numeric(rank_source[col], errors="coerce").to_numpy(),
                rank_isins=rank_source["isin"].to_numpy(),
            )
        book = optimize_portfolio(
            fc.terminal, isins,
            max_names=cfg.max_names,
            min_weight=cfg.min_weight,
            objective=cfg.objective,
            cap=cfg.weight_cap,
            kelly_multiplier=cfg.kelly_multiplier,
            eligible=eligible,
            rank_by=arm,
            groups=groups,
            sector_cap=cfg.sector_cap,
            group_labels=label_map,
            group_caps=cfg.group_caps,
            relative_denominator_q=cfg.relative_denominator_q,
            random_seed=cfg.random_seed,
            **kwargs,
        )
        books[arm] = book

        frame = book.analytics.copy()
        # Label columns attached HERE, by ISIN, rather than only at export: the
        # figure layer reads the in-memory frame, and a sector panel that silently
        # skips because the column arrives one step later is worse than no panel.
        if groups is not None:
            keyed_groups = pd.Series(groups, index=isins)
            frame[cfg.sector_col] = keyed_groups.reindex(
                frame["isin"].to_numpy()).to_numpy()
        if screen is not None and "name" in screen.columns:
            keyed = screen.drop_duplicates("isin").set_index("isin")["name"]
            frame["name"] = keyed.reindex(frame["isin"].to_numpy()).to_numpy()
        frame["rank_by"] = arm
        # The precedence declaration, on the row rather than in a docstring.
        frame["book_role"] = "recommendation" if position == 0 else "contrast"
        # The same three columns `10b_risk_book_v2` now carries, so the two books
        # can be read side by side and told apart on their own rows rather than
        # by knowing which file they came from.
        frame["book_engine"] = f"log_growth:{arm}"
        frame["book_universe"] = "unscreened"
        frame["backend"] = fc.backend
        frame["factor_share"] = fc.factor_share
        frame["sector_cap"] = cfg.sector_cap if cfg.sector_cap is not None else np.nan
        frame["size_down_veto_applied"] = bool(cfg.apply_size_down_veto)
        frames.append(frame)

        _report_book_gates(book, arm, veto_names, report, cfg)

    out: dict[str, Any] = {"books": books}
    if frames:
        out["decision_frame"] = pd.concat(frames, ignore_index=True)
    if len(books) > 1:
        out["agreement"] = _book_agreement(books, cfg.max_names, report)
    if books:
        first = books[cfg.rank_arms[0]] if cfg.rank_arms[0] in books else next(iter(books.values()))
        out["ergodicity"] = ergodicity_report(fc.terminal)
        held = [i for i, s in enumerate(isins) if s in set(first.weights.index)]
        if held:
            w = first.weights.reindex(isins[held]).to_numpy()
            out["wealth_curve"] = terminal_wealth_curve(w @ fc.terminal[held])
            # The mean-variance CONTRAST, over the recommendation's own holdings.
            # Not over the universe: the quadratic objectives need a covariance
            # and cannot take 6,500 names, and the interesting question is what a
            # different objective does with the SAME candidates rather than what
            # it picks from a pool this book never saw.
            out["frontier"], out["tangency"] = _frontier_contrast(
                fc.terminal[held], isins[held], first, cfg
            )
    return out


def _frontier_contrast(
    draws: np.ndarray,
    isins: np.ndarray,
    book: Any,
    cfg: "KalmanPortfolioConfig",
) -> tuple[Optional[pd.DataFrame], Optional[dict[str, Any]]]:
    """Solved frontier and tangency point over the recommendation's holdings.

    A labelled contrast, never the recommendation. Mean-variance treats
    volatility as total risk -- symmetric, so it charges a name for its upside --
    optimises one period, and is famously unstable in the return estimates, which
    here are posterior means of a latent that moves between refreshes. The book
    ships on expected log growth; this says what the single-period answer would
    have been on the same draws.

    Returns ``(None, None)`` when the contrast cannot be drawn, with the reason
    logged: a silently absent panel reads as a frontier that was not worth
    drawing.
    """
    if len(isins) < 3:
        logger.info("frontier contrast skipped: %d holdings", len(isins))
        return None, None
    try:
        frontier = efficient_frontier(
            draws, isins, n_points=30, cap=cfg.weight_cap,
            group_caps=cfg.group_caps or None,
        )
        tangency = tangency_portfolio(
            draws, isins, cap=cfg.weight_cap, group_caps=cfg.group_caps or None,
        )
    except Exception as exc:
        logger.info("frontier contrast skipped: %s", exc)
        return None, None

    frontier = frontier.assign(
        objective="target_return",
        book_objective=cfg.objective,
        book_log_growth=float(book.summary.get("log_growth", float("nan"))),
        book_return=float(book.summary.get("port_expected", float("nan"))),
        book_vol=float(book.summary.get("port_vol", float("nan"))),
        tangency_sharpe=float(tangency["sharpe"]),
        tangency_n_holdings=float(tangency["n_holdings"]),
    )
    logger.info(
        "frontier contrast over %d holdings: tangency Sharpe %.3f on %d names "
        "against the shipped book's %.3f return / %.3f vol",
        len(isins), tangency["sharpe"], tangency["n_holdings"],
        frontier["book_return"].iloc[0], frontier["book_vol"].iloc[0],
    )
    return frontier, tangency


def _group_labels(
    handoff: ForecastHandoff,
    screen: Optional[pd.DataFrame],
    column: str,
    isins: np.ndarray,
) -> Optional[np.ndarray]:
    """Per-name group labels aligned to ``isins`` **by key**, from screen or handoff.

    **Never filled.** An ISIN the source does not carry gets ``None``, not
    ``"Unknown"``. The fill was source-level rather than per-row, so a screen
    matching 6,362 of 6,507 names fully shadowed a handoff carrying all 6,507 and
    manufactured a 145-name group that then cleared ``MIN_GROUP_N``, took the
    largest shrunk excess on the run and hid a 30 % sector cap being satisfied at
    a true 59.1 %. ``None`` propagates: it matches no group in
    ``group_allocation_signals``, receives no spill in the cap projection, and is
    counted by ``unlabelled_<dim>_weight`` so the concentration gate can block.

    **The handoff comes first.** It carries the full 42-column identity block,
    written from ``panel.frame`` by the fit itself, so it is in-vintage by
    construction and covers every name the posterior does. The screen is a
    derived frame and is consulted only for a column the handoff does not have.
    The order used to be the other way round, and the fallback was source-level
    rather than per-row: a screen matching 6,362 of 6,507 names fully shadowed a
    complete handoff, and the 145 it did not match were filled to ``"Unknown"``.
    """
    def _keyed(frame: pd.DataFrame) -> np.ndarray:
        keyed = frame.drop_duplicates("isin").set_index("isin")[column]
        aligned = keyed.reindex(isins)
        present = aligned.notna().to_numpy()
        missing = int((~present).sum())
        if missing:
            logger.warning(
                "%d of %d names carry no %r label and are left UNLABELLED. They "
                "are not a group -- a cap on this dimension cannot reach them and "
                "the concentration gate will say so.",
                missing, len(present), column,
            )
        return np.where(present, aligned.astype(str).to_numpy(), None)

    ident = handoff.identity
    if ident is not None and column in getattr(ident, "columns", []):
        return _keyed(ident)
    if screen is not None and column in screen.columns and "isin" in screen.columns:
        return _keyed(screen)
    codes = handoff.coord_idx.get(column)
    uniques = handoff.coord_uniques.get(column)
    if codes is not None and uniques is not None:
        # Codes index a complete level vocabulary, so there is nothing to miss.
        return np.asarray(uniques, dtype=object)[np.asarray(codes)]
    logger.info("no %r labels available; sector concentration will not be reported", column)
    return None


def _report_book_gates(
    book: Any, arm: str, veto_names: set[str],
    report: GateReport, cfg: KalmanPortfolioConfig,
) -> None:
    """The four per-book findings, each recorded rather than thresholded."""
    summary = book.summary
    held = set(book.weights.index.astype(str))

    n_book = float(summary.get("n_book", float("nan")))
    n_elig = float(summary.get("n_eligible", float("nan")))
    eff_n = float(summary.get("effective_n", float("nan")))
    report.add(GateResult(
        name="portfolio_solver_breadth",
        passed=True,
        value=f"{int(n_book) if np.isfinite(n_book) else '?'} of "
              f"{int(n_elig) if np.isfinite(n_elig) else '?'} eligible, "
              f"effective N {eff_n:.1f}",
        threshold=(f"ceiling {int(summary['max_names'])}"
                   if np.isfinite(summary.get("max_names", float("nan")))
                   else "no ceiling set"),
        blocking=False,
        detail=(
            f"[{arm}] objective {summary.get('objective', '?')}, minimum weight "
            f"{summary.get('min_weight', float('nan')):.2%}"
            f"{', CEILING BINDING' if summary.get('breadth_binding') else ''}. "
            f"Effective N is the Herfindahl reciprocal: the gap between it and "
            f"the position count is how much of the book is rounding error."
        ),
    ))

    report.add(GateResult(
        name="portfolio_kelly_interior",
        passed=True,
        value=f"{summary.get('book_kelly_interior_share', float('nan')):.1%} of book, "
              f"{summary.get('kelly_interior_share', float('nan')):.1%} of universe",
        threshold="reported, not gated",
        blocking=False,
        detail=(
            f"[{arm}] a pinned fraction means E[log(1+f*r)] never turned over, i.e. "
            f"no draw loses money -- a statement about the simulation's left tail, "
            f"not about the opportunity"
        ),
    ))

    pctile = summary.get("book_denominator_pctile_max", float("nan"))
    report.add(GateResult(
        name="portfolio_rank_denominator",
        passed=True,
        value="n/a (bounded rule)" if not np.isfinite(pctile) else f"max pctile {pctile:.3f}",
        threshold="reported, not gated",
        blocking=False,
        detail=(
            f"[{arm}] every book name sits at or below this percentile of the "
            f"eligible universe's ranking denominator. A bounded rule has no "
            f"denominator, which is precisely why it cannot fail this way."
        ),
    ))

    tie = float(summary.get("rank_tie_span", 0.0))
    report.add(GateResult(
        name="portfolio_rank_tie_span",
        passed=True,
        value=f"{int(tie)} tied at the cut",
        threshold="reported, not gated",
        blocking=False,
        detail=f"[{arm}] broken on {summary.get('rank_tiebreak') or 'nothing'}",
    ))

    if "top_group_weight" in summary:
        # A cap enforced on a label that can be absent is not a cap. `groupby`
        # drops null keys, so unlabelled weight leaves the total rather than
        # joining a bucket, and the maximum is taken over what survives: run
        # `b00f8d8ca093` reported 30.0 % in Health Care against a 30 % cap while
        # the true weight was 59.1 %, because three of the sector's names had no
        # label. Reported on every run; BLOCKING only where it changes an answer,
        # which is when a cap is set on the dimension and some weight escaped it.
        unlabelled = float(summary.get("unlabelled_group_weight", 0.0))
        capped = cfg.sector_cap is not None
        clean = unlabelled <= 0.0
        report.add(GateResult(
            name="portfolio_sector_concentration",
            passed=clean or not capped,
            value=(
                f"{summary['top_group_weight']:.1%} in "
                f"{summary.get('top_group', '?')}"
                + (f", {unlabelled:.1%} unlabelled" if unlabelled > 0 else "")
            ),
            threshold=(f"cap {cfg.sector_cap:.0%}, 0% unlabelled" if capped
                       else "no cap set"),
            blocking=capped and not clean,
            detail=(
                f"[{arm}] {int(summary.get('n_groups', 0))} groups held. No cap is "
                f"also a decision -- it should just never be taken by omission."
                + (
                    f" {unlabelled:.1%} of the book carries no "
                    f"{summary.get('top_group_dimension', 'group')} label, so the "
                    f"reported weight is a maximum over the labelled part only and "
                    f"the cap was NOT enforced on that share. Fix the labels; do "
                    f"not raise the cap."
                    if unlabelled > 0 else ""
                )
            ),
        ))

    if veto_names:
        if cfg.apply_size_down_veto:
            value = f"{len(veto_names)} names removed from eligibility (veto APPLIED)"
            note = ("the book holds none of them by construction, so the informative "
                    "number is how many the veto took out of the running")
        else:
            value = f"{len(held & veto_names)} of {len(held)} book names flagged"
            note = ("the veto is NOT applied; this is how much the ranking and the "
                    "watch disagree")
        report.add(GateResult(
            name="portfolio_size_down_overlap",
            passed=True,
            value=value,
            threshold="reported, not gated",
            blocking=False,
            detail=(
                f"[{arm}] the size-down watch (wide posterior, or an analyst panel of "
                f"two) is a veto orthogonal to the ranking; those conditions are what "
                f"produce a forward simulation with no credible left tail. {note}."
            ),
        ))


def _book_agreement(
    books: dict[str, Any], max_names: Optional[int], report: GateReport
) -> pd.DataFrame:
    """Pairwise membership overlap between the arms, and a gate on the worst.

    Jaccard, not a raw count over a shared ``k``: the arms no longer hold the
    same number of names, because breadth is solved per arm. Comparing
    ``overlap`` against one nominal ``k`` would make a small disciplined book
    look like it disagreed with a large one when it may be a subset of it.
    """
    rows: list[dict[str, Any]] = []
    names = list(books)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa = set(books[a].weights.index.astype(str))
            sb = set(books[b].weights.index.astype(str))
            union = len(sa | sb)
            rows.append({
                "arm_a": a, "arm_b": b,
                "overlap": len(sa & sb),
                "jaccard": len(sa & sb) / union if union else float("nan"),
                "n_a": len(sa), "n_b": len(sb),
                # Of the SMALLER book, how much the larger one contains. A subset
                # relationship and a genuine disagreement have very different
                # Jaccards at unequal sizes, and only this separates them.
                "containment": (len(sa & sb) / min(len(sa), len(sb))
                                if min(len(sa), len(sb)) else float("nan")),
                "max_names": float(max_names) if max_names is not None else float("nan"),
            })
    frame = pd.DataFrame(rows)
    if len(frame):
        worst = frame.loc[frame["jaccard"].idxmin()]
        report.add(GateResult(
            name="portfolio_book_agreement",
            passed=True,
            value=f"Jaccard {worst['jaccard']:.2f}, {int(worst['overlap'])} shared "
                  f"of {int(worst['n_a'])}/{int(worst['n_b'])} "
                  f"({worst['arm_a']} vs {worst['arm_b']})",
            threshold="reported, not gated",
            blocking=False,
            detail=(
                f"{len(names)} arms on one posterior. A low overlap is a statement "
                f"about how underdetermined the ranking choice is, not about which "
                f"arm is right -- that is a question about realised returns."
            ),
        ))
    return frame


# =========================================================================== #
# §P4  Mean-model arms (Max-and-Smooth)                                       #
# =========================================================================== #


def run_mean_model_arms(
    handoff: ForecastHandoff,
    cfg: KalmanPortfolioConfig,
    report: GateReport,
    *,
    arms: Sequence[str],
) -> Optional[pd.DataFrame]:
    """Does book membership survive a change of MEAN model?

    Max-and-Smooth regenerates the latent under an arm in seconds. It **cannot**
    produce a forecast-ready posterior on its own: ``COVARIANCE_FIELDS`` freezes
    ``nu``, ``ou_length_scale_days_*``, ``log_sigma_*`` and ``likelihood`` -- exactly
    the quantities the forward simulation reads. Here that freeze is a *feature*: it
    isolates the mean-structure effect instead of confounding it with a refitted noise
    model, and the covariance is borrowed from this handoff's baseline fit.

    It **ranks arms; it does not decide them.** Every frame is stamped
    ``backend='max_and_smooth'``, the convention the fast comparison already
    established, and ``drift_strict`` is refused here for the same reason the fast
    screener refuses it: it changes the design matrix, which is the quantity the Max
    step froze.

    Returns ``None`` when the stage cannot run, with the reason logged -- a contrast
    that silently degrades into a comparison of one arm against itself is worse than
    no contrast.
    """
    refused = [a for a in arms if a == "drift_strict"]
    if refused:
        raise ValueError(
            "drift_strict changes the design matrix, which is the quantity the "
            "Max-and-Smooth step conditioned on. Screening it here would contrast "
            "two arms against one arm's noise model. Use the exact harness: "
            "`python pymc_kalman_filter_pt_v2.py --compare baseline,drift_strict`."
        )
    logger.warning(
        "mean-model arms require a live panel and posterior, which a handoff "
        "deliberately does not carry (it stores the four quantities the SIMULATOR "
        "reads, not the model graph). Run this stage from the v2 workflow with "
        "--compare-fast; requested arms: %s",
        list(arms),
    )
    report.add(GateResult(
        name="portfolio_mean_model_arms",
        passed=True,
        value="not run from a replay",
        threshold="needs a live panel",
        blocking=False,
        detail=(
            "Max-and-Smooth screens MEAN-structure arms against a frozen covariance. "
            "The handoff carries no panel or model graph, so this stage runs from the "
            "v2 workflow rather than from a replay."
        ),
    ))
    return None


# =========================================================================== #
# §P5  Recommendations                                                        #
# =========================================================================== #


def run_recommendations_v2(
    forecast: dict[str, Any],
    handoff: ForecastHandoff,
    decision: dict[str, Any],
    cfg: KalmanPortfolioConfig,
    *,
    screen: Optional[pd.DataFrame] = None,
    diagnostics: Optional[pd.DataFrame] = None,
    panel_frame: Optional[pd.DataFrame] = None,
    report: Optional[GateReport] = None,
    render: bool = True,
) -> dict[str, Any]:
    """Turn the ranked frames into a posture: groups, actions, vetoes, reliability.

    The group signals run over the forecast's **terminal** returns, which is the
    quantity the books are sized from -- so the posture and the book answer the same
    question. v1's version runs over the upside posterior; both are legitimate and the
    two are a contrast.
    """
    fc = forecast["draws"]
    isins = np.asarray(fc.isins)

    coords = pd.DataFrame({"isin": isins})
    for level in ("sector", "trading_region", "country_name", "size_class", "style_class"):
        labels = _group_labels(handoff, screen, level, isins)
        if labels is not None:
            coords[level] = labels

    confidence = None
    if screen is not None and "shrink_gain" in screen.columns:
        confidence = (
            screen.drop_duplicates("isin").set_index("isin")["shrink_gain"]
            .reindex(isins).to_numpy(dtype="float64")
        )

    out: dict[str, Any] = {}
    out["group_signals"] = recs.group_allocation_signals(
        fc.terminal, coords, confidence=confidence
    )
    out["reliability"] = recs.reliability_posture(diagnostics=diagnostics)

    if screen is not None:
        conf_scale = float(np.nanmean(confidence)) if confidence is not None else 1.0
        # Vestigial since 2026-08-31 and kept as a floor, not as a path: this
        # file builds the screen now and `run_screen` always emits
        # `feat_analyst_rating` from the handoff's panel vectors, so `main` passes
        # `panel_frame=None`. It stays because `consensus_gap` is the one column
        # that says where this model DIFFERS from the panel it reproduces at
        # Spearman 0.992, and losing it silently is what this guard is for.
        actions_input = screen
        if "feat_analyst_rating" not in screen.columns and panel_frame is not None:
            if {"isin", "feat_analyst_rating"} <= set(panel_frame.columns):
                rating = (panel_frame.drop_duplicates("isin")
                          .set_index("isin")["feat_analyst_rating"])
                actions_input = screen.assign(
                    feat_analyst_rating=pd.to_numeric(
                        rating.reindex(screen["isin"].astype(str)).to_numpy(),
                        errors="coerce",
                    )
                )
                logger.info(
                    "analyst consensus read from the panel frame; re-export the "
                    "screen to carry it directly."
                )
        try:
            out["actions"] = recs.name_action_list(
                actions_input, confidence_scale=conf_scale
            )
        except KeyError as exc:
            logger.info("name actions skipped: %s", exc)
        if report is not None and out.get("actions") is not None:
            _report_action_ladder(out["actions"], conf_scale, report)
        out["watch"] = recs.size_down_watch(screen)
        out["demoted"] = recs.demotion_list(screen)

    if render:
        first = cfg.rank_arms[0] if cfg.rank_arms else DEFAULT_RANKING_RULE
        book = decision.get("books", {}).get(first)
        recs.render_recommendations(
            reliability=out.get("reliability"),
            group_signals=out.get("group_signals"),
            actions=out.get("actions"),
            watch=out.get("watch"),
            demoted=out.get("demoted"),
            book=(book.analytics.loc[book.analytics["weight"] > 0]
                  if book is not None else None),
            book_summary=book.summary if book is not None else None,
            title="KALMAN PORTFOLIO - FORECAST + DECISION REPLAY",
        )
    return out



def _report_action_ladder(
    actions: pd.DataFrame, confidence_scale: float, report: GateReport
) -> None:
    """Record how the universe distributes over the five rungs.

    The finding this exists to prevent going unnoticed: the previous
    three-valued list returned 83.5 % BUY on run 807df55e7158 and nothing said
    so. Five rungs do not fix that on their own -- the gates are scaled by the
    universe-mean confidence, so a low mean pulls the STRONG threshold down onto
    what used to be the ordinary one, and the top rung inherits the same
    population under a stronger name.
    """
    counts = actions["action"].value_counts()
    total = float(len(actions)) or float("nan")
    top = float(counts.get("STRONG BUY", 0)) / total
    gap = (pd.to_numeric(actions["consensus_gap"], errors="coerce")
           if "consensus_gap" in actions.columns else None)
    report.add(GateResult(
        name="portfolio_action_ladder",
        passed=True,
        value=" / ".join(
            f"{int(counts.get(a, 0))} {a}" for a in recs.ACTIONS
        ),
        threshold="reported, not gated",
        blocking=False,
        detail=(
            f"{top:.1%} of the universe on the top rung; gates scaled by "
            f"confidence {confidence_scale:.3f}, so STRONG BUY sits at "
            f"p >= {float(actions['gate_strong_hi'].iloc[0]):.3f} rather than at "
            f"its nominal 0.90. A top rung holding most of the universe is a "
            f"statement about the forward simulation's left tail, not about "
            f"conviction."
            + (f" Median consensus_gap {gap.median():+.2f} on the 1-5 analyst "
               f"scale, {float((gap > 0).mean()):.0%} more bullish than the panel."
               if gap is not None and gap.notna().any() else
               " consensus_gap unavailable: no analyst rating on the screen or "
               "panel frame.")
        ),
    ))


#: Screen/risk-book column -> the name it is PUBLISHED under.
#:
#: Moved here with the export on 2026-08-31, verbatim. Two of these entries are
#: one-release guards rather than live renames: `compute_cvar_aware_book` stopped
#: emitting the `expected_sharpe` alias on 2026-08-24 and `exp_vol` now keeps its
#: own name until `EXPORT_REDUNDANT_COLUMNS` drops it as a duplicate of `er_sd`.
#: They stay because a stale or pinned `RiskBookModel` that still emitted both
#: would otherwise rename one onto the other and produce TWO columns under one
#: name -- not merely untidy: `export_ranking_range` then hands `pd.to_numeric` a
#: DataFrame instead of a Series and the whole export dies, after the fit has
#: already been paid for.
_CANONICAL_RENAMES: dict[str, str] = {
    "starr": "reward_to_cvar",
    "cvar05": "cvar_5pct_kalman",
    "book_weight": "cvar_book_weight",
    "expected_upside": "expected_return_kalman",
    "expected_pt": "price_target_kalman",
    "last_price": "original_price",
    "observed_pt": "original_target",
}


# =========================================================================== #
# §P6  Export                                                                 #
# =========================================================================== #

#: Frame stem -> what it holds. The stems continue the v2 section numbering, so a
#: reader who knows that tree knows where these land.
PORTFOLIO_FRAMES: dict[str, str] = {
    "15c_forecast_summary": "per-name forward returns, pooled and terminal",
    "15c_forecast_engines": "this engine against the shipped AR simulator, by ISIN",
    "15d_factor_share_sweep": "book dispersion and membership across the factor_share grid",
    "15d_multiplier_sweep": "cross-sectional spread across the forecast-error grid",
    "15e_decision_books": ("one row per (isin, rank_by); book_role names the "
                          "recommendation, book_engine/book_universe distinguish "
                          "it from the CVaR book at 10b"),
    "15e_book_agreement": "pairwise membership overlap between the ranking arms",
    "15e_frontier": ("the solved mean-variance frontier over the book's "
                     "holdings, with the tangency point -- a labelled "
                     "CONTRAST, not the recommendation"),
    "14b_group_signals": "shrunk over/underweight postures by coordinate",
    "14b_name_actions": ("per-name action on the five-point analyst scale, "
                         "scored 5-1 and gapped against analyst consensus"),
    "14b_size_down_watch": "the veto list: wide posterior or thin analyst coverage",
    "09_gate_report_portfolio": "every gate this replay recorded",
}


#: Schema the replay appends to. Read at point of use, the project pattern.
PORTFOLIO_SCHEMA_ENV = "DB_PORTFOLIO_SCHEMA"
DEFAULT_PORTFOLIO_SCHEMA = "kalman_portfolio"


def _portfolio_engine(
    cfg: KalmanPortfolioConfig, run_id: str
) -> tuple[Optional[Any], str]:
    """Resolve the append target, refusing a ``run_id`` already stored.

    Returns ``(None, schema)`` whenever the write is off, `DB_URL` is unset, the
    database is unreachable, or this replay's ``run_id`` is already present --
    each of which falls back to CSV rather than failing the replay, because the
    frames are the point and the table is a convenience.

    The duplicate check is the append-only half that is easy to skip. Appending a
    second copy of one ``run_id`` does not error, does not warn, and quietly
    doubles the weight of that replay in every aggregate over the history.
    """
    if not cfg.write_analytics:
        return None, DEFAULT_PORTFOLIO_SCHEMA

    from probabilistic_ml_model.data_utils.inference_schema import (
        _validate_schema_name,
    )

    schema = _validate_schema_name(
        os.environ.get(PORTFOLIO_SCHEMA_ENV, DEFAULT_PORTFOLIO_SCHEMA)
    )
    url = os.environ.get("DB_URL")
    if not url:
        logger.info("DB_URL is not set; the replay writes CSV only")
        return None, schema
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(url)
        with engine.connect() as conn:
            if not conn.execute(
                text("SELECT 1 FROM information_schema.schemata "
                     "WHERE schema_name = :s"), {"s": schema},
            ).first():
                logger.error(
                    "schema %s does not exist; run "
                    "`python scripts/apply_kalman_portfolio_schema.py` first. "
                    "Writing CSV only.", schema,
                )
                return None, schema
            # Every table carries `run_id`, so any one of them answers this.
            existing = list(conn.execute(
                text("SELECT table_name FROM information_schema.tables "
                     "WHERE table_schema = :s"), {"s": schema},
            ))
            for (table,) in existing:
                hit = conn.execute(
                    text(f'SELECT 1 FROM {schema}."{table}" '
                         f"WHERE run_id = :r LIMIT 1"), {"r": run_id},
                ).first()
                if hit:
                    logger.error(
                        'run_id %s is already stored in %s."%s". REFUSING to '
                        "append a second copy -- an append-only store whose rows "
                        "can be silently doubled is not a history. Writing CSV "
                        "only.", run_id, schema, table,
                    )
                    return None, schema
        return engine, schema
    except Exception as exc:
        logger.error("cannot reach %s (%s); the replay writes CSV only",
                     schema, exc)
        return None, schema


def publish_canonical_table(
    frame: pd.DataFrame,
    cfg: KalmanPortfolioConfig,
    fit_run_id: str,
    replay_id: str,
    *,
    identity: Optional[pd.DataFrame] = None,
) -> dict[str, int]:
    """Publish ``analytics.kalman_filtered_price_targets_v2`` -- GEIB's only source.

    The one frame this replay writes that is NOT its own. It is the current state
    of one fit, so it stays in the `analytics` schema with `if_exists='replace'`
    and carries the **FIT's** ``run_id``.

    That is load-bearing rather than tidy. `dashboards/geib/data.py::_join_panel`
    asserts this table's `run_id` equals `analytics."04_panel_frame_v2"`'s, and
    the fit writes that panel frame. Stamping this table with the REPLAY's id
    would fail that assertion on every run, and `_join_panel` drops the panel
    block on a mismatch -- so beta, the Piotroski set, the rating mix and the
    price ladders would all silently disappear from the board, which is exactly
    the degrade-to-empty the assertion exists to produce.

    ``replay_id`` / ``replayed_at`` name the replay. Two ids because there are
    two runs behind the row, and one id cannot answer both questions.
    """
    out = frame
    if identity is not None and "isin" in out.columns:
        out = attach_identity_columns(out, identity, label=_ANALYTICS_TABLE_V2)
    out = stamp_export_provenance(out, fit_run_id, pd.Timestamp.now("UTC"))
    if "replay_id" not in out.columns:
        out = out.assign(replay_id=replay_id, replayed_at=pd.Timestamp.now("UTC"))

    path = section_path(cfg.results_path, _ANALYTICS_TABLE_V2)
    out.to_csv(path, index=False)
    if not cfg.write_analytics:
        logger.info("wrote %s (%d rows); --write publishes it to analytics",
                    path, len(out))
        return {_ANALYTICS_TABLE_V2: len(out)}

    schema = os.environ.get("DB_ANALYTICS_SCHEMA", "analytics")
    url = os.environ.get("DB_URL")
    if not url:
        logger.warning("DB_URL is not set; %s stays CSV-only and GEIB will read "
                       "the previous run", _ANALYTICS_TABLE_V2)
        return {_ANALYTICS_TABLE_V2: len(out)}
    try:
        from sqlalchemy import create_engine

        engine = create_engine(url)
        out.to_sql(_ANALYTICS_TABLE_V2, engine, schema=schema,
                   if_exists="replace", index=False, chunksize=2000)
        # The ONE file in `sql_scripts/analytics/` carrying per-column
        # documentation and the raw-decimal unit header. Regenerated here
        # because this script now produces the table -- and only on a publish,
        # so the committed schema always describes something that was published.
        write_analytics_ddl_v2(out, table=_ANALYTICS_TABLE_V2, schema=schema)
        logger.info(
            'published %s."%s" (%d rows, fit %s / replay %s). GEIB reads this '
            "table -- deploy the dashboard if the schema changed.",
            schema, _ANALYTICS_TABLE_V2, len(out), fit_run_id, replay_id,
        )
    except Exception as exc:
        logger.error("could not publish %s (%s); the CSV at %s is current and "
                     "the table is NOT", _ANALYTICS_TABLE_V2, exc, path)
    return {_ANALYTICS_TABLE_V2: len(out)}


def export_frames(
    frames: dict[str, pd.DataFrame],
    cfg: KalmanPortfolioConfig,
    run_id: str,
    *,
    identity: Optional[pd.DataFrame] = None,
) -> dict[str, int]:
    """Write the frames, stamped and identity-attached.

    Database first, CSV as the fallback -- the shape ``export_analytics`` uses,
    including the memoised ``sql_ok``: one dead connection is not eleven errors.

    **APPEND-ONLY, into ``kalman_portfolio``.** Not the analytics schema, which
    the fit DROPs and RECREATEs each export: a replay writing there would destroy
    the export it was replaying, which is why ``--write`` was a documented no-op
    until 2026-08-31. And not ``replace`` here either, because a replay is one
    observation about a fit rather than the current state of one -- three ranking
    arms and two prior sweeps only mean something against each other, and a
    replay stops being reproducible the moment its handoff is superseded.

    A ``run_id`` already present is REFUSED rather than duplicated, in the spirit
    of ``capture_panel_vintage.py``: an append-only store whose rows can be
    silently rewritten is not append-only, and one whose rows can be silently
    doubled is not a history.
    """
    # One section directory per stem, resolved through the shared layout SSOT --
    # not one `15_portfolio` bucket. The bucket was the whole replay in a single
    # folder: a forecast summary, two prior sweeps, the sized books and three
    # recommendation frames, which is four stages under one name. The stems
    # already carried the section numbers; only the directories were missing.
    root = cfg.results_path
    root.mkdir(parents=True, exist_ok=True)
    stamped = pd.Timestamp.now('UTC')
    counts: dict[str, int] = {}

    engine, schema = _portfolio_engine(cfg, run_id)
    sql_ok = engine is not None

    for stem, frame in frames.items():
        if frame is None or not len(frame):
            continue
        written = frame
        if identity is not None and "isin" in written.columns:
            written = attach_identity_columns(written, identity, label=stem)
        written = stamp_export_provenance(written, run_id, stamped)
        counts[stem] = len(written)

        # DDL only on a PUBLISHING run. The fit renders it unconditionally,
        # because it exports once and an offline run still deserves a reviewable
        # schema. A replay is different on both counts: it runs many times over
        # one fit, so re-rendering identical files is noise, and it is the thing
        # test suites drive with synthetic frames -- which is not hypothetical.
        # `sql_scripts/` is committed source, and `write_analytics_ddl_v2`
        # resolves its default path relative to the CWD, so an unconditional
        # render let a 40-name fixture overwrite a real 79-column schema.
        if cfg.write_analytics:
            try:
                write_analytics_ddl_v2(written, table=stem, schema=schema)
            except Exception as exc:  # pragma: no cover - docs are not the run
                logger.info("could not render DDL for %s: %s", stem, exc)

        wrote_table = False
        if sql_ok:
            try:
                written.to_sql(stem, engine, schema=schema, if_exists="append",
                               index=False, chunksize=2000)
                wrote_table = True
                logger.info('wrote %s."%s" (%d rows, appended)',
                            schema, stem, len(written))
            except Exception as exc:
                logger.error(
                    "SQL export failed for %s (%s); CSV from here on", stem, exc
                )
                sql_ok = False
        # CSV is written EITHER WAY. The analytics export treats it as a
        # fallback, but a replay is meant to be read from the tree it wrote --
        # the figures sit beside these frames, and a reader browsing a section
        # directory should not find half of it in a database.
        path = section_path(root, stem)
        written.to_csv(path, index=False)
        if not wrote_table:
            logger.info("wrote %s (%d rows)", path, len(written))

    return counts


# =========================================================================== #
# §P7  Orchestration                                                          #
# =========================================================================== #


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a v2 artifact if it happens to be there; never fail because it is not."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("could not read %s: %s", path, exc)
    return None


def _read_v2_artifact(
    cfg: KalmanPortfolioConfig, stem: str
) -> Optional[pd.DataFrame]:
    """Read one v2 export frame: section directory first, flat root second.

    The flat path is where every v2 frame lived before the 2026-08-27 layout
    migration. Keeping the fallback means a replay still works against a results
    tree written by an older build, and the log line says which one it read --
    a silent degradation here costs the replay its screen, and with it the sector
    labels, the `p_upside_pos_cond` arm and the size-down watch.
    """
    sectioned = section_path(cfg.v2_results_path, stem)
    frame = _read_optional_csv(sectioned)
    if frame is not None:
        return frame
    legacy = cfg.v2_results_path / f"{stem}.csv"
    frame = _read_optional_csv(legacy)
    if frame is not None:
        logger.info(
            "read %s from the pre-migration flat path; run "
            "`python pymc_kalman_filter_pt_v2.py --migrate-layout --apply` to move "
            "it into %s", stem, sectioned.parent,
        )
    else:
        logger.info("no %s under %s; stages needing it will be skipped",
                    stem, cfg.v2_results_path)
    return frame


def main(
    *,
    config: Optional[KalmanPortfolioConfig] = None,
    sweeps: Sequence[str] = (),
    render: bool = True,
    export: bool = True,
) -> dict[str, Any]:
    """Run the replay end to end.

    Returns
    -------
    dict[str, Any]
        ``run_id``, ``handoff``, ``screen``, ``screen_draws``, ``risk_book``,
        ``kalman_results``, ``forecast``, ``sweeps``, ``decision``,
        ``recommendations``, ``report``, ``export_counts``.

        The first six are new on 2026-08-31: the screen and the CVaR risk book
        moved here from the fit script, so this is where they are produced now.
    """
    cfg = config or KalmanPortfolioConfig.from_env()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    GATE_CATALOGUE.update(PORTFOLIO_GATES)

    run_id = uuid.uuid4().hex[:12]
    sha, dirty = resolve_source_revision()
    logger.info(
        "kalman_portfolio replay %s -- source %s%s",
        run_id, (sha or "?")[:7], " (dirty)" if dirty else "",
    )

    report = GateReport()
    handoff = load_handoff(cfg, report)

    # Still read, and now only for what the handoff genuinely does not carry:
    # per-parameter R-hat / ESS, which is one row per PARAMETER and has no ISIN
    # axis at all. Resolved through the section tree with a flat-path fallback for
    # a pre-2026-08-27 tree.
    #
    # Checked against the handoff's own run_id BEFORE anything reads it. This is
    # a small surface now -- the screen and the identity block moved into the
    # handoff -- but it is the surface that was never checked, and the gate is
    # what makes reading anything from beside the handoff safe to add back.
    read_frames = check_input_vintage(
        handoff, {"09_diagnostics_v2": _read_v2_artifact(cfg, "09_diagnostics_v2")},
        cfg, report,
    )
    diagnostics = read_frames["09_diagnostics_v2"]

    # THE SCREEN IS BUILT HERE, off the handoff, not read from a CSV.
    #
    # It used to be `_read_v2_artifact(cfg, "10_screen_results_v2")`, and nothing
    # checked that the file came from the fit being replayed. On run
    # `b00f8d8ca093` it did not: a 2026-08-30 handoff of 6,507 names met a
    # 2026-08-27 screen of 6,513 from a DIFFERENT fit, 145 names had no row, and
    # those 145 became a phantom sector that evaded a 30 % cap and hid three
    # unnamed positions in an eleven-name book. Building it from the handoff
    # removes the seam rather than guarding it.
    screen, screen_draws = run_screen(handoff, cfg, report)
    risk_book = run_risk_book(screen, cfg, screen_draws)


    forecast = run_forecast(
        handoff, cfg, report,
        mc_summary=screen[[c for c in (
            "isin", "er_mean", "er_sd", "er_p05", "er_p50", "er_p95", "mc_prob_pos"
        ) if c in screen.columns]],
    )

    rank_values = None
    if "p_upside_pos_cond" in screen.columns:
        keyed = screen.drop_duplicates("isin").set_index("isin")["p_upside_pos_cond"]
        rank_values = pd.to_numeric(
            keyed.reindex(np.asarray(forecast["draws"].isins)), errors="coerce"
        ).to_numpy()

    sweep_frames = run_prior_sweeps(
        handoff, cfg, report, which=sweeps, rank_values=rank_values
    ) if sweeps else {}

    decision = run_decision_books(forecast, handoff, cfg, report, screen=screen)

    if cfg.mean_model_arms:
        run_mean_model_arms(handoff, cfg, report, arms=cfg.mean_model_arms)

    recommendations = run_recommendations_v2(
        forecast, handoff, decision, cfg,
        screen=screen, diagnostics=diagnostics, panel_frame=None,
        report=report, render=render,
    )

    # The canonical frame: the risk book's analytics if we have it (it is the
    # screen plus the risk columns), otherwise the screen alone. Renames and
    # out-of-support suppression are `pymc_kalman_filter_pt_v2`'s SSOT, imported
    # rather than re-spelled -- the published column names must not depend on
    # which script happens to write them.
    kalman_results = (
        risk_book.analytics.copy() if risk_book is not None else screen.copy()
    )
    kalman_results = kalman_results.drop(columns=["expected_sharpe"], errors="ignore")
    kalman_results = kalman_results.rename(columns=_CANONICAL_RENAMES)
    # Suppression runs BEFORE anything is written, so every consumer -- including
    # the intermediate risk table -- sees the same guarded values.
    kalman_results = apply_out_of_support(kalman_results)
    # GEIB reads this table and asserts its `run_id` matches
    # `analytics."04_panel_frame_v2"`, which the FIT still writes. So the
    # canonical frame carries the FIT's id, not this replay's -- otherwise the
    # dashboard's vintage assertion fails on every run and the whole descriptive
    # block (beta, Piotroski, rating mix, price ladders) silently drops.
    #
    # `replay_id` / `replayed_at` say which replay produced the row. Two ids
    # because there are genuinely two runs behind it, and collapsing them would
    # make one of the two unanswerable.
    kalman_results = kalman_results.assign(
        replay_id=run_id, replayed_at=pd.Timestamp.now("UTC")
    )

    fit_run_id = str(handoff.attrs.get("run_id") or run_id)
    frames: dict[str, pd.DataFrame] = {
        "10_screen_results_v2": screen,
        "10_screen_mc_summary_v2": screen[[c for c in (
            "isin", "er_mean", "er_sd", "er_p05", "er_p50", "er_p95", "mc_prob_pos"
        ) if c in screen.columns]],
        _ANALYTICS_TABLE_V2: kalman_results,
        "15c_forecast_summary": forecast.get("summary"),
        "15c_forecast_engines": forecast.get("engines"),
        "15d_factor_share_sweep": sweep_frames.get("factor_share"),
        "15d_multiplier_sweep": sweep_frames.get("multiplier"),
        "15e_decision_books": decision.get("decision_frame"),
        "15e_book_agreement": decision.get("agreement"),
        "15e_frontier": decision.get("frontier"),
        "14b_group_signals": recommendations.get("group_signals"),
        "14b_name_actions": recommendations.get("actions"),
        "14b_size_down_watch": recommendations.get("watch"),
        "09_gate_report_portfolio": report.to_frame(),
    }
    if risk_book is not None:
        frames["10b_risk_analytics_v2"] = apply_out_of_support(
            risk_book.analytics.drop(columns=["expected_sharpe"], errors="ignore")
        )
        # PRECEDENCE, declared on the row. One posterior produces two books --
        # a CVaR-aware large-cap book here and a growth-optimal micro-cap book at
        # §15e -- and for four editions they shared no name at all while nothing
        # in either export said which was the recommendation. A reader handed two
        # books with no statement of precedence uses whichever they find first.
        #
        # `15e_decision_books` has carried `book_role` since it gained ranking
        # arms; this is the other half. The two are not competing answers to one
        # question: they screen DISJOINT universes (`mcap_country_r <= 0.03`
        # against no screen at all), which is why neither is simply better.
        frames["10b_risk_book_v2"] = risk_book.book.assign(
            book_role="contrast",
            book_engine="cvar_starr",
            book_universe=f"mcap_global_r <= {cfg.mcap_global_r_max}",
        )
    # `handoff.identity`, never the screen. The handoff's block is written from
    # `panel.frame` by the fit itself, so it is in-vintage by construction and
    # complete for every name the posterior covers -- which is what the stale CSV
    # was not.
    # The canonical table is published separately: it goes to `analytics` with
    # `if_exists='replace'` under the FIT's run_id, because it is the current
    # state of one fit and GEIB joins it to a table the fit wrote. Everything
    # else is this replay's own and appends to `kalman_portfolio` under this
    # replay's id. One frame, one destination, two different meanings of "the
    # latest".
    canonical = frames.pop(_ANALYTICS_TABLE_V2, None)
    counts = export_frames(
        {k: v for k, v in frames.items() if v is not None},
        cfg, run_id, identity=handoff.identity,
    ) if export else {}
    if export and canonical is not None:
        counts.update(
            publish_canonical_table(canonical, cfg, fit_run_id, run_id,
                                    identity=handoff.identity)
        )

    print(report.render())
    return {
        "run_id": run_id,
        "handoff": handoff,
        "screen": screen,
        "screen_draws": screen_draws,
        "risk_book": risk_book,
        "kalman_results": kalman_results,
        "forecast": forecast,
        "sweeps": sweep_frames,
        "decision": decision,
        "recommendations": recommendations,
        "report": report,
        "export_counts": counts,
    }


def migrate_portfolio_layout(
    root: Optional[str] = None,
    *,
    from_root: Optional[str] = None,
    dry_run: bool = True,
) -> dict[str, str]:
    """Move this replay's artifacts out of the fit's tree into its own.

    The replay wrote into ``pymc_kalman_filter_pt_v2_results`` until 2026-08-31,
    so a fit and every replay of it interleaved in one directory. A fit happens
    once; a replay happens many times over that fit, and a reader browsing the
    tree could not tell which run had produced what.

    **Moves only what it can prove it owns.** A file in the fit's tree belongs to
    the replay when its stem ends ``_portfolio`` or resolves to
    :data:`PORTFOLIO_ONLY_SECTION_DIRS`. That was checked rather than assumed:
    neither the v2 fit nor v1 writes any ``14b_*`` stem into the v2 tree (v1 has
    its own ``14b_recommendations`` under its own root), and
    ``09_gate_report_portfolio`` carries a suffix that distinguishes it from
    ``09_gate_report_v2`` -- which is why ``09_gates`` is not an owned section and
    the fit's gate report stays where it is.

    Idempotent, and by the same means as the v2 migration: a planned move whose
    target resolves to its source is skipped, so a second run is a no-op. Nothing
    is recorded on disk.

    Parameters
    ----------
    root
        Destination. Defaults to the configured portfolio results root.
    from_root
        Source to sweep. Defaults to the v2 fit's tree, which is the only place
        these files have ever been.
    dry_run
        Report the moves without making them. **The default**, because this
        rewrites a results tree and the first thing anyone should see is the list.

    Returns
    -------
    dict[str, str]
        ``source -> destination`` for every move planned or made.
    """
    import contextlib
    import shutil

    cfg = KalmanPortfolioConfig.from_env()
    dest_root = Path(root) if root else cfg.results_path
    src_root = Path(from_root) if from_root else cfg.v2_results_path
    dest_root.mkdir(parents=True, exist_ok=True)
    moves: dict[str, str] = {}
    emptied: list[Path] = []

    def _owned(stem: str) -> bool:
        return stem.endswith("_portfolio") or export_dir_for(stem) in PORTFOLIO_ONLY_SECTION_DIRS

    def _plan(path: Path) -> None:
        if not _owned(path.stem):
            return
        target = section_path(dest_root, path.stem, suffix=path.suffix)
        if target.resolve() == path.resolve():
            return
        moves[str(path)] = str(target)
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))

    if src_root.is_dir() and src_root.resolve() != dest_root.resolve():
        # Recursive: the frames landed in section directories and so did the
        # figures beside them. A root-only sweep would leave every PNG behind.
        for entry in sorted(src_root.rglob("*")):
            if entry.is_file():
                _plan(entry)
        # A section directory the replay owned OUTRIGHT is empty afterwards.
        # `09_gates` and `04_panel` are shared and must survive, which is what
        # the ownership set above is for.
        for name in sorted(PORTFOLIO_ONLY_SECTION_DIRS):
            candidate = src_root / name
            if candidate.is_dir():
                emptied.append(candidate)

    # Anything already under the destination but in the wrong section, e.g. a
    # tree migrated by hand.
    for entry in sorted(dest_root.iterdir()) if dest_root.is_dir() else []:
        if entry.is_file():
            _plan(entry)

    if not dry_run:
        for directory in emptied:
            # Only when empty. A directory still holding something holds a file
            # this migration could not place, and removing it would destroy
            # exactly what most needs looking at.
            with contextlib.suppress(OSError):
                directory.rmdir()
                logger.info("removed the now-empty %s", directory)

    verb = "would move" if dry_run else "moved"
    if moves:
        logger.info("%s %d artifact(s) from %s into %s",
                    verb, len(moves), src_root, dest_root)
        for src, dst in sorted(moves.items()):
            logger.info("  %s %s -> %s", verb, src, dst)
        if dry_run:
            logger.info("dry run: re-run with --apply to make these moves")
    else:
        logger.info("portfolio layout is already current under %s", dest_root)
    return moves


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--handoff", default=None,
                        help=f"path to {_HANDOFF_STEM} (default: under the results dir)")
    parser.add_argument("--results-dir", default=None,
                        help="where the replay WRITES (default: kalman_portfolio_results)")
    parser.add_argument("--v2-results-dir", default=None,
                        help="where it READS the fit's artifacts (default: the v2 tree)")
    parser.add_argument("--fit", action="store_true",
                        help="run the v2 workflow first to produce a handoff")
    parser.add_argument("--rank-arms", default=DEFAULT_RANKING_RULE,
                        help="comma-separated, or 'all'. The FIRST is the recommendation")
    parser.add_argument("--sweep", default="",
                        help="comma-separated: factor_share, multiplier")
    parser.add_argument("--migrate-layout", action="store_true",
                        help=("move this replay's artifacts out of the fit's tree "
                              "into kalman_portfolio_results (dry run by default)"))
    parser.add_argument("--apply", action="store_true",
                        help="with --migrate-layout: actually move the files")
    parser.add_argument("--allow-stale-inputs", action="store_true",
                        help=("replay against v2 artifacts from a DIFFERENT fit; "
                              "the mismatched frames are dropped, not joined"))
    parser.add_argument("--arms", default="",
                        help="mean-model arms to contrast (needs a live panel)")
    parser.add_argument("--max-names", type=int, default=None,
                        help="ceiling on positions; breadth is otherwise solved")
    parser.add_argument("--min-weight", type=float, default=None,
                        help="drop weights below this and re-solve (default 0.005)")
    parser.add_argument("--objective", default=None,
                        help=f"weighting objective: {', '.join(PORTFOLIO_OBJECTIVES)}")
    parser.add_argument("--group-cap", default="",
                        help="comma-separated dim=cap, e.g. sector=0.30,country=0.35")
    parser.add_argument("--k-book", type=int, default=None,
                        help="deprecated alias for --max-names")
    parser.add_argument("--cap", type=float, default=None, help="per-name weight cap")
    parser.add_argument("--sector-cap", type=float, default=None,
                        help="max weight in any one sector; unset means no cap")
    parser.add_argument("--size-down-veto", action="store_true",
                        help="feed the size-down watch into the eligibility mask")
    parser.add_argument("--relative-denominator-q", type=float, default=None,
                        help="relative ranking-denominator floor; 0 reproduces today")
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="skip the console report")
    parser.add_argument("--write", action="store_true",
                        help="opt in to the analytics write (currently a no-op; see export_frames)")
    args = parser.parse_args()

    if args.migrate_layout:
        # Before any config is built and before anything is read: this rewrites a
        # results tree and must not be entangled with a replay that might fail
        # halfway through it.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        )
        migrate_portfolio_layout(
            root=args.results_dir, from_root=args.v2_results_dir,
            dry_run=not args.apply,
        )
        return 0

    cfg = KalmanPortfolioConfig.from_env()
    overrides: dict[str, Any] = {}
    if args.handoff:
        overrides["handoff_path"] = args.handoff
    if args.results_dir:
        overrides["results_dir"] = args.results_dir
    if args.v2_results_dir:
        overrides["v2_results_dir"] = args.v2_results_dir
    if args.k_book is not None:
        logger.warning(
            "--k-book is deprecated: breadth is solved, not chosen. Applying it "
            "as --max-names, a CEILING."
        )
        overrides["max_names"] = args.k_book
    if args.max_names is not None:
        overrides["max_names"] = args.max_names
    if args.min_weight is not None:
        overrides["min_weight"] = args.min_weight
    if args.objective is not None:
        if args.objective not in PORTFOLIO_OBJECTIVES:
            parser.error(
                f"unknown --objective {args.objective!r}; "
                f"valid: {sorted(PORTFOLIO_OBJECTIVES)}"
            )
        overrides["objective"] = args.objective
    if args.group_cap.strip():
        caps: dict[str, float] = {}
        for token in args.group_cap.split(","):
            if not token.strip():
                continue
            dim, _, value = token.partition("=")
            if not _:
                parser.error(f"--group-cap expects dim=cap, got {token!r}")
            try:
                caps[dim.strip()] = float(value)
            except ValueError:
                parser.error(f"--group-cap value {value!r} is not a number")
        # `sector` goes to the dedicated field, which the concentration gate and
        # the figure layer already read; anything else joins `group_caps`.
        if "sector" in caps:
            overrides["sector_cap"] = caps.pop("sector")
        if caps:
            overrides["group_caps"] = caps
    if args.cap is not None:
        overrides["weight_cap"] = args.cap
    if args.sector_cap is not None:
        overrides["sector_cap"] = args.sector_cap
    if args.relative_denominator_q is not None:
        overrides["relative_denominator_q"] = args.relative_denominator_q
    if args.scenarios is not None:
        overrides["scenarios"] = args.scenarios
    if args.size_down_veto:
        overrides["apply_size_down_veto"] = True
    if args.write:
        overrides["write_analytics"] = True
    if args.allow_stale_inputs:
        overrides["allow_stale_inputs"] = True
    arms = (tuple(RANKING_RULES) if args.rank_arms.strip() == "all"
            else tuple(a.strip() for a in args.rank_arms.split(",") if a.strip()))
    unknown = [a for a in arms if a not in RANKING_RULES]
    if unknown:
        parser.error(f"unknown rank arms {unknown}; valid: {sorted(RANKING_RULES)}")
    overrides["rank_arms"] = arms
    if args.arms:
        overrides["mean_model_arms"] = tuple(
            a.strip() for a in args.arms.split(",") if a.strip()
        )
    cfg = replace(cfg, **overrides)

    if args.fit:
        logger.info("--fit: running the v2 workflow to produce a handoff")
        import pymc_kalman_filter_pt_v2 as v2

        v2.main()

    sweeps = tuple(s.strip() for s in args.sweep.split(",") if s.strip())
    try:
        main(config=cfg, sweeps=sweeps, render=not args.quiet,
             export=not args.no_export)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except VintageMismatch as exc:
        # Non-zero and NOTHING WRITTEN. Reporting the mismatch in a gate table
        # beside a sized book would be reporting it; refusing to produce the book
        # is the point.
        logger.error("REFUSING TO REPLAY: %s", exc)
        return 3
    except ValueError as exc:
        # `run_screen`'s refusals: a handoff that cannot be screened from, or
        # shrinkage asked for without the inputs it needs. Both carry a message
        # that says what to do, so a traceback adds nothing but noise.
        logger.error("%s", exc)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(_cli())