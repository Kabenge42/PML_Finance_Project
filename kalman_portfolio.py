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
analyst trail it was fitted to; that is why none of them blocks.

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
    GateReport,
    GateResult,
    KalmanRunConfigV2,
    PROVENANCE_COLUMNS,
    _HANDOFF_STEM,
    attach_identity_columns,
    resolve_source_revision,
    stamp_export_provenance,
)
from probabilistic_ml_model.export_layout import (  # noqa: E402
    DEFAULT_RESULTS_DIRNAME_V2,
    RESULTS_DIR_ENV_V2,
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
    "load_handoff",
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
    "portfolio_factor_sensitivity": (
        "Spread of book membership and dispersion across the `factor_share` grid. "
        "The split is variance-preserving, so per-name marginals are invariant and "
        "only the JOINT distribution moves -- harmless for the screen, decisive for "
        "every portfolio statistic."
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
    results_dir: Optional[str] = None
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
    write_analytics: bool = True

    @property
    def results_path(self) -> Path:
        """Root of the artifact tree — the same one the v2 workflow writes to."""
        return resolve_results_root(
            self.results_dir,
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
        sectioned = section_path(self.results_path, stem, suffix=suffix)
        if sectioned.exists():
            return sectioned
        legacy = self.results_path / _HANDOFF_STEM
        if legacy.exists():
            logger.info(
                "reading the handoff from the pre-migration flat path %s; "
                "`python pymc_kalman_filter_pt_v2.py --migrate-layout --apply` "
                "moves it into %s", legacy, sectioned.parent,
            )
            return legacy
        return sectioned

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
            # v2's variable, not v1's: a replay must land beside the run it
            # replays, and `set_env.ps1` points KALMAN_PT_RESULTS_DIR at v1.
            results_dir=os.environ.get(RESULTS_DIR_ENV_V2) or None,
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
        sizeable = recs.size_down_mask(screen, isins)
        veto_names = set(isins[~sizeable].tolist())
        if cfg.apply_size_down_veto:
            eligible = sizeable

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
    """Per-name group labels aligned to ``isins`` **by key**, from screen or handoff."""
    if screen is not None and column in screen.columns and "isin" in screen.columns:
        keyed = screen.drop_duplicates("isin").set_index("isin")[column]
        return keyed.reindex(isins).fillna("Unknown").astype(str).to_numpy()
    ident = handoff.identity
    if ident is not None and column in getattr(ident, "columns", []):
        keyed = ident.drop_duplicates("isin").set_index("isin")[column]
        return keyed.reindex(isins).fillna("Unknown").astype(str).to_numpy()
    codes = handoff.coord_idx.get(column)
    uniques = handoff.coord_uniques.get(column)
    if codes is not None and uniques is not None:
        return np.asarray(uniques)[np.asarray(codes)]
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
        report.add(GateResult(
            name="portfolio_sector_concentration",
            passed=True,
            value=f"{summary['top_group_weight']:.1%} in {summary.get('top_group', '?')}",
            threshold=(f"cap {cfg.sector_cap:.0%}" if cfg.sector_cap is not None
                       else "no cap set"),
            blocking=False,
            detail=(
                f"[{arm}] {int(summary.get('n_groups', 0))} groups held. No cap is "
                f"also a decision -- it should just never be taken by omission."
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
        # The analyst consensus reached the screen frame only from 2026-08-28, so
        # a replay against an older export would silently lose `consensus_gap` --
        # the one column that says where this model DIFFERS from the panel it
        # reproduces at Spearman 0.992. The panel frame has always carried it.
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
    "15e_decision_books": "one row per (isin, rank_by); book_role names the recommendation",
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


def export_frames(
    frames: dict[str, pd.DataFrame],
    cfg: KalmanPortfolioConfig,
    run_id: str,
    *,
    identity: Optional[pd.DataFrame] = None,
) -> dict[str, int]:
    """Write the frames as CSV, stamped and identity-attached.

    CSV by default and by design. This script exists to be run many times over one
    fit; a workflow that writes to the analytics schema on every run is not that, and
    the v2 tables are DROP-and-RECREATE, so a replay that wrote would destroy the
    export it was replaying.
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

    for stem, frame in frames.items():
        if frame is None or not len(frame):
            continue
        written = frame
        if identity is not None and "isin" in written.columns:
            written = attach_identity_columns(written, identity, label=stem)
        written = stamp_export_provenance(written, run_id, stamped)
        path = section_path(root, stem)
        written.to_csv(path, index=False)
        counts[stem] = len(written)
        logger.info("wrote %s (%d rows)", path, len(written))

    if cfg.write_analytics:
        logger.warning(
            "--write is accepted but intentionally does nothing yet: the v2 analytics "
            "tables are DROP-and-RECREATE, so a replay writing into them would "
            "destroy the export it replayed. Declare precedence in the v2 DDL first."
        )
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
    sectioned = section_path(cfg.results_path, stem)
    frame = _read_optional_csv(sectioned)
    if frame is not None:
        return frame
    legacy = cfg.results_path / f"{stem}.csv"
    frame = _read_optional_csv(legacy)
    if frame is not None:
        logger.info(
            "read %s from the pre-migration flat path; run "
            "`python pymc_kalman_filter_pt_v2.py --migrate-layout --apply` to move "
            "it into %s", stem, sectioned.parent,
        )
    else:
        logger.info("no %s under %s; stages needing it will be skipped",
                    stem, cfg.results_path)
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
        ``run_id``, ``handoff``, ``forecast``, ``sweeps``, ``decision``,
        ``recommendations``, ``report``, ``export_counts``.
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

    # Resolved through the section tree, with a flat-path fallback: the v2 export
    # moved into subdirectories on 2026-08-27 and these three reads are what give
    # the replay its sector labels, its bounded ranking arm and its size-down
    # watch. Reading only the new path would have degraded all three SILENTLY
    # against an older results tree -- `_read_optional_csv` never fails, which is
    # exactly why the fallback has to be explicit.
    screen = _read_v2_artifact(cfg, "10_screen_results_v2")
    mc_summary = _read_v2_artifact(cfg, "10_screen_mc_summary_v2")
    diagnostics = _read_v2_artifact(cfg, "09_diagnostics_v2")
    # Read only for the analyst consensus the older screens lack; a
    # missing panel frame costs `consensus_gap` and nothing else.
    panel_frame = _read_v2_artifact(cfg, "04_panel_frame_v2")

    forecast = run_forecast(handoff, cfg, report, mc_summary=mc_summary)

    rank_values = None
    if screen is not None and "p_upside_pos_cond" in screen.columns:
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
        screen=screen, diagnostics=diagnostics, panel_frame=panel_frame,
        report=report, render=render,
    )

    frames: dict[str, pd.DataFrame] = {
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
    counts = export_frames(
        {k: v for k, v in frames.items() if v is not None},
        cfg, run_id, identity=screen if screen is not None else handoff.identity,
    ) if export else {}

    print(report.render())
    return {
        "run_id": run_id,
        "handoff": handoff,
        "forecast": forecast,
        "sweeps": sweep_frames,
        "decision": decision,
        "recommendations": recommendations,
        "report": report,
        "export_counts": counts,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--handoff", default=None,
                        help=f"path to {_HANDOFF_STEM} (default: under the results dir)")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--fit", action="store_true",
                        help="run the v2 workflow first to produce a handoff")
    parser.add_argument("--rank-arms", default=DEFAULT_RANKING_RULE,
                        help="comma-separated, or 'all'. The FIRST is the recommendation")
    parser.add_argument("--sweep", default="",
                        help="comma-separated: factor_share, multiplier")
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

    cfg = KalmanPortfolioConfig.from_env()
    overrides: dict[str, Any] = {}
    if args.handoff:
        overrides["handoff_path"] = args.handoff
    if args.results_dir:
        overrides["results_dir"] = args.results_dir
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
    return 0


if __name__ == "__main__":
    sys.exit(_cli())