"""The replay workflow, end to end over a synthetic handoff.

Covers the two behaviours that are easy to get quietly wrong: three arms coming out of
one posterior with their roles declared, and the Max-and-Smooth stage refusing an arm it
cannot honestly screen rather than degrading into a comparison of one arm against
itself.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

import kalman_portfolio as kp  # noqa: E402
from probabilistic_ml_model.pymc_models.KalmanForecast import (  # noqa: E402
    save_forecast_handoff,
)
from probabilistic_ml_model.pymc_models.PortfolioOptimizationModel import (  # noqa: E402
    RANKING_RULES,
)

N_ISIN, N_CHAIN, N_DRAW = 120, 2, 120


class _Panel:
    def __init__(self, isins, coord_idx, coord_uniques):
        self.isins = isins
        self.response_mean = 0.11
        self.response_std = 0.42
        self.coord_idx = coord_idx
        self.coord_uniques = coord_uniques


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    """A synthetic v2 results directory: a handoff plus the CSVs the replay reads."""
    out = tmp_path_factory.mktemp("v2_results")
    rng = np.random.default_rng(21)
    isins = np.array([f"FX{i:05d}" for i in range(N_ISIN)])
    sectors = rng.choice(["Info Tech", "Health Care", "Industrials", "Financials"],
                         N_ISIN, p=[.34, .24, .26, .16])
    regions = rng.choice(["APAC", "EU", "NA"], N_ISIN, p=[.42, .30, .28])

    _shape = (N_CHAIN, N_DRAW, N_ISIN)
    post = xr.Dataset(
        {
            "sigma_isin": (("chain", "draw", "isin"),
                           np.abs(rng.normal(0.28, 0.06, _shape))),
            "nu": (("chain", "draw"), rng.uniform(9, 13, (N_CHAIN, N_DRAW))),
            "ou_length_scale_days": (("chain", "draw"),
                                     rng.uniform(74, 84, (N_CHAIN, N_DRAW))),
            # The screen's own inputs. The replay BUILDS the screen off the
            # handoff since 2026-08-31 rather than reading a CSV beside it, so a
            # fixture posterior that carries only the forward simulation's four
            # quantities is refused -- which is the point: a partial handoff
            # cannot produce a decision quantity with a definition.
            "state_now_mean": (("chain", "draw", "isin"), rng.normal(0.25, 0.55, _shape)),
            "state_now_sd": (("chain", "draw", "isin"),
                             np.abs(rng.normal(0.18, 0.04, _shape))),
            "mu_scaled": (("chain", "draw", "isin"), rng.normal(0.12, 0.20, _shape)),
            "risk_adj_return": (("chain", "draw", "isin"),
                                rng.normal(0.06, 0.30, _shape)),
            "variance_weights": (("chain", "draw", "vw"),
                                 rng.dirichlet([3, 2, 1], (N_CHAIN, N_DRAW))),
        },
        coords={"chain": np.arange(N_CHAIN), "draw": np.arange(N_DRAW), "isin": isins},
    )
    idata = xr.DataTree()
    idata["posterior"] = xr.DataTree(post)

    panel = _Panel(
        isins,
        {"sector": pd.factorize(sectors)[0], "trading_region": pd.factorize(regions)[0]},
        {"sector": np.asarray(pd.factorize(sectors)[1]),
         "trading_region": np.asarray(pd.factorize(regions)[1])},
    )
    latent = np.asarray(post["state_now_mean"])
    identity = pd.DataFrame({"isin": isins, "sector": sectors,
                             "trading_region": regions,
                             "name": [f"Company {i}" for i in range(N_ISIN)],
                             "ticker": [f"CO{i}" for i in range(N_ISIN)]})
    # The per-ISIN vectors the screen reads: `dispersion_cv` is a panel attribute,
    # the rest are frame columns. Without them `run_screen` refuses rather than
    # screening with a forecast error of zero, which would reproduce consensus.
    _last = rng.uniform(4.0, 180.0, N_ISIN)
    panel.dispersion_cv = np.abs(rng.normal(0.24, 0.07, N_ISIN))
    panel.frame = pd.DataFrame({
        "isin": isins,
        "n_analysts": rng.integers(1, 30, N_ISIN).astype(float),
        "last_price": _last,
        "observed_pt": _last * (1.0 + rng.uniform(-0.15, 1.10, N_ISIN)),
        "feat_analyst_rating": rng.uniform(1.0, 5.0, N_ISIN),
        "market_cap": rng.uniform(6e7, 4e10, N_ISIN),
        "feat_mcap_global_r": rng.uniform(0.0, 1.0, N_ISIN),
        "feat_mcap_country_r": rng.uniform(0.0, 1.0, N_ISIN),
    })

    save_forecast_handoff(
        out / "07_forecast_handoff_v2.nc", idata, panel, latent=latent,
        n_samples=200, identity=identity,
        provenance={"run_id": "fixture0cafe", "source_sha": "0badc0de",
                    "source_dirty": True},
    )

    # NO screen CSV. The replay BUILDS the screen off the handoff since
    # 2026-08-31; writing one here would be a file nothing reads, and the version
    # of this fixture that did write one is exactly the arrangement that let a
    # 2026-08-27 screen meet a 2026-08-30 posterior in production.
    #
    # `09_diagnostics_v2.csv` below is still written FLAT, and deliberately: it is
    # one row per model PARAMETER, has no ISIN axis, is the one frame the handoff
    # cannot carry, and its flat path is what exercises the pre-migration read
    # fallback.
    pd.DataFrame({"index": ["nu"], "mean": [11.2], "r_hat": [1.002],
                  "ess_bulk": [1637]}).to_csv(out / "09_diagnostics_v2.csv",
                                              index=False)
    return out


@pytest.fixture(scope="module")
def cfg(results_dir):
    return kp.KalmanPortfolioConfig(
        # TWO roots since 2026-08-31. `results_dir` is where the replay WRITES;
        # `v2_results_dir` is the fit's tree it reads. Setting only the first
        # leaves the read pointed at the real repository tree, whose diagnostics
        # frame carries another fit's run_id -- which the vintage gate then
        # correctly refuses.
        results_dir=str(results_dir / "replay"),
        v2_results_dir=str(results_dir),
        handoff_path=str(results_dir / "07_forecast_handoff_v2.nc"),
        rank_arms=tuple(RANKING_RULES),
        max_names=20,
        scenarios=200,
    )


@pytest.fixture(scope="module")
def result(cfg):
    return kp.main(config=cfg, sweeps=("factor_share", "multiplier"),
                   render=False, export=True)


def test_three_arms_from_one_posterior(result, cfg):
    books = result["decision"]["books"]
    assert set(books) == set(RANKING_RULES)
    frame = result["decision"]["decision_frame"]
    assert set(frame["rank_by"]) == set(RANKING_RULES)


def test_precedence_is_declared_on_the_row(result, cfg):
    """A reader handed two books with no statement of precedence uses whichever
    they find first."""
    frame = result["decision"]["decision_frame"]
    recommended = set(frame.loc[frame["book_role"] == "recommendation", "rank_by"])
    assert recommended == {cfg.rank_arms[0]}
    assert set(frame.loc[frame["book_role"] == "contrast", "rank_by"]) == \
        set(cfg.rank_arms[1:])


def test_arms_disagree_and_the_disagreement_is_measured(result):
    """Overlap is measured against each arm's OWN size, not against one nominal k.

    Breadth is solved per arm since 2026-08-28, so the arms no longer hold the
    same number of names. Comparing a shared ``k`` would make a small disciplined
    book look like it disagreed with a large one when it may be a subset of it --
    which is why ``containment`` ships beside ``jaccard``.
    """
    agreement = result["decision"]["agreement"]
    assert len(agreement) == 3          # one row per unordered pair of three arms
    assert (agreement["overlap"] <= agreement[["n_a", "n_b"]].min(axis=1)).all()
    assert ((agreement["jaccard"] >= 0) & (agreement["jaccard"] <= 1)).all()
    assert ((agreement["containment"] >= 0) & (agreement["containment"] <= 1)).all()
    # A ceiling was set on this fixture, so no arm may exceed it.
    assert (agreement[["n_a", "n_b"]].max(axis=1) <= agreement["max_names"]).all()


def test_identity_survives_the_join_by_isin(result):
    """The screen is sorted by expected_upside while the draws are in universe
    order; a positional attach passes every length check and is still wrong.

    Graded against the HANDOFF's identity block, which is the source now -- the
    fit writes it from `panel.frame`, so it is in-vintage by construction. It used
    to be graded against a CSV sitting beside the handoff, which is the very thing
    that could disagree with it.
    """
    frame = result["decision"]["decision_frame"]
    ident = result["handoff"].identity
    truth = ident.drop_duplicates("isin").set_index("isin")["sector"]
    got = frame.drop_duplicates("isin").set_index("isin")["sector"]
    assert (truth.reindex(got.index) == got).all()


def test_the_screen_covers_every_name_in_the_posterior(result):
    """The 145-name gap, as a property.

    A screen read from a file could cover a different universe from the posterior
    it was joined to, and nothing measured the difference. Built from the handoff
    it cannot: one row per name, no name missing, and no label invented for one.
    """
    handoff, screen = result["handoff"], result["screen"]
    assert len(screen) == handoff.n_isin
    assert set(screen["isin"]) == set(handoff.isins)
    assert screen["sector"].notna().all()


#: The one gate here that blocks, and why it is allowed to.
_BLOCKING_GATES = {"portfolio_input_vintage", "portfolio_sector_concentration"}


def test_every_gate_that_grades_the_model_is_non_blocking(result):
    """Nothing here can fail a run on the strength of a MODEL judgement.

    Every measurement in this replay scores the model against the analyst trail it
    was fitted to, and no such measurement can be decisive -- which is the whole
    reason the vintage harness exists. The exceptions are the gates that grade
    something other than the model: `portfolio_input_vintage` grades LINEAGE (did
    these files come from one fit -- a fact about provenance, checkable now) and
    `portfolio_sector_concentration` grades whether a stated CONSTRAINT was
    actually applied. Neither is an opinion about whether the model is any good.
    """
    report = result["report"]
    assert report.ok
    blocking = {g.name for g in report.results if g.blocking}
    assert blocking <= _BLOCKING_GATES, (
        f"{blocking - _BLOCKING_GATES} blocks a run on a model judgement"
    )


def test_gates_are_documented(result):
    """A gate whose rationale is not in the catalogue reports a bare number."""
    from pymc_kalman_filter_pt_v2 import GATE_CATALOGUE

    for gate in result["report"].results:
        assert GATE_CATALOGUE.get(gate.name), f"{gate.name} has no rationale"


def test_frames_are_exported_into_their_section_directories(result, cfg):
    """Every frame lands under the section its stem resolves to, not in a bucket.

    The replay used to write all ten frames into one ``15_portfolio`` directory --
    a forecast summary, two prior sweeps, the sized books and three recommendation
    frames, which is four stages under one name. The stems already carried the
    section numbers; only the directories were missing.
    """
    from probabilistic_ml_model.export_layout import export_dir_for

    counts = result["export_counts"]
    write_root = cfg.results_path
    assert counts, "nothing was exported"
    assert not (write_root / "15_portfolio").exists(), "the legacy bucket is back"
    for stem in counts:
        path = write_root / export_dir_for(stem) / f"{stem}.csv"
        assert path.exists(), f"{stem} is not under {export_dir_for(stem)}/"
        frame = pd.read_csv(path)
        assert len(frame) == counts[stem]
        assert {"run_id", "exported_at", "source_sha"} <= set(frame.columns)
    # No exported frame left loose in the root beside the tree.
    assert not [stem for stem in counts if (write_root / f"{stem}.csv").exists()]


def test_the_replay_writes_its_own_tree_and_never_the_fits(result, cfg):
    """Two roots, and the separation is the point.

    A fit happens once; a replay happens many times over that fit. Sharing a root
    meant a reader browsing the tree could not tell which run had produced what,
    and it is how a 2026-08-27 screen came to be sitting beside a 2026-08-30
    handoff in the first place.
    """
    from probabilistic_ml_model.export_layout import export_dir_for

    write_root, read_root = cfg.results_path, cfg.v2_results_path
    assert write_root.resolve() != read_root.resolve()
    for stem in result["export_counts"]:
        assert not (read_root / export_dir_for(stem) / f"{stem}.csv").exists(), (
            f"{stem} was written into the fit's tree"
        )
    # The fit's own artifacts are untouched.
    assert (read_root / "07_forecast_handoff_v2.nc").exists()


def test_sweeps_report_the_priors_consequence_not_its_posterior(result):
    sweep = result["sweeps"]["factor_share"]
    assert "book_sd_ratio" in sweep.columns
    assert "er_sd_max_abs_diff" in sweep.columns
    multiplier = result["sweeps"]["multiplier"]
    # Shrinking harder narrows the cross-section; that is the whole departure from
    # consensus, as a function of one chosen number.
    assert multiplier["cross_sectional_sd"].is_monotonic_decreasing


def test_recommendation_layer_produces_a_posture(result):
    recs = result["recommendations"]
    assert len(recs["group_signals"])
    assert recs["reliability"]["posture"] in {"OK", "ACCEPTABLE", "CONCERN"}
    assert "action" in recs["actions"].columns


def test_drift_strict_is_refused_by_the_screening_stage(cfg, result):
    """It changes the design matrix, which is the quantity the Max step froze.
    Screening it would contrast two arms against one arm's noise model."""
    from pymc_kalman_filter_pt_v2 import GateReport

    with pytest.raises(ValueError, match="drift_strict"):
        kp.run_mean_model_arms(result["handoff"], cfg, GateReport(),
                               arms=("drift_strict",))


def test_mean_model_arms_decline_rather_than_degrade(cfg, result):
    """A handoff carries what the SIMULATOR reads, not the model graph."""
    from pymc_kalman_filter_pt_v2 import GateReport

    report = GateReport()
    assert kp.run_mean_model_arms(result["handoff"], cfg, report,
                                  arms=("level_off",)) is None
    assert any(g.name == "portfolio_mean_model_arms" for g in report.results)


def test_size_down_veto_changes_eligibility_not_the_ranking(cfg):
    veto = replace(cfg, apply_size_down_veto=True, rank_arms=("reward_to_downside",))
    out = kp.main(config=veto, render=False, export=False)
    book = out["decision"]["books"]["reward_to_downside"]

    from probabilistic_ml_model.pymc_models._recommendations import (
        size_down_mask, size_down_watch,
    )

    flagged = set(size_down_watch(out["screen"])["isin"].astype(str))
    assert not (set(book.weights.index.astype(str)) & flagged)
    # And the watch SAW every name it was asked about. A name absent from the
    # frame the watch is scored on comes back sizeable, which is what put a
    # two-analyst name in the book at 9.13 % while the watch that defines it
    # reported nothing.
    _, unseen = size_down_mask(
        out["screen"], out["handoff"].isins, return_unseen=True
    )
    assert unseen == 0


# ---------------------------------------------------------------------------
# The vintage gate
#
# The one thing nothing checked. A results directory is a directory: the file
# system says nothing about whether the CSV beside a handoff came from the run
# that wrote it. On run `b00f8d8ca093` it did not, the join still succeeded, and
# the 145 names the two universes did not share were filled and sized.
# ---------------------------------------------------------------------------


def _stamped(run_id, n=4):
    return pd.DataFrame({
        "index": [f"p{i}" for i in range(n)],
        "r_hat": np.full(n, 1.002),
        "run_id": [run_id] * n,
    })


def test_a_frame_from_another_fit_refuses_the_replay(result, cfg):
    handoff = result["handoff"]
    report = kp.GateReport()
    with pytest.raises(kp.VintageMismatch) as excinfo:
        kp.check_input_vintage(
            handoff, {"09_diagnostics_v2": _stamped("someotherfit")}, cfg, report,
        )
    message = str(excinfo.value)
    # BOTH sides named. A verdict that reports only the offending frame costs its
    # reader a forensic pass to find what it was supposed to match.
    assert handoff.attrs["run_id"] in message
    assert "someotherfit" in message
    assert "09_diagnostics_v2" in message
    gate = [g for g in report.results if g.name == "portfolio_input_vintage"]
    assert gate and gate[0].blocking and not gate[0].passed


def test_allow_stale_inputs_drops_rather_than_joins(result, cfg):
    handoff = result["handoff"]
    report = kp.GateReport()
    stale = replace(cfg, allow_stale_inputs=True)
    out = kp.check_input_vintage(
        handoff, {"09_diagnostics_v2": _stamped("someotherfit")}, stale, report,
    )
    # Dropped, not joined. Degrading to empty is recoverable; degrading to wrong
    # is what this whole change is about.
    assert out["09_diagnostics_v2"] is None
    gate = [g for g in report.results if g.name == "portfolio_input_vintage"][0]
    assert not gate.passed and not gate.blocking


def test_a_matching_frame_passes(result, cfg):
    handoff = result["handoff"]
    report = kp.GateReport()
    frames = {"09_diagnostics_v2": _stamped(handoff.attrs["run_id"])}
    out = kp.check_input_vintage(handoff, frames, cfg, report)
    assert out["09_diagnostics_v2"] is not None
    gate = [g for g in report.results if g.name == "portfolio_input_vintage"][0]
    assert gate.passed


def test_an_unstamped_frame_is_warned_and_kept(result, cfg):
    """'No provenance' is not 'wrong provenance'.

    An export predating the provenance SSOT cannot be shown to be from another
    fit, and the only frame that can still reach this path has no ISIN axis -- it
    is one row per model PARAMETER -- so it cannot mis-attribute a company. A
    per-ISIN frame in this position would have to be refused.
    """
    handoff = result["handoff"]
    report = kp.GateReport()
    unstamped = pd.DataFrame({"index": ["nu"], "r_hat": [1.001]})
    out = kp.check_input_vintage(
        handoff, {"09_diagnostics_v2": unstamped}, cfg, report,
    )
    assert out["09_diagnostics_v2"] is not None
    gate = [g for g in report.results if g.name == "portfolio_input_vintage"][0]
    assert gate.passed
    assert "unstamped" in gate.value


def test_an_absent_frame_is_not_a_mismatch(result, cfg):
    report = kp.GateReport()
    out = kp.check_input_vintage(
        result["handoff"], {"09_diagnostics_v2": None}, cfg, report,
    )
    assert out["09_diagnostics_v2"] is None
    assert [g for g in report.results if g.name == "portfolio_input_vintage"][0].passed


def test_the_live_replay_passes_its_own_vintage_gate(result):
    gate = [g for g in result["report"].results
            if g.name == "portfolio_input_vintage"]
    assert gate and gate[0].passed
