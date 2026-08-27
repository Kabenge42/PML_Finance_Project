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

    post = xr.Dataset(
        {
            "sigma_isin": (("chain", "draw", "isin"),
                           np.abs(rng.normal(0.28, 0.06, (N_CHAIN, N_DRAW, N_ISIN)))),
            "nu": (("chain", "draw"), rng.uniform(9, 13, (N_CHAIN, N_DRAW))),
            "ou_length_scale_days": (("chain", "draw"),
                                     rng.uniform(74, 84, (N_CHAIN, N_DRAW))),
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
    latent = rng.normal(0.25, 0.55, (N_CHAIN, N_DRAW, N_ISIN))
    identity = pd.DataFrame({"isin": isins, "sector": sectors,
                             "trading_region": regions,
                             "name": [f"Company {i}" for i in range(N_ISIN)]})

    save_forecast_handoff(
        out / "07_forecast_handoff_v2.nc", idata, panel, latent=latent,
        n_samples=200, identity=identity,
        provenance={"run_id": "fixture0cafe", "source_sha": "0badc0de",
                    "source_dirty": True},
    )

    eu = latent.reshape(-1, N_ISIN).T * panel.response_std + panel.response_mean
    # Sorted by expected_upside, i.e. NOT universe order — a positional join would
    # hand every name someone else's sector while the row count still matched.
    pd.DataFrame({
        "isin": isins,
        "name": identity["name"],
        "sector": sectors,
        "trading_region": regions,
        "expected_upside": np.expm1(eu.mean(axis=1)),
        "p_upside_pos_cond": np.clip(rng.beta(7, 1.1, N_ISIN), 0, 1),
        "shrink_gain": np.clip(rng.beta(3, 5, N_ISIN), 0, 1),
        "band_width": np.abs(rng.normal(0.32, 0.13, N_ISIN)),
        "n_analysts": rng.integers(1, 30, N_ISIN),
        "ret_vol_ratio": np.abs(rng.normal(1.1, 0.45, N_ISIN)),
    }).sort_values("expected_upside", ascending=False).to_csv(
        out / "10_screen_results_v2.csv", index=False)

    pd.DataFrame({"index": ["nu"], "mean": [11.2], "r_hat": [1.002],
                  "ess_bulk": [1637]}).to_csv(out / "09_diagnostics_v2.csv",
                                              index=False)
    return out


@pytest.fixture(scope="module")
def cfg(results_dir):
    return kp.KalmanPortfolioConfig(
        results_dir=str(results_dir),
        handoff_path=str(results_dir / "07_forecast_handoff_v2.nc"),
        rank_arms=tuple(RANKING_RULES),
        k_book=20,
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
    agreement = result["decision"]["agreement"]
    assert len(agreement) == 3          # one row per unordered pair of three arms
    assert (agreement["overlap"] <= agreement["k_book"]).all()


def test_identity_survives_the_join_by_isin(result, results_dir):
    """The screen is sorted by expected_upside while the draws are in universe
    order; a positional attach passes every length check and is still wrong."""
    frame = result["decision"]["decision_frame"]
    screen = pd.read_csv(results_dir / "10_screen_results_v2.csv")
    truth = screen.drop_duplicates("isin").set_index("isin")["sector"]
    got = frame.drop_duplicates("isin").set_index("isin")["sector"]
    assert (truth.reindex(got.index) == got).all()


def test_every_gate_is_non_blocking(result):
    report = result["report"]
    assert report.ok
    assert not any(g.blocking for g in report.results)


def test_gates_are_documented(result):
    """A gate whose rationale is not in the catalogue reports a bare number."""
    from pymc_kalman_filter_pt_v2 import GATE_CATALOGUE

    for gate in result["report"].results:
        assert GATE_CATALOGUE.get(gate.name), f"{gate.name} has no rationale"


def test_frames_are_exported_into_their_section_directories(result, results_dir):
    """Every frame lands under the section its stem resolves to, not in a bucket.

    The replay used to write all ten frames into one ``15_portfolio`` directory --
    a forecast summary, two prior sweeps, the sized books and three recommendation
    frames, which is four stages under one name. The stems already carried the
    section numbers; only the directories were missing.
    """
    from probabilistic_ml_model.export_layout import export_dir_for

    counts = result["export_counts"]
    assert counts, "nothing was exported"
    assert not (results_dir / "15_portfolio").exists(), "the legacy bucket is back"
    for stem in counts:
        path = results_dir / export_dir_for(stem) / f"{stem}.csv"
        assert path.exists(), f"{stem} is not under {export_dir_for(stem)}/"
        frame = pd.read_csv(path)
        assert len(frame) == counts[stem]
        assert {"run_id", "exported_at", "source_sha"} <= set(frame.columns)
    # And no EXPORTED frame was left loose in the root beside the tree. Scoped to
    # the exports on purpose: this fixture writes the handoff and the screen flat,
    # which is the pre-migration layout and is what exercises the read fallback.
    assert not [stem for stem in counts
                if (results_dir / f"{stem}.csv").exists()]


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


def test_size_down_veto_changes_eligibility_not_the_ranking(cfg, results_dir):
    veto = replace(cfg, apply_size_down_veto=True, rank_arms=("reward_to_downside",))
    out = kp.main(config=veto, render=False, export=False)
    book = out["decision"]["books"]["reward_to_downside"]

    screen = pd.read_csv(results_dir / "10_screen_results_v2.csv")
    from probabilistic_ml_model.pymc_models._recommendations import size_down_watch

    flagged = set(size_down_watch(screen)["isin"].astype(str))
    assert not (set(book.weights.index.astype(str)) & flagged)
