"""The shrinkage must shrink, and the confidence argument must stay an argument.

Two properties carry the whole reason this layer exists:

* ``|shrunk| <= |raw|`` for every group, with ``lambda_g`` falling in the group's own
  posterior sd. That is the structural opposite of a reward-to-risk ratio, where thin
  evidence inflates a score instead of damping it.
* ``confidence`` is a parameter. v1 passes ``achieve_prob``; v2 REMOVED that variable
  and passes ``shrink_gain``. A silent fallback to 1.0 would leave the conditioning
  inert while every verdict still printed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probabilistic_ml_model.pymc_models._recommendations import (
    MIN_GROUP_N,
    VERDICTS,
    demotion_list,
    group_allocation_signals,
    name_action_list,
    reliability_posture,
    render_recommendations,
    size_down_mask,
    size_down_watch,
)

N_ISIN, N_SAMPLE = 400, 600


@pytest.fixture(scope="module")
def panel():
    """A universe with one deliberately noisy group and one below the coverage gate."""
    rng = np.random.default_rng(5)
    isins = np.array([f"X{i:04d}" for i in range(N_ISIN)])
    sector = rng.choice(["Tech", "Health", "Indust", "Fin", "Tiny"], N_ISIN,
                        p=[.30, .25, .25, .18, .02])
    region = rng.choice(["APAC", "EU", "NA"], N_ISIN)
    coords = pd.DataFrame({"isin": isins, "sector": sector,
                           "trading_region": region})

    true_mu = pd.Series({"Tech": .28, "Health": .20, "Indust": .18,
                         "Fin": .10, "Tiny": .90})
    mu = true_mu.reindex(sector).to_numpy() + rng.normal(0, 0.05, N_ISIN)
    sd = np.where(sector == "Fin", 0.9, 0.15)      # Fin is the noisy one
    draws = mu[:, None] + sd[:, None] * rng.standard_normal((N_ISIN, N_SAMPLE))
    confidence = np.clip(rng.beta(4, 2, N_ISIN), 0, 1)
    return isins, sector, coords, draws, confidence


@pytest.fixture(scope="module")
def signals(panel):
    _isins, _sector, coords, draws, confidence = panel
    return group_allocation_signals(draws, coords, confidence=confidence)


def test_shrinkage_never_expands(signals):
    assert (signals["excess_shrunk"].abs()
            <= signals["excess_raw"].abs() + 1e-12).all()


def test_lambda_falls_with_the_groups_own_noise(signals):
    """Within a level, a group the data resolves worse is shrunk harder."""
    for _level, block in signals.groupby("level"):
        ordered = block.sort_values("sd")
        assert (ordered["lambda_g"].diff().dropna() <= 1e-12).all()


def test_thin_evidence_is_damped_not_amplified(signals):
    """The finding this layer is built around, stated as a test: the group with the
    LARGEST raw excess gets no verdict because it is the least well resolved."""
    sector = signals[signals.level == "sector"].set_index("group")
    noisy, resolved = sector.loc["Fin"], sector.loc["Indust"]

    assert abs(noisy["excess_raw"]) > abs(resolved["excess_raw"])
    assert noisy["lambda_g"] < resolved["lambda_g"]
    assert abs(noisy["excess_shrunk"]) < abs(resolved["excess_shrunk"])
    assert noisy["verdict"] == "NEUTRAL"


def test_coverage_gate_excludes_thin_groups(signals, panel):
    _isins, sector, _coords, _draws, _conf = panel
    assert int((sector == "Tiny").sum()) < MIN_GROUP_N
    assert "Tiny" not in set(signals["group"])
    assert (signals["n"] >= MIN_GROUP_N).all()


def test_verdicts_are_from_the_declared_vocabulary(signals):
    assert set(signals["verdict"]) <= set(VERDICTS)


def test_confidence_is_a_parameter_not_a_posterior_read(panel):
    """Omitting it must leave the conditioning inert AND say so, rather than
    silently substituting a value that makes every gate look satisfied."""
    _isins, _sector, coords, draws, _conf = panel
    inert = group_allocation_signals(draws, coords, confidence=None)
    assert (inert["univ_confidence"] == 1.0).all()
    assert np.allclose(inert["p_pos_cond"], inert["p_pos"])


def test_misaligned_confidence_is_refused(panel):
    _isins, _sector, coords, draws, _conf = panel
    with pytest.raises(ValueError, match="confidence"):
        group_allocation_signals(draws, coords, confidence=np.ones(7))


def test_misaligned_coords_are_refused(panel):
    _isins, _sector, coords, draws, _conf = panel
    with pytest.raises(ValueError, match="aligned"):
        group_allocation_signals(draws, coords.head(10))


# --------------------------------------------------------------------------- #
#  Actions, watch, demotion, reliability
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def analytics(panel):
    isins, _sector, _coords, draws, _conf = panel
    rng = np.random.default_rng(6)
    return pd.DataFrame({
        "isin": isins,
        "name": [f"Co {i}" for i in range(N_ISIN)],
        "expected_upside": draws.mean(axis=1),
        "p_upside_pos_cond": np.clip(rng.beta(5, 2, N_ISIN), 0, 1),
        "band_width": np.abs(rng.normal(0.30, 0.12, N_ISIN)),
        "n_analysts": rng.integers(1, 25, N_ISIN),
        "ret_vol_ratio": np.abs(rng.normal(1.2, 0.5, N_ISIN)),
    })


def test_action_gates_scale_with_the_confidence_they_are_compared_against(analytics):
    """Comparing an unconditional gate against a conditional column is how a screen
    ends up with no high-conviction names and no explanation."""
    scaled = name_action_list(analytics, confidence_scale=0.5)
    assert scaled["gate_hi"].iloc[0] == pytest.approx(0.375)
    assert set(scaled["action"]) <= {"BUY", "HOLD", "AVOID"}
    assert len(scaled) == len(analytics)


def test_action_list_needs_its_columns(analytics):
    with pytest.raises(KeyError):
        name_action_list(analytics.drop(columns="p_upside_pos_cond"))


def test_watch_flags_both_legs_and_records_which_ran(analytics):
    watch = size_down_watch(analytics)
    assert watch["size_down_flag"].all()
    assert (watch["flag_wide_band"] | watch["flag_thin_coverage"]).all()
    assert len(watch.attrs["legs"]) == 2


def test_watch_with_no_columns_says_so_rather_than_reporting_clean(analytics):
    """An empty watch because the columns are absent is not the same as an empty
    watch because nothing was flagged."""
    bare = analytics[["isin", "name", "expected_upside"]]
    watch = size_down_watch(bare)
    assert watch.attrs["legs"] == []
    assert len(watch) == 0


def test_mask_is_the_complement_of_the_watch_and_keyed_by_isin(analytics):
    watch = size_down_watch(analytics)
    mask = size_down_mask(analytics, analytics["isin"].to_numpy())
    assert mask.sum() == len(analytics) - len(watch)
    flagged = set(watch["isin"])
    assert not (set(analytics.loc[mask, "isin"]) & flagged)

    # And it follows the ISINs it is given, not their order in the frame.
    shuffled = analytics["isin"].sample(frac=1.0, random_state=2).to_numpy()
    reordered = size_down_mask(analytics, shuffled)
    assert set(shuffled[~reordered]) == flagged


def test_demotion_names_what_the_screen_rejected(analytics):
    demoted = demotion_list(analytics)
    if len(demoted):
        median = float(demoted["universe_median_ratio"].iloc[0])
        assert (demoted["ret_vol_ratio"] < median).all()


def test_reliability_reads_a_diagnostics_frame(analytics):
    """A replay has no InferenceData at all, so the frame path is the one that has
    to work."""
    diagnostics = pd.DataFrame({
        "index": ["nu", "beta[0]"],
        "mean": [11.2, 0.3],
        "r_hat": [1.004, 1.008],
        "ess_bulk": [1600, 900],
    })
    posture = reliability_posture(diagnostics=diagnostics, n_divergences=0)
    assert posture["posture"] == "OK"
    assert posture["nu"] == pytest.approx(11.2)
    assert posture["max_r_hat"] == pytest.approx(1.008)

    bad = reliability_posture(
        diagnostics=diagnostics.assign(r_hat=[1.09, 1.20]), n_divergences=12)
    assert bad["posture"] == "CONCERN"


def test_low_nu_adds_the_scale_caveat():
    diagnostics = pd.DataFrame({"index": ["nu"], "mean": [3.1],
                                "r_hat": [1.001], "ess_bulk": [2000]})
    posture = reliability_posture(diagnostics=diagnostics, n_divergences=0)
    assert "tail-aware" in posture["advice"]


def test_renderer_only_formats(signals, analytics):
    lines: list[str] = []
    render_recommendations(
        reliability=reliability_posture(n_divergences=0),
        group_signals=signals,
        actions=name_action_list(analytics),
        watch=size_down_watch(analytics),
        demoted=demotion_list(analytics),
        printer=lines.append,
    )
    assert any("GROUP POSTURE" in line for line in lines)
    assert any("SIZE-DOWN WATCH" in line for line in lines)
    assert any("NOT investment advice" in line for line in lines)
    # Frames are untouched: the percent scaling lives in the renderer alone.
    assert signals["excess_shrunk"].abs().max() < 1.0


def test_renderer_tolerates_every_section_missing():
    lines: list[str] = []
    render_recommendations(printer=lines.append)
    assert lines and any("NOT investment advice" in line for line in lines)
