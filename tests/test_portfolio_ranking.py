"""Three ranking arms, and the guarantee that adding them moved no default.

The first test is the one that matters most: every other test here describes NEW
behaviour, and only this one pins the behaviour that already shipped. A refactor that
quietly re-selects the book is the failure this file exists to catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probabilistic_ml_model.pymc_models.PortfolioOptimizationModel import (
    DEFAULT_RANKING_RULE,
    MIN_RATIO_DENOMINATOR,
    RANKING_RULES,
    downside_deviation,
    kelly_report,
    optimize_portfolio,
)

N_ISIN, N_SCEN, K_BOOK = 200, 1500, 25


@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(11)
    mu = rng.normal(0.18, 0.10, N_ISIN)
    sd = np.abs(rng.normal(0.22, 0.08, N_ISIN)) + 0.02
    return mu[:, None] + sd[:, None] * rng.standard_normal((N_ISIN, N_SCEN))


@pytest.fixture(scope="module")
def isins():
    return np.array([f"X{i:04d}" for i in range(N_ISIN)])


@pytest.fixture(scope="module")
def sectors():
    rng = np.random.default_rng(12)
    return rng.choice(["Tech", "Health", "Indust", "Fin"], N_ISIN, p=[.5, .2, .2, .1])


def _legacy_selection(draws, isins, k):
    """The pre-change rule, reimplemented: rank on expected / downside_dev with the
    absolute floor EXCLUDING anything beneath it."""
    dd = np.array([downside_deviation(draws[i]) for i in range(len(isins))])
    er = draws.mean(axis=1)
    ratio = np.where(dd >= MIN_RATIO_DENOMINATOR, er / np.maximum(dd, 1e-300), np.nan)
    frame = pd.DataFrame({"isin": isins, "r": ratio, "er": er})
    frame = frame[(frame.er > 0) & np.isfinite(frame.r)]
    return set(frame.sort_values("r", ascending=False).head(k)["isin"])


def test_default_arm_reproduces_the_shipped_selection(draws, isins):
    """No default moved. If this fails, the refactor changed the book."""
    book = optimize_portfolio(draws, isins, k_book=K_BOOK, cap=0.10)
    assert book.summary["rank_by"] == DEFAULT_RANKING_RULE == "reward_to_downside"
    assert set(book.weights.index) == _legacy_selection(draws, isins, K_BOOK)


def test_relative_floor_is_off_by_default(draws, isins):
    off = optimize_portfolio(draws, isins, k_book=K_BOOK)
    explicit = optimize_portfolio(draws, isins, k_book=K_BOOK,
                                  relative_denominator_q=0.0)
    assert set(off.weights.index) == set(explicit.weights.index)


def test_relative_floor_excludes_rather_than_clamps(draws, isins):
    """Clamping would hand every sub-floor name the same capped-but-large ratio and
    leave it in the running, admitting exactly what the floor exists to keep out."""
    strict = optimize_portfolio(draws, isins, k_book=K_BOOK,
                                relative_denominator_q=0.9)
    analytics = strict.analytics
    # The floor is reported as a BOOLEAN since 2026-08-27. It used to be the
    # masked denominator itself, `downside_dev_admitted`, which is `downside_dev`
    # wherever the floor did not bind -- a byte-identical copy of its own source
    # under the default `relative_denominator_q = 0.0`, and one of the two pairs
    # `export_duplicate_content` flagged on run 6efb530d5881. The boolean carries
    # the only thing the float ever added: WHICH names the floor cut.
    floored = analytics["downside_dev_floored"]
    assert floored.dtype == bool
    assert floored.any(), "a 0.9 relative floor must exclude something"
    # Everything below the floor is masked out, so it cannot be ranked at all.
    assert analytics.loc[floored, "reward_to_downside"].isna().all()
    assert strict.summary["n_eligible"] < optimize_portfolio(
        draws, isins, k_book=K_BOOK).summary["n_eligible"]


@pytest.mark.parametrize("arm", sorted(RANKING_RULES))
def test_every_arm_labels_itself(draws, isins, arm):
    kwargs = {}
    if arm == "p_upside_pos_cond":
        rng = np.random.default_rng(3)
        kwargs = dict(rank_values=np.clip(rng.beta(6, 1.2, N_ISIN), 0, 1),
                      rank_isins=isins)
    book = optimize_portfolio(draws, isins, k_book=K_BOOK, rank_by=arm, **kwargs)
    assert book.summary["rank_by"] == arm
    assert len(book.weights) <= K_BOOK


def test_bounded_arm_has_no_denominator(draws, isins):
    """The structural point: a probability cannot be inflated by a vanishing
    denominator because it has none."""
    rng = np.random.default_rng(4)
    book = optimize_portfolio(
        draws, isins, k_book=K_BOOK, rank_by="p_upside_pos_cond",
        rank_values=np.clip(rng.beta(6, 1.2, N_ISIN), 0, 1), rank_isins=isins,
    )
    assert np.isnan(book.summary["book_denominator_pctile_max"])
    assert book.analytics["rank_denominator_pctile"].isna().all()
    # Which column ranked is ONE fact about the run, recorded in the summary. It
    # used to be a `rank_denominator` column holding a verbatim per-name copy of
    # `downside_dev`, which is a duplicate rather than information.
    assert book.summary["rank_denominator_col"] == ""
    assert "rank_denominator" not in book.analytics.columns


def test_external_arm_refuses_to_invent_its_column(draws, isins):
    with pytest.raises(ValueError, match="rank_values"):
        optimize_portfolio(draws, isins, rank_by="p_upside_pos_cond")


def test_unknown_arm_is_refused(draws, isins):
    with pytest.raises(ValueError, match="Unknown rank_by"):
        optimize_portfolio(draws, isins, rank_by="reward_to_vibes")


def test_saturated_ranking_is_broken_deterministically(draws, isins):
    """With most of a universe tied at 1.0 the cut lands inside the tie, and without
    an explicit rule argsort's ordering silently becomes the selection."""
    rng = np.random.default_rng(3)
    p_cond = np.where(rng.random(N_ISIN) < 0.60, 1.0,
                      rng.uniform(0.3, 0.99, N_ISIN))

    first = optimize_portfolio(draws, isins, k_book=K_BOOK,
                               rank_by="p_upside_pos_cond",
                               rank_values=p_cond, rank_isins=isins)
    perm = rng.permutation(N_ISIN)
    shuffled = optimize_portfolio(draws[perm], isins[perm], k_book=K_BOOK,
                                  rank_by="p_upside_pos_cond",
                                  rank_values=p_cond, rank_isins=isins)

    assert first.summary["rank_tie_span"] > K_BOOK
    assert set(first.weights.index) == set(shuffled.weights.index)

    # And it is broken on the recorded key, not on position.
    held = first.analytics.set_index("isin").loc[list(first.weights.index)]
    tied_out = first.analytics[
        (~first.analytics["isin"].isin(first.weights.index))
        & (first.analytics["p_upside_pos_cond"] == 1.0)
    ]
    assert held["expected_return"].min() >= tied_out["expected_return"].max() - 1e-12


def test_kelly_pin_is_distinguishable_from_a_solution():
    rng = np.random.default_rng(9)
    no_loss = np.abs(rng.normal(0.3, 0.05, 2000))
    pinned = kelly_report(no_loss)
    assert pinned["kelly_fraction"] == 1.0
    assert pinned["kelly_interior"] == 0.0
    assert not np.isfinite(pinned["kelly_max_feasible"])
    # Growth was still RISING at the cap: the signature of a pin, not an optimum.
    assert pinned["kelly_endpoint_score"] > 0

    two_sided = rng.normal(0.05, 0.40, 4000)
    solved = kelly_report(two_sided)
    assert np.isfinite(solved["kelly_max_feasible"])
    if solved["kelly_fraction"] not in (0.0, 1.0):
        assert solved["kelly_interior"] == 1.0


def test_sector_cap_binds_without_breaching_the_name_cap(draws, isins, sectors):
    uncapped = optimize_portfolio(draws, isins, k_book=K_BOOK, cap=0.10,
                                  groups=sectors)
    capped = optimize_portfolio(draws, isins, k_book=K_BOOK, cap=0.10,
                                groups=sectors, sector_cap=0.30)

    assert uncapped.summary["top_group_weight"] > 0.30
    assert capped.summary["top_group_weight"] <= 0.30 + 1e-6
    assert capped.weights.max() <= 0.10 + 1e-9
    assert capped.weights.sum() == pytest.approx(1.0)


def test_sector_cap_is_off_by_default(draws, isins, sectors):
    """Its absence is a decision too; it just must never be taken by omission."""
    book = optimize_portfolio(draws, isins, k_book=K_BOOK, groups=sectors)
    assert np.isnan(book.summary["sector_cap"])


def test_eligibility_mask_is_honoured(draws, isins):
    mask = np.zeros(N_ISIN, dtype=bool)
    mask[:40] = True
    book = optimize_portfolio(draws, isins, k_book=K_BOOK, eligible=mask)
    assert set(book.weights.index) <= set(isins[:40])
