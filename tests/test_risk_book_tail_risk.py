"""Regression tests for the ``tail_risk`` denominator in :mod:`RiskBookModel`.

Two properties are pinned here, and they are different in kind.

The first is a semantics fix (2026-08-23): ``starr`` is exported as
``reward_to_cvar`` and documented as reward per unit *expected shortfall*, but
its loss leg was ``-er_p05`` — the 5 % quantile, a VaR. Since ``cvar05 <=
er_p05`` always holds for one distribution, switching the leg to ``-cvar05`` can
only raise ``tail_risk``, never lower it. That is asserted directly.

The second is a *negative* result worth keeping, because it is the thing a
future reader is most likely to try to "fix" again: on the favourable side of
the tail the leg is clipped away by the relative floor, so reshaping it cannot
change which names the book selects. The test constructs names with a strictly
positive 5 % tail and asserts that three candidate legs give an identical
denominator. If someone makes a leg sensitive there, this test fails — and the
failure is the signal to check whether the double-reward has been reintroduced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from probabilistic_ml_model.pymc_models.RiskBookModel import (
    DEFAULT_TAIL_RISK_VOL_FLOOR_K,
    MIN_TAIL_RISK,
    compute_cvar_aware_book,
)

N_NAMES = 30
N_SAMPLES = 600
RNG = np.random.default_rng(11)


def _case(loc: float):
    """Return ``(screen, draws, isins)`` for a universe centred on ``loc``.

    ``loc`` shifts the whole return distribution, so a large positive value
    yields names whose simulated 5 % quantile is above zero — the regime the
    published book lives in.
    """
    isins = np.array(["ISIN%04d" % i for i in range(N_NAMES)])
    scale = np.linspace(0.05, 0.25, N_NAMES)
    draws = RNG.normal(loc, 1.0, size=(N_NAMES, N_SAMPLES)) * scale[:, None]
    # Posterior upside draws, keyed by an ``isin`` coord as the workflow supplies
    # them. Always correctly aligned; present so the function under test runs.
    eu = xr.DataArray(
        RNG.normal(0.2, 0.05, size=(2, 40, N_NAMES)),
        dims=("chain", "draw", "isin"),
        coords={"chain": [0, 1], "draw": np.arange(40), "isin": isins},
    )
    screen = pd.DataFrame({
        "isin": isins,
        "expected_upside": np.linspace(0.05, 0.60, N_NAMES),
        "expected_upside_sd": np.full(N_NAMES, 0.04),
        "expected_pt": 100.0 * (1.0 + np.linspace(0.05, 0.60, N_NAMES)),
        "expected_pt_hdi_lo": np.full(N_NAMES, 90.0),
        "expected_pt_hdi_hi": np.full(N_NAMES, 110.0),
        "prob_pos": np.full(N_NAMES, 0.9),
        "mc_prob_pos": np.full(N_NAMES, 0.9),
        "p_upside_pos_cond": np.full(N_NAMES, 0.9),
        "p_upside_pos": np.full(N_NAMES, 0.9),
        "kalman_gain": np.full(N_NAMES, 1.0),
        "mcap_global_r": np.full(N_NAMES, 0.001),
        "er_mean": draws.mean(axis=1),
        "er_sd": draws.std(axis=1),
        "er_p05": np.quantile(draws, 0.05, axis=1),
        "er_p50": np.quantile(draws, 0.50, axis=1),
        "er_p95": np.quantile(draws, 0.95, axis=1),
    })
    return screen, draws, isins, eu


def _tail_of(screen, draws, isins, eu):
    rb = compute_cvar_aware_book(
        idata=None, eu=eu, results=screen,
        return_draws=draws, return_draws_isins=isins,
    )
    return rb.analytics.set_index("isin")


def test_loss_leg_is_the_shortfall_not_the_quantile():
    """``tail_risk`` must charge at least the old quantile leg, never less."""
    screen, draws, isins, eu = _case(loc=-0.4)          # genuine losing tails
    a = _tail_of(screen, draws, isins, eu)

    # The identity the switch rests on: shortfall is never milder than the
    # quantile it is taken beyond.
    assert (a["cvar05"] <= a["er_p05"] + 1e-12).all()

    old_leg = np.maximum.reduce([
        -a["er_p05"].to_numpy(),
        DEFAULT_TAIL_RISK_VOL_FLOOR_K * a["er_sd"].to_numpy(),
        np.full(len(a), MIN_TAIL_RISK),
    ])
    assert (a["tail_risk"].to_numpy() >= old_leg - 1e-12).all()

    # And for at least some names it is strictly larger, or the switch is inert.
    assert (a["tail_risk"].to_numpy() > old_leg + 1e-9).any()


def test_shortfall_leg_is_live_where_the_tail_is_a_loss():
    """On a losing tail the shortfall leg — not the floor — sets the denominator."""
    screen, draws, isins, eu = _case(loc=-0.4)
    a = _tail_of(screen, draws, isins, eu)
    floor = np.maximum(
        DEFAULT_TAIL_RISK_VOL_FLOOR_K * a["er_sd"].to_numpy(),
        MIN_TAIL_RISK,
    )
    live = ~np.isclose(a["tail_risk"].to_numpy(), floor, rtol=1e-9)
    assert live.mean() > 0.5, "shortfall leg should bind for most losing names"


def test_favourable_tail_collapses_every_candidate_leg_to_the_floor():
    """The negative result: no leg reshaping moves the book where it lives.

    With a strictly positive 5 % tail all three candidate legs are non-positive,
    so the relative floor takes the whole denominator and the three forms are
    numerically identical. This is why recommendation 01's stated target — cut
    the book's favourable-tail enrichment — is not reachable by editing this
    expression.
    """
    screen, draws, isins, eu = _case(loc=3.0)           # 5% quantile above zero (loc > 1.645 sd)
    a = _tail_of(screen, draws, isins, eu)
    assert (a["er_p05"] > 0).all(), "fixture must produce favourable tails"

    sd = a["er_sd"].to_numpy()
    floor = np.maximum(DEFAULT_TAIL_RISK_VOL_FLOOR_K * sd, MIN_TAIL_RISK)

    shipped = np.maximum(-a["cvar05"].to_numpy(), floor)          # current
    quantile = np.maximum(-a["er_p05"].to_numpy(), floor)         # pre-change
    additive = np.maximum(-a["er_p05"].to_numpy(), 0.0) + \
        DEFAULT_TAIL_RISK_VOL_FLOOR_K * sd                        # candidate (a)
    additive = np.maximum(additive, MIN_TAIL_RISK)

    np.testing.assert_allclose(shipped, quantile, rtol=0, atol=1e-12)
    np.testing.assert_allclose(shipped, additive, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["tail_risk"].to_numpy(), floor, rtol=0, atol=1e-12)

    # And therefore starr is exactly the rescaled reward-to-variability ratio.
    expected = a["expected_upside"].to_numpy() / (DEFAULT_TAIL_RISK_VOL_FLOOR_K * sd)
    np.testing.assert_allclose(a["starr"].to_numpy(), expected, rtol=1e-9)


def test_fallback_path_keeps_the_quantile_leg_and_the_dispersion_leg():
    """v1 takes ``return_draws=None``; that path must not change."""
    screen, draws, isins, eu = _case(loc=-0.4)
    rb = compute_cvar_aware_book(idata=None, eu=eu, results=screen,
                                 return_draws=None)
    a = rb.analytics.set_index("isin")
    # The dispersion leg is present only on the fallback, and dominates there.
    disp = (a["expected_upside"] - a["cvar05"]).clip(lower=0.0).to_numpy()
    assert (a["tail_risk"].to_numpy() >= disp - 1e-12).all()
    assert np.isclose(a["tail_risk"].to_numpy(), disp, rtol=1e-9).any()
