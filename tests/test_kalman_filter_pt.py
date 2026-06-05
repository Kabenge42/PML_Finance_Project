"""Tests for :class:`KalmanFilterPriceTarget` (``KalmanFilterModel.py``).

Covers the ``feat_implied_upside`` integration:
  * :meth:`implied_upside_from_state` — pure-NumPy helper (no sampler needed).
  * The ``last_price`` anchor + ``implied_upside`` Deterministic emitted by
    :meth:`fit` (requires a sampler; guarded by ``importorskip``).

The single-ISIN time-series model mirrors the SQL ``feat_implied_upside`` feature
(``calc_change_ratio(price_target, last_price)``) and the cross-sectional
notebook's ``expected_upside`` so both Kalman variants report the same metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from probabilistic_ml_model.pymc_models.KalmanFilterModel import KalmanFilterPriceTarget


# ---------------------------------------------------------------------------
# implied_upside_from_state (pure NumPy — no PyMC required)
# ---------------------------------------------------------------------------
class TestImpliedUpsideFromState:
    def test_matches_calc_change_ratio_semantics(self):
        state = np.array([100.0, 110.0, 120.0, 90.0])
        out = KalmanFilterPriceTarget.implied_upside_from_state(state, 100.0)
        np.testing.assert_allclose(out, [0.0, 0.10, 0.20, -0.10])

    def test_non_positive_last_price_returns_nan(self):
        state = np.array([100.0, 110.0])
        assert np.isnan(
            KalmanFilterPriceTarget.implied_upside_from_state(state, 0.0)
        ).all()
        assert np.isnan(
            KalmanFilterPriceTarget.implied_upside_from_state(state, float("nan"))
        ).all()


# ---------------------------------------------------------------------------
# fit() — last_price anchor + implied_upside Deterministic
# ---------------------------------------------------------------------------
pytest.importorskip("pymc")

_PTS = np.array([100, 102, 101, 105, 108, 107, 110, 112], dtype="float64")
_FIT_KW = dict(samples=40, tune=40, chains=1, nuts_sampler="pymc",
               progressbar=False, random_seed=7)


@pytest.mark.parametrize("parameterization", ["marginalized", "non_centered"])
def test_fit_emits_implied_upside_when_last_price_given(parameterization):
    kf = KalmanFilterPriceTarget()
    idata, _ = kf.fit(
        price_targets=_PTS, isin="TEST", last_price=100.0,
        parameterization=parameterization, **_FIT_KW,
    )
    post = idata.posterior
    assert "implied_upside" in post
    assert "state" in post
    assert post["implied_upside"].dims[-1] == "time"
    # implied_upside == state / last_price - 1 holds exactly per draw.
    np.testing.assert_allclose(
        post["implied_upside"].values, post["state"].values / 100.0 - 1.0
    )


def test_fit_without_last_price_omits_implied_upside():
    kf = KalmanFilterPriceTarget()
    idata, _ = kf.fit(
        price_targets=_PTS, isin="TEST", parameterization="marginalized", **_FIT_KW,
    )
    assert "implied_upside" not in idata.posterior
    assert "state" in idata.posterior  # unchanged baseline behaviour preserved