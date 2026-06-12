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
import pandas as pd
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
# build_price_target_history — fiscal-calendar anchoring (pure pandas)
# ---------------------------------------------------------------------------
class TestFiscalAnchoring:
    @staticmethod
    def _snap():
        return pd.DataFrame(
            {
                "isin": ["A", "B"],
                "price_target": [100.0, 50.0],
                "price_target_1m_ago": [98.0, 49.0],
                "price_target_3m_ago": [95.0, 48.0],
                "income_statement_report_date": ["2025-03-31", "2024-12-31"],
            }
        )

    def test_per_isin_anchor_offsets_from_report_date(self):
        long_df, _eligible, date_col = KalmanFilterPriceTarget.build_price_target_history(
            self._snap(),
            now_cols=("price_target",),
            fiscal_anchor_col="income_statement_report_date",
        )
        assert date_col == "asof_date"
        # Each ISIN's "now" lands on its own report date; lags step back from it.
        a = long_df[long_df["isin"] == "A"].set_index("asof_date")["price_target"]
        assert a.loc[pd.Timestamp("2025-03-31")] == 100.0  # now
        assert a.loc[pd.Timestamp("2025-02-28")] == 98.0  # 1m before 2025-03-31
        assert a.loc[pd.Timestamp("2024-12-31")] == 95.0  # 3m before 2025-03-31
        b_max = long_df.loc[long_df["isin"] == "B", "asof_date"].max()
        assert b_max == pd.Timestamp("2024-12-31")  # B anchored independently

    def test_no_anchor_shares_single_global_now(self):
        long_df, _eligible, _ = KalmanFilterPriceTarget.build_price_target_history(
            self._snap(), now_cols=("price_target",)
        )
        a_max = long_df.loc[long_df["isin"] == "A", "asof_date"].max()
        b_max = long_df.loc[long_df["isin"] == "B", "asof_date"].max()
        assert a_max == b_max  # both anchored at the same global ref_now


# ---------------------------------------------------------------------------
# forecast() — structural projection invariants (pure; fabricated posterior)
# ---------------------------------------------------------------------------
def _fake_fitted_model(*, trend=True, chain=2, draw=40, n_time=6, seed=0, last_price=98.0):
    """A KalmanFilterPriceTarget with a fabricated posterior so forecast() can be
    exercised without running MCMC (the draw order is fixed so trend/no-trend
    share an identical terminal state)."""
    import xarray as xr

    rng = np.random.default_rng(seed)
    data = {
        "sigma_state": (("chain", "draw"), np.abs(rng.normal(0.1, 0.02, (chain, draw)))),
        "sigma_obs": (("chain", "draw"), np.abs(rng.normal(0.05, 0.01, (chain, draw)))),
        "log_state": (
            ("chain", "draw", "time"),
            rng.normal(np.log(100.0), 0.05, (chain, draw, n_time)),
        ),
    }
    if trend:
        data["beta_trend"] = (("chain", "draw"), rng.normal(0.2, 0.05, (chain, draw)))
    post = xr.Dataset(
        data,
        coords={"chain": np.arange(chain), "draw": np.arange(draw), "time": np.arange(n_time)},
    )

    class _ID:
        pass

    fake = _ID()
    fake.posterior = post
    kf = KalmanFilterPriceTarget()
    kf._fit_idata_ = fake
    kf._fit_last_price_ = last_price
    kf._fit_has_trend_ = trend
    return kf


class TestForecast:
    def test_shapes_positivity_and_labels(self):
        kf = _fake_fitted_model()
        pred = kf.forecast([30, 90, 180], labels=["a", "b", "c"])
        ds = pred.predictions
        assert ds.sizes["time_future"] == 3
        assert {"forecast_state", "forecast_pt", "predictive_sd_log",
                "implied_upside_future"}.issubset(set(ds.data_vars))
        assert (ds["forecast_pt"].values > 0).all()
        assert list(ds["label"].values) == ["a", "b", "c"]

    def test_predictive_variance_non_decreasing_in_horizon(self):
        kf = _fake_fitted_model()
        sd = (
            kf.forecast([10, 30, 60, 120, 365])
            .predictions["predictive_sd_log"]
            .mean(("chain", "draw"))
            .values
        )
        assert np.all(np.diff(sd) >= -1e-9)

    @pytest.mark.parametrize("bad", [[], [0.0], [-5.0], [np.inf], [np.nan]])
    def test_rejects_bad_horizons(self, bad):
        with pytest.raises(ValueError):
            _fake_fitted_model().forecast(bad)

    def test_requires_a_completed_fit(self):
        with pytest.raises(RuntimeError):
            KalmanFilterPriceTarget().forecast([30])

    def test_fiscal_dates_must_align_with_horizons(self):
        with pytest.raises(ValueError):
            _fake_fitted_model().forecast([30, 60], fiscal_dates=pd.to_datetime(["2026-07-01"]))

    def test_positive_trend_lifts_long_horizon_projection(self):
        # Identical terminal state (fixed draw order) -> the trend term adds
        # beta_trend * delta_tau, so the long-horizon forecast is strictly higher.
        ft = (
            _fake_fitted_model(trend=True)
            .forecast([365]).predictions["forecast_state"].mean(("chain", "draw")).values[0]
        )
        fn = (
            _fake_fitted_model(trend=False)
            .forecast([365]).predictions["forecast_state"].mean(("chain", "draw")).values[0]
        )
        assert ft > fn


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


def test_fit_with_trend_emits_beta_trend_and_forecasts():
    """trend=True adds the beta_trend slope, and forecast() projects to fiscal dates."""
    kf = KalmanFilterPriceTarget()
    dates = pd.date_range("2025-01-01", periods=len(_PTS), freq="30D")
    idata, _ = kf.fit(
        price_targets=_PTS, isin="TEST", dates=dates, last_price=109.0,
        trend=True, parameterization="marginalized", **_FIT_KW,
    )
    assert "beta_trend" in idata.posterior

    pred = kf.forecast(
        [30, 90], labels=["next_earnings", "expected_report"],
        fiscal_dates=pd.to_datetime(["2025-09-01", "2025-11-01"]),
    )
    ds = pred.predictions
    assert ds.sizes["time_future"] == 2
    assert (ds["forecast_pt"].values > 0).all()
    assert "implied_upside_future" in ds.data_vars  # last_price was supplied