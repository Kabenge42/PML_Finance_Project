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
def _fake_fitted_model(
        *, trend=True, chain=2, draw=40, n_time=6, seed=0, last_price=98.0,
        stochastic_volatility=False,
):
    """A KalmanFilterPriceTarget with a fabricated posterior so forecast() can be
    exercised without running MCMC (the draw order is fixed so trend/no-trend
    share an identical terminal state).

    When ``stochastic_volatility`` is True the fabricated ``sigma_obs`` carries a
    ``time`` dim (mirroring the SV path) and a ``log_vol`` variable is added, so
    the dim-sniffing branch in :meth:`forecast` is exercised."""
    import xarray as xr

    rng = np.random.default_rng(seed)
    data = {
        "sigma_state": (("chain", "draw"), np.abs(rng.normal(0.1, 0.02, (chain, draw)))),
        "log_state": (
            ("chain", "draw", "time"),
            rng.normal(np.log(100.0), 0.05, (chain, draw, n_time)),
        ),
    }
    if stochastic_volatility:
        data["sigma_obs"] = (
            ("chain", "draw", "time"),
            np.abs(rng.normal(0.05, 0.01, (chain, draw, n_time))),
        )
        data["log_vol"] = (
            ("chain", "draw", "time"),
            rng.normal(np.log(0.05), 0.1, (chain, draw, n_time)),
        )
    else:
        data["sigma_obs"] = (
            ("chain", "draw"), np.abs(rng.normal(0.05, 0.01, (chain, draw)))
        )
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

    def test_forecast_handles_time_varying_sigma_obs(self):
        # Under stochastic volatility sigma_obs carries a `time` dim; forecast()
        # must dim-sniff it (holding the terminal volatility flat) and still
        # produce the same shapes/positivity as the scalar path, with a
        # non-decreasing predictive sd in the horizon.
        kf = _fake_fitted_model(stochastic_volatility=True)
        pred = kf.forecast([10, 30, 60, 120, 365])
        ds = pred.predictions
        assert ds.sizes["time_future"] == 5
        assert (ds["forecast_pt"].values > 0).all()
        sd = ds["predictive_sd_log"].mean(("chain", "draw")).values
        assert np.all(np.diff(sd) >= -1e-9)


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
    idata, _ = kf.fit(price_targets=_PTS, isin="TEST", last_price=100.0, parameterization=parameterization, **_FIT_KW)
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
    idata, _ = kf.fit(price_targets=_PTS, isin="TEST", parameterization="marginalized", **_FIT_KW)
    assert "implied_upside" not in idata.posterior
    assert "state" in idata.posterior  # unchanged baseline behaviour preserved


def test_default_fit_is_opt_in_marginalized_not_sv():
    """Regression guard: fit() defaults must be opt-in (trend=False,
    stochastic_volatility=False). A flipped default silently routes every caller
    through the SV / non_centered path — dropping ``log_state_init`` and the
    scalar ``sigma_obs`` (the §12 notebook KeyError). The other fit tests only
    assert vars common to both paths, so they do not catch this; this one does."""
    kf = KalmanFilterPriceTarget()
    idata, _ = kf.fit(price_targets=_PTS, isin="TEST", **_FIT_KW)
    post = idata.posterior
    # "auto" on a short series resolves to the marginalized closed form, which
    # exposes the scalar initial-level RV and a *scalar* sigma_obs.
    assert "log_state_init" in post
    assert "sigma_obs" in post and "time" not in post["sigma_obs"].dims
    # No stochastic-volatility latents must leak in by default.
    assert not ({"log_vol", "vol_step_size", "z_vol", "nu_obs"} & set(post.data_vars))
    # No structural trend by default.
    assert "beta_trend" not in post
    assert kf._fit_stochastic_volatility_ is False
    assert kf._fit_has_trend_ is False


def test_fit_with_trend_emits_beta_trend_and_forecasts():
    """trend=True adds the beta_trend slope, and forecast() projects to fiscal dates."""
    kf = KalmanFilterPriceTarget()
    dates = pd.date_range("2025-01-01", periods=len(_PTS), freq="30D")
    idata, _ = kf.fit(price_targets=_PTS, isin="TEST", dates=dates, last_price=109.0, trend=True,
                      parameterization="marginalized", **_FIT_KW)
    assert "beta_trend" in idata.posterior

    pred = kf.forecast(
        [30, 90], labels=["next_earnings", "expected_report"],
        fiscal_dates=pd.to_datetime(["2025-09-01", "2025-11-01"]),
    )
    ds = pred.predictions
    assert ds.sizes["time_future"] == 2
    assert (ds["forecast_pt"].values > 0).all()
    assert "implied_upside_future" in ds.data_vars  # last_price was supplied


# ---------------------------------------------------------------------------
# fit() — stochastic volatility (opt-in, time-varying observation noise)
# ---------------------------------------------------------------------------
def test_sv_emits_per_time_sigma_obs_and_priors():
    """stochastic_volatility=True swaps the scalar sigma_obs for a per-time
    log-volatility walk feeding a Student-t likelihood."""
    kf = KalmanFilterPriceTarget()
    idata, _ = kf.fit(price_targets=_PTS, isin="TEST", last_price=100.0, stochastic_volatility=True,
                      parameterization="non_centered", **_FIT_KW)
    post = idata.posterior
    assert {"log_vol", "vol_step_size", "nu_obs"}.issubset(set(post.data_vars))
    assert "sigma_obs" in post and post["sigma_obs"].dims[-1] == "time"
    assert kf._fit_stochastic_volatility_ is True


def test_sv_overrides_marginalized_with_warning(caplog):
    """SV is incompatible with the marginalized closed-form; the requested
    marginalized parameterization is silently downgraded to non_centered."""
    import logging

    kf = KalmanFilterPriceTarget()
    with caplog.at_level(logging.WARNING):
        idata, _ = kf.fit(
            price_targets=_PTS, isin="TEST",
            stochastic_volatility=True, parameterization="marginalized", **_FIT_KW,
        )
    assert "log_vol" in idata.posterior  # fell back to the explicit-state path
    assert any("marginalized" in rec.message for rec in caplog.records)


def test_sv_with_realized_vol_anchor_emits_offset():
    kf = KalmanFilterPriceTarget()
    realized_vol = np.full(_PTS.shape, 0.25, dtype="float64")
    idata, _ = kf.fit(
        price_targets=_PTS, isin="TEST", stochastic_volatility=True,
        realized_vol=realized_vol, parameterization="non_centered", **_FIT_KW,
    )
    assert "vol_anchor_offset" in idata.posterior


@pytest.mark.parametrize(
    "bad_vol",
    [
        np.full(len(_PTS) + 1, 0.2),  # length mismatch
        np.array([0.2] * (len(_PTS) - 1) + [-0.1]),  # non-positive entry
    ],
)
def test_sv_rejects_invalid_realized_vol(bad_vol):
    kf = KalmanFilterPriceTarget()
    with pytest.raises(ValueError):
        kf.fit(
            price_targets=_PTS, isin="TEST", stochastic_volatility=True,
            realized_vol=bad_vol, parameterization="non_centered", **_FIT_KW,
        )


def test_sv_composes_with_trend():
    kf = KalmanFilterPriceTarget()
    dates = pd.date_range("2025-01-01", periods=len(_PTS), freq="30D")
    idata, _ = kf.fit(
        price_targets=_PTS, isin="TEST", dates=dates, last_price=109.0,
        trend=True, stochastic_volatility=True, parameterization="non_centered",
        **_FIT_KW,
    )
    post = idata.posterior
    assert "beta_trend" in post
    assert "log_vol" in post


# ---------------------------------------------------------------------------
# Fused-panel coverage guard + ICM zero-variance assertion.
#
# Regression guard for the freeze (max R-hat 4.45, min ESS 4.3, 0 divergences):
# a sparsely-populated second response series (feat_pt_drift, NULL whenever the
# price_target_*_ago trail is empty) standardised to ~0 and left its rank-1 ICM
# loading unidentified. The coverage guard drops such series upstream; the builder
# asserts no zero-variance series can enter the ICM.
# ---------------------------------------------------------------------------
def _kalman_panel_frame(n=40, *, sparse_drift=True, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "isin": [f"X{i:04d}" for i in range(n)],
            "observed_pt": rng.uniform(10, 200, n),
            "last_price": rng.uniform(10, 200, n),
            "n_analysts": rng.integers(1, 20, n).astype(float),
            "feat_implied_upside": rng.normal(0.1, 0.3, n),
            "feat_avg_beta": rng.normal(1.0, 0.2, n),
            # EPS drift signal (replaced feat_mcap_vs_3yavg when the price-derived
            # market-cap / EV family left mv_pymc_kalman_pt).
            "feat_net_eps_drift": rng.normal(0.05, 0.3, n),
            "feat_pt_noise_sigma": rng.uniform(1, 5, n),
            "sector": rng.choice(["Tech", "Energy"], n),
            "region": rng.choice(["US", "EU"], n),
        }
    )
    # feat_pt_drift: mostly-NULL (below the 0.60 coverage gate) vs fully dense.
    df["feat_pt_drift"] = (
        np.where(np.arange(n) < 4, rng.normal(0, 0.1, n), np.nan)
        if sparse_drift
        else rng.normal(0, 0.1, n)
    )
    return df


# feat_pt_drift is no longer a DEFAULT response (it is a drift PREDICTOR, and its
# rank-1 ICM loading on the single-snapshot MV was a divergence driver — see
# KALMAN_PANEL_RESPONSE_COLS). These tests therefore exercise the coverage guard via
# an EXPLICIT multi-response request, which is the supported way to add a genuine
# second series (e.g. for a future collapse_time=False panel).
_EXPLICIT_RESP = ("feat_log_uplift", "feat_pt_drift")


def test_coverage_guard_drops_sparse_response_series():
    import pymc_kalman_filter_pt as kf

    panel = kf.prepare_kalman_panel_inputs(
        _kalman_panel_frame(sparse_drift=True), drift_features=["feat_pt_drift"],
        response_cols=_EXPLICIT_RESP,
    )
    assert panel.response_names == ["feat_log_uplift"]
    assert panel.Y.shape[-1] == 1


def test_coverage_guard_keeps_dense_response_series():
    import pymc_kalman_filter_pt as kf

    panel = kf.prepare_kalman_panel_inputs(
        _kalman_panel_frame(sparse_drift=False), drift_features=["feat_pt_drift"],
        response_cols=_EXPLICIT_RESP,
    )
    assert panel.response_names == ["feat_log_uplift", "feat_pt_drift"]
    assert panel.Y.shape[-1] == 2


def test_default_response_set_is_single_series():
    """The default fused panel models the single primary log-uplift response.

    Regression guard for the convergence fix: feat_pt_drift must NOT enter the
    default response set (it is a drift predictor + a divergence-driving sparse
    ICM series). The rank-1 ICM stays available via an explicit response_cols.
    """
    import pymc_kalman_filter_pt as kf

    panel = kf.prepare_kalman_panel_inputs(
        _kalman_panel_frame(sparse_drift=False), drift_features=["feat_pt_drift"],
    )
    assert panel.response_names == ["feat_log_uplift"]
    assert panel.Y.shape[-1] == 1


def test_builder_rejects_zero_variance_nonprimary_series():
    import dataclasses

    import pymc_kalman_filter_pt as kf
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        build_fused_kalman_pt_model,
    )

    panel = kf.prepare_kalman_panel_inputs(
        _kalman_panel_frame(sparse_drift=False), drift_features=["feat_pt_drift"]
    )
    # Force a degenerate (all-zero) second series past the data guard.
    y_primary = panel.Y[:, :, :1]
    bad = dataclasses.replace(
        panel,
        Y=np.concatenate([y_primary, np.zeros_like(y_primary)], axis=-1),
        response_names=["feat_log_uplift", "dead"],
    )
    with pytest.raises(ValueError, match="zero variance"):
        build_fused_kalman_pt_model(bad)


# ---------------------------------------------------------------------------
# De-standardisation moments (Finding 2).
#
# The fused model standardises the response tensor on the POOLED (isin x time)
# moments of the genuine *_ago trails. _panel_response_stats used to recompute
# them by tiling the SNAPSHOT column across T — correct only for the removed
# tile-based panel — which inflated every exported expected_upside /
# expected_pt by ~1.5-2.3 percentage points. These guard the exact inverse.
# ---------------------------------------------------------------------------
def _history_panel_frame(n=60, seed=3):
    """Frame carrying genuine price_target_{lb}_ago / price_{lb}_ago trails."""
    rng = np.random.default_rng(seed)
    df = _kalman_panel_frame(n=n, sparse_drift=False, seed=seed)
    # Trails drift away from the snapshot so pooled != snapshot moments.
    for lb, shift in (("6m", 0.85), ("3m", 0.90), ("1m", 0.95)):
        df[f"price_{lb}_ago"] = df["last_price"] * shift
        df[f"price_target_{lb}_ago"] = df["observed_pt"] * rng.uniform(
            0.8, 1.0, len(df)
        )
        df[f"price_target_stddev_{lb}_ago"] = rng.uniform(1, 5, len(df))
    return df


def test_panel_records_the_moments_it_standardised_with():
    import pymc_kalman_filter_pt as kf

    panel = kf.prepare_kalman_panel_inputs(
        _history_panel_frame(), drift_features=["feat_pt_drift"],
        history_lookbacks=("6m", "3m", "1m"),
    )
    assert panel.Y.shape[1] == 4, "expected a T=4 history panel"
    # The recorded moments must reproduce the standardisation exactly.
    flat = panel.Y.reshape(-1, panel.Y.shape[-1])
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-9)
    stats = kf._panel_response_stats(panel)
    mean, std = stats[panel.response_names[0]]
    assert mean == pytest.approx(float(panel.response_mean[0]))
    assert std == pytest.approx(float(panel.response_std[0]))


def test_pooled_moments_differ_from_snapshot_moments_on_a_history_panel():
    """The bug this guards is silent: both paths return finite, plausible numbers."""
    import pymc_kalman_filter_pt as kf

    df = _history_panel_frame()
    panel = kf.prepare_kalman_panel_inputs(
        df, drift_features=["feat_pt_drift"],
        history_lookbacks=("6m", "3m", "1m"),
    )
    pooled_mean, pooled_std = kf._panel_response_stats(panel)["feat_log_uplift"]
    snap = panel.frame["feat_log_uplift"].to_numpy(dtype="float64")
    # The legacy (snapshot-tiling) computation.
    assert pooled_mean != pytest.approx(float(np.mean(snap)), abs=1e-6), (
        "pooled and snapshot moments coincide; this fixture no longer "
        "exercises the de-standardisation bug"
    )
    assert pooled_mean == pytest.approx(float(panel.response_mean[0]))
    assert pooled_std == pytest.approx(float(panel.response_std[0]))


def test_t1_panel_moments_match_the_snapshot():
    """With T == 1 the pooled moments ARE the snapshot moments — no regression."""
    import pymc_kalman_filter_pt as kf

    panel = kf.prepare_kalman_panel_inputs(
        _history_panel_frame(), drift_features=["feat_pt_drift"],
        history_lookbacks=(),
    )
    assert panel.Y.shape[1] == 1
    mean, std = kf._panel_response_stats(panel)["feat_log_uplift"]
    snap = panel.frame["feat_log_uplift"].to_numpy(dtype="float64")
    assert mean == pytest.approx(float(np.mean(snap)))
    assert std == pytest.approx(float(np.std(snap)))


def test_panel_response_stats_warns_without_recorded_moments(caplog):
    """A pre-0.9.9.14 panel must fall back LOUDLY, never silently."""
    import dataclasses
    import logging

    import pymc_kalman_filter_pt as kf

    panel = kf.prepare_kalman_panel_inputs(
        _history_panel_frame(), drift_features=["feat_pt_drift"],
        history_lookbacks=("6m", "3m", "1m"),
    )
    legacy = dataclasses.replace(
        panel, response_mean=np.empty(0), response_std=np.empty(0)
    )
    with caplog.at_level(logging.WARNING):
        kf._panel_response_stats(legacy)
    assert any("fit-time response moments" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Local-level state (Finding 1).
#
# beta_t was identically zero because prepare_kalman_panel_inputs tiles the
# calendar time axis, so it is isin-CONSTANT and a per-series slope is exactly
# spanned by the T free alpha_level intercepts. It is no longer materialised
# there; per-ISIN time structure is carried by the state random walk instead.
# ---------------------------------------------------------------------------
def _built_model(**kw):
    import pymc_kalman_filter_pt as kf
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        build_fused_kalman_pt_model,
    )

    lookbacks = kw.pop("history_lookbacks", ("6m", "3m", "1m"))
    panel = kf.prepare_kalman_panel_inputs(
        _history_panel_frame(), drift_features=["feat_pt_drift"],
        history_lookbacks=lookbacks,
    )
    return panel, build_fused_kalman_pt_model(panel, **kw)


def test_no_beta_t_on_an_isin_constant_time_axis():
    panel, model = _built_model()
    assert np.ptp(panel.t_scaled, axis=0).max() < 1e-12, (
        "fixture must have a tiled (isin-constant) time axis"
    )
    assert "beta_t" not in model.named_vars
    assert "beta_slope" not in model.named_vars


def test_beta_t_returns_on_an_isin_varying_time_axis():
    import dataclasses

    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        build_fused_kalman_pt_model,
    )

    panel, _ = _built_model()
    rng = np.random.default_rng(0)
    varying = dataclasses.replace(
        panel, t_scaled=rng.normal(size=panel.t_scaled.shape)
    )
    model = build_fused_kalman_pt_model(varying)
    assert "beta_t" in model.named_vars
    assert "beta_slope" in model.named_vars


def test_per_isin_latent_is_restored_on_a_panel():
    """T>1 identifies a per-ISIN intercept; T==1 genuinely does not."""
    _, model = _built_model()
    assert "sigma_isin_level" in model.named_vars
    assert "z_isin_level" in model.named_vars
    assert tuple(model.named_vars["z_isin_level"].shape.eval()) == (60,)

    _, flat = _built_model(history_lookbacks=())
    assert "sigma_isin_level" not in flat.named_vars, (
        "the per-ISIN intercept must stay off on the T=1 cross-section, where it "
        "is non-identified"
    )


def test_per_isin_latent_can_be_pinned_off_for_baseline_comparison():
    _, model = _built_model(isin_level_scale=0.0)
    assert "sigma_isin_level" not in model.named_vars
    assert "z_isin_level" not in model.named_vars


def test_ar_state_layer_is_off_by_default():
    """It bought +0.013 recovery for min ESS 14 vs 69 at T=4 -- see the builder."""
    _, model = _built_model()
    assert "sigma_state" not in model.named_vars
    assert "state_rho" not in model.named_vars
    # The decision latent is still emitted, so consumers resolve unchanged.
    assert "state_now" in model.named_vars
    assert "state_path" in model.named_vars


def test_ar_state_layer_emits_its_parameters_when_enabled():
    _, model = _built_model(state_innovation_scale=0.1)
    for name in ("state_path", "state_now", "sigma_state", "state_rho", "z_state"):
        assert name in model.named_vars, f"{name} missing"
    assert tuple(model.named_vars["state_now"].shape.eval()) == (60,)
    # PyMC's ZeroSumNormal constrains the TRAILING axis, so the field is laid out
    # (time, isin) to zero-sum across names at each step.
    assert tuple(model.named_vars["z_state"].shape.eval()) == (4, 60)
    assert "time_innov" not in model.coords


def test_ar_state_is_zero_sum_across_isin_at_every_step():
    """Zero-sum across names is what keeps the field from aliasing alpha_level."""
    import pymc as pm

    _, model = _built_model(state_innovation_scale=0.1)
    with model:
        draw = pm.draw(
            [model.named_vars["state_path"], model.named_vars["mu_isin"]],
            draws=1, random_seed=5,
        )
    dev = np.asarray(draw[0]) - np.asarray(draw[1])[:, None]
    assert np.allclose(dev.mean(axis=0), 0.0, atol=1e-8)


def test_ar_state_marginal_variance_is_flat_across_time():
    """Stationarity: the property the cumulative random walk violated.

    A walk's marginal sd grows as sqrt(t); on the 2026-08-10 full-scale run that
    ramped per-time predictive coverage 89.9% -> 98.2% against a 94% target. The
    sqrt(1 - rho^2) innovation scaling is what holds it flat.
    """
    import pymc as pm

    _, model = _built_model(state_innovation_scale=0.1)
    with model:
        z = np.asarray(pm.draw(model.named_vars["z_state"], draws=600,
                               random_seed=11))
    sd_per_t = z.std(axis=(0, 2))  # (draws, time, isin) -> per time step
    assert sd_per_t.max() / sd_per_t.min() < 1.25, (
        f"marginal sd varies across time steps: {sd_per_t.round(3)}"
    )


def test_ar_state_rho_is_bounded_away_from_one():
    """rho -> 1 degenerates the AR into a second per-name intercept."""
    import pymc as pm

    from probabilistic_ml_model.pymc_models.KalmanFilterModel import _STATE_RHO_MAX

    _, model = _built_model(state_innovation_scale=0.1)
    with model:
        rho = np.asarray(pm.draw(model.named_vars["state_rho"], draws=500,
                                 random_seed=3))
    assert rho.max() < _STATE_RHO_MAX + 1e-9
    assert rho.min() >= 0.0


def test_state_collapses_to_the_anchor_when_pinned_off():
    _, model = _built_model(state_innovation_scale=0.0)
    assert "sigma_state" not in model.named_vars
    assert "z_state" not in model.named_vars
    assert "time_innov" not in model.coords
    # state_now still exists, so downstream consumers resolve unchanged.
    assert "state_now" in model.named_vars


def test_state_collapses_on_a_t1_cross_section():
    _, model = _built_model(history_lookbacks=())
    assert "sigma_state" not in model.named_vars
    assert "state_now" in model.named_vars


def test_screen_latent_resolves_with_fallback():
    """state_now wins; risk_adj_return keeps pre-state-layer idata readable."""
    import xarray as xr

    import pymc_kalman_filter_pt as kf

    both = xr.Dataset({"state_now": ("isin", [1.0, 2.0]),
                       "risk_adj_return": ("isin", [9.0, 9.0])})
    assert kf.resolve_screen_latent(both).values.tolist() == [1.0, 2.0]

    legacy = xr.Dataset({"risk_adj_return": ("isin", [3.0, 4.0])})
    assert kf.resolve_screen_latent(legacy).values.tolist() == [3.0, 4.0]

    with pytest.raises(KeyError):
        kf.resolve_screen_latent(xr.Dataset({"other": ("isin", [0.0])}))


# ---------------------------------------------------------------------------
# Optional second response series (D > 1) — opt-in ICM activation.
# ---------------------------------------------------------------------------
def test_response_extra_builds_a_second_series_with_a_real_trail():
    import pymc_kalman_filter_pt as kf

    panel = kf.prepare_kalman_panel_inputs(
        _history_panel_frame(),
        drift_features=["feat_pt_drift", "feat_pt_noise_drift"],
        history_lookbacks=("6m", "3m", "1m"),
        response_extra=("pt_dispersion",),
    )
    assert panel.response_names == ["feat_log_uplift", "pt_dispersion"]
    assert panel.Y.shape[-1] == 2
    # A genuine trail, not a snapshot-only column padded with NaN -> 0.
    assert panel.Y[:, :, 1].std() > 1e-3
    assert np.isfinite(panel.Y[:, :, 1]).all()
    # Response <-> predictor disjointness: the level's own first difference
    # must have been dropped from the drift design.
    assert "feat_pt_noise_drift" not in panel.drift_names


def test_response_extra_rejects_an_unknown_key():
    import pymc_kalman_filter_pt as kf

    with pytest.raises(ValueError, match="Unknown response_extra"):
        kf.prepare_kalman_panel_inputs(
            _history_panel_frame(), drift_features=["feat_pt_drift"],
            response_extra=("not_a_series",),
        )


# ---------------------------------------------------------------------------
# Drift-matrix collinearity pruning.
#
# The 21-column design had condition number 1580 / max VIF 162 because four
# price-target drift columns and four analyst-sentiment columns each restated a
# single signal. That left `beta` on thin ridges: R-hat 1.026 / bulk-ESS 140 at
# ZERO divergences on the 2026-08-10 full-scale run -- the last failing gate.
# Pruning to one representative apiece gives 15 columns / cond 23 / VIF 3.8.
# ---------------------------------------------------------------------------
_PRUNED_DRIFT_ALIASES = (
    "feat_pt_median_drift", "feat_pt_high_drift", "feat_pt_low_drift",
    "feat_analyst_bullish_pct", "feat_analyst_bearish_pct",
    "feat_analyst_conviction",
)
_RETAINED_REPRESENTATIVES = ("feat_pt_drift", "feat_analyst_rating")

# 2026-08-13: the price-derived market-cap / EV family (feat_mv_ev_drift,
# feat_mcap_trend_1y, feat_mcap_vs_3yavg, feat_ev_vs_3yavg) left
# mv_pymc_kalman_pt entirely and was replaced by the EPS family below. Only the
# valid-pair COUNTER is excluded from the drift matrix; the five signals are
# retained. Guarding both directions catches the two ways this regresses: a
# re-added counter (a metadata column masquerading as a signal) and a silently
# dropped signal.
_EPS_DRIFT_SIGNALS = (
    "feat_net_eps_drift", "feat_last_q_surprise", "feat_last_y_surprise",
    "feat_eps_beat_rate", "feat_eps_beat_rate_annual",
)


@pytest.mark.parametrize("alias", _PRUNED_DRIFT_ALIASES)
def test_collinear_drift_aliases_are_excluded(alias):
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        KALMAN_DRIFT_EXCLUDED_FEATURES,
    )

    assert alias in KALMAN_DRIFT_EXCLUDED_FEATURES


@pytest.mark.parametrize("alias", _RETAINED_REPRESENTATIVES)
def test_drift_family_representatives_are_retained(alias):
    """Each collapsed family must keep exactly one column in the design."""
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        KALMAN_DRIFT_EXCLUDED_FEATURES,
    )

    assert alias not in KALMAN_DRIFT_EXCLUDED_FEATURES


def test_select_drift_features_drops_the_collinear_families():
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        KalmanFilterPriceTarget,
    )

    candidates = [
        "feat_pt_drift", "feat_pt_median_drift", "feat_pt_high_drift",
        "feat_pt_low_drift", "feat_analyst_rating", "feat_analyst_bullish_pct",
        "feat_analyst_bearish_pct", "feat_analyst_conviction",
        "feat_analyst_neutral_pct", "feat_price_drift", "feat_coverage_drift",
    ]
    kept = KalmanFilterPriceTarget.select_drift_features(candidates)
    assert set(kept) == {
        "feat_pt_drift", "feat_analyst_rating", "feat_price_drift",
        "feat_coverage_drift",
    }


@pytest.mark.parametrize("alias", _EPS_DRIFT_SIGNALS)
def test_eps_drift_signals_are_retained(alias):
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        KALMAN_DRIFT_EXCLUDED_FEATURES,
    )

    assert alias not in KALMAN_DRIFT_EXCLUDED_FEATURES


def test_eps_drift_support_counter_is_excluded():
    """feat_net_eps_drift_n is a coverage diagnostic, not a signal."""
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        KALMAN_DRIFT_SUPPORT_COUNTERS,
        KALMAN_DRIFT_EXCLUDED_FEATURES,
    )

    assert "feat_net_eps_drift_n" in KALMAN_DRIFT_SUPPORT_COUNTERS
    assert "feat_net_eps_drift_n" in KALMAN_DRIFT_EXCLUDED_FEATURES


def test_select_drift_features_keeps_eps_family_without_the_counter():
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        KalmanFilterPriceTarget,
    )

    kept = KalmanFilterPriceTarget.select_drift_features(
        [*_EPS_DRIFT_SIGNALS, "feat_net_eps_drift_n"]
    )
    assert set(kept) == set(_EPS_DRIFT_SIGNALS)


def test_drift_fallback_matches_the_shipped_selection():
    """The offline literal must survive select_drift_features unchanged.

    ``_DRIFT_FEATURE_FALLBACK`` is the last resort when neither the catalogue
    frame nor the database is reachable. It drifted from the SSOT once already
    (it listed the analyst composition legs long after
    ``KALMAN_COLLINEAR_COMPOSITION_FEATURES`` started excluding them), which made
    the literal misleading without changing behaviour. Assert the fixed point.
    """
    import pymc_kalman_filter_pt as kf
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        KalmanFilterPriceTarget,
    )

    fallback = list(kf._DRIFT_FEATURE_FALLBACK)
    assert KalmanFilterPriceTarget.select_drift_features(fallback) == fallback
    assert len(fallback) == 16
    assert set(_EPS_DRIFT_SIGNALS).issubset(fallback)


def test_pruned_drift_design_is_well_conditioned():
    """The guard that actually regresses if a sibling feature is re-added.

    Builds a synthetic design reproducing the measured correlation structure --
    the target-band columns move almost rigidly together (r ~ 0.85) and the
    analyst columns likewise -- then checks that the retained subset is far
    better conditioned than the full one. Thresholds are loose because the
    fixture is synthetic; the real numbers are 1580 -> 23.
    """
    from probabilistic_ml_model.pymc_models.KalmanFilterModel import (
        KALMAN_DRIFT_EXCLUDED_FEATURES,
    )

    rng = np.random.default_rng(0)
    n = 2000
    pt_signal = rng.normal(size=n)
    analyst_signal = rng.normal(size=n)

    def near(sig, rho=0.88):
        return rho * sig + np.sqrt(1 - rho ** 2) * rng.normal(size=n)

    cols = {
        "feat_pt_drift": near(pt_signal),
        "feat_pt_median_drift": near(pt_signal),
        "feat_pt_high_drift": near(pt_signal),
        "feat_pt_low_drift": near(pt_signal),
        "feat_analyst_rating": near(analyst_signal),
        "feat_analyst_bullish_pct": near(analyst_signal),
        "feat_analyst_bearish_pct": -near(analyst_signal),
        "feat_analyst_conviction": near(analyst_signal),
        "feat_price_drift": rng.normal(size=n),
        "feat_coverage_drift": rng.normal(size=n),
    }
    df = pd.DataFrame(cols)

    def cond(frame):
        z = (frame - frame.mean()) / frame.std(ddof=0)
        ev = np.linalg.eigvalsh(z.corr().values)
        return float(ev.max() / max(ev.min(), 1e-12))

    kept = [c for c in df.columns if c not in KALMAN_DRIFT_EXCLUDED_FEATURES]
    full_cond, kept_cond = cond(df), cond(df[kept])
    assert kept_cond < full_cond / 5, (
        f"pruning barely helped: {full_cond:.0f} -> {kept_cond:.0f}"
    )
    assert kept_cond < 100, f"retained design still ill-conditioned: {kept_cond:.0f}"


def test_log_uplift_clip_band_matches_the_input_winsorisation():
    """The inverse map must be bounded by the same support as the forward one."""
    import pymc_kalman_filter_pt as kf

    assert kf.LOG_UPLIFT_CLIP_LO == pytest.approx(np.log1p(kf.UPLIFT_CLIP_LO))
    assert kf.LOG_UPLIFT_CLIP_HI == pytest.approx(np.log1p(kf.UPLIFT_CLIP_HI))
    # Decimal returns recovered from the clipped band stay inside it.
    assert np.expm1(kf.LOG_UPLIFT_CLIP_HI) == pytest.approx(kf.UPLIFT_CLIP_HI)
    assert np.expm1(kf.LOG_UPLIFT_CLIP_LO) == pytest.approx(kf.UPLIFT_CLIP_LO)


def test_clipping_bounds_expm1_blowup():
    """Guard for the 2026-08-10 export: er_sd reached 1.32e15 without this.

    expm1 is exponential, so an unclipped log-space tail becomes an astronomical
    decimal return. Clipping in log space caps it at the winsorisation band and
    is sign-preserving, so prob_pos is unaffected.
    """
    import pymc_kalman_filter_pt as kf

    rng = np.random.default_rng(0)
    # A heavy log-space tail of the kind a widened per-name posterior produces.
    draws = np.r_[rng.normal(0, 0.3, 5000), np.array([12.0, 20.0, 35.0, -18.0])]

    unclipped = np.expm1(draws)
    clipped = np.expm1(np.clip(draws, kf.LOG_UPLIFT_CLIP_LO, kf.LOG_UPLIFT_CLIP_HI))

    assert unclipped.max() > 1e5, "fixture no longer reproduces the blow-up"
    assert clipped.max() == pytest.approx(kf.UPLIFT_CLIP_HI)
    assert clipped.min() >= kf.UPLIFT_CLIP_LO - 1e-12
    # Sign preservation: clipping in log space cannot flip P(return > 0).
    assert ((unclipped > 0) == (clipped > 0)).all()


def test_out_of_support_suppression_rule():
    """Names pinned at the uplift cap must not carry a ranking score.

    Truncation collapses er_sd for a name whose whole forward distribution sits
    beyond the training support, so er_mean/er_sd explodes -- the 2026-08-10
    re-export produced Sharpe 717.7 on er_sd 0.007. That failure sorts to the TOP
    of a risk-adjusted screen while marking the least-understood names, which is
    strictly more dangerous than an obviously-broken 1e15.
    """
    import pymc_kalman_filter_pt as kf

    # Detect on er_p05, NOT er_mean. er_mean averages the clipped draws, so it
    # sits ~1e-4 below the cap even when the whole distribution is pinned -- a
    # first attempt used `er_mean >= cap - 1e-6` and matched ZERO of the 18 real
    # cases. er_p05 lands exactly on the cap once ~95% of draws are clipped.
    # Values below are the real ones from the 2026-08-10 export.
    df = pd.DataFrame({
        "isin": ["A", "B", "C"],
        "er_mean": [0.25, 4.999901, 4.998807],
        "er_p05": [0.0015, kf.UPLIFT_CLIP_HI, kf.UPLIFT_CLIP_HI],
        "er_sd": [0.20, 0.006966, 0.044114],
        "expected_sharpe_ratio": [1.25, 717.7, 113.3],
        "reward_to_cvar": [3.6, 500.0, 500.0],
        "cvar_book_weight": [0.04, 0.05, 0.05],
    })
    naive = (pd.to_numeric(df["er_mean"], errors="coerce")
             >= kf.UPLIFT_CLIP_HI - 1e-6).fillna(False)
    assert not naive.any(), "regression guard: er_mean detection silently misses"

    oos = (pd.to_numeric(df["er_p05"], errors="coerce")
           >= kf.UPLIFT_CLIP_HI - 1e-6).fillna(False)
    assert oos.tolist() == [False, True, True]

    for col in ("expected_sharpe_ratio", "reward_to_cvar", "cvar_book_weight"):
        df.loc[oos, col] = np.nan
    # The in-support name keeps its score; the pinned ones carry none.
    assert df.loc[0, "expected_sharpe_ratio"] == pytest.approx(1.25)
    assert df.loc[1:, "expected_sharpe_ratio"].isna().all()
    assert df.loc[1:, "reward_to_cvar"].isna().all()
    # er_* are RETAINED -- suppression hides the ranking, not the evidence.
    assert df["er_mean"].notna().all()
    assert df["er_sd"].notna().all()


def test_comparison_subsample_keeps_arrays_aligned():
    import pymc_kalman_filter_pt as kf

    panel = kf.prepare_kalman_panel_inputs(
        _history_panel_frame(n=60), drift_features=["feat_pt_drift"],
        history_lookbacks=("6m", "3m", "1m"),
    )
    sub = kf._subsample_panel(panel, 20, random_seed=1)
    assert len(sub.isins) == 20
    assert sub.Y.shape == (20, 4, 1)
    assert sub.X_drift.shape == (20, panel.X_drift.shape[1])
    assert len(sub.frame) == 20
    assert sub.frame["isin"].tolist() == list(sub.isins)
    for col, idx in sub.coord_idx.items():
        assert len(idx) == 20
        # Re-derived uniques: every index must address a real level.
        assert idx.max() < len(sub.coord_uniques[col])
    # The de-standardisation moments must survive the slice unchanged.
    assert np.allclose(sub.response_mean, panel.response_mean)
    # A cap at/above n is a no-op.
    assert kf._subsample_panel(panel, 10_000) is panel


# ---------------------------------------------------------------------------
# compute_cvar_aware_book — market-cap pre-selection gate (pure pandas/xarray;
# the achieve_prob path is defensively wrapped, so a stub idata falls back to
# kalman_gain = 1.0 and no sampler is required).
# ---------------------------------------------------------------------------
def _risk_book_inputs(*, drop_mcap_col=False, seed=0):
    """Minimal (idata, panel, screen, results) for compute_cvar_aware_book.

    Four names: A/D pass the 0.02 default gate, B fails on size, C fails on a
    missing rank (strict NaN policy).
    """
    import types

    import xarray as xr

    rng = np.random.default_rng(seed)
    isins = np.array(["A", "B", "C", "D"])
    eu = xr.DataArray(
        rng.normal(0.20, 0.05, (2, 100, len(isins))),
        dims=("chain", "draw", "isin"),
        coords={"isin": isins},
    )
    results = pd.DataFrame(
        {
            "isin": isins,
            "expected_pt": [110.0, 120.0, 130.0, 140.0],
            "expected_pt_hdi_lo": [100.0, 110.0, 120.0, 130.0],
            "expected_pt_hdi_hi": [120.0, 130.0, 140.0, 150.0],
            "expected_upside": [0.20, 0.22, 0.18, 0.25],
            "mc_prob_pos": [0.9, 0.9, 0.9, 0.9],
            "mcap_country_r": [0.01, 0.05, np.nan, 0.015],
        }
    )
    if drop_mcap_col:
        results = results.drop(columns=["mcap_country_r"])
    screen = types.SimpleNamespace(eu=eu)
    return object(), None, screen, results


def test_cvar_book_mcap_gate_default_excludes_small_and_unranked():
    import pymc_kalman_filter_pt as kf

    idata, panel, screen, results = _risk_book_inputs()
    rb = kf.compute_cvar_aware_book(
        idata, panel, screen, results, config=kf.KalmanRunConfig(),
    )
    assert set(rb.book["isin"]) == {"A", "D"}
    zeroed = rb.analytics.set_index("isin").loc[["B", "C"], "book_weight"]
    assert (zeroed == 0.0).all()
    assert rb.summary["mcap_r_max"] == pytest.approx(0.02)
    assert rb.summary["n_mcap_eligible"] == 2.0
    assert rb.summary["n_book"] == 2.0


def test_cvar_book_mcap_gate_loose_threshold_keeps_ranked_names_only():
    import pymc_kalman_filter_pt as kf

    idata, panel, screen, results = _risk_book_inputs()
    rb = kf.compute_cvar_aware_book(
        idata, panel, screen, results,
        mcap_r_max=1.0, config=kf.KalmanRunConfig(),
    )
    # B re-enters under the loose gate; C stays out (NaN rank is strict).
    assert set(rb.book["isin"]) == {"A", "B", "D"}
    assert rb.summary["mcap_r_max"] == pytest.approx(1.0)
    assert rb.summary["n_mcap_eligible"] == 3.0


def test_cvar_book_mcap_gate_skipped_when_column_missing(caplog):
    import pymc_kalman_filter_pt as kf

    idata, panel, screen, results = _risk_book_inputs(drop_mcap_col=True)
    with caplog.at_level("WARNING", logger="pymc_kalman_filter_pt"):
        rb = kf.compute_cvar_aware_book(
            idata, panel, screen, results, config=kf.KalmanRunConfig(),
        )
    # Pre-0.9.9.12 frame: gate is skipped (all eligible) with a warning.
    assert set(rb.book["isin"]) == {"A", "B", "C", "D"}
    assert rb.summary["n_mcap_eligible"] == 4.0
    assert any("mcap_country_r" in rec.getMessage() for rec in caplog.records)


def test_run_config_mcap_gate_default():
    import pymc_kalman_filter_pt as kf

    assert kf.KalmanRunConfig().mcap_country_r_max == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 0.9.9.17 figure-payload + PPC-thinning helpers (pure NumPy / pandas)
#
# These guard the notebook-size regression that put the v4 notebook at 233 MB,
# 207.7 MB of it in ONE prior-predictive figure whose ``go.Histogram`` traces
# shipped ~6.5 M raw float64 each to be binned client-side.
# ---------------------------------------------------------------------------
class TestBinnedDensityTrace:
    def test_payload_is_bounded_by_bins_not_sample_size(self):
        import pymc_kalman_filter_pt as kf

        rng = np.random.default_rng(0)
        small = kf._binned_density_trace(rng.normal(size=1_000), bins=80,
                                         color="#56b4e9", name="x")
        large = kf._binned_density_trace(rng.normal(size=500_000), bins=80,
                                         color="#56b4e9", name="x")
        # The whole point: 500x the sample, same number of plotted coordinates.
        assert len(small.x) == len(large.x) == 80
        assert len(small.y) == len(large.y) == 80

    def test_density_normalises_and_counts_do_not(self):
        import pymc_kalman_filter_pt as kf

        v = np.repeat([0.0, 1.0], 500)
        dens = kf._binned_density_trace(v, bins=10, color="#ffffff", density=True)
        cnts = kf._binned_density_trace(v, bins=10, color="#ffffff", density=False)
        # A count trace must sum to n, so a "count" axis label stays truthful.
        assert float(np.sum(cnts.y)) == pytest.approx(1000.0)
        assert float(np.sum(dens.y)) != pytest.approx(1000.0)

    def test_clip_is_applied_before_binning(self):
        import pymc_kalman_filter_pt as kf

        v = np.array([-50.0, 0.0, 0.5, 1.0, 50.0])
        tr = kf._binned_density_trace(v, bins=4, color="#ffffff", clip=(0.0, 1.0))
        assert float(np.min(tr.x)) >= 0.0
        assert float(np.max(tr.x)) <= 1.0

    def test_degenerate_sample_returns_none(self):
        import pymc_kalman_filter_pt as kf

        assert kf._binned_density_trace(np.array([]), color="#ffffff") is None
        assert kf._binned_density_trace(np.array([np.nan, np.inf]),
                                        color="#ffffff") is None

    def test_add_binned_density_skips_a_degenerate_series(self):
        import plotly.graph_objects as go

        import pymc_kalman_filter_pt as kf

        fig = go.Figure()
        kf._add_binned_density(fig, np.array([np.nan]), color="#ffffff")
        assert len(fig.data) == 0  # never fig.add_trace(None)


class TestEcdfGrid:
    def test_grid_reproduces_the_empirical_quantiles(self):
        import pymc_kalman_filter_pt as kf

        rng = np.random.default_rng(1)
        v = rng.normal(size=25_948)  # the real (isin x time) response cell count
        xs, ys = kf._ecdf_xy(v, n=512)
        assert xs.size == ys.size == 512
        # Sampling the quantile function IS the ECDF, so the curve is exact at
        # every plotted point -- the reduction is in points, not in fidelity.
        np.testing.assert_allclose(xs, np.quantile(v, ys), atol=1e-12)

    def test_empty_input_returns_empty_arrays(self):
        import pymc_kalman_filter_pt as kf

        xs, ys = kf._ecdf_xy(np.array([np.nan, np.nan]))
        assert xs.size == 0 and ys.size == 0


class TestDecimateFrame:
    @staticmethod
    def _frame(n=5_000):
        rng = np.random.default_rng(2)
        return pd.DataFrame({"v": rng.normal(size=n),
                             "sector": rng.choice(list("ABCDEF"), n)})

    def test_no_op_below_the_cap(self):
        import pymc_kalman_filter_pt as kf

        df = self._frame(100)
        out, thinned = kf._decimate_frame(df, 1_000, by="sector")
        assert thinned is False
        assert out is df  # not even a copy

    def test_disabled_by_non_positive_cap(self):
        import pymc_kalman_filter_pt as kf

        df = self._frame(100)
        out, thinned = kf._decimate_frame(df, 0)
        assert thinned is False and out is df

    def test_stratified_sample_keeps_the_grouping_column(self):
        import pymc_kalman_filter_pt as kf

        # Regression guard: pandas 3 excludes grouping columns from the frame
        # handed to ``groupby.apply``, so an apply-based implementation silently
        # dropped the very column it stratified on.
        out, thinned = kf._decimate_frame(self._frame(), 1_200, by="sector")
        assert thinned is True
        assert "sector" in out.columns
        assert len(out) == 1_200
        assert out["sector"].nunique() == 6  # every stratum survives

    def test_sample_is_uniform_not_a_top_n_cut(self):
        import pymc_kalman_filter_pt as kf

        # A rank-based cut would delete one tail entirely and leave a cloud that
        # misrepresents the screen; a uniform sample preserves both tails.
        df = self._frame()
        out, _ = kf._decimate_frame(df, 1_200, by="sector")
        assert out["v"].min() < df["v"].quantile(0.02)
        assert out["v"].max() > df["v"].quantile(0.98)

    def test_is_reproducible_for_a_fixed_seed(self):
        import pymc_kalman_filter_pt as kf

        df = self._frame()
        a, _ = kf._decimate_frame(df, 900, by="sector", seed=7)
        b, _ = kf._decimate_frame(df, 900, by="sector", seed=7)
        pd.testing.assert_frame_equal(a, b)


class TestThinPosterior:
    @staticmethod
    def _idata(chains=4, draws=2000):
        azb = pytest.importorskip("arviz_base")
        rng = np.random.default_rng(3)
        return azb.from_dict({
            "posterior": {"beta": rng.normal(size=(chains, draws, 5))},
            "prior": {"beta": rng.normal(size=(1, 1000, 5))},
            "observed_data": {"target_pct_obs": rng.normal(size=(50, 4))},
        })

    def test_thins_to_approximately_the_target_total(self):
        import pymc_kalman_filter_pt as kf

        out = kf.thin_posterior(self._idata(), 1_000)
        sizes = out.posterior.sizes
        assert sizes["chain"] * sizes["draw"] == 1_000

    def test_leaves_the_production_object_untouched(self):
        import pymc_kalman_filter_pt as kf

        # §8 must not weld its predictive group onto the real idata: it would
        # ride into 07_posterior_idata.nc and be swept again by §9.
        idata = self._idata()
        kf.thin_posterior(idata, 1_000)
        assert idata.posterior.sizes["draw"] == 2000

    def test_preserves_groups_without_a_draw_dim(self):
        import pymc_kalman_filter_pt as kf

        out = kf.thin_posterior(self._idata(), 1_000)
        assert out.observed_data["target_pct_obs"].shape == (50, 4)

    @pytest.mark.parametrize("target", [0, -1, 10 ** 9])
    def test_no_op_returns_the_same_object(self, target):
        import pymc_kalman_filter_pt as kf

        idata = self._idata()
        assert kf.thin_posterior(idata, target) is idata


def test_azp_backend_split_and_env_override(monkeypatch):
    import pymc_kalman_filter_pt as kf

    monkeypatch.delenv(kf._AZP_HEAVY_BACKEND_ENV, raising=False)
    assert kf._azp_backend() == "plotly"
    assert kf._azp_backend(heavy=True) == "matplotlib"

    monkeypatch.setenv(kf._AZP_HEAVY_BACKEND_ENV, "plotly")
    assert kf._azp_backend(heavy=True) == "plotly"
    assert kf._azp_backend() == "plotly"  # light panels are never overridden

    monkeypatch.setenv(kf._AZP_HEAVY_BACKEND_ENV, "nonsense")
    assert kf._azp_backend(heavy=True) == "matplotlib"  # junk falls back


def test_mpl_figure_of_resolves_raw_and_wrapped_figures():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pymc_kalman_filter_pt as kf

    fig = plt.figure()
    try:
        assert kf._mpl_figure_of(fig) is fig
        assert kf._plotly_figure_of(fig) is None
        assert kf._mpl_figure_of(object()) is None
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Forecast-error shrinkage (2026-08-20). Pure NumPy/xarray — no sampler.
#
# These guard the change that stopped the v2 screen reproducing analyst
# consensus. Run 49e84d7e9d59 shipped a median revision of 0.03pp and a Spearman
# of 0.999995 against its own input, so the properties asserted here (a gain
# strictly below 1, monotone in coverage, and an exact identity at kappa = 0)
# are the difference between a filter and a pass-through.
# ---------------------------------------------------------------------------
def _fe_panel(n=80, seed=0):
    """Minimal KalmanPanelV2 carrying only what the shrinkage helper reads."""
    from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import KalmanPanelV2

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "isin": [f"X{i:03d}" for i in range(n)],
            "n_analysts": rng.integers(2, 40, n).astype(float),
        }
    )
    zeros = np.zeros(n)
    return KalmanPanelV2(
        frame=frame,
        isins=frame["isin"].to_numpy(),
        Y=np.zeros((n, 4)),
        time_days=np.array([365.0, 91.0, 7.0, 0.0]),
        X_drift=np.zeros((n, 1)),
        drift_names=["a"],
        dispersion_cv=np.clip(rng.normal(0.12, 0.05, n), 0.01, None),
        precision_weight=np.ones(n),
        vol_level=zeros,
        log_mcap=zeros,
        range_norm=zeros,
        avg_beta=zeros,
        size_ratio=zeros,
        volume_ratio=zeros,
        response_mean=0.19,
        response_std=0.2207,
    )


def _fe_idata(panel, seed=1, state_sd=0.02):
    """Stub posterior with the five variables the shrinkage helper reads."""
    import types

    import xarray as xr

    rng = np.random.default_rng(seed)
    n, c, d = len(panel.isins), 2, 60

    def _da(v):
        return xr.DataArray(
            v, dims=("chain", "draw", "isin"), coords={"isin": panel.isins}
        )

    post = xr.Dataset(
        {
            "mu_scaled": _da(rng.normal(0.0, 0.3, (c, d, n))),
            "state_now_mean": _da(rng.normal(0.0, 0.8, (c, d, n))),
            "state_now_sd": _da(np.full((c, d, n), state_sd)),
            "sigma_isin": _da(np.full((c, d, n), 0.55)),
            "variance_weights": xr.DataArray(
                np.tile([0.0044, 0.9940, 0.0016], (c, d, 1)),
                dims=("chain", "draw", "variance_component"),
            ),
        }
    )
    return types.SimpleNamespace(posterior=post)


def test_forecast_error_variance_is_zero_at_zero_multiplier():
    from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
        forecast_error_variance,
    )

    panel = _fe_panel()
    np.testing.assert_allclose(
        forecast_error_variance(panel, multiplier=0.0), 0.0
    )
    assert (forecast_error_variance(panel, multiplier=2.0) > 0).all()


@pytest.mark.parametrize("bad", [-0.1, -1.0])
def test_forecast_error_variance_rejects_a_negative_multiplier(bad):
    from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
        forecast_error_variance,
    )

    with pytest.raises(ValueError, match="non-negative"):
        forecast_error_variance(_fe_panel(), multiplier=bad)


def test_zero_multiplier_returns_the_unshrunk_latent_exactly():
    """The opt-out path must be bit-exact, not merely close.

    ``enable_forecast_error_shrinkage=False`` is the comparison arm for every
    before/after measurement, so a gain of 1 that only approximately reproduces
    the old latent would put noise into the very deltas the change is judged on.
    """
    from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
        apply_forecast_error_shrinkage,
    )

    panel = _fe_panel()
    idata = _fe_idata(panel)
    theta, gain = apply_forecast_error_shrinkage(idata, panel, multiplier=0.0)

    np.testing.assert_allclose(gain, 1.0)
    s_mean = np.asarray(idata.posterior["state_now_mean"])
    s_sd = np.asarray(idata.posterior["state_now_sd"])
    rng = np.random.default_rng(42)  # the helper's default random_seed
    expected = s_mean + s_sd * rng.standard_normal(s_mean.shape)
    np.testing.assert_allclose(np.asarray(theta), expected)


def test_gain_rises_with_coverage_and_falls_with_dispersion():
    """The coverage_gradient gate reads this monotonicity.

    Run 49e84d7e9d59 had ``kalman_gain`` correlating -0.004 with analyst count,
    i.e. the exported confidence term was blind to how much information a name
    carried. The shrinkage weight must not repeat that.
    """
    from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
        apply_forecast_error_shrinkage,
    )

    panel = _fe_panel()
    _, gain = apply_forecast_error_shrinkage(panel and _fe_idata(panel), panel,
                                             multiplier=2.0)
    n = panel.frame["n_analysts"]
    cv = pd.Series(panel.dispersion_cv)
    assert pd.Series(gain).corr(n, method="spearman") > 0.5
    assert pd.Series(gain).corr(cv, method="spearman") < -0.5
    assert (gain > 0).all() and (gain < 1).all()


def test_shrinkage_pulls_the_latent_toward_the_fitted_mean():
    from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
        apply_forecast_error_shrinkage,
        forecast_error_variance,
    )

    panel = _fe_panel()
    idata = _fe_idata(panel)
    _, gain = apply_forecast_error_shrinkage(idata, panel, multiplier=2.0)

    # Assert on the closed form rather than on sampled draws: theta carries an
    # added sd of ~sqrt(g * fe_var), so a 120-draw sample mean cannot resolve a
    # shift this size (a real false negative seen while writing these tests).
    mu = np.asarray(idata.posterior["mu_scaled"]).mean((0, 1))
    s_mean = np.asarray(idata.posterior["state_now_mean"]).mean((0, 1))
    theta_mean = mu + gain * (s_mean - mu)
    assert np.all(np.abs(theta_mean - mu) <= np.abs(s_mean - mu) + 1e-12)
    assert np.any(np.abs(theta_mean - mu) < np.abs(s_mean - mu))


def test_shrinkage_widens_the_per_name_posterior():
    """The prob_pos_degenerate warning is downstream of this.

    With a 0.47pp posterior sd against an 18pp median upside, P(upside > 0)
    saturates: 87.4% of the universe sat at exactly 1.0 on run 49e84d7e9d59.
    """
    from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
        apply_forecast_error_shrinkage,
    )

    panel = _fe_panel()
    idata = _fe_idata(panel, state_sd=0.02)
    theta, _ = apply_forecast_error_shrinkage(idata, panel, multiplier=2.0)
    assert np.all(np.asarray(theta).std((0, 1)) > 0.02)


def test_shrinkage_rejects_a_panel_from_another_run():
    from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
        apply_forecast_error_shrinkage,
    )

    panel = _fe_panel(n=80)
    idata = _fe_idata(_fe_panel(n=40))
    with pytest.raises(ValueError, match="not from the same run"):
        apply_forecast_error_shrinkage(idata, panel, multiplier=2.0)


# ---------------------------------------------------------------------------
# Risk book: return-space CVaR and the relative tail-risk floor (2026-08-20)
# ---------------------------------------------------------------------------
def _rb_frame(n=40, seed=3):
    """Screen frame + posterior upside draws + MC return draws for the book."""
    import xarray as xr

    rng = np.random.default_rng(seed)
    isins = np.array([f"N{i:02d}" for i in range(n)])
    upside = rng.uniform(0.05, 0.60, n)
    # Posterior draws of the MEAN: tight, as the real ones are (0.47pp median).
    eu = xr.DataArray(
        upside[None, None, :] + rng.normal(0.0, 0.005, (2, 200, n)),
        dims=("chain", "draw", "isin"),
        coords={"isin": isins},
    )
    # Forward-return draws: wide, as the real ones are (19pp median sd).
    ret = upside[:, None] + rng.normal(0.0, 0.20, (n, 400))
    results = pd.DataFrame(
        {
            "isin": isins,
            "expected_pt": 100 * (1 + upside),
            "expected_pt_hdi_lo": 100 * (1 + upside - 0.01),
            "expected_pt_hdi_hi": 100 * (1 + upside + 0.01),
            "expected_upside": upside,
            "mc_prob_pos": np.clip(rng.uniform(0.6, 0.99, n), 0, 1),
            "mcap_global_r": rng.uniform(0.0, 0.015, n),
            "er_mean": upside,
            "er_sd": ret.std(axis=1),
            "er_p05": np.quantile(ret, 0.05, axis=1),
        }
    )
    return object(), eu, results, ret


def test_cvar_from_return_draws_is_a_real_loss_quantile():
    """cvar05 must sit at or below er_p05 — a tail MEAN cannot exceed its own
    tail QUANTILE. On run 49e84d7e9d59 it exceeded it for 88.4% of names,
    because it was computed from the posterior of the mean instead."""
    from probabilistic_ml_model.pymc_models.RiskBookModel import compute_cvar_aware_book

    idata, eu, results, ret = _rb_frame()
    rb = compute_cvar_aware_book(idata, eu, results, return_draws=ret)
    a = rb.analytics
    assert (a["cvar05"] <= a["er_p05"] + 1e-9).all()
    np.testing.assert_allclose(a["exp_vol"], a["er_sd"], rtol=1e-9)


def test_cvar_falls_back_to_the_posterior_and_says_so(caplog):
    """The degraded path must be loud: silently sizing on estimation
    uncertainty is exactly the failure this change exists to end."""
    from probabilistic_ml_model.pymc_models.RiskBookModel import compute_cvar_aware_book

    idata, eu, results, _ = _rb_frame()
    with caplog.at_level(
        "WARNING", logger="probabilistic_ml_model.pymc_models.RiskBookModel"
    ):
        rb = compute_cvar_aware_book(idata, eu, results, return_draws=None)
    assert "return_draws" in caplog.text
    # The posterior-derived vol is ~40x tighter than the return sd — the
    # discrepancy that made the exported book report a positive CVaR.
    assert (rb.analytics["exp_vol"] < rb.analytics["er_sd"] / 10).all()


def test_relative_tail_floor_unpins_names_with_no_simulated_loss():
    """The floor guards the POSTERIOR-derived path, which is where it bit.

    Worth being precise about, because it is not what the plan assumed:
    repointing ``cvar05`` at the return draws (R4) makes
    ``expected_upside - cvar05`` roughly ``2 * er_sd``, which dominates any
    sensible fraction of ``er_sd`` — so on the primary path the absolute floor
    is already unreachable and the relative one never binds.

    It binds exactly where the shipped book was broken: the fallback path, where
    ``cvar05`` comes from the tight posterior-of-the-mean draws, the mean-to-CVaR
    dispersion is ~1pp, and a name whose simulated 5% quantile is positive has no
    downside term at all. That was 29.6% of the universe and 14 of the 25 book
    names on run 49e84d7e9d59, where STARR became 100 x expected_upside.
    """
    from probabilistic_ml_model.pymc_models.RiskBookModel import (
        MIN_TAIL_RISK,
        compute_cvar_aware_book,
    )

    idata, eu, results, _ = _rb_frame()
    results = results.copy()
    # Entirely positive simulated distribution => no MC loss leg.
    results["er_p05"] = results["er_p05"].abs() + 0.05

    pinned = compute_cvar_aware_book(
        idata, eu, results, return_draws=None, tail_risk_vol_floor_k=0.0
    ).analytics
    floored = compute_cvar_aware_book(
        idata, eu, results, return_draws=None, tail_risk_vol_floor_k=0.25
    ).analytics

    # Without the relative floor everything sits in the floor regime: the only
    # other live term is the ~1pp posterior mean-to-CVaR dispersion.
    assert pinned["tail_risk"].max() < 2 * MIN_TAIL_RISK
    assert (pinned["tail_risk"] >= MIN_TAIL_RISK - 1e-12).all()
    # With it, every name is charged a real fraction of its own dispersion.
    assert (floored["tail_risk"] > 2 * MIN_TAIL_RISK).all()
    # STARR stops being ~100 x expected_upside, which is what made it bimodal.
    assert floored["starr"].max() < pinned["starr"].max() / 2


def test_screen_supplied_probability_columns_win_over_the_legacy_sigmoid():
    """p_upside_pos_cond passed in by the workflow must not be overwritten by
    the mc_prob_pos * kalman_gain product the risk book falls back to."""
    from probabilistic_ml_model.pymc_models.RiskBookModel import compute_cvar_aware_book

    idata, eu, results, ret = _rb_frame()
    results = results.copy()
    results["kalman_gain"] = 0.5
    results["p_upside_pos_cond"] = 0.42
    rb = compute_cvar_aware_book(idata, eu, results, return_draws=ret)
    np.testing.assert_allclose(rb.analytics["p_upside_pos_cond"], 0.42)


# ---------------------------------------------------------------------------
# The tightened shrinkage gate (2026-08-20)
# ---------------------------------------------------------------------------
def test_shrinkage_gate_band_excludes_identity():
    """Regression guard on the thresholds themselves.

    The gate passed run 49e84d7e9d59 at slope 0.979 / intercept +0.0051 while
    the screen reproduced consensus at Spearman 0.999995 and a 0.03pp median
    revision. Slope and intercept alone cannot see that: an exact copy scores a
    perfect 1.0 / 0.0. If someone widens the band back past 1.0, or drops either
    companion statistic, this fails.
    """
    import pymc_kalman_filter_pt_v2 as kf2

    cfg = kf2.KalmanRunConfigV2()
    assert cfg.gate_shrinkage_slope_hi < 1.0
    assert cfg.gate_shrinkage_rho_max < 1.0
    assert cfg.gate_shrinkage_revision_min_pp > 0.0

    # The shipped run's own numbers must be rejected.
    rho, revision_pp = 0.999995, 0.03
    assert rho > cfg.gate_shrinkage_rho_max
    assert revision_pp < cfg.gate_shrinkage_revision_min_pp


def test_the_gate_grades_center_shift_not_the_raw_intercept():
    """The y-intercept cannot separate an offset from shrinkage.

    Shrinking a cloud toward its centre ``c`` gives ``eu = c + s*(iu - c)``, so
    ``intercept = (1 - s) * c`` identically. Measured across a multiplier sweep
    on the production fit, ``intercept / (1 - slope)`` came out 0.2018, 0.2016,
    0.2010, 0.2006, 0.2006, 0.2007 — the response centre, every time.

    Pairing ``|intercept| <= 0.02`` with ``rho <= 0.995`` therefore had an EMPTY
    feasible set on a response centred at +20%: the first needs slope >= 0.90,
    the second needs slope <= ~0.88. This asserts the arithmetic that makes the
    old threshold unusable, so nobody reinstates it.
    """
    import numpy as np

    import pymc_kalman_filter_pt_v2 as kf2

    cfg = kf2.KalmanRunConfigV2()
    assert not hasattr(cfg, "gate_shrinkage_intercept_max")
    assert cfg.gate_shrinkage_center_shift_max > 0

    rng = np.random.default_rng(0)
    iu = 0.20 + rng.normal(0.0, 0.30, 20_000)
    # The SAMPLE centre: shrinking toward the population value would move the
    # sample mean by (1-slope)*(population - sample), which is sampling error,
    # not an offset, and would make the zero-shift assertion below a statement
    # about the RNG rather than about shrinkage.
    centre = float(iu.mean())
    for slope in (0.70, 0.80, 0.90, 0.95):
        eu = centre + slope * (iu - centre)          # pure shrinkage, no offset
        fit_slope, intercept = np.polyfit(iu, eu, 1)
        assert fit_slope == pytest.approx(slope, abs=1e-9)
        assert intercept == pytest.approx((1 - slope) * centre, abs=1e-9)
        # The intercept grows without bound as shrinkage strengthens...
        assert intercept > 0
        # ...while the quantity the gate actually grades stays at zero, because
        # pure shrinkage moves no name's centre.
        assert abs(eu.mean() - iu.mean()) < 1e-9
        assert abs(eu.mean() - iu.mean()) <= cfg.gate_shrinkage_center_shift_max

    # And a genuine universe-wide lift IS caught.
    lifted = iu + 0.05
    assert abs(lifted.mean() - iu.mean()) > cfg.gate_shrinkage_center_shift_max


@pytest.mark.parametrize(
    "field, bad",
    [
        ("forecast_error_multiplier", -1.0),
        ("forecast_error_n_exponent", -0.5),
        ("tail_risk_vol_floor_k", -0.1),
        ("gate_shrinkage_rho_max", 1.5),
    ],
)
def test_run_config_rejects_invalid_decision_knobs(field, bad):
    import dataclasses

    import pymc_kalman_filter_pt_v2 as kf2

    with pytest.raises(ValueError):
        dataclasses.replace(kf2.KalmanRunConfigV2(), **{field: bad})


def test_shrinkage_recovers_the_latent_better_than_the_raw_consensus():
    """Synthetic recovery: the whole claim, on data with a known truth.

    Deliberately NOT built by extending ``_simulate_panel``. That simulator
    generates the response from the fitted model, which has no forecast-error
    component at all — adding one would change the generative process the
    existing variance-split recovery selftest validates, and risk trading a
    working acceptance check for a new one. The claim here is about the decision
    layer, so it is tested on the decision layer directly.

    The setup mirrors the real situation exactly:

        theta_i     ~ N(mu_i, struct_sd)        the fair uplift (unobserved)
        consensus_i  = theta_i + fe_i           analysts, off by forecast error
        fe_i        ~ N(0, kappa * cv_i / sqrt(n_i))

    The fitted model reproduces ``consensus_i`` — that is what run 49e84d7e9d59
    did, at Spearman 0.999995. So ``state_now_mean = consensus`` is the honest
    stand-in, and the question is whether shrinking it toward ``mu`` gets closer
    to ``theta``. It must, and by the textbook amount: the posterior mean of a
    normal-normal update is the minimum-MSE estimator.
    """
    import types

    import xarray as xr

    from probabilistic_ml_model.pymc_models.KalmanFilterModel_v2 import (
        apply_forecast_error_shrinkage,
    )

    n, kappa = 4000, 2.0
    panel = _fe_panel(n=n, seed=11)
    rng = np.random.default_rng(11)

    struct_sd_std = 0.5559                     # sigma_state of the reference fit
    sd = panel.response_std
    mu = rng.normal(0.0, 0.30, n)              # the pooled drift + hierarchy view
    theta = mu + rng.normal(0.0, struct_sd_std, n)

    cv, n_an = panel.dispersion_cv, panel.frame["n_analysts"].to_numpy()
    fe_sd = kappa * (cv / np.sqrt(n_an)) / sd  # standardised, as the helper builds it
    consensus = theta + rng.normal(0.0, fe_sd)

    c, d = 2, 40
    def _da(v):
        return xr.DataArray(
            np.broadcast_to(v, (c, d, n)).copy(),
            dims=("chain", "draw", "isin"), coords={"isin": panel.isins},
        )

    idata = types.SimpleNamespace(
        posterior=xr.Dataset(
            {
                "mu_scaled": _da(mu),
                "state_now_mean": _da(consensus),
                "state_now_sd": _da(np.zeros(n)),   # isolate the shrinkage term
                "sigma_isin": _da(np.full(n, struct_sd_std / np.sqrt(0.9984))),
                "variance_weights": xr.DataArray(
                    np.tile([0.0044, 0.9940, 0.0016], (c, d, 1)),
                    dims=("chain", "draw", "variance_component"),
                ),
            }
        )
    )
    _, gain = apply_forecast_error_shrinkage(
        idata, panel, multiplier=kappa, random_seed=11
    )
    shrunk = mu + gain * (consensus - mu)

    rmse_raw = float(np.sqrt(np.mean((consensus - theta) ** 2)))
    rmse_shrunk = float(np.sqrt(np.mean((shrunk - theta) ** 2)))
    assert rmse_shrunk < rmse_raw, (rmse_shrunk, rmse_raw)

    # Not just better — near-optimal. The Bayes RMSE for this update is
    # sqrt(mean(g * fe_var)), and the estimator should be within a few percent
    # of it. A helper that shrank in the right direction by the wrong amount
    # would pass the inequality above and fail here.
    fe_var = fe_sd ** 2
    bayes = float(np.sqrt(np.mean(gain * fe_var)))
    assert rmse_shrunk == pytest.approx(bayes, rel=0.10)

    # And it must beat the other degenerate choice: ignoring the name entirely.
    rmse_pooled = float(np.sqrt(np.mean((mu - theta) ** 2)))
    assert rmse_shrunk < rmse_pooled
