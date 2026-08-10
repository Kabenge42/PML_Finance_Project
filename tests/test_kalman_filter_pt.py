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
            "feat_mcap_vs_3yavg": rng.normal(1.0, 0.2, n),
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