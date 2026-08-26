"""The handoff must replay the live path, not merely resemble it.

The check that earns its keep is the round-trip: a replay whose ``mu_log`` drifts is a
replay describing a differently-scaled model while every gate still passes, which is the
exact failure ``run_forecast_layer`` already documents once for the live path. The
tolerance is the float32 storage width and nothing looser.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from probabilistic_ml_model.pymc_models.KalmanForecast import (  # noqa: E402
    ForecastConfig,
    compare_forecast_engines,
    forecast_from_posterior,
    load_forecast_handoff,
    prepare_forecast_inputs,
    save_forecast_handoff,
    summarize_forecast,
    sweep_factor_share,
)

N_ISIN, N_CHAIN, N_DRAW = 40, 2, 60


class _Panel:
    """The three things ``prepare_forecast_inputs`` reads off a KalmanPanelV2."""

    def __init__(self, isins, coord_idx, coord_uniques):
        self.isins = isins
        self.response_mean = 0.12
        self.response_std = 0.44
        self.coord_idx = coord_idx
        self.coord_uniques = coord_uniques


@pytest.fixture(scope="module")
def fit():
    """A synthetic posterior, panel and shrunk latent from one seeded draw."""
    rng = np.random.default_rng(0)
    isins = np.array([f"X{i:03d}" for i in range(N_ISIN)])
    post = xr.Dataset(
        {
            "sigma_isin": (("chain", "draw", "isin"),
                           np.abs(rng.normal(0.3, 0.05, (N_CHAIN, N_DRAW, N_ISIN)))),
            "nu": (("chain", "draw"), rng.uniform(8, 14, (N_CHAIN, N_DRAW))),
            "ou_length_scale_days": (("chain", "draw"),
                                     rng.uniform(70, 90, (N_CHAIN, N_DRAW))),
        },
        coords={"chain": np.arange(N_CHAIN), "draw": np.arange(N_DRAW), "isin": isins},
    )
    idata = xr.DataTree()
    idata["posterior"] = xr.DataTree(post)
    panel = _Panel(
        isins,
        {"trading_region": rng.integers(0, 4, N_ISIN),
         "sector": rng.integers(0, 6, N_ISIN)},
        {"trading_region": np.array(list("ABCD")),
         "sector": np.array([f"S{i}" for i in range(6)])},
    )
    latent = rng.normal(0.2, 0.5, (N_CHAIN, N_DRAW, N_ISIN))
    return idata, panel, latent


@pytest.fixture(scope="module")
def cfg():
    return ForecastConfig(n_scenarios=300, random_seed=7)


def test_replay_reproduces_the_live_inputs(tmp_path_factory, fit, cfg):
    """De-standardisation must survive the round trip to float32 and no further."""
    idata, panel, latent = fit
    live = prepare_forecast_inputs(idata, panel, config=cfg, latent=latent)

    path = tmp_path_factory.mktemp("handoff") / "h.nc"
    save_forecast_handoff(path, idata, panel, latent=latent, n_samples=None)
    replay = prepare_forecast_inputs(load_forecast_handoff(path), config=cfg)

    assert np.allclose(live.mu_log, replay.mu_log, atol=1e-5)
    assert np.allclose(live.sigma_log, replay.sigma_log, atol=1e-6)
    assert live.ou_length_scale_days == replay.ou_length_scale_days
    assert list(live.group_index) == list(replay.group_index)
    for level, codes in live.group_index.items():
        assert np.array_equal(codes, replay.group_index[level])


def test_replay_reproduces_the_draws(tmp_path_factory, fit, cfg):
    """Same posterior, same seed, same scenarios: the replay IS the run."""
    idata, panel, latent = fit
    live = forecast_from_posterior(idata, panel, config=cfg, latent=latent)

    path = tmp_path_factory.mktemp("handoff") / "h.nc"
    save_forecast_handoff(path, idata, panel, latent=latent, n_samples=None)
    replay = forecast_from_posterior(load_forecast_handoff(path), config=cfg)

    assert np.allclose(live.terminal, replay.terminal, atol=1e-4)
    assert np.array_equal(live.isins, replay.isins)


def test_thinning_is_seeded_and_recorded(tmp_path_factory, fit):
    """A thinned handoff must be reproducible and must say how far it was thinned."""
    idata, panel, latent = fit
    path = tmp_path_factory.mktemp("handoff") / "thin.nc"

    save_forecast_handoff(path, idata, panel, latent=latent, n_samples=30)
    first = load_forecast_handoff(path)
    save_forecast_handoff(path, idata, panel, latent=latent, n_samples=30)
    second = load_forecast_handoff(path)

    assert first.n_samples == 30
    assert np.array_equal(first.mu_std, second.mu_std)
    assert first.attrs["thin_factor"] == pytest.approx(
        (N_CHAIN * N_DRAW) / 30
    )
    assert first.attrs["n_samples_original"] == N_CHAIN * N_DRAW


def test_provenance_round_trips(tmp_path_factory, fit):
    """A handoff whose revision cannot be attributed cannot be contrasted later."""
    idata, panel, latent = fit
    path = tmp_path_factory.mktemp("handoff") / "p.nc"
    save_forecast_handoff(
        path, idata, panel, latent=latent, n_samples=None,
        provenance={"run_id": "deadbeef", "source_sha": "abc1234",
                    "source_dirty": True},
    )
    handoff = load_forecast_handoff(path)
    assert handoff.attrs["run_id"] == "deadbeef"
    assert handoff.attrs["source_sha"] == "abc1234"
    # A bool would round-trip through NetCDF as 0/1 and stop reading as a flag.
    assert bool(handoff.attrs["source_dirty"]) is True
    assert "deadbeef" in handoff.describe()


def test_missing_handoff_names_both_ways_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_forecast_handoff(tmp_path / "absent.nc")
    assert "--fit" in str(excinfo.value)


def test_factor_share_preserves_per_name_variance(tmp_path_factory, fit, cfg):
    """The property that makes factor_share harmless to the screen and decisive
    for the book: the split is variance-preserving, so per-name marginals do not
    move and only the JOINT distribution does."""
    idata, panel, latent = fit
    path = tmp_path_factory.mktemp("handoff") / "sweep.nc"
    save_forecast_handoff(path, idata, panel, latent=latent, n_samples=None)
    handoff = load_forecast_handoff(path)

    sweep = sweep_factor_share(handoff, [0.0, 0.2, 0.35, 0.6], config=cfg,
                               baseline_share=0.35, k_book=10)

    assert sweep["er_sd_max_abs_diff"].max() < 5e-2      # Monte-Carlo noise only
    assert sweep["book_sd_ratio"].is_monotonic_increasing
    # Independent shocks give diversification away for free, so the book is tighter.
    assert float(sweep.loc[sweep.factor_share == 0.0, "book_sd_ratio"].iloc[0]) < 1.0
    assert float(sweep.loc[sweep.factor_share == 0.35, "book_sd_ratio"].iloc[0]) == \
        pytest.approx(1.0)


def test_engine_contrast_joins_by_isin(fit, cfg):
    """A positional merge would contrast each name against a different one while
    every row count still matched."""
    idata, panel, latent = fit
    draws = forecast_from_posterior(idata, panel, config=cfg, latent=latent)
    mc = summarize_forecast(draws).sample(frac=0.6, random_state=1)

    merged = compare_forecast_engines(draws, mc)

    assert len(merged) == len(mc)
    assert set(merged["isin"]) == set(mc["isin"])
    # Contrasted against itself, the ratio is exactly one for every matched name.
    assert np.allclose(merged["sd_ratio"].to_numpy(), 1.0)


def test_inference_data_without_a_panel_is_refused(fit, cfg):
    idata, _panel, latent = fit
    with pytest.raises(TypeError, match="ForecastHandoff"):
        prepare_forecast_inputs(idata, config=cfg, latent=latent)
