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


# ---------------------------------------------------------------------------
# The handoff as the BOUNDARY, not just the forecast's input
#
# Once the decision layer moves out of the fit script, the screen has to be
# rebuildable from this file alone. That needs three more posterior arrays and
# the variance shares -- paired with `mu_std` DRAW FOR DRAW, because the
# shrinkage and the risk-adjusted probability are both per-draw operations and
# independently thinned arrays would pair unrelated draws with every shape still
# matching.
# ---------------------------------------------------------------------------

from probabilistic_ml_model.pymc_models.KalmanForecast import (  # noqa: E402
    HANDOFF_PANEL_VECTORS,
    ForecastHandoff,
)


@pytest.fixture(scope="module")
def screen_fit():
    """A posterior carrying everything the screen reads, plus a panel frame."""
    rng = np.random.default_rng(4)
    isins = np.array([f"Y{i:03d}" for i in range(N_ISIN)])
    shape = (N_CHAIN, N_DRAW, N_ISIN)
    post = xr.Dataset(
        {
            "sigma_isin": (("chain", "draw", "isin"), np.abs(rng.normal(0.3, .05, shape))),
            "state_now_mean": (("chain", "draw", "isin"), rng.normal(0.2, 0.4, shape)),
            "state_now_sd": (("chain", "draw", "isin"), np.abs(rng.normal(.2, .04, shape))),
            "mu_scaled": (("chain", "draw", "isin"), rng.normal(0.1, 0.3, shape)),
            "risk_adj_return": (("chain", "draw", "isin"), rng.normal(0.05, 0.3, shape)),
            "variance_weights": (("chain", "draw", "vw"),
                                 rng.dirichlet([2, 2, 2], (N_CHAIN, N_DRAW))),
            "nu": (("chain", "draw"), rng.uniform(8, 14, (N_CHAIN, N_DRAW))),
        },
        coords={"chain": np.arange(N_CHAIN), "draw": np.arange(N_DRAW), "isin": isins},
    )
    idata = xr.DataTree()
    idata["posterior"] = xr.DataTree(post)

    panel = _Panel(isins, {"sector": rng.integers(0, 4, N_ISIN)},
                   {"sector": np.array([f"S{i}" for i in range(4)])})
    panel.dispersion_cv = np.abs(rng.normal(0.2, 0.05, N_ISIN))
    panel.frame = pd.DataFrame({
        "isin": isins,
        "n_analysts": rng.integers(1, 25, N_ISIN).astype(float),
        "last_price": rng.uniform(5, 200, N_ISIN),
        "observed_pt": rng.uniform(5, 300, N_ISIN),
        "feat_analyst_rating": rng.uniform(1, 5, N_ISIN),
        "feat_mcap_global_r": rng.uniform(0, 1, N_ISIN),
        "feat_mcap_country_r": rng.uniform(0, 1, N_ISIN),
    })
    latent = rng.normal(0.2, 0.5, shape)
    return idata, panel, latent


def test_handoff_carries_the_screens_posterior_arrays(tmp_path_factory, screen_fit):
    idata, panel, latent = screen_fit
    path = tmp_path_factory.mktemp("h") / "screen.nc"
    save_forecast_handoff(path, idata, panel, latent=latent, n_samples=None)
    h = load_forecast_handoff(path)

    assert h.screen_ready
    for name in ("latent_mean", "latent_sd", "fitted_mean", "rar"):
        arr = getattr(h, name)
        assert arr is not None and arr.shape == h.mu_std.shape
    assert h.variance_weights is not None
    assert h.variance_weights.shape == (h.n_samples, 3)


def test_the_new_arrays_are_thinned_on_the_same_index_as_mu_std(
    tmp_path_factory, screen_fit
):
    """Draw i of one must be draw i of the others, not an independent sample."""
    idata, panel, latent = screen_fit
    path = tmp_path_factory.mktemp("h") / "thin.nc"
    save_forecast_handoff(path, idata, panel, latent=latent, n_samples=25)
    h = load_forecast_handoff(path)

    assert h.n_samples == 25
    for name in ("latent_mean", "latent_sd", "fitted_mean", "rar"):
        assert getattr(h, name).shape == (N_ISIN, 25)
    assert h.variance_weights.shape[0] == 25

    # The pairing itself: recover which source draws survived by matching one
    # array, then check a DIFFERENT array agrees on the same positions.
    flat = lambda v: np.asarray(  # noqa: E731
        idata["posterior"][v]
    ).reshape(-1, N_ISIN).T
    src_mu, src_rar = flat("state_now_sd"), flat("risk_adj_return")
    keep = [
        int(np.argmin(np.abs(src_mu[0] - h.latent_sd[0, j])))
        for j in range(h.n_samples)
    ]
    assert np.allclose(src_rar[0][keep], h.rar[0], atol=1e-5)
    assert sorted(keep) == keep, "thinning must preserve draw order"


def test_panel_vectors_round_trip(tmp_path_factory, screen_fit):
    idata, panel, latent = screen_fit
    path = tmp_path_factory.mktemp("h") / "vec.nc"
    save_forecast_handoff(path, idata, panel, latent=latent, n_samples=None)
    h = load_forecast_handoff(path)

    assert set(h.panel_vectors) == set(HANDOFF_PANEL_VECTORS)
    assert np.allclose(h.panel_vectors["dispersion_cv"], panel.dispersion_cv)
    assert np.allclose(h.panel_vectors["last_price"], panel.frame["last_price"])


def test_a_pre_2026_08_31_handoff_loads_and_says_it_cannot_screen(
    tmp_path_factory, fit
):
    """The old four quantities alone: forecastable, not screenable."""
    idata, panel, latent = fit          # no state_now_sd / mu_scaled / rar
    path = tmp_path_factory.mktemp("h") / "old.nc"
    save_forecast_handoff(path, idata, panel, latent=latent, n_samples=None)
    h = load_forecast_handoff(path)

    assert not h.screen_ready
    assert h.latent_sd is None and h.rar is None
    # And it must still forecast, which is the whole point of degrading rather
    # than raising.
    assert forecast_from_posterior(h, config=ForecastConfig(n_scenarios=50)).n_isin


def test_mismatched_extra_arrays_are_refused_rather_than_broadcast():
    rng = np.random.default_rng(1)
    mu = rng.normal(size=(6, 20))
    with pytest.raises(ValueError, match="must match mu_std"):
        ForecastHandoff(
            isins=np.array([f"Z{i}" for i in range(6)]),
            mu_std=mu,
            sigma_std=np.abs(rng.normal(size=(6, 20))),
            nu=np.array([9.0]),
            rar=rng.normal(size=(6, 19)),      # one draw short
        )


def test_a_panel_vector_of_the_wrong_length_is_refused():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="entries for"):
        ForecastHandoff(
            isins=np.array([f"Z{i}" for i in range(6)]),
            mu_std=rng.normal(size=(6, 20)),
            sigma_std=np.abs(rng.normal(size=(6, 20))),
            nu=np.array([9.0]),
            panel_vectors={"n_analysts": np.arange(5.0)},
        )


# ---------------------------------------------------------------------------
# Identity encoding
#
# NetCDF has no null for a fixed-width string and no nullable integer, so every
# identity column has to be flattened to something. Three of the four kinds
# round-trip WRONG under the obvious `astype(str)`, and the text case is the one
# that matters most: `""` coming back as a value rather than as missing is the
# "Unknown" bucket again under a quieter name.
# ---------------------------------------------------------------------------


def _identity_frame(isins):
    """One column of each kind, each with a hole in it."""
    n = len(isins)
    return pd.DataFrame({
        "isin": isins,
        "sector": ["Health Care"] * (n - 3) + [None] * 3,
        "fy_end_date": pd.to_datetime(
            ["2026-03-31"] * (n - 2) + [None] * 2
        ),
        "market_cap_country_r": pd.array(
            list(range(1, n - 1)) + [None] * 2, dtype="Int64"
        ),
        "market_cap": np.concatenate([np.arange(n - 1, dtype=float), [np.nan]]),
    })


def test_identity_nulls_come_back_as_nulls_not_as_a_group(
    tmp_path_factory, screen_fit
):
    idata, panel, latent = screen_fit
    ident = _identity_frame(panel.isins)
    path = tmp_path_factory.mktemp("h") / "ident.nc"
    save_forecast_handoff(
        path, idata, panel, latent=latent, n_samples=None, identity=ident
    )
    out = load_forecast_handoff(path).identity

    # The whole point: an empty string is NOT a sector.
    assert out["sector"].isna().sum() == 3
    assert "" not in set(out["sector"].dropna())
    assert set(out["sector"].dropna()) == {"Health Care"}


def test_identity_dates_survive_as_dates(tmp_path_factory, screen_fit):
    idata, panel, latent = screen_fit
    ident = _identity_frame(panel.isins)
    path = tmp_path_factory.mktemp("h") / "date.nc"
    save_forecast_handoff(
        path, idata, panel, latent=latent, n_samples=None, identity=ident
    )
    out = load_forecast_handoff(path).identity

    assert pd.api.types.is_datetime64_any_dtype(out["fy_end_date"])
    assert out["fy_end_date"].isna().sum() == 2
    assert out["fy_end_date"].dropna().iloc[0] == pd.Timestamp("2026-03-31")


def test_identity_nullable_ints_survive_as_nullable_ints(
    tmp_path_factory, screen_fit
):
    """`Int64.to_numpy()` yields an OBJECT array holding `pd.NA`.

    Stringified, a rank of 3 becomes "3" and a missing one the literal "<NA>" --
    a text column where a ranking should be.
    """
    idata, panel, latent = screen_fit
    ident = _identity_frame(panel.isins)
    path = tmp_path_factory.mktemp("h") / "int.nc"
    save_forecast_handoff(
        path, idata, panel, latent=latent, n_samples=None, identity=ident
    )
    out = load_forecast_handoff(path).identity

    assert str(out["market_cap_country_r"].dtype) == "Int64"
    assert out["market_cap_country_r"].isna().sum() == 2
    assert out["market_cap_country_r"].dropna().iloc[0] == 1
    assert "<NA>" not in {str(v) for v in out["market_cap_country_r"].dropna()}


def test_identity_floats_keep_their_nan(tmp_path_factory, screen_fit):
    idata, panel, latent = screen_fit
    ident = _identity_frame(panel.isins)
    path = tmp_path_factory.mktemp("h") / "float.nc"
    save_forecast_handoff(
        path, idata, panel, latent=latent, n_samples=None, identity=ident
    )
    out = load_forecast_handoff(path).identity

    assert pd.api.types.is_float_dtype(out["market_cap"])
    assert out["market_cap"].isna().sum() == 1
