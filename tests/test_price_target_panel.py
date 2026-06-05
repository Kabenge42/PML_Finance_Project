"""Tests for the fused price-target panel helper and PyMC model builder.

Covers:
  * :func:`prepare_price_target_panel_inputs` — PyMC-free assembly of the 3-D
    ``(isin, time, y_series)`` response tensor, the ``(isin, time)`` fiscal-anchor
    time matrix, ``sqrt(n_analysts)``, the standardised predictor matrix and the
    per-coord unique/index arrays (column-aligned with the MV DDL).
  * :func:`build_fused_price_target_model` — the single fused ``pm.Model``
    builder (skipped when PyMC is unavailable). Sampling is *not* exercised;
    we only assert the graph builds with the expected named variables / dims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probabilistic_ml_model.pymc_models._price_target_mc import (
    PANEL_DAY_COUNT_COLS,
    PANEL_FISCAL_ANCHOR_COLS,
    PANEL_RESPONSE_COLS,
    PriceTargetPanelInputs,
    prepare_price_target_panel_inputs,
)


def _sample_panel_df() -> pd.DataFrame:
    n = 6
    isins = [f"ISIN{i}" for i in range(n)]
    base = pd.to_datetime("2024-01-31")
    df = pd.DataFrame(
        {
            "isin": isins,
            "sector": ["Tech", "Tech", "Energy", "Energy", "Health", "Health"],
            "region": ["NA", "NA", "EU", "EU", "APAC", "APAC"],
            "industry": ["SW", "HW", "Oil", "Gas", "Bio", "Pharma"],
            "size_class": ["L", "L", "M", "M", "S", "S"],
            "style_class": ["G", "V", "G", "V", "G", "V"],
            "last_price": [100.0, 50.0, 80.0, 25.0, 200.0, 10.0],
            "price_target": [120.0, 45.0, 90.0, 30.0, 220.0, 12.0],
            "price_target_median": [118.0, 46.0, 88.0, 31.0, 215.0, 11.5],
            "price_target_high": [130.0, 55.0, 100.0, 35.0, 240.0, 14.0],
            "price_target_low": [110.0, 40.0, 80.0, 26.0, 200.0, 10.0],
            "price_target_stddev": [5.0, 2.5, 4.0, 1.5, 8.0, 0.5],
            "n_analysts": [10, 4, 6, 3, 8, 1],
            "num_buys_ratings": [6, 1, 3, 2, 5, 1],
            "observed_target_pct": [0.20, -0.10, 0.125, 0.20, 0.10, 0.20],
            "observed_target_pct_med": [0.18, -0.08, 0.10, 0.24, 0.075, 0.15],
            "feat_implied_upside": [0.2, -0.1, 0.125, 0.2, 0.1, 0.2],
            "feat_net_buy_sentiment": [4, 0, 1, 1, 3, 0],
        }
    )
    # Fiscal-calendar anchors (ordered) + numeric day-count horizons.
    for k, col in enumerate(PANEL_FISCAL_ANCHOR_COLS):
        df[col] = base + pd.to_timedelta(30 * k + np.arange(n), unit="D")
    for col in PANEL_DAY_COUNT_COLS:
        df[col] = np.linspace(5, 90, n)
    return df


class TestPreparePriceTargetPanelInputs:
    def test_returns_panel_dataclass(self):
        panel = prepare_price_target_panel_inputs(_sample_panel_df())
        assert isinstance(panel, PriceTargetPanelInputs)

    def test_tensor_shapes(self):
        panel = prepare_price_target_panel_inputs(_sample_panel_df())
        n_isin = len(panel.isins)
        T = len(PANEL_FISCAL_ANCHOR_COLS)
        D = len(panel.response_names)
        assert panel.Y.shape == (n_isin, T, D)
        assert panel.t_scaled.shape == (n_isin, T)
        assert panel.X_std.shape == (n_isin, len(panel.predictor_names))
        assert panel.n_analysts.shape == (n_isin,)
        assert panel.sqrt_n_analysts.shape == (n_isin,)

    def test_response_series_uses_mv_columns(self):
        panel = prepare_price_target_panel_inputs(_sample_panel_df())
        # All response names must be a subset of the canonical MV list.
        assert set(panel.response_names).issubset(set(PANEL_RESPONSE_COLS))
        assert "observed_target_pct" in panel.response_names

    def test_sqrt_n_analysts_precomputed(self):
        panel = prepare_price_target_panel_inputs(_sample_panel_df())
        np.testing.assert_allclose(
            panel.sqrt_n_analysts, np.sqrt(panel.n_analysts)
        )

    def test_day_count_columns_have_no_feat_prefix(self):
        # Regression guard for the notebook's wrong `feat_days_*` names.
        assert all(not c.startswith("feat_") for c in PANEL_DAY_COUNT_COLS)
        assert "days_to_next_earnings" in PANEL_DAY_COUNT_COLS

    def test_categorical_coords_built(self):
        panel = prepare_price_target_panel_inputs(_sample_panel_df())
        for col in ("sector", "region", "industry", "size_class", "style_class"):
            assert col in panel.coord_uniques
            assert col in panel.coord_idx
            assert panel.coord_idx[col].shape == (len(panel.isins),)
            # Index values address valid unique labels.
            assert panel.coord_idx[col].max() < len(panel.coord_uniques[col])

    def test_time_matrix_is_finite_and_standardised(self):
        panel = prepare_price_target_panel_inputs(_sample_panel_df())
        assert np.isfinite(panel.t_scaled).all()

    def test_response_tensor_broadcasts_across_time(self):
        panel = prepare_price_target_panel_inputs(_sample_panel_df())
        # Each y_series is broadcast across the T anchors → constant along axis 1.
        for d in range(panel.Y.shape[2]):
            slab = panel.Y[:, :, d]
            np.testing.assert_allclose(slab, np.repeat(slab[:, [0]], slab.shape[1], axis=1))

    def test_explicit_predictor_cols(self):
        df = _sample_panel_df()
        panel = prepare_price_target_panel_inputs(
            df, predictor_cols=["feat_implied_upside"]
        )
        assert panel.predictor_names == ["feat_implied_upside"]
        assert panel.X_std.shape == (len(panel.isins), 1)

    def test_missing_response_raises(self):
        df = _sample_panel_df().drop(columns=["observed_target_pct"])
        with pytest.raises(KeyError, match="observed_target_pct"):
            prepare_price_target_panel_inputs(df)

    def test_empty_coord_guard_skips_absent_columns(self):
        df = _sample_panel_df().drop(columns=["industry"])
        panel = prepare_price_target_panel_inputs(df)
        assert "industry" not in panel.coord_idx
        assert "sector" in panel.coord_idx


# ---------------------------------------------------------------------------
# Fused PyMC model builder (skipped when PyMC is not installed).
# ---------------------------------------------------------------------------
pm = pytest.importorskip("pymc")
from probabilistic_ml_model.pymc_models.PriceTargetModel import (  # noqa: E402
    build_fused_price_target_model,
)


class TestBuildFusedPriceTargetModel:
    def test_builds_and_exposes_named_variables(self):
        panel = prepare_price_target_panel_inputs(_sample_panel_df())
        model = build_fused_price_target_model(panel)

        names = set(model.named_vars)
        # Model A latents fused into Model B spine.
        for v in (
            "achieve_prob",
            "expected_return",
            "risk_adj_return",
            "mu_isin",
            "sigma_isin",
            "nu",
            "alpha",
            "beta_t",
            "sigma_alpha_innov",
            "sigma_beta_innov",
            "target_pct_obs",
        ):
            assert v in names, f"missing model variable {v!r}"

    def test_per_isin_dims_preserved_for_mc_helper(self):
        panel = prepare_price_target_panel_inputs(_sample_panel_df())
        model = build_fused_price_target_model(panel)
        n_isin = len(panel.isins)
        # risk_adj_return / sigma_isin remain 1-D over isin so the MC helper
        # (per-isin mu/sigma + scalar nu) consumes them unchanged.
        assert tuple(model.named_vars["risk_adj_return"].shape.eval()) == (n_isin,)
        assert tuple(model.named_vars["sigma_isin"].shape.eval()) == (n_isin,)

    def test_pm_data_names_mirror_mv_aliases(self):
        panel = prepare_price_target_panel_inputs(_sample_panel_df())
        model = build_fused_price_target_model(panel)
        names = set(model.named_vars)
        for c in ("Y_obs", "t_scaled", "pt_features", "n_analysts",
                  "sqrt_n_analysts", "feat_analyst_conviction",
                  "feat_target_dispersion_cv", "sector_idx"):
            assert c in names, f"missing pm.Data container {c!r}"