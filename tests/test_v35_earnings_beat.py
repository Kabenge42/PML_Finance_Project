"""Tests for v3.5 EarningsBeatProbabilityModel improvements."""
import numpy as np
import pandas as pd
import pytest

from probabilistic_ml_model.statistical_functions.probability_models import (
    EarningsBeatProbabilityModel,
    PriorParameters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return EarningsBeatProbabilityModel(
        use_quality_adjustment=True,
        use_momentum_prior=True,
        momentum_prior_strength=0.3,
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "name": ["AA", "BB", "CC"],
        "sector": ["Energy", "Energy", "Information Technology"],
        "historical_beat_rate": [0.7, 0.5, 0.9],
        "dynamic_total_reports": [10, 8, 12],
        "eps_revision_momentum": [0.3, -0.2, 0.8],
        "composite_eps_trajectory_score": [0.5, -0.1, 0.7],
        "continuation_probability": [0.6, 0.4, 0.8],
        "eps_growth_accel": [0.2, -0.3, 0.5],
        "map_estimate": [0.65, 0.45, 0.85],
        "model_confidence": [0.4, 0.3, 0.5],
        "accounting_quality_score": [0.8, 0.3, 0.9],
        "gaap_adj_eps_gap_pct": [0.1, 0.6, 0.05],
        "eps_adjustment_pct": [0.1, 0.5, 0.05],
        "distress_risk_score": [0.1, 0.7, 0.05],
    })


# ---------------------------------------------------------------------------
# 1. Momentum prior adjustment
# ---------------------------------------------------------------------------

class TestMomentumPriorAdjustment:
    def test_no_signals_returns_original(self, model):
        prior = PriorParameters(2.0, 2.5)
        result = model._apply_momentum_prior_adjustment(prior)
        assert result.alpha == prior.alpha
        assert result.beta == prior.beta

    def test_positive_momentum_shifts_up(self, model):
        prior = PriorParameters(2.0, 2.5)
        result = model._apply_momentum_prior_adjustment(
            prior, eps_revision_momentum=1.0,
        )
        assert result.alpha / (result.alpha + result.beta) > prior.alpha / (prior.alpha + prior.beta)
        # Concentration preserved
        assert abs((result.alpha + result.beta) - (prior.alpha + prior.beta)) < 1e-10

    def test_negative_momentum_shifts_down(self, model):
        prior = PriorParameters(2.0, 2.5)
        result = model._apply_momentum_prior_adjustment(
            prior, eps_revision_momentum=-1.0,
        )
        assert result.alpha / (result.alpha + result.beta) < prior.alpha / (prior.alpha + prior.beta)

    def test_disabled_returns_original(self):
        m = EarningsBeatProbabilityModel(use_momentum_prior=False)
        prior = PriorParameters(2.0, 2.5)
        result = m._apply_momentum_prior_adjustment(prior, eps_revision_momentum=1.0)
        assert result.alpha == prior.alpha

    def test_shift_clamped(self, model):
        prior = PriorParameters(2.0, 2.5)
        result = model._apply_momentum_prior_adjustment(
            prior,
            eps_revision_momentum=1.0,
            composite_eps_trajectory_score=1.0,
            continuation_probability=1.0,
            eps_growth_accel=1.0,
        )
        new_mean = result.alpha / (result.alpha + result.beta)
        base_mean = prior.alpha / (prior.alpha + prior.beta)
        assert abs(new_mean - base_mean) <= model.momentum_prior_strength + 1e-10


# ---------------------------------------------------------------------------
# 2. Quality discount
# ---------------------------------------------------------------------------

class TestQualityDiscount:
    def test_no_signals_no_discount(self, model):
        assert model._apply_quality_discount(0.8) == 0.8

    def test_low_quality_discounts(self, model):
        result = model._apply_quality_discount(0.8, accounting_quality_score=0.0)
        assert result < 0.8
        assert result == pytest.approx(0.8 * 0.70, rel=1e-6)

    def test_high_distress_discounts(self, model):
        result = model._apply_quality_discount(0.8, distress_risk_score=1.0)
        assert result == pytest.approx(0.8 * 0.80, rel=1e-6)

    def test_disabled_no_discount(self):
        m = EarningsBeatProbabilityModel(use_quality_adjustment=False)
        assert m._apply_quality_discount(0.8, accounting_quality_score=0.0) == 0.8

    def test_combined_discounts(self, model):
        result = model._apply_quality_discount(
            1.0,
            accounting_quality_score=0.0,
            gaap_adj_eps_gap_pct=1.0,
            eps_adjustment_pct=1.0,
            distress_risk_score=1.0,
        )
        assert result < 0.5  # heavy combined discount


# ---------------------------------------------------------------------------
# 3. compute_beat_probability with new params
# ---------------------------------------------------------------------------

class TestComputeBeatProbability:
    def test_basic_call(self, model):
        r = model.compute_beat_probability(7, 10, sector="Energy")
        assert 0 < r["posterior_mean"] < 1
        assert 0 < r["confidence_score"] <= 1

    def test_momentum_shifts_posterior(self, model):
        base = model.compute_beat_probability(5, 10, sector="Energy")
        up = model.compute_beat_probability(5, 10, sector="Energy", eps_revision_momentum=1.0)
        assert up["posterior_mean"] > base["posterior_mean"]

    def test_streak_blending(self, model):
        no_blend = model.compute_beat_probability(5, 10)
        blended = model.compute_beat_probability(5, 10, streak_map_estimate=0.9, streak_model_confidence=0.5)
        assert blended["posterior_mean"] > no_blend["posterior_mean"]

    def test_quality_discount_lowers_confidence(self, model):
        clean = model.compute_beat_probability(7, 10)
        dirty = model.compute_beat_probability(7, 10, accounting_quality_score=0.0, distress_risk_score=1.0)
        assert dirty["confidence_score"] < clean["confidence_score"]


# ---------------------------------------------------------------------------
# 4. analyze_dataframe with new defaults
# ---------------------------------------------------------------------------

class TestAnalyzeDataframe:
    def test_basic_run(self, model, sample_df):
        result = model.analyze_dataframe(sample_df)
        assert len(result) == 3
        assert "posterior_beat_prob" in result.columns
        assert "confidence_score" in result.columns

    def test_beat_rate_converted_to_counts(self, model, sample_df):
        result = model.analyze_dataframe(sample_df)
        # Stock A: 0.7 * 10 = 7 beats
        assert result.iloc[0]["historical_beats"] == 7

    def test_sector_priors_applied(self, model, sample_df):
        result = model.analyze_dataframe(sample_df)
        # Energy and IT should have different prior_alpha
        assert result.iloc[0]["prior_alpha"] != result.iloc[2]["prior_alpha"]

    def test_momentum_adjusts_priors(self, sample_df):
        m_on = EarningsBeatProbabilityModel(use_momentum_prior=True)
        m_off = EarningsBeatProbabilityModel(use_momentum_prior=False)
        r_on = m_on.analyze_dataframe(sample_df)
        r_off = m_off.analyze_dataframe(sample_df)
        # Priors should differ when momentum is on
        assert not np.allclose(r_on["prior_alpha"].values, r_off["prior_alpha"].values)

    def test_quality_discount_applied(self, model, sample_df):
        r_qual = model.analyze_dataframe(sample_df)
        m_no = EarningsBeatProbabilityModel(use_quality_adjustment=False)
        r_no = m_no.analyze_dataframe(sample_df)
        # Stock B has low quality + high distress → lower confidence
        b_qual = r_qual[r_qual["ticker"] == "B"]["confidence_score"].values[0]
        b_no = r_no[r_no["ticker"] == "B"]["confidence_score"].values[0]
        assert b_qual < b_no

    def test_streak_blending_applied(self, model, sample_df):
        result = model.analyze_dataframe(sample_df)
        # All rows have map_estimate and model_confidence → blending active
        assert all(result["posterior_beat_prob"].notna())

    def test_fallback_without_new_columns(self, model):
        """Model still works with legacy columns only."""
        df = pd.DataFrame({
            "ticker": ["X", "Y"],
            "name": ["XX", "YY"],
            "sector": ["Energy", "Energy"],
            "eps_trajectory_score": [70, 40],
        })
        result = model.analyze_dataframe(df)
        assert len(result) == 2

    def test_empty_df(self, model):
        result = model.analyze_dataframe(pd.DataFrame())
        assert result.empty
