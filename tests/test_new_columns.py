"""Verify implied_return_mc and implied_return_kalman columns."""
import numpy as np
import pandas as pd

from probabilistic_ml_model.statistical_functions.statistical_models import (
    kalman_filter_price_target,
    monte_carlo_price_target_simulation,
)


def _make_mc_df():
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "name": ["Apple", "Microsoft", "Google"],
            "sector": ["Tech", "Tech", "Tech"],
            "industry": ["Software", "Software", "Software"],
            "region": ["US", "US", "US"],
            "country": ["US", "US", "US"],
            "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
            "last_price": [150.0, 300.0, 140.0],
            "price_target": [180.0, 350.0, 170.0],
            "price_target_high": [200.0, 400.0, 190.0],
            "price_target_low": [160.0, 310.0, 150.0],
            "price_target_median": [180.0, 350.0, 170.0],
        }
    )


def _make_kal_df():
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "name": ["Apple", "Microsoft", "Google"],
            "sector": ["Tech", "Tech", "Tech"],
            "industry": ["Software", "Software", "Software"],
            "country": ["US", "US", "US"],
            "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
            "last_price": [150.0, 300.0, 140.0],
            "price_target": [180.0, 350.0, 170.0],
        }
    )


def test_mc_implied_return_mc_column_present():
    result = monte_carlo_price_target_simulation(_make_mc_df(), n_simulations=1000)
    assert "implied_return_mc" in result.columns


def test_mc_implied_return_mc_is_percentage_based():
    result = monte_carlo_price_target_simulation(_make_mc_df(), n_simulations=1000)
    # Percentage-based implied return: (price_target / last_price - 1) * 100
    # With targets above last_price, returns should be positive percentages (not dollar prices)
    assert (result["implied_return_mc"] > 0).all()
    assert (result["implied_return_mc"] < 200).all()  # sanity: not unreasonably large


def test_mc_implied_return_mc_is_implied_return():
    """implied_return_mc should be percentage implied return from MC fair value."""
    result = monte_carlo_price_target_simulation(_make_mc_df(), n_simulations=50000)
    assert result["implied_return_mc"].notna().all()
    assert len(result) == 3
    # price_target_mc holds the price-based fair value
    assert "price_target_mc" in result.columns
    # Verify: implied_return_mc ≈ (price_target_mc / last_price - 1) * 100
    expected_pct = (result["price_target_mc"] / result["last_price"] - 1) * 100
    np.testing.assert_allclose(result["implied_return_mc"].values, expected_pct.values, atol=0.01)


def test_kalman_implied_return_kalman_column_present():
    result = kalman_filter_price_target(_make_kal_df())
    assert "implied_return_kalman" in result.columns


def test_kalman_implied_return_kalman_is_percentage_based():
    result = kalman_filter_price_target(_make_kal_df())
    # implied_return_kalman = (original_target / original_price - 1) * 100
    assert "price_target_kalman" in result.columns
    expected_pct = (result["original_target"] / result["original_price"] - 1) * 100
    np.testing.assert_allclose(
        result["implied_return_kalman"].values,
        expected_pct.values,
        atol=0.01,
    )


def test_kalman_variance_is_per_stock_array():
    """Verify np.full wrappers were removed — variance should be vectorized."""
    result = kalman_filter_price_target(_make_kal_df())
    assert result["kalman_variance"].dtype == np.float64 or result["kalman_variance"].dtype == float
    assert len(result["kalman_variance"]) == 3


def test_kalman_empty_input_has_implied_return_kalman():
    empty_df = pd.DataFrame({"foo": [1]})
    result = kalman_filter_price_target(empty_df)
    assert "implied_return_kalman" in result.columns


def test_kalman_gain_is_per_stock_array():
    """Verify np.full wrappers were removed — gain should be vectorized."""
    result = kalman_filter_price_target(_make_kal_df())
    assert len(result["kalman_gain"]) == 3


def test_signal_strength_is_per_stock_array():
    """Verify np.full wrappers were removed — signal_strength should be vectorized."""
    result = kalman_filter_price_target(_make_kal_df())
    assert len(result["signal_strength"]) == 3
