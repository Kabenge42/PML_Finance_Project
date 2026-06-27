"""Shared, horizon-correct risk metrics for the GEIB charts.

The ``analytics.kalman_filtered_price_targets`` columns the charts consume are
produced by :class:`KalmanFilterPriceTarget` in ``pymc_kalman_filter_pt.py``:

* ``expected_return_kalman`` — posterior-mean implied upside
  (``expected_pt / last_price - 1``, the de-standardised ``expected_upside``).
  A *total* return to the analyst price target, not a per-period (daily) return.
* ``kalman_variance``        — posterior variance of the smoothed price-target
  *level* (``var(expected_pt)``), i.e. in currency-squared units (not a return
  variance).

Both are next-twelve-month (NTM) quantities: analyst price targets are NTM, and
the ``*_ago`` history the smoother is fit on is anchored at the snapshot "now"
(``KalmanFilterPriceTarget.build_price_target_history``). They therefore already
sit on a ~1-year horizon, so the daily-frequency annualization the original
chart stubs applied (``* 252`` on the return, ``* sqrt(252)`` on the volatility)
inflated returns by ~252x and mis-scaled volatility — which is additionally in
price units until divided by the spot price.

These helpers centralise the correct conversion so every chart stays consistent.
Accept :class:`pandas.Series` (the common case, index preserved), NumPy arrays,
or scalars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Analyst price targets / the Kalman expected upside are next-twelve-month (NTM)
# quantities — already on a ~1-year horizon.
PRICE_TARGET_HORIZON_YEARS = 1.0

# z-score for the 95th percentile; under a normal approximation a 5th–95th
# percentile range spans ``2 * _Z_95`` standard deviations.
_Z_95 = 1.6448536269514722


def annualized_return_pct(
        expected_return_kalman: object,
        horizon_years: float = PRICE_TARGET_HORIZON_YEARS,
) -> object:
    """Return the annualized expected return, in percent.

    Parameters
    ----------
    expected_return_kalman
        Total NTM implied upside (decimal, e.g. ``0.15`` for +15%).
    horizon_years
        Horizon the upside spans, in years. Defaults to
        :data:`PRICE_TARGET_HORIZON_YEARS`.

    Returns
    -------
    object
        ``expected_return_kalman / horizon_years * 100`` (same container type as
        the input).
    """
    return expected_return_kalman / horizon_years * 100.0


def return_volatility(
        kalman_variance: object,
        price: object,
        horizon_years: float = PRICE_TARGET_HORIZON_YEARS,
) -> object:
    """Return the annualized return standard deviation, as a decimal.

    ``kalman_variance`` is the variance of the price-target *level*; dividing its
    square root by the spot ``price`` converts the currency dispersion into a
    return (the units cancel exactly because the implied upside is
    ``expected_pt / price - 1`` with ``price`` constant per name), which is then
    de-annualized by ``sqrt(horizon_years)``.

    Parameters
    ----------
    kalman_variance
        Posterior variance of the price-target level (currency-squared).
    price
        Spot/anchor price the upside is measured from (``original_price``). A
        non-positive price yields ``NaN`` — the name is unusable for risk scaling.
    horizon_years
        Horizon the variance spans, in years. Defaults to
        :data:`PRICE_TARGET_HORIZON_YEARS`.

    Returns
    -------
    object
        Annualized return standard deviation (decimal). Returns a
        :class:`pandas.Series` (index preserved) when ``kalman_variance`` is a
        Series, otherwise a NumPy array / scalar.
    """
    variance = np.asarray(kalman_variance, dtype="float64")
    price_arr = np.asarray(price, dtype="float64")
    safe_price = np.where(price_arr > 0, price_arr, np.nan)
    sigma = np.sqrt(variance) / safe_price / np.sqrt(horizon_years)
    if isinstance(kalman_variance, pd.Series):
        return pd.Series(sigma, index=kalman_variance.index, name="return_volatility")
    return sigma


def annualized_volatility_pct(
        kalman_variance: object,
        price: object,
        horizon_years: float = PRICE_TARGET_HORIZON_YEARS,
) -> object:
    """Return the annualized return volatility, in percent.

    Thin wrapper over :func:`return_volatility` (see it for the unit reasoning).
    """
    return return_volatility(kalman_variance, price, horizon_years) * 100.0


def quantile_return_volatility(
        return_p05: object,
        return_p95: object,
        horizon_years: float = PRICE_TARGET_HORIZON_YEARS,
) -> object:
    """Return the return-scale volatility implied by a 5th–95th percentile range.

    Inverts the Monte-Carlo *return* distribution's 5th/95th percentiles into a
    standard deviation under a normal approximation: a normal ``[p05, p95]`` range
    spans ``2 * z_0.95`` standard deviations, so
    ``sigma = (p95 - p05) / (2 * z_0.95)``, de-annualized by ``sqrt(horizon_years)``.

    This is the asset's *return* dispersion — the proper mean-variance risk input —
    unlike :func:`return_volatility`, which converts ``kalman_variance`` (the
    posterior variance of the price-target *level*, i.e. estimation uncertainty of
    the mean) and so understates risk by ~1-2 orders of magnitude.

    Parameters
    ----------
    return_p05, return_p95
        5th and 95th percentiles of the per-name return distribution (decimal,
        e.g. ``er_p05`` / ``er_p95`` from
        ``analytics.kalman_filtered_price_targets``).
    horizon_years
        Horizon the percentiles span, in years. Defaults to
        :data:`PRICE_TARGET_HORIZON_YEARS`.

    Returns
    -------
    object
        Return standard deviation (decimal). Returns a :class:`pandas.Series`
        (index preserved) when the inputs are Series, otherwise a NumPy array /
        scalar. A non-positive spread yields ``NaN`` (an unusable risk estimate).
    """
    p05 = np.asarray(return_p05, dtype="float64")
    p95 = np.asarray(return_p95, dtype="float64")
    spread = p95 - p05
    spread = np.where(spread > 0, spread, np.nan)
    sigma = spread / (2.0 * _Z_95) / np.sqrt(horizon_years)
    if isinstance(return_p95, pd.Series):
        return pd.Series(sigma, index=return_p95.index, name="return_volatility")
    return sigma


def quantile_volatility_pct(
        return_p05: object,
        return_p95: object,
        horizon_years: float = PRICE_TARGET_HORIZON_YEARS,
) -> object:
    """Return the percentile-implied return volatility, in percent.

    Thin wrapper over :func:`quantile_return_volatility` (see it for the method).
    """
    return quantile_return_volatility(return_p05, return_p95, horizon_years) * 100.0
