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
