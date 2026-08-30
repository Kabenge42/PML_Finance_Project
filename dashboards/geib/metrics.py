"""Shared, horizon-correct risk metrics for the GEIB charts.

The ``analytics.kalman_filtered_price_targets`` columns the charts consume are
produced by :class:`KalmanFilterPriceTarget` in ``pymc_kalman_filter_pt.py``:

* ``expected_return_kalman`` — posterior-mean implied upside
  (``expected_pt / last_price - 1``, the de-standardised ``expected_upside``).
  A *total* return to the analyst price target, not a per-period (daily) return.
* ``er_sd``                  — pooled standard deviation of the forward-return
  Monte-Carlo draws: a *return* dispersion in raw decimal, on the same NTM
  horizon as the mean above. This replaced v1's ``kalman_variance`` (a
  currency-squared price-target *level* variance), which every consumer had to
  divide by spot before it meant anything.

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

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Analyst price targets / the Kalman expected upside are next-twelve-month (NTM)
# quantities — already on a ~1-year horizon.
PRICE_TARGET_HORIZON_YEARS = 1.0

# Suffix -> fixed calendar-day lookback for the ``*_{suffix}_ago`` ladders in
# ``analytics.kalman_filtered_price_targets``. The anchored suffixes (``mtd`` /
# ``qtd`` / ``ytd``) are month/quarter/year starts resolved at call time by
# :func:`lookback_date`.
_FIXED_LOOKBACK_DAYS: dict[str, int] = {
    "5d": 5,
    "1w": 7,
    "1m": 30,
    "3m": 91,
    "6m": 182,
    "1y": 365,
    "3y": 1095,
    "5y": 1826,
}

# Ladder suffixes present in the DDL, ordered longest lookback -> shortest so a
# melted ladder comes out chronologically.
PRICE_TARGET_SUFFIXES: tuple[str, ...] = ("1y", "ytd", "6m", "3m", "qtd", "1m", "mtd", "1w")
PRICE_SUFFIXES: tuple[str, ...] = ("5y", "3y", "1y", "6m", "3m", "qtd", "1m", "1w", "5d")

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


# ``return_volatility`` and ``annualized_volatility_pct`` lived here until the v2
# repoint. Both took ``kalman_variance`` -- the price-target LEVEL variance in
# currency-squared -- and divided its square root by the spot price to recover a
# return sd. v2 does not export that column and does not need to: ``er_sd`` is
# the forward-return standard deviation directly, in raw decimal, over the same
# NTM horizon. The conversion those two functions existed to get right no longer
# has to happen, so use ``er_sd`` and delete the round trip.
#
# For a name whose ``er_sd`` is missing, :func:`quantile_return_volatility` below
# recovers a sd from the ``er_p05`` / ``er_p95`` spread under a normal
# approximation -- that is the supported fallback, and the efficient-frontier and
# VaR/CVaR cards already use it as their primary.


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
    unlike ``er_sd``, which is already a return sd (the
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


def lookback_date(suffix: str, asof: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Return the calendar date a ``*_{suffix}_ago`` column refers to.

    Parameters
    ----------
    suffix
        Ladder suffix from the DDL naming (``"1w"``, ``"3m"``, ``"mtd"`` …).
        Fixed-window suffixes resolve via :data:`_FIXED_LOOKBACK_DAYS`; the
        anchored suffixes ``mtd`` / ``qtd`` / ``ytd`` resolve to the start of
        the *asof* month / quarter / year.
    asof
        Snapshot date the ladder is anchored at. Defaults to today (normalised).

    Returns
    -------
    pandas.Timestamp
        The lookback date.

    Raises
    ------
    KeyError
        On an unknown suffix (a programming error, not a data condition).
    """
    anchor = (asof if asof is not None else pd.Timestamp.now()).normalize()
    if suffix == "mtd":
        return anchor.replace(day=1)
    if suffix == "qtd":
        return pd.Timestamp(anchor.year, 3 * ((anchor.month - 1) // 3) + 1, 1)
    if suffix == "ytd":
        return pd.Timestamp(anchor.year, 1, 1)
    return anchor - pd.Timedelta(days=_FIXED_LOOKBACK_DAYS[suffix])


def history_ladder(
        row: "pd.Series",
        prefix: str,
        suffixes: Sequence[str],
        today_value: Optional[float] = None,
        asof: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Melt a row's ``{prefix}_{suffix}_ago`` ladder into a tidy (date, value) frame.

    Reads ``{prefix}_{suffix}_ago`` for each suffix, dates it via
    :func:`lookback_date`, drops missing columns / NaN values, and optionally
    appends ``(asof, today_value)`` as the terminal "today" point. Anchored
    suffixes (``mtd``/``qtd``/``ytd``) can collide with a fixed-window point or
    with today early in a period; exact-date collisions keep the
    shortest-lookback (latest-appended) value.

    Parameters
    ----------
    row
        Name -> value mapping. A single table row, or an *aggregate* Series
        (e.g. per-column means keyed by the same ``{prefix}_{suffix}_ago``
        names) — so universe-average timelines and realized-return ladders
        reuse this unchanged.
    prefix
        Column prefix (``"price"`` or ``"price_target"``).
    suffixes
        Ladder suffixes to read, longest lookback first
        (:data:`PRICE_SUFFIXES` / :data:`PRICE_TARGET_SUFFIXES`).
    today_value
        Terminal value at *asof* (e.g. ``original_price``); skipped when
        ``None`` / non-finite.
    asof
        Snapshot date the ladder is anchored at. Defaults to today (normalised).

    Returns
    -------
    pandas.DataFrame
        Columns ``["date", "value"]``, sorted by date; empty (with those
        columns) when nothing survives.
    """
    anchor = (asof if asof is not None else pd.Timestamp.now()).normalize()
    records: list[tuple[pd.Timestamp, float]] = []
    for suffix in suffixes:
        raw = row.get(f"{prefix}_{suffix}_ago")
        if isinstance(raw, pd.Series):
            # Duplicate labels make ``row.get`` return a Series, whose
            # ``pd.isna`` result has an ambiguous truth value — collapse first.
            raw = raw.iloc[0] if len(raw) else None
        if raw is None or pd.isna(raw):
            continue
        records.append((lookback_date(suffix, anchor), float(raw)))
    if today_value is not None and np.isfinite(today_value):
        records.append((anchor, float(today_value)))

    ladder = pd.DataFrame(records, columns=["date", "value"])
    if len(ladder) == 0:
        return ladder
    # Stable sort preserves append order within a date, so ``keep="last"``
    # resolves collisions to the shortest lookback / the today point.
    ladder = ladder.sort_values("date", kind="stable")
    return ladder.drop_duplicates("date", keep="last").reset_index(drop=True)
