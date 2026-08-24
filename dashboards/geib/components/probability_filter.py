"""Shared probability metric selector + percentage range slider.

Replaces the per-chart "minimum probability" dropdowns (each of which silently
hard-coded ``p_upside_pos_cond`` and offered four or five coarse steps) with one
reusable control pair:

* a **metric dropdown** choosing which probability column to gate on, and
* a two-handle **percentage range slider** giving an inclusive ``lo <= x <= hi``
  band at 0.05 resolution.

Every call site derives its DOM ids, callback inputs, and applied filtering from
the single :data:`PROBABILITY_METRICS` registry below, mirroring the registry
pattern of the sibling :mod:`~dashboards.geib.components.filter_component`.

Units
-----
Slider *values* are raw decimals (``0.8`` = 80%), matching the storage
convention of ``analytics.kalman_filtered_price_targets``; percent formatting
happens only at the display boundary (tick marks and the live readout). See the
"Unit convention" section of ``CLAUDE.md``.

Bounds
------
Three of the four metrics are probabilities in ``[0, 1]``, but
``analyst_conviction`` is **not**. The analytics DDL documents it as::

    COMMENT ON COLUMN analytics.kalman_filtered_price_targets."analyst_conviction"
        IS 'Net buy-minus-sell analyst conviction (decimal in [-1, 1]).';

so it is registered with a ``[-1, 1]`` range and the slider re-ranges when the
metric changes. Do **not** "normalise" it to ``[0, 1]``: a 0-1 slider silently
drops every net-bearish name, which is exactly the half of the distribution a
user selecting "Analyst Conviction" is most likely looking for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from ..theme import COLORWAY, control

# --- Metric registry -------------------------------------------------------
# Single source of truth for the selectable probability columns. ``column`` must
# match the analytics DDL exactly (sql_scripts/analytics/kalman_filtered_price_targets.sql).


@dataclass(frozen=True)
class ProbabilityMetric:
    """A selectable probability column and the slider range it spans.

    Parameters
    ----------
    column
        Column name in ``analytics.kalman_filtered_price_targets``.
    label
        Human-readable label shown in the metric dropdown.
    lo
        Slider minimum (the metric's documented lower bound).
    hi
        Slider maximum (the metric's documented upper bound).
    """

    column: str
    label: str
    lo: float
    hi: float


# ``kalman_gain`` was removed from this tuple on 2026-08-24 and deliberately
# NOT replaced. Two reasons, either sufficient:
#
#   * It does not order the universe. Under the v2 definition
#     (P(risk_adj_return > 0)) 54.0% of names sat at exactly 0 or exactly 1 on
#     run 0aa3397b1d01, up from 50.6%; under the v1 definition this board
#     actually reads, it is ``sigmoid(risk_adj_return)`` -- a sigmoid of a
#     standardised log-uplift, which is not the probability of any event and
#     correlated -0.004 with analyst count. Degenerate under one definition and
#     meaningless under the other.
#   * Its label named a variable that no longer exists. "Achieve Prob." was
#     ``achieve_prob``, the Deterministic v2 removed for exactly that reason.
#
# The column is still exported and still loads (``data.NUMERIC_COLUMNS``); it is
# simply no longer offered as something to filter or rank on. Anything that
# needs a probability should use ``p_upside_pos_cond``, which is the screen's
# primary column and the one furthest from a consensus sort.
#
# ``shrink_gain`` is the successor -- it is the quantity the retired name always
# suggested (the weight the forecast-error update puts on the name's own
# smoothed observation, ``struct_var / (struct_var + fe_var)``, so [0, 1] by
# construction, averaging ~0.847 and rising monotonically with analyst
# coverage). It is NOT added here yet: this board reads
# ``analytics.kalman_filtered_price_targets``, the **v1** table, and v1 does not
# compute ``shrink_gain`` at all. Add the entry below in the same edit that
# points ``data.TABLE_NAME`` at the v2 table, not before.
PROBABILITY_METRICS: tuple[ProbabilityMetric, ...] = (
    ProbabilityMetric("p_upside_pos_cond", "Conditional Prob. Positive", 0.0, 1.0),
    ProbabilityMetric("mc_prob_pos", "Monte Carlo Prob. Positive", 0.0, 1.0),
    ProbabilityMetric("analyst_conviction", "Analyst Conviction", -1.0, 1.0),
)

METRICS_BY_COLUMN: dict[str, ProbabilityMetric] = {m.column: m for m in PROBABILITY_METRICS}

METRIC_OPTIONS: list[dict] = [{"label": m.label, "value": m.column} for m in PROBABILITY_METRICS]


def available_metrics(df: pd.DataFrame) -> tuple[ProbabilityMetric, ...]:
    """Return the registry entries whose column is present in *df*.

    The loader issues ``SELECT *`` and coerces only the columns that came back
    (``data._coerce_dtypes``), so a registry entry naming a column the live
    analytics table does not have yields a slider that filters nothing and says
    nothing about why. That is a live hazard rather than a hypothetical: the
    board reads the v1 table, and half the columns documented in the v2 DDL do
    not exist there.

    Parameters
    ----------
    df
        The loaded analytics frame.

    Returns
    -------
    tuple[ProbabilityMetric, ...]
        The subset of :data:`PROBABILITY_METRICS` backed by a real column,
        registry order preserved. Empty only if the frame carries none of them.
    """
    cols = set(df.columns)
    return tuple(m for m in PROBABILITY_METRICS if m.column in cols)


def metric_options(df: Optional[pd.DataFrame] = None) -> list[dict]:
    """Dropdown options, restricted to metrics *df* can actually support.

    Falls back to the full registry when no frame is supplied, so an import-time
    caller with nothing loaded yet behaves as before.
    """
    if df is None:
        return METRIC_OPTIONS
    return [{"label": m.label, "value": m.column} for m in available_metrics(df)]

DEFAULT_METRIC = "p_upside_pos_cond"

# Slider resolution (5 percentage points).
STEP = 0.05

# Number of intervals between rendered tick marks, so a [0, 1] metric marks at
# 0/25/50/75/100% and a [-1, 1] metric marks at -100/-50/0/50/100%.
_MARK_INTERVALS = 4

_SLIDER_STYLE = {"minWidth": "320px", "paddingRight": "20px"}
_READOUT_STYLE = {"fontWeight": "bold", "color": COLORWAY[3], "marginBottom": "6px"}

# component_ids whose sync callbacks are already registered (import-time guard,
# so a module imported twice cannot raise DuplicateCallback).
_REGISTERED: set[str] = set()


# --- Id helpers ------------------------------------------------------------


def metric_id(component_id: str) -> str:
    """DOM id of the metric dropdown for *component_id*."""
    return f"{component_id}_prob_metric"


def range_id(component_id: str) -> str:
    """DOM id of the probability range slider for *component_id*."""
    return f"{component_id}_prob_range"


def display_id(component_id: str) -> str:
    """DOM id of the live band readout for *component_id*."""
    return f"{component_id}_prob_range_display"


# --- Internals -------------------------------------------------------------


def resolve_metric(column: Optional[str]) -> ProbabilityMetric:
    """Return the registry entry for *column*, falling back to the default.

    Guards against a cleared dropdown (``None``) and against a stale/unknown
    column arriving from a persisted client state.
    """
    return METRICS_BY_COLUMN.get(column or "", METRICS_BY_COLUMN[DEFAULT_METRIC])


def _marks(spec: ProbabilityMetric) -> dict:
    """Return percent-formatted tick marks spanning *spec*'s range."""
    width = (spec.hi - spec.lo) / _MARK_INTERVALS
    return {
        round(spec.lo + i * width, 2): f"{round(spec.lo + i * width, 2):.0%}"
        for i in range(_MARK_INTERVALS + 1)
    }


def _clamp_band(value: Any, spec: ProbabilityMetric) -> list[float]:
    """Return *value* as a ``[lo, hi]`` pair clamped into *spec*'s range.

    Preserves the user's chosen band across a metric change wherever it still
    fits (e.g. ``[0.7, 1.0]`` survives a switch to a ``[-1, 1]`` metric) instead
    of resetting the slider to full width. Falls back to the metric's full range
    when *value* is missing or malformed.
    """
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        return [spec.lo, spec.hi]
    lo = min(max(lo, spec.lo), spec.hi)
    hi = min(max(hi, spec.lo), spec.hi)
    if lo > hi:
        lo, hi = hi, lo
    return [lo, hi]


def _readout(spec: ProbabilityMetric, band: list[float]) -> str:
    """Return the ``"Label: 70% - 100%"`` live readout text."""
    return f"{spec.label}: {band[0]:.0%} – {band[1]:.0%}"


# --- Public API ------------------------------------------------------------


def probability_controls(
        component_id: str,
        *,
        metric: str = DEFAULT_METRIC,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
) -> list[html.Div]:
    """Return the ``[metric dropdown, range slider]`` controls for a chart.

    Splat the result into a ``geib-controls-row`` children list in place of the
    chart's former "Min Probability" dropdown.

    Parameters
    ----------
    component_id
        The chart's ``component_id``; all three DOM ids are derived from it.
    metric
        Initially selected probability column.
    lo
        Initial low handle. Defaults to the metric's minimum; pass the chart's
        former threshold so its default universe is unchanged.
    hi
        Initial high handle. Defaults to the metric's maximum.

    Returns
    -------
    list[dash.html.Div]
        Two themed control blocks.
    """
    spec = resolve_metric(metric)
    band = _clamp_band([spec.lo if lo is None else lo, spec.hi if hi is None else hi], spec)

    dropdown = control("Probability Metric:", dcc.Dropdown(
        id=metric_id(component_id), options=METRIC_OPTIONS, value=spec.column,
        searchable=False, clearable=False, style={"minWidth": "230px"}))

    slider = html.Div(
        className="geib-control",
        style=_SLIDER_STYLE,
        children=[
            html.Label("Probability Range:", className="geib-control-label"),
            html.Div(id=display_id(component_id), children=_readout(spec, band),
                     style=_READOUT_STYLE),
            dcc.RangeSlider(
                id=range_id(component_id), min=spec.lo, max=spec.hi, step=STEP,
                value=band, marks=_marks(spec),
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode="drag", allowCross=False),
        ],
    )
    return [dropdown, slider]


def probability_inputs(component_id: str) -> dict[str, Input]:
    """Return the callback ``Input`` map to spread into a chart's ``inputs``.

    Mirrors :data:`~dashboards.geib.components.filter_component.FILTER_CALLBACK_INPUTS`:
    keys are the DOM ids, so the values arrive in the chart's ``**kwargs`` under
    those ids and are read back by :func:`apply_probability_filter`.
    """
    return {
        metric_id(component_id): Input(metric_id(component_id), "value"),
        range_id(component_id): Input(range_id(component_id), "value"),
    }


def apply_probability_filter(
        df: pd.DataFrame,
        component_id: str,
        kwargs: dict[str, Any],
) -> pd.DataFrame:
    """Narrow *df* to rows whose selected probability metric falls in the band.

    Call this **before** any fixed column projection: the four selectable
    metrics are not all present in the per-chart projection lists, so filtering
    afterwards would raise ``KeyError`` for three of the four.

    The band is inclusive at both ends (``Series.between``); ``NaN`` rows drop
    out, matching the ``>=`` semantics of the dropdowns this replaces. Missing
    columns and empty frames pass through unchanged, so a chart still renders
    against a source that predates a metric.

    Parameters
    ----------
    df
        Frame to filter (typically straight out of ``filter_data``).
    component_id
        The chart's ``component_id``, used to look the control values up.
    kwargs
        The chart callback's ``**kwargs``.

    Returns
    -------
    pandas.DataFrame
        The filtered frame.
    """
    if df is None or len(df) == 0:
        return df

    spec = resolve_metric(kwargs.get(metric_id(component_id)))
    if spec.column not in df.columns:
        return df

    band = _clamp_band(kwargs.get(range_id(component_id)), spec)
    return df[df[spec.column].between(band[0], band[1])]


def register(component_id: str) -> None:
    """Register the metric/slider sync callbacks for *component_id*.

    Call once at module import next to the chart's other id constants (GEIB
    chart modules self-register their callbacks on import). Idempotent, so a
    re-imported module cannot raise a duplicate-callback error.

    Two callbacks are registered rather than one, because a single callback
    reading the slider value *and* writing it back would be circular:

    * bounds — metric change re-ranges ``min``/``max``/``step``/``marks`` and
      clamps the current band into the new range (band read as ``State``);
    * readout — metric or slider change refreshes the percent text.
    """
    if component_id in _REGISTERED:
        return
    _REGISTERED.add(component_id)

    @callback(
        output=[
            Output(range_id(component_id), "min"),
            Output(range_id(component_id), "max"),
            Output(range_id(component_id), "step"),
            Output(range_id(component_id), "marks"),
            Output(range_id(component_id), "value"),
        ],
        inputs=[Input(metric_id(component_id), "value")],
        state=[State(range_id(component_id), "value")],
        prevent_initial_call=True,
    )
    def _sync_bounds(metric, band):
        spec = resolve_metric(metric)
        return spec.lo, spec.hi, STEP, _marks(spec), _clamp_band(band, spec)

    @callback(
        output=Output(display_id(component_id), "children"),
        inputs=[
            Input(metric_id(component_id), "value"),
            Input(range_id(component_id), "value"),
        ],
    )
    def _sync_readout(metric, band):
        spec = resolve_metric(metric)
        return _readout(spec, _clamp_band(band, spec))
