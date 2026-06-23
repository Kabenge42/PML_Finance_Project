"""Global Equity Investment Board (GEIB) dashboard package.

A focused Dash + dash-bootstrap-components board driven by a single analytics
table, ``analytics.kalman_filtered_price_targets`` (exported by
``pymc_kalman_filter_pt.py`` from
:class:`probabilistic_ml_model.pymc_models.KalmanFilterModel`).

Run via ``python dashboards/global_equity_investment_dashboard.py``.
"""

__all__ = ["app"]
