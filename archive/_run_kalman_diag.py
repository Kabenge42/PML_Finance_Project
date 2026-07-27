"""Focused driver: fit the fused Kalman panel model and print §9 diagnostics only.

Skips EDA and the downstream single-ISIN / cohort / forest sections so we can
re-run the MCMC and inspect convergence after the non-centred group-effect fix.
Plots are routed to a non-interactive Agg backend and silently dropped.
"""
from __future__ import annotations

import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless: .show() becomes a no-op
import matplotlib.pyplot as plt

from sqlalchemy import create_engine

import pymc_kalman_filter_pt as K

# Silence display() side effects (tables) to keep stdout focused on diagnostics.
K.display = lambda *a, **k: None
# Drop figures without trying to render them.
plt.show = lambda *a, **k: None


def run() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"))
    K.setup_plotting()
    plt.show = lambda *a, **k: None  # setup_plotting may reset rcParams, not plt.show

    engine = create_engine(K.resolve_db_url())
    kalman_df = K.load_kalman_df(engine)
    feature_catalogue = K.load_feature_catalogue(engine)
    roles = K.resolve_feature_roles(kalman_df, feature_catalogue)

    drift_features, _ = K.map_state_space_features(kalman_df)
    panel = K.prepare_kalman_panel_inputs(kalman_df, roles, drift_features)

    model = K.build_panel_model(panel, robust=True)
    prior_idata = K.run_prior_predictive(model, panel)
    idata = K.sample_posterior(model, prior_idata)

    print("\n" + "=" * 70)
    print("§9 MCMC DIAGNOSTICS (post non-centred group-effect fix)")
    print("=" * 70)
    K.run_diagnostics(idata, panel)


if __name__ == "__main__":
    run()
