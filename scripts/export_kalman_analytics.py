#!/usr/bin/env python
"""Run the Kalman workflow through the analytics write, and stop there.

The same path :func:`pymc_kalman_filter_pt.main` takes up to and including §10c,
without the six post-export sections (universe fit, single-ISIN + SV twin,
mingled cohort + SV twin, granular forest) that each fit further models and are
not needed to refresh ``analytics.kalman_filtered_price_targets``.

Every step calls the module's own function in the module's own order — this is a
narrower entry point, not a reimplementation.

**Two deliberate differences from a bare ``main()`` call:**

* ``robust=True``. ``main()`` defaults to ``robust=False`` (Normal likelihood),
  but the 2026-08-10 validation — 0 divergences, max R-hat 1.0090, min ESS 678.3,
  per-time coverage 91.6-92.5 % — was measured on the **Student-t** panel
  likelihood, which is also ``build_fused_kalman_pt_model``'s own default.
  Exporting on the Normal variant would ship numbers no gate has seen.
* It stops after §10c.

Usage::

    . .\\set_env.ps1
    python scripts\\export_kalman_analytics.py            # writes the table
    python scripts\\export_kalman_analytics.py --dry-run  # everything but the write

The write is a DROP-and-RECREATE of ``analytics.kalman_filtered_price_targets``
and regenerates its checked-in DDL. That table is the GEIB dashboard's only
source, so deploy the dashboard after this completes — they ship as a pair.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pymc_kalman_filter_pt as K  # noqa: E402

ROBUST = True          # matches validation; see the module docstring
VOLUME_PENALTY = 0.25  # main()'s default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true',
                    help='run everything except the analytics table write')
    ap.add_argument('--no-eda', action='store_true', help='skip the §2 EDA panels')
    args = ap.parse_args()

    from sqlalchemy import create_engine

    cfg = K.get_run_config()
    logging.basicConfig(level=cfg.log_level)
    K.setup_plotting()
    K.enable_artifact_export()
    print(f'Artifact export -> {K.get_export_state().root}')
    print(f'Config: draws={cfg.draws} tune={cfg.tune} chains={cfg.chains} '
          f'cores={cfg.cores} target_accept={cfg.target_accept}')
    print(f'        panel_lookbacks={cfg.panel_lookbacks} '
          f'state_innovation_scale={cfg.state_innovation_scale}')
    print(f'        robust={ROBUST} (Student-t)  volume_penalty={VOLUME_PENALTY}')
    print(f'        write_analytics={not args.dry_run}')

    engine = create_engine(K.resolve_db_url())

    with K.export_section('01_data'):
        kalman_df = K.load_kalman_df(engine, cfg)
        feature_catalogue = K.load_feature_catalogue(engine)
        roles = K.resolve_feature_roles(kalman_df, feature_catalogue)

    if not args.no_eda:
        with K.export_section('02_eda'):
            K.run_eda(kalman_df, roles)

    with K.export_section('03_features'):
        drift_features, _ = K.map_state_space_features(kalman_df, feature_catalogue)
        panel = K.prepare_kalman_panel_inputs(
            kalman_df, roles, drift_features,
            history_lookbacks=cfg.panel_lookbacks,
            response_extra=cfg.panel_response_extra)

    model = K.build_panel_model(panel, robust=ROBUST,
                                volume_penalty=VOLUME_PENALTY, config=cfg)
    with K.export_section('06_prior'):
        prior_idata = K.run_prior_predictive(model, panel, cfg)
    with K.export_section('07_posterior'):
        idata = K.sample_posterior(model, prior_idata, panel=panel, config=cfg)
    with K.export_section('08_ppc'):
        K.run_posterior_predictive(model, idata, panel)
    with K.export_section('09_diagnostics'):
        K.run_diagnostics(idata, panel)

    with K.export_section('10_screen'):
        screen = K.summarize_panel_screen(idata, panel,
                                          horizon=cfg.mc_horizon, rho=cfg.mc_rho)
    with K.export_section('10b_risk'):
        risk_book = K.compute_cvar_aware_book(idata, panel, screen,
                                              screen.results, config=cfg)
    with K.export_section('10c_analytics'):
        kalman_results = K.export_analytics(
            idata, panel, screen, risk_book=risk_book,
            write=not args.dry_run)

    n = len(kalman_results) if kalman_results is not None else 0
    print('')
    print('=' * 62)
    if args.dry_run:
        print(f'DRY RUN complete — {n} rows prepared, nothing written.')
    else:
        print(f'EXPORT complete — {n} rows written to '
              f'analytics.kalman_filtered_price_targets.')
        print('NEXT: deploy the GEIB dashboard. The export and the deploy ship '
              'as a pair (see CLAUDE.md).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
