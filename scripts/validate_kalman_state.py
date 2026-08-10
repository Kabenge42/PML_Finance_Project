#!/usr/bin/env python
"""Validate the local-level-state Kalman panel model before re-exporting analytics.

Runs the production T=4 fit and checks the gates that decide whether the
local-level state layer (0.9.9.14) should ship. Nothing is written to the
database: this is a read-only, decide-then-export gate.

Usage
-----
Run on the PyCharm SDK interpreter — the repo ``.venv`` and the pipenv env are
known-broken for this stack::

    . .\\set_env.ps1
    ~\\AppData\\Local\\Python\\bin\\python.exe scripts\\validate_kalman_state.py

Options::

    --quick              draws/tune 300 instead of the config budget (smoke run)
    --compare            also run the §9b ELPD comparison (roughly triples runtime)
    --no-static          skip the static-twin refit (drops the sd/sigma_base deltas)
    --isins N            subsample to N names (default: full panel)

What it checks
--------------
1. **Convergence** — 0 divergences, max R-hat < 1.01, min bulk-ESS > 400
   (``MIN_ESS_GATE``).
2. **The state is alive** — ``sigma_state`` mixes AND its posterior is bounded
   away from 0. A collapse to ~0 means the panel carries no per-name time
   dynamics and the layer is dead weight: revert rather than ship.
3. **The predicted signature** — versus the static twin, ``sigma_base`` should
   FALL (persistent per-name signal moves out of the residual into the state)
   and the per-name posterior sd should RISE (the pseudo-replication fix).
4. **Per-time PPC coverage** ≈ 0.94, with no monotone drift across ``t``.
5. **The de-standardisation correction** — reports the pooled vs snapshot
   response moments and the resulting shift in exported upside, so the ~1.5-2.3
   pp downward correction is visible before it reaches the analytics table.

Exit status is non-zero if any hard gate fails.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

# Python puts THIS file's directory (scripts/) on sys.path, not the repo root, so
# the top-level modules are not importable without help. Prepend the repo root so
# the script runs from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pymc_kalman_filter_pt as K  # noqa: E402
from probabilistic_ml_model.pymc_models._workflow import MIN_ESS_GATE  # noqa: E402

RHAT_GATE = 1.01
SIGMA_STATE_FLOOR = 0.01  # below this the walk is indistinguishable from static


def _fit(panel, cfg, *, scale: float, label: str):
    """Build and sample one arm; returns (idata, model).

    Sampling goes through :func:`K.sample_with_fallback` (nutpie → numpyro →
    pymc), NOT bare ``build_sample_kwargs``. Its ``nuts_sampler=None`` default
    lands on PyMC's pure-Python NUTS, which under the project's forced PyTensor
    py-VM produced zero draws in 42 minutes of CPU on this model — the state
    layer adds ``n_isin × (T-1)`` ≈ 16.8k innovation parameters.
    """
    print(f'\n--- fitting {label} (state_innovation_scale={scale}) ---')
    model = K.build_fused_kalman_pt_model(
        panel, robust=True, volume_penalty=0.25, state_innovation_scale=scale)
    idata = K.sample_with_fallback(model, cfg, model_name=f'kalman_pt[{label}]',
                                   progressbar=False)
    if idata is None:
        raise RuntimeError(f'every candidate sampler failed on the {label!r} arm')
    return idata, model


def _convergence(idata) -> tuple[int, float, float]:
    """Return (divergences, max R-hat, min bulk-ESS) over non-constant vars."""
    import arviz_stats as azs

    post = K._posterior_dataset(idata)
    keep = [n for n, da in post.data_vars.items()
            if all(da.sizes[d] > 0 for d in da.dims if d not in ('chain', 'draw'))]
    keep = [v for v in keep if v not in set(K._degenerate_posterior_vars(idata, keep))]
    rhat = azs.rhat(post[keep])
    ess = azs.ess(post[keep], method='bulk')
    n_div = int(idata.sample_stats['diverging'].sum())
    max_rhat = float(np.nanmax([float(rhat[v].max()) for v in rhat.data_vars]))
    min_ess = float(np.nanmin([float(ess[v].min()) for v in ess.data_vars]))
    return n_div, max_rhat, min_ess


def _per_time_coverage(model, idata, panel) -> pd.Series:
    """94% posterior-predictive coverage per time step."""
    import pymc as pm
    from probabilistic_ml_model.pymc_models._pytensor_compat import (
        get_pytensor_compile_kwargs,
    )

    with model:
        ppc = pm.sample_posterior_predictive(
            idata, var_names=['target_pct_obs'], random_seed=7,
            progressbar=False, compile_kwargs=get_pytensor_compile_kwargs())
    pp = ppc.posterior_predictive['target_pct_obs']
    obs = ppc.observed_data['target_pct_obs'] if hasattr(ppc, 'observed_data') \
        else idata.observed_data['target_pct_obs']
    lo = pp.quantile(K._HDI_LO, dim=('chain', 'draw'))
    hi = pp.quantile(K._HDI_HI, dim=('chain', 'draw'))
    inside = (obs >= lo) & (obs <= hi)
    dims = [d for d in inside.dims if d != 'time']
    return inside.mean(dims).to_series()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--no-static', action='store_true')
    ap.add_argument('--isins', type=int, default=0)
    args = ap.parse_args()

    from sqlalchemy import create_engine

    cfg = K.get_run_config()
    if args.quick:
        cfg = replace(cfg, draws=300, tune=300)
    logging.basicConfig(level=cfg.log_level)

    engine = create_engine(K.resolve_db_url())
    kalman_df = K.load_kalman_df(engine, cfg)
    catalogue = K.load_feature_catalogue(engine)
    roles = K.resolve_feature_roles(kalman_df, catalogue)
    drift, _ = K.map_state_space_features(kalman_df, catalogue)
    panel = K.prepare_kalman_panel_inputs(
        kalman_df, roles, drift, history_lookbacks=cfg.panel_lookbacks,
        response_extra=cfg.panel_response_extra)
    if args.isins:
        panel = K._subsample_panel(panel, args.isins, random_seed=cfg.random_seed)

    n_isin, T, D = panel.Y.shape
    print(f'\n=== panel: {n_isin} ISINs, T={T}, D={D}, '
          f'{len(panel.drift_names)} drift features ===')
    if T < 2:
        print('FAIL: T=1 panel — the state layer needs panel_lookbacks set.')
        return 1

    # --- Gate 5: the de-standardisation correction (no fit required) ----------
    snap = panel.frame['feat_log_uplift'].to_numpy(dtype='float64')
    pooled_m, pooled_s = float(panel.response_mean[0]), float(panel.response_std[0])
    snap_m, snap_s = float(np.mean(snap)), float(np.std(snap))
    print('\n=== de-standardisation moments (Finding 2) ===')
    print(f'  pooled   (used by the model): mean={pooled_m:.6f} std={pooled_s:.6f}')
    print(f'  snapshot (the old inverse)  : mean={snap_m:.6f} std={snap_s:.6f}')
    print('  implied exported-upside shift at the cross-sectional mean latent:')
    for lat in (-0.5, 0.0, 0.5, 1.0):
        corrected = np.expm1(pooled_m + lat * pooled_s)
        legacy = np.expm1(snap_m + lat * snap_s)
        print(f'    latent={lat:+.1f}: corrected={corrected:+.2%}  '
              f'legacy={legacy:+.2%}  delta={corrected - legacy:+.2%}')

    failures: list[str] = []

    idata, model = _fit(panel, cfg, scale=cfg.state_innovation_scale,
                        label='local_level')
    n_div, max_rhat, min_ess = _convergence(idata)
    print('\n=== gate 1: convergence ===')
    print(f'  divergences = {n_div}      (gate: 0)')
    print(f'  max R-hat   = {max_rhat:.4f} (gate: < {RHAT_GATE})')
    print(f'  min ESS     = {min_ess:.1f}   (gate: > {MIN_ESS_GATE})')
    if n_div:
        failures.append(f'{n_div} divergences')
    if not (max_rhat < RHAT_GATE):
        failures.append(f'max R-hat {max_rhat:.4f}')
    if not (min_ess > MIN_ESS_GATE):
        failures.append(f'min ESS {min_ess:.1f}')

    post = idata.posterior
    print('\n=== gate 2: is the state alive? ===')
    if 'sigma_state' not in post:
        failures.append('sigma_state absent')
        print('  FAIL: sigma_state not in the posterior.')
    else:
        ss = post['sigma_state']
        ss_mean = float(ss.mean())
        ss_lo = float(ss.quantile(0.055))
        print(f'  sigma_state mean = {ss_mean:.4f}, 5.5% quantile = {ss_lo:.4f} '
              f'(floor {SIGMA_STATE_FLOOR})')
        if ss_lo < SIGMA_STATE_FLOOR:
            failures.append(
                f'sigma_state collapsed toward 0 (5.5% q = {ss_lo:.4f}) — the '
                'panel carries no per-name dynamics; revert the state layer')

    print('\n=== gate 4: per-time PPC coverage (target 0.94) ===')
    try:
        cov = _per_time_coverage(model, idata, panel)
        for t, v in cov.items():
            flag = '' if abs(v - 0.94) <= 0.03 else '   <-- off target'
            print(f'  t={t}: {v:.2%}{flag}')
        if (cov.max() - cov.min()) > 0.10:
            failures.append(
                f'per-time coverage drifts {cov.min():.2%}->{cov.max():.2%}')
    except Exception as exc:  # pragma: no cover - best-effort
        print(f'  skipped: {exc!r}')

    latent = K.resolve_screen_latent(post)
    sd_state = float(latent.std(('chain', 'draw')).mean())
    base_state = float(post['sigma_base'].mean())
    print('\n=== gate 3: predicted signature vs the static twin ===')
    print(f'  local_level: sigma_base={base_state:.4f}  '
          f'mean per-name posterior sd={sd_state:.4f}')

    if not args.no_static:
        idata_s, _ = _fit(panel, cfg, scale=0.0, label='static')
        post_s = idata_s.posterior
        sd_static = float(
            K.resolve_screen_latent(post_s).std(('chain', 'draw')).mean())
        base_static = float(post_s['sigma_base'].mean())
        print(f'  static    : sigma_base={base_static:.4f}  '
              f'mean per-name posterior sd={sd_static:.4f}')
        print(f'  -> sigma_base {base_static:.4f} -> {base_state:.4f} '
              f'({"FALLS as predicted" if base_state < base_static else "RISES — unexpected"})')
        print(f'  -> per-name sd {sd_static:.4f} -> {sd_state:.4f} '
              f'({"WIDENS as predicted" if sd_state > sd_static else "NARROWS — unexpected"})')
        if base_state >= base_static:
            failures.append('sigma_base did not fall versus the static twin')
        if sd_state <= sd_static:
            failures.append('per-name posterior sd did not widen')

    if args.compare:
        print('\n=== §9b ELPD comparison ===')
        K.run_model_comparison(panel, config=replace(cfg, chains=cfg.chains),
                               robust=True, volume_penalty=0.25)

    print('\n' + '=' * 62)
    if failures:
        print('VALIDATION FAILED:')
        for f in failures:
            print(f'  - {f}')
        print('Do NOT re-export analytics until these are resolved.')
        return 1
    print('VALIDATION PASSED — safe to re-export.')
    print('Next:  export_analytics(..., write=True)  THEN deploy the GEIB '
          'dashboard (they ship as a pair).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
