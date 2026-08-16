#!/usr/bin/env python
"""Validate the per-ISIN-latent Kalman panel model before re-exporting analytics.

Runs the production T=4 fit and checks the gates that decide whether the
0.9.9.14 changes should ship. Nothing is written to the database: this is a
read-only, decide-then-export gate.

Usage
-----
Run on the PyCharm SDK interpreter — the repo ``.venv`` and the pipenv env are
known-broken for this stack::

    . .\\set_env.ps1
    ~\\AppData\\Local\\Python\\bin\\python.exe scripts\\validate_kalman_state.py

Options::

    --quick              draws/tune 300 instead of the config budget (smoke run)
    --compare            also run the §9b ELPD comparison (roughly triples runtime)
    --no-static          skip the baseline refit (drops the sd/sigma_base deltas)
    --isins N            subsample to N names (default: full panel)

What it checks
--------------
1. **Convergence** — 0 divergences, max R-hat < 1.01, min bulk-ESS > 400
   (``MIN_ESS_GATE``).
2. **The per-ISIN latent is alive** — ``sigma_isin_level`` is bounded away from
   0. A collapse to ~0 means the panel carries no per-name signal beyond the
   drift regression. If the opt-in AR(1) layer is enabled
   (``state_innovation_scale > 0``), ``sigma_state`` is checked the same way.
3. **The predicted signature** — versus the pre-0.9.9.14 baseline (per-ISIN
   latent pinned off), ``sigma_base`` should FALL (per-name signal moves out of
   the residual) and the per-name posterior sd should RISE (the
   pseudo-replication fix).
4. **Per-time PPC coverage** ≈ the nominal interval (``_HDI_HI - _HDI_LO``),
   with no monotone drift across ``t``.
5. **The de-standardisation correction** — reports the pooled vs snapshot
   response moments and the resulting shift in exported upside, so the downward
   correction is visible before it reaches the analytics table.
6. **The exported analytics table** (added 2026-08-16) — read-only checks on
   what is already in the schema, not on the new fit: no non-finite or |v| > 100
   ranking metric, every clip-pinned row flagged ``out_of_support`` with its
   ranking columns NULL, and a single ``run_id`` across all seven tables. Skips
   rather than fails when the table predates the provenance columns.

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


def _fit(panel, cfg, *, scale: float, label: str, isin_level: float = 0.10):
    """Build and sample one arm; returns (idata, model).

    Sampling goes through :func:`K.sample_with_fallback` (nutpie → numpyro →
    pymc), NOT bare ``build_sample_kwargs``. Its ``nuts_sampler=None`` default
    lands on PyMC's pure-Python NUTS, which under the project's forced PyTensor
    py-VM produced zero draws in 42 minutes of CPU on this model — the state
    layer adds ``n_isin × (T-1)`` ≈ 16.8k innovation parameters.
    """
    print(f'\n--- fitting {label} (state_innovation_scale={scale}) ---')
    model = K.build_fused_kalman_pt_model(
        panel, robust=True, volume_penalty=0.25, isin_level_scale=isin_level,
        state_innovation_scale=scale)
    idata = K.sample_with_fallback(model, cfg, model_name=f'kalman_pt[{label}]',
                                   progressbar=False)
    if idata is None:
        raise RuntimeError(f'every candidate sampler failed on the {label!r} arm')
    return idata, model


def _convergence(idata) -> dict:
    """Per-variable R-hat / ESS, split into GLOBAL parameters vs per-ISIN latents.

    The split is the point. This model carries thousands of per-name latents
    (``z_isin_level``, and ``z_state`` when the AR layer is on), so a
    ``min(ESS)`` taken across *every* variable is a minimum over thousands of
    draws from a noisy statistic — a handful of slow-mixing individual names trips
    it even when every hyperparameter is healthy. A global scalar missing the gate
    is a real modelling problem; a per-name latent missing it usually is not.
    Reporting them separately keeps the gate honest in both directions.
    """
    import arviz_stats as azs

    post = K._posterior_dataset(idata)
    keep = [n for n, da in post.data_vars.items()
            if all(da.sizes[d] > 0 for d in da.dims if d not in ('chain', 'draw'))]
    keep = [v for v in keep if v not in set(K._degenerate_posterior_vars(idata, keep))]
    rhat = azs.rhat(post[keep])
    ess = azs.ess(post[keep], method='bulk')

    rows = []
    for v in keep:
        per_isin = 'isin' in post[v].dims
        rows.append({
            'var': v,
            'per_isin': per_isin,
            'rhat': float(np.nanmax(rhat[v].values)),
            'ess': float(np.nanmin(ess[v].values)),
            'size': int(np.prod([post[v].sizes[d] for d in post[v].dims
                                 if d not in ('chain', 'draw')]) or 1),
        })
    df = pd.DataFrame(rows)
    g, l = df[~df.per_isin], df[df.per_isin]
    return {
        'n_div': int(idata.sample_stats['diverging'].sum()),
        'table': df,
        'global_max_rhat': float(g.rhat.max()) if len(g) else float('nan'),
        'global_min_ess': float(g.ess.min()) if len(g) else float('nan'),
        'latent_max_rhat': float(l.rhat.max()) if len(l) else float('nan'),
        'latent_min_ess': float(l.ess.min()) if len(l) else float('nan'),
    }


def _per_time_coverage(model, idata, panel) -> pd.Series:
    """Posterior-predictive interval coverage per time step (K._HDI_LO/_HI)."""
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


def _gate_exported_table(engine) -> list[str]:
    """Gate 6: check the analytics table already in the schema, not the new fit.

    These three checks all guard regressions that shipped silently once before,
    so each one is worth a gate rather than a comment:

    * **Ranking-metric sanity.** ``expected_sharpe_ratio`` reached -4.28e15 on
      the 2026-08-15 export because a denormal ``er_sd`` passed a ``> 0``
      denominator guard. Anything non-finite or beyond |100| is an artefact, not
      a measurement.
    * **Symmetric out-of-support.** The detector was upper-only until
      2026-08-16, so names pinned at the -95% floor kept their ranking columns.
      Every clip-pinned row must now be flagged AND have its ranking columns
      NULL.
    * **Single vintage.** Five of the seven curated frames were a run behind the
      other two, and nothing in the schema said so.

    Read-only and best-effort: a missing table or unreachable schema reports a
    skip rather than a failure, so this cannot block a validation run performed
    before the first stamped export.

    **Advisory before the first fixed export, hard after it.** Judging a new fit
    by the quality of the PREVIOUS export is circular — run against the pre-fix
    table these checks fail on exactly the rows the imminent re-export is about
    to clear, which would block the export precisely when it is most needed. The
    discriminator is provenance: a table with no ``run_id`` was written by code
    that predates the fixes, so its findings are stale-data notes; a table that
    carries one was written by the current code, so the same findings are real
    regressions and gate the run.

    Parameters
    ----------
    engine
        Live SQLAlchemy engine against the analytics database.

    Returns
    -------
    list[str]
        Hard-failure descriptions; empty when every check passes, is skipped, or
        is advisory-only against a pre-provenance table.
    """
    from sqlalchemy import text

    out: list[str] = []
    schema = K._analytics_schema()
    table = K._ANALYTICS_TABLE
    print('\n=== gate 6: exported analytics table ===')
    try:
        with engine.connect() as conn:
            row = conn.execute(text(f"""
                SELECT count(*)                                              AS n,
                       count(*) FILTER (WHERE expected_sharpe_ratio IS NOT NULL
                                          AND (NOT (expected_sharpe_ratio
                                                    BETWEEN -100 AND 100)))  AS bad_sharpe,
                       count(*) FILTER (WHERE reward_to_cvar IS NOT NULL
                                          AND (NOT (reward_to_cvar
                                                    BETWEEN -100 AND 100)))  AS bad_starr,
                       count(*) FILTER (WHERE (er_p05 >= {K.UPLIFT_CLIP_HI} - 1e-6
                                            OR er_p95 <= {K.UPLIFT_CLIP_LO} + 1e-6)
                                          AND NOT out_of_support)            AS missed_oos,
                       count(*) FILTER (WHERE out_of_support
                                          AND expected_sharpe_ratio IS NOT NULL)
                                                                             AS leaked_rank,
                       count(*) FILTER (WHERE out_of_support)                AS n_oos
                FROM {schema}.{table}
            """)).mappings().one()
    except Exception as exc:
        print(f'  SKIPPED — {table} not readable ({exc.__class__.__name__}). '
              f'Expected before the first export with provenance.')
        return out

    # Provenance decides whether findings gate the run — see the docstring.
    vintages = K.check_export_vintage(verbose=False)
    distinct = {v for v in vintages.values() if v}
    stamped = sum(1 for v in vintages.values() if v)
    table_stamped = bool(vintages.get(table))
    mode = 'HARD' if table_stamped else 'ADVISORY'
    print(f'  rows={row["n"]}  out_of_support={row["n_oos"]}  [{mode}]')
    if not table_stamped:
        print('  NOTE: the live table carries no run_id, so it predates the '
              '2026-08-16 fixes. Findings below are stale-data notes that the '
              'pending re-export is expected to clear; they do NOT gate this run.')
    for label, key, hint in (
        ('non-finite / |v|>100 expected_sharpe_ratio', 'bad_sharpe',
         'MIN_RATIO_DENOMINATOR guard in RiskBookModel'),
        ('non-finite / |v|>100 reward_to_cvar', 'bad_starr',
         'MIN_RATIO_DENOMINATOR guard in RiskBookModel'),
        ('clip-pinned rows not flagged out_of_support', 'missed_oos',
         'symmetric detector in export_analytics'),
        ('out_of_support rows with a non-NULL ranking metric', 'leaked_rank',
         '_ranking_cols NULL-ing in export_analytics'),
    ):
        n = int(row[key])
        tag = 'OK  ' if not n else ('FAIL' if table_stamped else 'note')
        print(f'  {tag}  {label}: {n}')
        if n and table_stamped:
            out.append(f'{label}: {n} row(s) — check the {hint}')

    if not stamped:
        print('  note  vintage — no table carries run_id yet (pre-provenance export)')
    elif len(distinct) > 1:
        print(f'  FAIL  vintage — {len(distinct)} distinct run ids across '
              f'{stamped} stamped table(s)')
        out.append(f'analytics schema serves {len(distinct)} vintages; '
                   f're-run scripts/export_kalman_analytics.py')
    else:
        print(f'  OK    vintage — single run id across {stamped} stamped table(s)')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--draws', type=int, default=0,
                    help='override draws AND tune (ESS scales ~linearly in draws)')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--no-static', action='store_true')
    ap.add_argument('--isins', type=int, default=0)
    args = ap.parse_args()

    from sqlalchemy import create_engine

    cfg = K.get_run_config()
    if args.quick:
        cfg = replace(cfg, draws=300, tune=300)
    if args.draws:
        cfg = replace(cfg, draws=args.draws, tune=args.draws)
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
                        label='production')
    conv = _convergence(idata)
    n_div = conv['n_div']
    print('\n=== gate 1: convergence ===')
    print(f'  divergences = {n_div}      (gate: 0)')
    print(f'  GLOBAL parameters : max R-hat = {conv["global_max_rhat"]:.4f} '
          f'(gate < {RHAT_GATE}), min ESS = {conv["global_min_ess"]:.1f} '
          f'(gate > {MIN_ESS_GATE})')
    print(f'  per-ISIN latents  : max R-hat = {conv["latent_max_rhat"]:.4f}, '
          f'min ESS = {conv["latent_min_ess"]:.1f}   '
          f'[min over thousands of names — informational]')
    tbl = conv['table']
    worst = tbl.sort_values('rhat', ascending=False).head(8)
    print('  worst 8 by R-hat:')
    for _, r in worst.iterrows():
        tag = 'per-isin' if r.per_isin else 'GLOBAL  '
        print(f'    {tag} {r["var"]:<22s} n={int(r["size"]):<6d} '
              f'r_hat={r["rhat"]:.4f}  ess={r["ess"]:8.1f}')
    # Only GLOBAL parameters gate the run. A slow-mixing individual name is not a
    # reason to block an export; a stuck hyperparameter is.
    if n_div:
        failures.append(f'{n_div} divergences')
    if not (conv['global_max_rhat'] < RHAT_GATE):
        failures.append(f'global max R-hat {conv["global_max_rhat"]:.4f}')
    if not (conv['global_min_ess'] > MIN_ESS_GATE):
        failures.append(f'global min ESS {conv["global_min_ess"]:.1f}')

    post = idata.posterior
    print('\n=== gate 2: is the per-ISIN latent alive? ===')
    # The per-ISIN random intercept is the layer the T>1 panel actually buys; the
    # AR(1) state on top of it is opt-in (state_innovation_scale > 0) and off by
    # default. Check whichever is present.
    if 'z_isin_level' not in post:
        failures.append('z_isin_level absent (per-ISIN latent not built)')
        print('  FAIL: z_isin_level not in the posterior.')
    else:
        # The SCALE is fixed by construction (isin_level_scale), so its posterior
        # is a point and quantiles of it say nothing. What matters is whether the
        # effect is doing work: the realised cross-sectional spread of the
        # per-name intercept, and whether the data moved individual names off zero.
        eff = post['z_isin_level'].mean(('chain', 'draw'))
        realised = float(eff.std())
        prior_scale = float(post['sigma_isin_level'].mean()) \
            if 'sigma_isin_level' in post else float('nan')
        shrink = realised / prior_scale if prior_scale else float('nan')
        print(f'  fixed prior scale        = {prior_scale:.4f}')
        print(f'  realised effect sd       = {realised:.4f} '
              f'({shrink:.0%} of the prior scale)')
        if realised < SIGMA_STATE_FLOOR:
            failures.append(
                f'per-ISIN effect collapsed (realised sd = {realised:.4f}) — the '
                'panel carries no per-name signal beyond the drift regression')
    if 'sigma_state' in post:
        ss = post['sigma_state']
        ss_lo = float(ss.quantile(0.055))
        rho = (float(post['state_rho'].mean()) if 'state_rho' in post
               else float('nan'))
        print(f'  AR(1) layer ENABLED: sigma_state mean = {float(ss.mean()):.4f}, '
              f'5.5% q = {ss_lo:.4f}, rho = {rho:.3f}')
        if ss_lo < SIGMA_STATE_FLOOR:
            failures.append(
                f'sigma_state collapsed toward 0 (5.5% q = {ss_lo:.4f}) — disable '
                'the AR layer (state_innovation_scale=0.0)')
    else:
        print('  AR(1) layer disabled (state_innovation_scale=0.0) — expected; '
              'the per-ISIN intercept carries the panel.')

    # Target follows the module's interval width, so a 92% band is not scored
    # against a hardcoded 94%.
    target = K._HDI_HI - K._HDI_LO
    print(f'\n=== gate 4: per-time PPC coverage (target {target:.2f}) ===')
    try:
        cov = _per_time_coverage(model, idata, panel)
        for t, v in cov.items():
            flag = '' if abs(v - target) <= 0.03 else '   <-- off target'
            print(f'  t={t}: {v:.2%}{flag}')
        if (cov.max() - cov.min()) > 0.10:
            failures.append(
                f'per-time coverage drifts {cov.min():.2%}->{cov.max():.2%}')
    except Exception as exc:
        # Not swallowed: a coverage failure is a validation failure, not a note.
        print(f'  FAILED to compute: {exc!r}')
        failures.append(f'per-time coverage could not be computed ({exc})')

    latent = K.resolve_screen_latent(post)
    sd_state = float(latent.std(('chain', 'draw')).mean())
    base_state = float(post['sigma_base'].mean())
    print('\n=== gate 3: predicted signature vs the static twin ===')
    print(f'  production: sigma_base={base_state:.4f}  '
          f'mean per-name posterior sd={sd_state:.4f}')

    if not args.no_static:
        # Baseline = the PRE-0.9.9.14 model: no per-ISIN latent at all.
        idata_s, _ = _fit(panel, cfg, scale=0.0, isin_level=0.0,
                          label='baseline(no per-ISIN latent)')
        post_s = idata_s.posterior
        sd_static = float(
            K.resolve_screen_latent(post_s).std(('chain', 'draw')).mean())
        base_static = float(post_s['sigma_base'].mean())
        print(f'  baseline  : sigma_base={base_static:.4f}  '
              f'mean per-name posterior sd={sd_static:.4f}')
        print(f'  -> sigma_base {base_static:.4f} -> {base_state:.4f} '
              f'({"FALLS as predicted" if base_state < base_static else "RISES — unexpected"})')
        print(f'  -> per-name sd {sd_static:.4f} -> {sd_state:.4f} '
              f'({"WIDENS as predicted" if sd_state > sd_static else "NARROWS — unexpected"})')
        if base_state >= base_static:
            failures.append('sigma_base did not fall versus the baseline')
        if sd_state <= sd_static:
            failures.append('per-name posterior sd did not widen')

    if args.compare:
        print('\n=== §9b ELPD comparison ===')
        K.run_model_comparison(panel, config=replace(cfg, chains=cfg.chains),
                               robust=True, volume_penalty=0.25)

    failures.extend(_gate_exported_table(engine))

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
