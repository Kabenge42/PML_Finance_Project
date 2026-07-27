import json

path = 'pymc_kalman_filter_pt.ipynb'
nb = json.load(open(path, encoding='utf-8'))

md_source = r"""## 14. Comprehensive Summary - Recent Earnings Period vs Historical Data

This closing section consolidates the notebook's outputs into a single decision-oriented read
on the **expected price targets for names reporting within +/- 5 days** (`next_earnings`),
benchmarked against the broader **historical / baseline** data:

- **Cross-sectional benchmark** - the earnings cohort vs the rest of the modelled universe
  (the names *not* reporting this week), on expected upside, share positive, credible-band
  width (uncertainty) and Kalman shrinkage vs raw consensus - read from the section-10
  `results` table.
- **Time-series benchmark** - the mingled cohort's analyst-target trail reconstructed from the
  embedded `*_ago` history (section 12): the oldest historical consensus vs the most recent,
  and the latest Kalman-smoothed target's implied upside vs `last_price`.
- **Distributional view** - an `arviz_plots` KDE overlay of the posterior cohort-average vs
  universe-average expected upside, with a 0 % break-even reference line.

All blocks are guarded: the summary degrades to whatever upstream artifacts (`results`,
`cohort_meta`, `comparison`) are present in the kernel, so it still produces a partial read if
the DB-dependent sections were skipped.
"""

code_source = r'''# Section 14: comprehensive earnings-cohort vs historical-data summary. Reuses `results`
# (section 10), `cohort_meta` (section 13), `comparison` (section 12) and `idata` when present.
def _fmt(x, nd=1, suf=''):
    """Format a possibly-NaN/None scalar for narrative output."""
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return 'n/a'
        return f'{x:.{nd}f}{suf}'
    except Exception:
        return 'n/a'


def _label(row):
    t = row.get('ticker')
    return t if isinstance(t, str) and t.strip() else str(row['isin'])


def _band_width_pct(df):
    denom = df['expected_pt'].replace(0, np.nan)
    return (df['expected_pt_hdi_hi'] - df['expected_pt_hdi_lo']) / denom * 100.0


def _shrink_pct(df):
    denom = df['observed_pt'].replace(0, np.nan)
    return (df['expected_pt'] / denom - 1.0) * 100.0


try:
    results
except NameError:
    print('Section 10 `results` not in scope - run sections 5-10 first.')
else:
    universe = results.copy()

    # ---- A. Cross-sectional: earnings cohort vs the rest of the modelled universe ----
    try:
        _cohort_ids = set(cohort_meta['isin'].astype(str))
    except NameError:
        _cohort_ids = None

    if _cohort_ids:
        _in = universe['isin'].astype(str).isin(_cohort_ids)
        cohort, rest = universe[_in].copy(), universe[~_in].copy()
        groups = [('Earnings cohort (+/-5d)', cohort),
                  ('Historical baseline (not reporting)', rest),
                  ('Full universe', universe)]
    else:
        cohort = rest = None
        groups = [('Full universe', universe)]
        print('Section 13 `cohort_meta` not in scope - showing the universe only. '
              'Run Section 13 to populate the earnings cohort comparison.')

    _rows = []
    for label, df in groups:
        if df is None or len(df) == 0:
            continue
        _rows.append({
            'group': label,
            'n_names': int(len(df)),
            'median_upside_%': df['expected_upside_pct'].median(),
            'mean_upside_%': df['expected_upside_pct'].mean(),
            'positive_upside_%': (df['expected_upside_pct'] > 0).mean() * 100.0,
            'median_band_width_%': _band_width_pct(df).median(),
            'median_shrink_vs_consensus_%': _shrink_pct(df).median(),
            'median_n_analysts': (df['n_analysts'].median()
                                  if 'n_analysts' in df else np.nan),
        })
    summary_tbl = pd.DataFrame(_rows).set_index('group').round(2)
    print('Cross-sectional summary - expected price targets by group:')
    display(summary_tbl)

    # Cohort sector tilt (composition of the names reporting this week).
    if cohort is not None and len(cohort) and 'sector' in cohort.columns:
        sector_mix = (cohort.assign(sector=cohort['sector'].fillna('Unknown'))
                      .groupby('sector')
                      .agg(n=('isin', 'size'),
                           median_upside_pct=('expected_upside_pct', 'median'))
                      .sort_values('n', ascending=False).round(2))
        print('\nEarnings-cohort sector tilt:')
        display(sector_mix.head(10))

    # ---- B. Time-series: recent vs historical mingled cohort price-target trail ----
    try:
        _cmp = comparison
    except NameError:
        _cmp = None
    hist_drift = implied_now = first = last = None
    if _cmp is not None and len(_cmp) >= 2:
        first, last = _cmp.iloc[0], _cmp.iloc[-1]
        hist_drift = ((last['observed_pt'] / first['observed_pt'] - 1.0) * 100.0
                      if first['observed_pt'] else np.nan)
        implied_now = ((last['expected_pt'] / last['last_price'] - 1.0) * 100.0
                       if last['last_price'] else np.nan)

    # ---- C. Headline narrative ----
    print('\n' + '=' * 74)
    print('KEY INSIGHTS - recent earnings period vs historical data')
    print('=' * 74)
    if cohort is not None and len(cohort):
        cu = cohort['expected_upside_pct'].median()
        ru = rest['expected_upside_pct'].median() if rest is not None and len(rest) else np.nan
        print(f'- {len(cohort)} names report within +/-5d. Median expected upside '
              f'{_fmt(cu, 1, "%")} vs {_fmt(ru, 1, "%")} for non-reporting names '
              f'(delta {_fmt(cu - ru, 1, " pp")}).')
        print(f'- {_fmt((cohort["expected_upside_pct"] > 0).mean() * 100, 0, "%")} of the cohort '
              f'has positive expected upside; median credible band width '
              f'{_fmt(_band_width_pct(cohort).median(), 1, "%")} '
              f'(universe {_fmt(_band_width_pct(universe).median(), 1, "%")}).')
        sh = _shrink_pct(cohort).median()
        print(f'- Kalman-smoothed targets sit {_fmt(abs(sh), 1, "%")} '
              f'{"above" if sh >= 0 else "below"} raw consensus (median) - shrinkage toward '
              f'the hierarchical group mean.')
        _top = cohort.sort_values('expected_upside_pct', ascending=False).head(3)
        _bot = cohort.sort_values('expected_upside_pct').head(3)
        _names = lambda d: ', '.join(f'{_label(r)} ({_fmt(r["expected_upside_pct"], 0, "%")})'
                                     for _, r in d.iterrows())
        print(f'- Highest expected upside: {_names(_top)}.')
        print(f'- Lowest / most downside : {_names(_bot)}.')
    else:
        print('- Earnings cohort not available (Section 13 was skipped); '
              'cross-sectional cohort insights omitted.')
    if first is not None and last is not None:
        print(f'- Historical target trail ({first["asof_date"]} -> {last["asof_date"]}): mingled '
              f'cohort consensus {"rose" if (hist_drift or 0) >= 0 else "fell"} '
              f'{_fmt(abs(hist_drift), 1, "%")}; latest Kalman-smoothed target implies '
              f'{_fmt(implied_now, 1, "%")} upside vs cohort last price.')
    else:
        print('- Historical mingled trail not available (Section 12 was skipped); '
              'time-series drift insight omitted.')

    # ---- D. Distributional view: cohort vs universe expected upside (arviz_plots KDE) ----
    try:
        if _cohort_ids:
            _eu = idata.posterior['expected_upside'] * 100.0
            _modelled = set(_eu.coords['isin'].values.astype(str).tolist())
            _cohort_post = [i for i in _cohort_ids if i in _modelled]
            if _cohort_post:
                _cohort_avg = _eu.sel(isin=_cohort_post).mean('isin')
                _univ_avg = _eu.mean('isin')
                _stacked = xr.concat(
                    [_cohort_avg, _univ_avg],
                    dim=pd.Index(['earnings_cohort', 'universe'], name='group'),
                ).rename('avg_expected_upside_pct')
                pc_sum = azp.plot_dist(
                    _stacked.to_dataset(), kind='kde',
                    var_names=['avg_expected_upside_pct'],
                    sample_dims=['chain', 'draw'], backend='matplotlib',
                )
                pc_sum.add_title('Expected upside (%): earnings cohort vs universe '
                                 '(posterior cross-sectional average)')
                pc_sum = azp.add_lines(
                    pc_sum, values=0.0,
                    visuals={'ref_line': {'color': '#bbbbbb',
                                          'linestyle': '--', 'linewidth': 1.3}},
                )
                pc_sum.show()
    except Exception as _e:  # pragma: no cover - plot is best-effort
        print(f'Summary KDE overlay skipped: {_e!r}')
'''

def to_lines(s):
    return s.splitlines(keepends=True)

cells = [
    {'cell_type': 'markdown', 'metadata': {}, 'source': to_lines(md_source),
     'id': 'sec14summarymd001'},
    {'cell_type': 'code', 'metadata': {}, 'source': to_lines(code_source),
     'id': 'sec14summarycd001', 'outputs': [], 'execution_count': None},
]

nb['cells'].extend(cells)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print('Appended', len(cells), 'cells; total now', len(nb['cells']))
