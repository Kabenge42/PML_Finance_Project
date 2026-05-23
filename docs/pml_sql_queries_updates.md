Below is a complete, aligned set of changes for the three SQL files. All identifiers stay in the `pml` schema. The
strategy:

- Add a `feature_alias` column to `pml.pml_df_metadata` (the alias used inside each `mv_pymc_*` MV, e.g.
  `feat_eps_fy1e`, `n_beats`, `observed_pt`, `region`, `sector`, …).
- Populate `feature_alias` per `(column_name, model_target)` pair — because the *same* source column (e.g.
  `price_target`) is exposed as `observed_pt` in `mv_pymc_kalman_pt` / `mv_pymc_dcf_pt` but only feeds
  `feat_implied_upside` in `mv_pymc_price_target`. The cleanest way to keep `pml_df_metadata` keyed by `column_name`
  only is to store the alias *per model* in a side table `pml.pml_df_feature_alias` and surface it via the catalogue
  view. (Option B keeps the table simple and exact.) A simpler "good-enough" Option A stores a single canonical alias on
  `pml_df_metadata` — fine if you accept that aliases are model-scoped only when joined. Both are shown.
- Add the full classification block (
  `region, country, trading_country, exchange, unit, style_class, size_class, sector, industry`) to every `mv_pymc_*` MV
  and to every model's `model_targets` wiring + alias mapping.
- Update `vw_pymc_feature_catalogue` to expose `feature_alias`.

---

### 1) `pml_df_metadata.sql` — add `feature_alias`

```sql
DROP TABLE IF EXISTS pml.pml_df_metadata CASCADE;

CREATE TABLE IF NOT EXISTS pml.pml_df_metadata
(
    column_name      TEXT PRIMARY KEY,
    category         TEXT   NOT NULL DEFAULT 'n/a',
    feature_role     TEXT   NOT NULL,
    feature_alias    TEXT, -- NEW: canonical MV alias
    ordinal_position INTEGER,
    description      TEXT,
    data_type        TEXT,
    pymc_role        TEXT,
    model_targets    TEXT[] NOT NULL DEFAULT ARRAY []::TEXT[],
    updated_at       TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_feature_role ON pml.pml_df_metadata (feature_role);
CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_category ON pml.pml_df_metadata (category);
CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_pymc_role ON pml.pml_df_metadata (pymc_role);
CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_feature_alias ON pml.pml_df_metadata (feature_alias);
CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_model_targets ON pml.pml_df_metadata USING gin (model_targets);

-- Optional (recommended): per-model alias side table, since the same source
-- column maps to different aliases across MVs (e.g. price_target ->
-- observed_pt in kalman_pt/dcf_pt and feat_implied_upside numerator in
-- price_target). Keep pml_df_metadata.feature_alias as the *default*
-- alias and override per model here.
CREATE TABLE IF NOT EXISTS pml.pml_df_feature_alias
(
    column_name   TEXT NOT NULL REFERENCES pml.pml_df_metadata (column_name) ON DELETE CASCADE,
    model_target  TEXT NOT NULL,
    feature_alias TEXT NOT NULL,
    PRIMARY KEY (column_name, model_target)
);

CREATE INDEX IF NOT EXISTS idx_pml_df_feature_alias_model ON pml.pml_df_feature_alias (model_target);

COMMENT ON COLUMN pml.pml_df_metadata.feature_alias IS 'Default (model-agnostic) alias used inside pml.mv_pymc_* materialized views. For model-specific overrides, see pml.pml_df_feature_alias.';
```

(Leave the existing `COMMENT ON TABLE` and `pymc_role` / `model_targets` comments as in the current file.)

---

### 2) `pml_df_metadata_populate.sql` — populate `feature_alias` + wire classification columns + per-model overrides

Add the following blocks **before `COMMIT;`** (keep all existing logic).

#### 2a. Wire classification columns into every model

```sql
-- Classification columns are coords for ALL pymc models.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT
                                         unnest(model_targets ||
                                                ARRAY ['earnings_beat','price_target','kalman_pt', 'dcf_pt','dividend_safety','credit_risk', 'accounting_anomaly']))
                    )
WHERE column_name IN ('region', 'country', 'trading_country', 'exchange',
                      'unit', 'style_class', 'size_class', 'sector', 'industry');
```

#### 2b. Default `feature_alias` (model-agnostic) for classification coords

```sql
UPDATE pml.pml_df_metadata
SET feature_alias = column_name
WHERE column_name IN ('isin', 'ticker',
                      'region', 'country', 'trading_country', 'exchange',
                      'unit', 'style_class', 'size_class', 'sector', 'industry');
```

#### 2c. Per-model alias overrides (mirror the MV column aliases exactly)

```sql
-- Clear any stale rows so reruns are idempotent
TRUNCATE pml.pml_df_feature_alias;

INSERT INTO pml.pml_df_feature_alias (column_name,                        model_target,         feature_alias                    )
VALUES
    -- Coords shared by all MVs (classification block)
                                     ('isin',                             'earnings_beat',      'isin'                           ),
                                     ('ticker',                           'earnings_beat',      'ticker'                         ),
                                     ('region',                           'earnings_beat',      'region'                         ),
                                     ('country',                          'earnings_beat',      'country'                        ),
                                     ('trading_country',                  'earnings_beat',      'trading_country'                ),
                                     ('exchange',                         'earnings_beat',      'exchange'                       ),
                                     ('unit',                             'earnings_beat',      'unit'                           ),
                                     ('style_class',                      'earnings_beat',      'style_class'                    ),
                                     ('size_class',                       'earnings_beat',      'size_class'                     ),
                                     ('sector',                           'earnings_beat',      'sector'                         ),
                                     ('industry',                         'earnings_beat',      'industry'                       ),

    -- ---- earnings_beat aliases (from mv_pymc_earnings_beat) ----
                                     ('eps_norm_est_avg_fy1e',            'earnings_beat',      'feat_eps_fy1e'                  ),
                                     ('eps_norm_est_avg_fq1e',            'earnings_beat',      'feat_eps_fq1e'                  ),
                                     ('eps_norm_est_num_fy1e',            'earnings_beat',      'n_eps_estimates'                ),
                                     ('eps_est_avg_rev_pct_fy1e_1w',      'earnings_beat',      'feat_rev_1w'                    ),
                                     ('eps_est_avg_rev_pct_fy1e_1m',      'earnings_beat',      'feat_rev_1m'                    ),
                                     ('eps_est_avg_rev_pct_fy1e_3m',      'earnings_beat',      'feat_rev_3m'                    ),
                                     ('eps_est_avg_rev_pct_fy1e_6m',      'earnings_beat',      'feat_rev_6m'                    ),
                                     ('eps_est_avg_rev_pct_fy1e_1y',      'earnings_beat',      'feat_rev_1y'                    ),
                                     ('eps_gaap_est_avg_rev_pct_fy1e_3m', 'earnings_beat',      'feat_rev_gaap_gap_3m'           ),
                                     ('eps_neg0fqsurprise_pct',           'earnings_beat',      'feat_last_q_surprise'           ),
                                     ('eps_neg0fysurprise_pct',           'earnings_beat',      'feat_last_y_surprise'           ),
                                     ('days_to_earnings',                 'earnings_beat',      'feat_days_to_earnings'          ),
                                     ('earnings_report_recency',          'earnings_beat',      'feat_report_recency'            ),
                                     ('next_earnings_status',             'earnings_beat',      'feat_next_earnings_status'      ),

    -- ---- price_target aliases (from mv_pymc_price_target) ----
                                     ('target_pct_avg',                   'price_target',       'observed_target_pct'            ),
                                     ('target_pct_med',                   'price_target',       'observed_target_pct_med'        ),
                                     ('last_price',                       'price_target',       'last_price'                     ),
                                     ('price_target_num',                 'price_target',       'n_analysts'                     ),
                                     ('num_hold_ratings',                 'price_target',       'feat_holds'                     ),
                                     ('num_no_opinion_ratings',           'price_target',       'feat_no_opinion'                ),
                                     ('price_target',                     'price_target',       'feat_implied_upside'            ),
                                     ('price_target_stddev',              'price_target',       'feat_target_dispersion_cv'      ),
                                     ('p_e_ntm',                          'price_target',       'feat_pe_ntm'                    ),
                                     ('ev_ebitda_ntm',                    'price_target',       'feat_ev_ebitda_ntm'             ),
                                     ('volatility_3m',                    'price_target',       'feat_vol_3m'                    ),
                                     ('analyst_rating',                   'price_target',       'feat_analyst_rating'            ),
                                     ('w_52high_adj',                     'price_target',       'feat_52w_range_position_high'   ),
                                     ('w_52low_adj',                      'price_target',       'feat_52w_range_position_low'    ),
                                     ('price_target_3m_ago',              'price_target',       'feat_pt_momentum_3m'            ),
                                     ('price_target_num_3m_ago',          'price_target',       'feat_coverage_change_3m'        ),
                                     ('target_pct_high',                  'price_target',       'feat_target_range_width_high'   ),
                                     ('target_pct_low',                   'price_target',       'feat_target_range_width_low'    ),

    -- ---- kalman_pt aliases (from mv_pymc_kalman_pt) ----
                                     ('price_target',                     'kalman_pt',          'observed_pt'                    ),
                                     ('last_price',                       'kalman_pt',          'last_price'                     ),
                                     ('price_target_high',                'kalman_pt',          'price_target_high'              ),
                                     ('price_target_low',                 'kalman_pt',          'price_target_low'               ),
                                     ('price_target_num',                 'kalman_pt',          'n_analysts'                     ),
                                     ('price_target_stddev',              'kalman_pt',          'feat_pt_noise_sigma'            ),
                                     ('volatility_1m',                    'kalman_pt',          'feat_vol_1m'                    ),
                                     ('volatility_3m',                    'kalman_pt',          'feat_vol_3m'                    ),
                                     ('volatility_6m',                    'kalman_pt',          'feat_vol_6m'                    ),
                                     ('volatility_1y',                    'kalman_pt',          'feat_vol_1y'                    ),
                                     ('total_return_ytd',                 'kalman_pt',          'feat_total_return_ytd'          ),

    -- ---- dcf_pt aliases (from mv_pymc_dcf_pt) ----
                                     ('price_target',                     'dcf_pt',             'observed_pt'                    ),
                                     ('market_cap',                       'dcf_pt',             'market_cap'                     ),
                                     ('enterprise_value',                 'dcf_pt',             'enterprise_value'               ),
                                     ('shrs_out',                         'dcf_pt',             'shrs_out'                       ),
                                     ('fcf_ltm',                          'dcf_pt',             'feat_fcf_ltm'                   ),
                                     ('fcf_est_avg_fy1e',                 'dcf_pt',             'feat_fcf_fy1e'                  ),
                                     ('fcf_est_avg_fy2e',                 'dcf_pt',             'feat_fcf_fy2e'                  ),
                                     ('fcf_est_avg_fy3e',                 'dcf_pt',             'feat_fcf_fy3e'                  ),
                                     ('fcf_est_avg_fy4e',                 'dcf_pt',             'feat_fcf_fy4e'                  ),
                                     ('fcf_est_avg_fy5e',                 'dcf_pt',             'feat_fcf_fy5e'                  ),
                                     ('cfo_ltm',                          'dcf_pt',             'feat_cfo_ltm'                   ),
                                     ('capital_expenditure_ltm',          'dcf_pt',             'feat_capex_to_fcf'              ),
                                     ('tot_return_pct_cagr_3y',           'dcf_pt',             'feat_tr_cagr_3y'                ),
                                     ('tot_return_pct_cagr_10y',          'dcf_pt',             'feat_tr_cagr_10y'               ),
                                     ('peg_ntm',                          'dcf_pt',             'feat_peg_ntm'                   ),
                                     ('ev_sales_ltm',                     'dcf_pt',             'feat_ev_sales_ltm'              ),
                                     ('ev_ebitda_ntm',                    'dcf_pt',             'feat_ev_ebitda_ntm'             ),
                                     ('return_on_assets_roa_pct_ltm',     'dcf_pt',             'feat_roa_ltm'                   ),
                                     ('gross_profit_margin_pct_ltm',      'dcf_pt',             'feat_gpm_ltm'                   ),
                                     ('beta_5y',                          'dcf_pt',             'feat_beta_5y'                   ),

    -- ---- dividend_safety aliases (from mv_pymc_dividend_safety) ----
                                     ('div_yield_ltm',                    'dividend_safety',    'observed_div_yield'             ),
                                     ('dividend_streak',                  'dividend_safety',    'n_streak'                       ),
                                     ('dividend_record_frequency',        'dividend_safety',    'feat_div_frequency'             ),
                                     ('fcf_ltm',                          'dividend_safety',    'feat_fcf_coverage'              ),
                                     ('cfo_ltm',                          'dividend_safety',    'feat_cfo_coverage'              ),
                                     ('common_dividends_paid_ltm',        'dividend_safety',    'feat_fcf_coverage_denom'        ),
                                     ('dividend_per_share_ltm',           'dividend_safety',    'feat_eps_payout_ratio'          ),
                                     ('net_eps_basic_ltm',                'dividend_safety',    'feat_eps_payout_ratio_denom'    ),
                                     ('dividend_per_share_neg1fy',        'dividend_safety',    'feat_dps_growth_1y'             ),
                                     ('dividend_per_share_neg3fy',        'dividend_safety',    'feat_dps_growth_3y'             ),
                                     ('dividend_per_share_neg5fy',        'dividend_safety',    'feat_dps_growth_5y'             ),
                                     ('buyback_yield_ltm',                'dividend_safety',    'feat_buyback_yield'             ),
                                     ('repurchase_common_stock_ltm',      'dividend_safety',    'feat_repurchases_ltm'           ),
                                     ('altman_z_score_ltm',               'dividend_safety',    'feat_altman_z'                  ),
                                     ('return_on_assets_roa_pct_ltm',     'dividend_safety',    'feat_roa_ltm'                   ),
                                     ('div_yield_5yavgltm',               'dividend_safety',    'feat_yield_spread_vs_5y'        ),

    -- ---- credit_risk aliases (from mv_pymc_credit_risk) ----
                                     ('altman_z_score_ltm',               'credit_risk',        'observed_altman_z'              ),
                                     ('altman_z_score_neg1fy',            'credit_risk',        'feat_z_trend_1y'                ),
                                     ('altman_z_score_neg3fy',            'credit_risk',        'feat_z_trend_3y'                ),
                                     ('cfo_ltm',                          'credit_risk',        'feat_cfo_capex_cov'             ),
                                     ('capital_expenditure_ltm',          'credit_risk',        'feat_cfo_capex_cov_denom'       ),
                                     ('fcf_ltm',                          'credit_risk',        'feat_fcf_yield'                 ),
                                     ('enterprise_value',                 'credit_risk',        'feat_fcf_yield_denom'           ),
                                     ('cff_ltm',                          'credit_risk',        'feat_cff_to_ev'                 ),
                                     ('issuance_common_stock_ltm',        'credit_risk',        'feat_net_equity_issuance'       ),
                                     ('repurchase_common_stock_ltm',      'credit_risk',        'feat_net_equity_issuance_offset'),
                                     ('market_cap',                       'credit_risk',        'feat_net_equity_issuance_denom' ),
                                     ('full_time_employees_fy',           'credit_risk',        'feat_employee_growth_1y'        ),
                                     ('full_time_employees_neg1fy',       'credit_risk',        'feat_employee_growth_1y_lag'    ),
                                     ('p_b_ltm',                          'credit_risk',        'feat_pb_ltm'                    ),
                                     ('beta_2y',                          'credit_risk',        'feat_beta_2y'                   ),
                                     ('volatility_6m',                    'credit_risk',        'feat_vol_6m'                    ),
                                     ('volatility_1y',                    'credit_risk',        'feat_vol_1y'                    ),

    -- ---- accounting_anomaly aliases (from mv_pymc_accounting_anomaly) ----
                                     ('eps_adj_ltm',                      'accounting_anomaly', 'observed_eps_adj'               ),
                                     ('net_eps_basic_ltm',                'accounting_anomaly', 'feat_accruals_ratio_ni'         ),
                                     ('cfo_ltm',                          'accounting_anomaly', 'feat_accruals_ratio_cfo'        ),
                                     ('enterprise_value',                 'accounting_anomaly', 'feat_accruals_ratio_scale'      ),
                                     ('gross_profit_margin_pct_ltm',      'accounting_anomaly', 'feat_gpm_change_1y'             ),
                                     ('gross_profit_margin_pct_neg1fy',   'accounting_anomaly', 'feat_gpm_change_1y_lag'         ),
                                     ('sales_neg0fyactual',               'accounting_anomaly', 'feat_sales_growth_1y'           ),
                                     ('sales_neg1fyactual',               'accounting_anomaly', 'feat_sales_growth_1y_lag'       ),
                                     ('ebit_neg0fyactual',                'accounting_anomaly', 'feat_ebit_growth_1y'            ),
                                     ('ebit_neg1fyactual',                'accounting_anomaly', 'feat_ebit_growth_1y_lag'        ),
                                     ('ebitda_neg0fyactual',              'accounting_anomaly', 'feat_ebitda_growth_1y'          ),
                                     ('ebitda_neg1fyactual',              'accounting_anomaly', 'feat_ebitda_growth_1y_lag'      ),
                                     ('capital_expenditure_ltm',          'accounting_anomaly', 'feat_capex_intensity'           ),
                                     ('cfi_ltm',                          'accounting_anomaly', 'feat_cfi_to_cfo'                ),
                                     ('cff_ltm',                          'accounting_anomaly', 'feat_cff_to_cfo'                ),
                                     ('shrs_out',                         'accounting_anomaly', 'feat_share_inflation_1y'        ),
                                     ('shrs_out_neg1fy',                  'accounting_anomaly', 'feat_share_inflation_1y_lag'    ),
                                     ('issuance_common_stock_ltm',        'accounting_anomaly', 'feat_issuance_intensity'        ),
                                     ('market_cap',                       'accounting_anomaly', 'feat_issuance_intensity_denom'  ),
                                     ('full_time_employees_fy',           'accounting_anomaly', 'feat_employee_growth_1y'        ),
                                     ('full_time_employees_neg1fy',       'accounting_anomaly', 'feat_employee_growth_1y_lag'    ),
                                     ('fcf_per_share_ltm',                'accounting_anomaly', 'feat_fcfps_vs_eps_gap'          ),
                                     ('peg_ntm',                          'accounting_anomaly', 'feat_peg_ntm'                   );

-- Replicate classification coords for every remaining model (price_target,
-- kalman_pt, dcf_pt, dividend_safety, credit_risk, accounting_anomaly).
INSERT INTO pml.pml_df_feature_alias (column_name, model_target, feature_alias)
SELECT col, m, col
FROM unnest(ARRAY ['isin','ticker','region','country','trading_country', 'exchange','unit','style_class','size_class','sector','industry']) col
         CROSS JOIN unnest(ARRAY ['price_target','kalman_pt','dcf_pt', 'dividend_safety','credit_risk','accounting_anomaly'])               m
ON CONFLICT DO NOTHING;
```

#### 2d. Rebuild `vw_pymc_feature_catalogue` to expose `feature_alias`

Replace the existing view in `pml_feature_catalogue.sql` (lines 559–569) with:

```sql
CREATE OR REPLACE VIEW pml.vw_pymc_feature_catalogue AS
SELECT m.model_name                                                 AS model_target,
       md.pymc_role,
       md.column_name,
       md.category,
       md.feature_role,
       COALESCE(fa.feature_alias, md.feature_alias, md.column_name) AS feature_alias,
       md.data_type,
       md.description
FROM pml.pml_df_metadata                                md
         CROSS JOIN LATERAL unnest(md.model_targets) AS m(model_name)
         LEFT JOIN  pml.pml_df_feature_alias            fa
                    ON fa.column_name = md.column_name AND fa.model_target = m.model_name
WHERE md.pymc_role <> 'excluded';
```

---

### 3) `pml_feature_catalogue.sql` — add classification columns to every per-model MV

For each `CREATE MATERIALIZED VIEW pml.mv_pymc_*`, expand the projected coord list from `isin, ticker, sector, industry`
to the full set. Pattern (apply to all 7 MVs):

```sql
SELECT isin,
       ticker,
       region,
       country,
       trading_country,
       exchange,
       unit,
       style_class,
       size_class,
       sector,
       industry, ... -- existing observed_* / feat_* / n_* columns
    FROM pml.pml_df;
```

Concretely:

- `mv_pymc_earnings_beat` — extend the `WITH beats AS (SELECT ... FROM pml.pml_df)` CTE to include the new columns, and
  add them to the final
  `SELECT b.isin, b.ticker, b.region, b.country, b.trading_country, b.exchange, b.unit, b.style_class, b.size_class, b.sector, b.industry, ...`.
- `mv_pymc_price_target`, `mv_pymc_kalman_pt`, `mv_pymc_dcf_pt`, `mv_pymc_dividend_safety`, `mv_pymc_credit_risk`,
  `mv_pymc_accounting_anomaly` — add the same 9 classification columns directly after `ticker` in their `SELECT` lists.

Example for `mv_pymc_price_target`:

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS pml.mv_pymc_price_target AS
SELECT isin,
       ticker,
       region,
       country,
       trading_country,
       exchange,
       unit,
       style_class,
       size_class,
       sector,
       industry,
       target_pct_avg   AS observed_target_pct,
       target_pct_med   AS observed_target_pct_med,
       last_price,
       price_target_num AS n_analysts,
...FROM pml.pml_df;
```

Example for `mv_pymc_earnings_beat` (CTE + outer SELECT):

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS pml.mv_pymc_earnings_beat AS
WITH beats AS (SELECT isin,
                      ticker,
                      region,
                      country,
                      trading_country,
                      exchange,
                      unit,
                      style_class,
                      size_class,
                      sector,
                      industry,
                      ARRAY [eps_neg0fqsurprise_pct, ...] AS eps_surprises_q, ...FROM pml.pml_df
              )
SELECT b.isin,
       b.ticker,
       b.region,
       b.country,
       b.trading_country,
       b.exchange,
       b.unit,
       b.style_class,
       b.size_class,
       b.sector,
       b.industry,
       bc_q.n_total AS n_total,
        ...FROM beats b, LATERAL pml.beat_counts(b.eps_surprises_q::NUMERIC []) bc_q, LATERAL pml.beat_counts(b.eps_surprises_y::NUMERIC []) bc_y;
```

Apply the same pattern to `mv_pymc_kalman_pt`, `mv_pymc_dcf_pt`, `mv_pymc_dividend_safety`, `mv_pymc_credit_risk`,
`mv_pymc_accounting_anomaly`.

---

### Verification queries

After applying the changes and `REFRESH MATERIALIZED VIEW pml.mv_pymc_*`:

```sql
-- 1) All 9 classification columns present on every MV
SELECT table_name,
       COUNT(*) FILTER (WHERE column_name IN ('region', 'country', 'trading_country', 'exchange', 'unit',
                                              'style_class', 'size_class', 'sector', 'industry')) AS n_class_cols
FROM information_schema.columns
WHERE table_schema = 'pml'
  AND table_name LIKE 'mv_pymc_%'
GROUP BY table_name;
-- expect n_class_cols = 9 for every MV.

-- 2) feature_alias coverage in the catalogue view
SELECT model_target, COUNT(*) AS n_rows, COUNT(feature_alias) AS n_with_alias
FROM pml.vw_pymc_feature_catalogue
GROUP BY model_target
ORDER BY model_target;

-- 3) Aliases match MV column names for a given model
SELECT column_name, feature_alias
FROM pml.vw_pymc_feature_catalogue
WHERE model_target = 'earnings_beat'
ORDER BY column_name;
```

These three coordinated edits keep the `postgres.pml` schema aligned: every classification coord is materialised inside
each `mv_pymc_*`, wired into `model_targets`, exposed with a deterministic `feature_alias`, and surfaced through
`vw_pymc_feature_catalogue` for the notebook's `MODEL_FEATURE_CONTAINERS` registry.