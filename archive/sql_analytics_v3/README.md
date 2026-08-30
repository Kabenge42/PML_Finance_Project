# Archived v3 analytics DDL

Schema snapshots (`create table analytics.<name> (...)`) for the tables the **v3**
pipeline writes. Moved here from `sql_scripts/analytics/` on 2026-08-31 so that
directory holds only what the live v2 workflow produces.

## Why archived rather than deleted

The pipeline that writes these tables is still reachable in the package —
`probabilistic_ml_model/pipeline_runners.py` (`_EXPORT_PAIRS`, `_SCREEN_TABLE_MAP`)
and `analytics/data_utils.py` (`ANALYTICS_EXPORT_TABLES`, `export_to_analytics_db`)
— so these files document behaviour that can still happen. What *was* retired is
the entry point: `expected_returns_v3.py` sits in `archive/` beside this
directory, and the `finance-ml*` console scripts in `pyproject.toml` do not
resolve because `cli.py` was never written. So there is no supported way to run
the v3 pipeline today, but the machinery has not been removed.

None of these tables currently exists in the database; `analytics` holds only the
`*_v2` set.

## What is NOT here

- `sql_scripts/analytics/*_v2.sql` — the live v2 export DDL, regenerated on every
  run by `write_analytics_ddl_v2`.
- `sql_scripts/analytics/kalman_panel_vintage.sql` — `analytics.panel_vintage_v2`,
  the point-in-time store the v2 vintage harness uses. Forward-looking, not legacy.
- The six `kalman_filtered_price_targets*.sql` files — **deleted**, not archived.
  They were GEIB's v1 DDL, superseded when the board moved to the v2 table, and
  the canonical one is a generated artifact that a v1 run recreates anyway.

## Restoring one

These are tracked, so they are simply files. To put one back:

```
git mv archive/sql_analytics_v3/<name>.sql sql_scripts/analytics/<name>.sql
```
