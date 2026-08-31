#!/usr/bin/env python
"""Capture today's Kalman v2 panel and decision outputs as a dated vintage.

Why
---
The price and price-target trails on ``pml.mv_pymc_kalman_pt_v2`` are not
versioned, and the seven pipeline analytics tables are DROP-and-RECREATEd on
every export. So the state the model saw on any given day is unrecoverable the
moment the next run finishes — which is why no gate in the workflow scores the
model against a realised return, and why a pass-through could clear 19 of 21
gates on run ``49e84d7e9d59``.

This writes one row per ISIN into ``analytics.panel_vintage_v2``, keyed by an
as-of date. Two captures far enough apart give
``scripts/score_panel_vintages.py`` something to score.

Run it **after** an export, on a schedule (monthly or quarterly is plenty — the
OU length scale is ~85 days). It does not fit anything and takes seconds.

Usage
-----
::

    . .\\set_env.ps1
    $env:PYTHONIOENCODING = "utf-8"
    ~\\AppData\\Local\\Python\\bin\\python.exe scripts\\capture_panel_vintage.py
    ~\\AppData\\Local\\Python\\bin\\python.exe scripts\\capture_panel_vintage.py --dry-run

Options::

    --asof YYYY-MM-DD   date to key the capture on (default: today)
    --run-id ID         export vintage to read (default: the latest)
    --replace           overwrite an existing asof_date instead of refusing
    --dry-run           build the frame and report, write nothing
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_VINTAGE_TABLE = "panel_vintage_v2"
_PANEL_TABLE = "04_panel_frame_v2"
_RESULTS_TABLE = "kalman_filtered_price_targets_v2"
# Renamed from `kalman_panel_vintage.sql` on 2026-08-31. The old name was
# still referenced here, so `ensure_table` would have exited with "missing
# DDL" on any database that did not already have the table -- which is the
# only case it exists to handle.
_DDL_PATH = _REPO_ROOT / "sql_scripts" / "analytics" / "panel_vintage_v2.sql"

#: Columns taken from the panel frame. Everything here is point-in-time by
#: construction and unrecoverable afterwards.
_PANEL_COLS = [
    "isin", "ticker", "name", "sector", "industry", "trading_region", "country",
    "style_class", "size_class", "market_cap",
    "last_price", "observed_pt", "n_analysts",
    "price_1w_ago", "price_1m_ago", "price_3m_ago", "price_6m_ago", "price_1y_ago",
    "price_target_1w_ago", "price_target_1m_ago", "price_target_3m_ago",
    "price_target_6m_ago", "price_target_1y_ago",
    "n_analysts_1w", "n_analysts_1m", "n_analysts_3m", "n_analysts_6m",
    "n_analysts_1y",
]

#: Columns taken from the exported decision table, so a vintage can be scored
#: without the run that produced it still existing.
_RESULT_COLS = [
    "isin", "implied_upside", "expected_return_kalman", "expected_upside_sd",
    "shrink_gain", "er_mean", "er_sd", "er_p05", "er_p50", "er_p95",
    "mc_prob_pos", "p_upside_pos_cond", "cvar_5pct_kalman", "out_of_support",
]


def _engine():
    from sqlalchemy import create_engine

    url = os.environ.get("DB_URL")
    if not url:
        raise SystemExit("DB_URL is not set. Run '. .\\set_env.ps1' first.")
    return create_engine(url)


def _table_exists(conn, table: str) -> bool:
    from sqlalchemy import text

    return bool(
        conn.execute(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'analytics' AND table_name = :t"
            ),
            {"t": table},
        ).scalar()
    )


def ensure_table(conn) -> None:
    """Create ``analytics.panel_vintage_v2`` from its checked-in DDL if absent.

    The DDL is the source of truth and is ``CREATE TABLE IF NOT EXISTS``, so
    running it against a live table is a no-op — but note that it will NOT pick
    up a changed definition, the same trap the ``mv_pymc_*`` views carry. Drop
    the table first if the schema moves.
    """
    from sqlalchemy import text

    if _table_exists(conn, _VINTAGE_TABLE):
        return
    if not _DDL_PATH.exists():
        raise SystemExit(f"missing DDL: {_DDL_PATH}")
    print(f"creating analytics.{_VINTAGE_TABLE} from {_DDL_PATH.name}")
    conn.execute(text(_DDL_PATH.read_text(encoding="utf-8")))
    conn.commit()


def build_vintage(conn, run_id: str | None, asof: _dt.date) -> tuple[pd.DataFrame, str]:
    """Join the panel frame and the decision table into one dated frame."""
    from sqlalchemy import text

    if run_id is None:
        run_id = conn.execute(
            text(
                f'SELECT run_id FROM analytics."{_PANEL_TABLE}" '
                "ORDER BY exported_at DESC LIMIT 1"
            )
        ).scalar()
        if run_id is None:
            raise SystemExit(f'analytics."{_PANEL_TABLE}" is empty — run an export first.')

    panel = pd.read_sql(
        text(f'SELECT * FROM analytics."{_PANEL_TABLE}" WHERE run_id = :r'),
        conn, params={"r": run_id},
    )
    if panel.empty:
        raise SystemExit(f"no panel rows for run_id {run_id!r}")

    have = [c for c in _PANEL_COLS if c in panel.columns]
    missing = sorted(set(_PANEL_COLS) - set(have))
    if missing:
        print(f"  note: panel frame lacks {len(missing)} column(s): {', '.join(missing)}")
    out = panel[have].copy()

    if _table_exists(conn, _RESULTS_TABLE):
        res = pd.read_sql(
            text(f'SELECT * FROM analytics."{_RESULTS_TABLE}" WHERE run_id = :r'),
            conn, params={"r": run_id},
        )
        if res.empty:
            print(f"  note: {_RESULTS_TABLE} has no rows for this run_id; "
                  "capturing the panel only")
        else:
            keep = [c for c in _RESULT_COLS if c in res.columns]
            gone = sorted(set(_RESULT_COLS) - set(keep))
            if gone:
                print(f"  note: decision table lacks {', '.join(gone)} — an export "
                      "predating the 2026-08-20 columns")
            out = out.merge(res[keep], on="isin", how="left")
    else:
        print(f"  note: analytics.{_RESULTS_TABLE} does not exist; capturing "
              "the panel only. The vintage will carry no predictions to score.")

    out.insert(0, "asof_date", asof)
    out["run_id"] = run_id
    return out, run_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    asof = (
        _dt.date.fromisoformat(args.asof) if args.asof else _dt.date.today()
    )
    from sqlalchemy import text

    eng = _engine()
    with eng.connect() as conn:
        if not args.dry_run:
            ensure_table(conn)
        frame, run_id = build_vintage(conn, args.run_id, asof)

        print(f"asof_date  {asof}")
        print(f"run_id     {run_id}")
        print(f"rows       {len(frame):,}")
        print(f"columns    {len(frame.columns)}")
        if "expected_return_kalman" in frame.columns:
            _er = pd.to_numeric(frame["expected_return_kalman"], errors="coerce")
            print(f"median expected upside  {_er.median():.2%}")
        if "shrink_gain" in frame.columns:
            _g = pd.to_numeric(frame["shrink_gain"], errors="coerce")
            if _g.notna().any():
                print(f"median shrink_gain      {_g.median():.3f}")

        if args.dry_run:
            print("\n--dry-run: nothing written.")
            return 0

        existing = conn.execute(
            text(
                f'SELECT count(*) FROM analytics."{_VINTAGE_TABLE}" '
                "WHERE asof_date = :d"
            ),
            {"d": asof},
        ).scalar()
        # A SECOND guard, on the export rather than the date. The vintage's prices
        # come from `04_panel_frame_v2`, which only changes when an export runs --
        # so capturing twice against one export writes IDENTICAL prices under two
        # dates, and `score_panel_vintages.py` would then compute
        # `last_price / last_price - 1` = 0 for every name and report a rank IC on
        # noise. Nothing downstream could tell that from a real null result.
        #
        # This is the failure a SCHEDULED capture walks into by construction: the
        # timer fires whether or not anything was re-exported. Refuse loudly, so a
        # scheduled run that has nothing new to record fails visibly instead of
        # quietly filling the table with duplicates.
        seen_before = conn.execute(
            text(
                f'SELECT min(asof_date) FROM analytics."{_VINTAGE_TABLE}" '
                "WHERE run_id = :r AND asof_date <> :d"
            ),
            {"r": run_id, "d": asof},
        ).scalar()
        if seen_before is not None and not args.replace:
            raise SystemExit(
                f"run_id {run_id} was already captured on {seen_before}. Capturing "
                f"it again under {asof} would store the same prices twice, and "
                "scoring the pair would return zero realised return for every "
                "name. "
                "Re-run the v2 export first, then capture — or pass --replace if "
                "you have a reason to want the duplicate."
            )

        if existing:
            if not args.replace:
                # An append-only store whose rows can be silently rewritten is
                # not append-only, and a vintage is worthless if it might have
                # been edited after the fact.
                raise SystemExit(
                    f"asof_date {asof} already has {existing:,} rows. Refusing to "
                    "overwrite a captured vintage — pass --replace if you really "
                    "mean to discard it."
                )
            conn.execute(
                text(f'DELETE FROM analytics."{_VINTAGE_TABLE}" WHERE asof_date = :d'),
                {"d": asof},
            )
            conn.commit()
            print(f"replaced: deleted {existing:,} existing rows for {asof}")

        frame.to_sql(
            _VINTAGE_TABLE, conn, schema="analytics",
            if_exists="append", index=False, chunksize=2000,
        )
        conn.commit()

        total = conn.execute(
            text(f'SELECT count(DISTINCT asof_date) FROM analytics."{_VINTAGE_TABLE}"')
        ).scalar()
        print(f"\nwrote {len(frame):,} rows. analytics.{_VINTAGE_TABLE} now holds "
              f"{total} vintage(s).")
        if total < 2:
            print("Scoring needs two vintages separated in time — capture again "
                  "in a quarter or so, then run scripts/score_panel_vintages.py.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
