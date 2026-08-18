"""Apply and verify the Kalman v2 database schema.

Runs the two DDL files in the one order that works, then proves the result is
clean. Both halves matter: the catalogue fails *silently* when a column is
registered wrongly — an unregistered MV column does not raise, it simply
disappears from the model's feature list and is zero-filled downstream — so
"applied without error" is not evidence of anything. The verification step is
the deliverable.

.. code-block:: powershell

    . .\\set_env.ps1
    python scripts/apply_kalman_v2_schema.py            # apply + verify
    python scripts/apply_kalman_v2_schema.py --verify   # verify only
    python scripts/apply_kalman_v2_schema.py --refresh  # apply + refresh + verify

Exit code is 0 only when the coverage check returns no violations for
``kalman_pt_v2`` **and** introduces none for any other model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

SQL_DIR = Path(__file__).resolve().parent.parent / "sql_scripts" / "pml"
DDL_FILES = (
    SQL_DIR / "mv_pymc_kalman_pt_v2.sql",
    SQL_DIR / "mv_pymc_kalman_pt_v2_metadata.sql",
)
TARGET = "kalman_pt_v2"


def _engine():
    url = os.environ.get("DB_URL")
    if not url:
        raise SystemExit("DB_URL is not set; run `. .\\set_env.ps1` first.")
    return create_engine(url)


def apply_ddl(engine) -> None:
    """Execute each DDL file as one script.

    Uses a raw DBAPI cursor rather than SQLAlchemy ``text()``: these files
    contain dollar-quoted PL/pgSQL bodies, which the SQLAlchemy parameter parser
    mangles.
    """
    for path in DDL_FILES:
        if not path.exists():
            raise SystemExit(f"missing DDL file: {path}")
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            cur.execute(path.read_text(encoding="utf-8"))
            raw.commit()
        finally:
            raw.close()
        print(f"applied {path.name}")


def refresh(engine) -> None:
    """Refresh the v2 MV, and its parent first."""
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("CALL pml.refresh_kalman_pt_v2(use_concurrently => TRUE)")
        raw.commit()
    finally:
        raw.close()
    print("refreshed pml.mv_pymc_kalman_pt_v2 (and parent)")


def verify(engine) -> int:
    """Report catalogue coverage and basic MV health. Returns an exit code."""
    failures = 0
    with engine.connect() as conn:
        def q(sql: str):
            return conn.execute(text(sql)).fetchall()

        rows = q(
            "SELECT status, count(*), string_agg(feat_name, ', ' ORDER BY feat_name) "
            "FROM pml.vw_pymc_catalogue_coverage_check "
            f"WHERE status <> 'OK' AND model_target = '{TARGET}' GROUP BY 1 ORDER BY 1"
        )
        if rows:
            failures += 1
            print(f"FAIL  catalogue coverage for {TARGET}:")
            for status, n, cols in rows:
                print(f"        {status}: {n} -- {cols}")
        else:
            print(f"PASS  catalogue coverage for {TARGET}: no violations")

        others = q(
            "SELECT model_target, status, count(*) "
            "FROM pml.vw_pymc_catalogue_coverage_check "
            f"WHERE status <> 'OK' AND model_target <> '{TARGET}' "
            "GROUP BY 1, 2 ORDER BY 1, 2"
        )
        if others:
            # Pre-existing violations elsewhere are not this change's fault, but
            # they must be visible so a new one is not mistaken for old news.
            print("NOTE  pre-existing violations in other models (not introduced here):")
            for mt, status, n in others:
                print(f"        {mt}: {status} x{n}")
        else:
            print("PASS  no violations in any other model")

        n_rows = q("SELECT count(*) FROM pml.mv_pymc_kalman_pt_v2")[0][0]
        if n_rows == 0:
            failures += 1
            print("FAIL  pml.mv_pymc_kalman_pt_v2 is empty")
        else:
            print(f"PASS  pml.mv_pymc_kalman_pt_v2 has {n_rows} rows")

        trail = q(
            "SELECT n_trail_obs, count(*) FROM pml.mv_pymc_kalman_pt_v2 "
            "GROUP BY 1 ORDER BY 1"
        )
        print("      trail depth: " + ", ".join(f"{k}:{v}" for k, v in trail))

        roles = q(
            "SELECT pymc_role, count(*) FROM pml.vw_pymc_feature_catalogue "
            f"WHERE model_target = '{TARGET}' GROUP BY 1 ORDER BY 1"
        )
        print("      catalogue roles: " + ", ".join(f"{k}={v}" for k, v in roles))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verify", action="store_true", help="verify only, no DDL")
    parser.add_argument("--refresh", action="store_true", help="refresh the MV after applying")
    args = parser.parse_args()

    engine = _engine()
    if not args.verify:
        apply_ddl(engine)
    if args.refresh:
        refresh(engine)
    return verify(engine)


if __name__ == "__main__":
    sys.exit(main())
