#!/usr/bin/env python
"""Create the ``kalman_portfolio`` schema and prove it is there.

Why this exists
---------------
Nothing else in this repository issues ``CREATE SCHEMA``. ``analytics`` has
always been assumed to pre-exist, which is fine for a schema that predates the
code and wrong for one this change introduces: ``to_sql(schema=...)`` fails with
a bare ``InvalidSchemaName`` if the schema is absent, after the replay has
already done its work.

Modelled on ``scripts/apply_kalman_v2_schema.py``: a raw DBAPI cursor rather than
SQLAlchemy ``text()`` (the DDL carries a dollar-quoted ``DO`` block, which the
parameter parser mangles), then apply-and-prove-clean rather than apply-and-hope.

Usage
-----
.. code-block:: powershell

    . .\\set_env.ps1
    ~\\AppData\\Local\\Python\\bin\\python.exe scripts\\apply_kalman_portfolio_schema.py
    ~\\AppData\\Local\\Python\\bin\\python.exe scripts\\apply_kalman_portfolio_schema.py --verify
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DDL_FILES = [_REPO / "sql_scripts" / "kalman_portfolio" / "00_schema.sql"]


def _schema() -> str:
    """The target schema, validated before it reaches an f-string.

    Reuses the project's existing guard rather than re-implementing it: a schema
    name interpolated into DDL is the one place an environment variable becomes
    executable SQL.
    """
    from probabilistic_ml_model.data_utils.inference_schema import (
        _validate_schema_name,
    )

    return _validate_schema_name(
        os.environ.get("DB_PORTFOLIO_SCHEMA", "kalman_portfolio")
    )


def _engine():
    from sqlalchemy import create_engine

    url = os.environ.get("DB_URL")
    if not url:
        raise SystemExit("DB_URL is not set; run `. .\\set_env.ps1` first.")
    return create_engine(url)


def apply_ddl(engine, schema: str) -> None:
    """Execute each DDL file as one script."""
    for path in DDL_FILES:
        if not path.exists():
            raise SystemExit(f"missing DDL file: {path}")
        sql = path.read_text(encoding="utf-8")
        if schema != "kalman_portfolio":
            # The file names the default; an overridden schema substitutes it.
            sql = sql.replace("kalman_portfolio", schema)
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            cur.execute(sql)
            raw.commit()
        finally:
            raw.close()
        print(f"applied {path.name} -> schema {schema}")


def verify(engine, schema: str) -> int:
    """Report what is there. Returns the process exit code."""
    from sqlalchemy import text

    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM information_schema.schemata WHERE schema_name = :s"),
            {"s": schema},
        ).first()
        if not exists:
            print(f"  ! schema {schema} does not exist; run without --verify first")
            return 1
        rows = list(conn.execute(
            text(
                "SELECT table_name, "
                "  (SELECT count(*) FROM information_schema.columns c "
                "   WHERE c.table_schema = t.table_schema "
                "     AND c.table_name = t.table_name) AS n_cols "
                "FROM information_schema.tables t "
                "WHERE t.table_schema = :s ORDER BY table_name"
            ),
            {"s": schema},
        ))

    print(f"schema {schema}: OK")
    if not rows:
        print("  (no tables yet -- they are created on the first "
              "`python kalman_portfolio.py --write`)")
        return 0
    for name, n_cols in rows:
        print(f"  {name:<34} {n_cols:>4} columns")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--verify", action="store_true",
                        help="report the schema's contents without applying the DDL")
    args = parser.parse_args()

    schema = _schema()
    engine = _engine()
    if not args.verify:
        apply_ddl(engine, schema)
    return verify(engine, schema)


if __name__ == "__main__":
    sys.exit(main())
