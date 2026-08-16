"""Global Equity Investment Board (GEIB) dashboard — launcher.

A focused Dash + dash-bootstrap-components board driven by the single analytics
table ``analytics.kalman_filtered_price_targets`` (exported by
``pymc_kalman_filter_pt.py`` from
:class:`probabilistic_ml_model.pymc_models.KalmanFilterModel`).

The implementation lives in the ``geib`` package next to this file; this module
just makes that package importable and starts the server, preserving the
documented run path::

    . .\\set_env.ps1            # sets DB_URL / DB_ANALYTICS_SCHEMA
    python dashboards/global_equity_investment_dashboard.py
    # open http://localhost:8050

Environment:
    DB_URL                  SQLAlchemy PostgreSQL URL (required for live data)
    DB_ANALYTICS_SCHEMA     analytics schema name (default "analytics")
    GEIB_DEBUG              set to "true" to run Dash in debug mode
    GEIB_PORT               override the listen port (default 8050)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the sibling ``geib`` package importable when this file is run directly.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ...and the REPO ROOT, so ``probabilistic_ml_model`` resolves too. Running
# ``python dashboards/global_equity_investment_dashboard.py`` puts THIS file's
# directory on sys.path[0], not the working directory, so the repo root is absent
# unless PYTHONPATH supplies it. Without this line
# ``geib.data``'s guarded import of ``get_analytics_engine`` falls back to None
# and the board renders every card against an EMPTY frame -- with nothing worse
# than one WARNING line to say so. Same pattern the scripts/ entry points use.
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from geib.app import app, server  # noqa: E402  (path setup must precede import)

__all__ = ["app", "server"]


if __name__ == "__main__":
    debug = os.environ.get("GEIB_DEBUG", "").lower() == "true"
    port = int(os.environ.get("GEIB_PORT", "8050"))
    app.run(debug=debug, host="0.0.0.0", port=port)
