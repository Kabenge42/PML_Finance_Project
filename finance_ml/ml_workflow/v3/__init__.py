"""v3 Expected Returns workflow split modules.

This package contains cohesive submodules extracted from the legacy
expected_returns_v3.py to improve maintainability and testing.

Only light-weight utilities are provided for now to avoid intrusive
changes. The legacy orchestrator can progressively delegate to these
modules without behavior changes.
"""

from . import utils as utils  # re-export for convenience
from . import cache as cache  # re-export for convenience

__all__ = [
    "utils",
    "cache",
]
