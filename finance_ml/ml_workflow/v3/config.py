from __future__ import annotations

"""
Thin configuration adapter for the v3 pipeline.

Exposes a stable import path while the legacy expected_returns_v3.PipelineConfig
remains the source of truth. This enables progressive migration without
behavior changes.
"""
from typing import Optional


class PipelineConfigAdapter:
    """Proxy to legacy PipelineConfig for backward compatibility."""

    def __init__(self, *args, **kwargs):
        from expected_returns_v3 import PipelineConfig as _Legacy

        self._cfg = _Legacy(*args, **kwargs)  # type: ignore[misc]

    def __getattr__(self, item):
        return getattr(self._cfg, item)

    def __repr__(self) -> str:  # pragma: no cover - simple proxy
        return f"PipelineConfigAdapter({self._cfg!r})"
