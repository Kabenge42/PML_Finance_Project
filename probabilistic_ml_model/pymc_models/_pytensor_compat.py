"""
PyTensor compilation compatibility helpers.

On Windows with MSVC-built CPython 3.14, MinGW g++ cannot link against
python314.dll (ABI mismatch).  Force the pure-Python linker at the PyMC
call site so it works regardless of when PYTENSOR_FLAGS was set relative
to the first pytensor import.
"""

from __future__ import annotations


def get_pytensor_compile_kwargs() -> dict:
    """Return compile_kwargs that force the Python-only (non-C) linker.

    Pass the returned dict as ``compile_kwargs=...`` to ``pm.sample()``
    and ``pm.sample_posterior_predictive()`` to avoid all C/g++ compilation.

    NOTE: This helper is OPT-IN — callers must explicitly pass the returned
    dict. It no longer mutates ``pytensor.config.cxx`` at import time, so the
    PYTENSOR_FLAGS env var (e.g. ``cxx=C:\\msys64\\mingw64\\bin\\g++.exe``)
    remains authoritative and the nutpie Rust NUTS sampler keeps working.
    """
    try:
        from pytensor.compile.mode import Mode

        # Pure-Python linker — no C/g++ compilation at sample time.
        return {"mode": Mode(linker="py")}
    except Exception:
        return {}