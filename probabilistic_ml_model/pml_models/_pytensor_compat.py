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
    """
    try:
        from pytensor.compile.mode import Mode
        from pytensor.link.basic import JITLinker

        # Use the Python-only JIT linker — avoids all C/g++ compilation
        mode = Mode(linker=JITLinker(allow_gc=False))
        return {"mode": mode}
    except Exception:
        return {}
