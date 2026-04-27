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
        import pytensor
        from pytensor.compile.mode import Mode

        # Globally disable the C backend so any graph compiled outside of the
        # returned `mode` (e.g. during model construction / logp caching) also
        # skips g++.  This is required on MSVC-built CPython 3.14 where the
        # bundled MinGW g++ cannot link against ``python314.dll``.
        try:
            pytensor.config.cxx = ""
        except Exception:
            pass

        # Pure-Python linker — no C/g++ compilation at sample time either.
        return {"mode": Mode(linker="py")}
    except Exception:
        return {}
