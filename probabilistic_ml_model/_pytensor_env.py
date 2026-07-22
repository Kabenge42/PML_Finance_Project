"""Authoritative PyTensor backend selection for the whole package.

Windows + MSVC-built CPython 3.14 + the MSYS2 MinGW ``g++`` C backend is
fragile: a fresh C compile (``lazylinker_ext`` / any op) silently fails with an
empty ``stderr`` and ``status=1`` whenever another MinGW runtime shadows
``C:\\msys64\\ucrt64\\bin`` on ``PATH``. nutpie (numba) and numpyro (JAX) do not
need PyTensor's C linker, and ``pm.model_to_graphviz`` / shape evaluation run
fine on the pure-Python VM, so the robust default is to disable the C backend.

This module normalises ``PYTENSOR_FLAGS`` to force ``cxx=`` (empty) **before any
``pytensor`` import**. It is imported at the very top of
:mod:`probabilistic_ml_model.__init__`, so importing the package is enough.

Crucially it *strips* an existing ``cxx=<path>`` value rather than only acting
when ``cxx`` is absent — a persistent ``PYTENSOR_FLAGS=...,cxx=<g++>`` User env
var previously defeated the naive ``"cxx=" in flags`` guards.

Opt back into the C backend by setting ``PML_ENABLE_PYTENSOR_C=1`` in the
environment before import (and ensure ``ucrt64\\bin`` is first on ``PATH``).
"""

from __future__ import annotations

import os

#: Env flag that re-enables PyTensor's C/g++ backend (advanced / opt-in).
ENABLE_C_ENV_VAR = "PYTENSOR_FLAGS"


def force_python_vm() -> str:
    """Rewrite ``PYTENSOR_FLAGS`` to force ``cxx=`` unless C is opted in.

    Returns
    -------
    str
        The resulting ``PYTENSOR_FLAGS`` value (unchanged if C is opted in).

    Notes
    -----
    Preserves any other flags (e.g. ``floatX``, ``device``), drops any existing
    ``cxx=<value>`` entry, and guarantees a trailing ``cxx=`` plus a
    ``floatX=float64`` default. Safe to call more than once (idempotent).
    """
    if os.environ.get(ENABLE_C_ENV_VAR) == "1":
        return os.environ.get("PYTENSOR_FLAGS", "")

    flags = os.environ.get("PYTENSOR_FLAGS", "")
    kept: dict[str, str] = {}
    order: list[str] = []
    for part in flags.split(","):
        part = part.strip()
        if not part:
            continue
        key = part.split("=", 1)[0].strip()
        if key == "cxx":
            continue  # drop any cxx=<path>; re-added empty below
        if key not in kept:
            order.append(key)
        kept[key] = part

    if "floatX" not in kept:
        order.append("floatX")
        kept["floatX"] = "floatX=float64"
    order.append("cxx")
    kept["cxx"] = "cxx="

    resolved = ",".join(kept[k] for k in order)
    os.environ["PYTENSOR_FLAGS"] = resolved
    return resolved


# Run on import — must happen before the first ``import pytensor`` anywhere.
force_python_vm()