"""rng-leak-audit: detect and neutralize PyTorch's DataLoader-iterator
global-RNG-state leak (pytorch/pytorch#11062, #122697, #107443)."""
from __future__ import annotations

__version__ = "0.2.3"

__all__ = ["isolated_iter", "diagnose", "__version__"]


def __getattr__(name):
    # Lazy import: importing rng_leak_audit itself must not require torch
    # to be installed (e.g. `--version` works without the 'torch' extra).
    if name in ("isolated_iter", "diagnose"):
        from . import core

        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
