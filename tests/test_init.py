"""Tests for rng_leak_audit's package-level lazy-attribute __getattr__:
importing the top-level package must not require torch (so --version
works without the 'torch' extra installed), but accessing
`isolated_iter`/`diagnose` must lazily pull them from .core, and any
other attribute name must raise AttributeError like a normal module."""
from __future__ import annotations

import pytest

import rng_leak_audit


def test_version_attribute_available_without_torch_import_path():
    assert rng_leak_audit.__version__ == "0.2.0"


def test_getattr_lazily_resolves_isolated_iter_from_core():
    torch = pytest.importorskip("torch")
    assert rng_leak_audit.isolated_iter is not None
    from rng_leak_audit.core import isolated_iter as core_isolated_iter

    assert rng_leak_audit.isolated_iter is core_isolated_iter


def test_getattr_lazily_resolves_diagnose_from_core():
    pytest.importorskip("torch")
    from rng_leak_audit.core import diagnose as core_diagnose

    assert rng_leak_audit.diagnose is core_diagnose


def test_getattr_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="module 'rng_leak_audit' has no attribute 'nonexistent_thing'"):
        rng_leak_audit.nonexistent_thing
