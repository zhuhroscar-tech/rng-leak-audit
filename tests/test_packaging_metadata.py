"""Regression tests for packaging metadata accepted by modern setuptools."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def test_pyproject_uses_current_spdx_license_metadata():
    text = PYPROJECT.read_text()

    assert 'requires = ["setuptools>=77"]' in text
    assert 'license = "MIT"' in text
    assert 'license-files = ["LICENSE"]' in text

    assert 'license = { text = "MIT" }' not in text
    assert "License :: OSI Approved :: MIT License" not in text
