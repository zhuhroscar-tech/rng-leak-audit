from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"
CHANGELOG = ROOT / "CHANGELOG.md"
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "rng_leak_audit" / "__init__.py"
CI = ROOT / ".github" / "workflows" / "ci.yml"
CODEQL = ROOT / ".github" / "workflows" / "codeql.yml"
MANIFEST = ROOT / "MANIFEST.in"
LICENSE = ROOT / "LICENSE"
SCREENSHOT = ROOT / "docs" / "images" / "example-output.png"


def _version_from_pyproject() -> str:
    match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(), re.MULTILINE)
    assert match, "pyproject.toml must declare project.version"
    return match.group(1)


def test_required_repository_files_exist():
    for path in (README, README_ZH, CHANGELOG, LICENSE, MANIFEST, CI, CODEQL, SCREENSHOT):
        assert path.exists(), f"missing required repository file: {path.relative_to(ROOT)}"


def test_readme_language_and_release_links_are_live():
    readme = README.read_text()
    readme_zh = README_ZH.read_text()

    assert "README.zh-CN.md" in readme
    assert "README.md" in readme_zh
    assert "CHANGELOG.md" in readme
    assert "CHANGELOG.md" in readme_zh
    assert "LICENSE" in readme
    assert "LICENSE" in readme_zh
    assert "docs/images/example-output.png" in readme
    assert "docs/images/example-output.png" in readme_zh


def test_current_version_is_documented_everywhere():
    version = _version_from_pyproject()
    init_text = INIT.read_text()
    changelog = CHANGELOG.read_text()

    assert f'__version__ = "{version}"' in init_text
    assert f"## v{version}" in changelog


def test_ci_builds_release_artifacts_and_runs_codeql():
    ci = CI.read_text()
    codeql = CODEQL.read_text()

    assert "python -m build" in ci
    assert "sha256sum" in ci
    assert "actions/upload-artifact" in ci
    assert "github/codeql-action/init" in codeql
    assert "github/codeql-action/analyze" in codeql


def test_manifest_includes_release_metadata_and_tests():
    manifest = MANIFEST.read_text()

    for required in (
        "include README.md",
        "include README.zh-CN.md",
        "include CHANGELOG.md",
        "recursive-include tests *.py",
        "recursive-include .github/workflows *.yml",
    ):
        assert required in manifest


def test_package_metadata_exposes_project_resource_links():
    pyproject = PYPROJECT.read_text()

    assert "[project.urls]" in pyproject
    assert 'Homepage = "https://github.com/zhuhroscar-tech/rng-leak-audit"' in pyproject
    assert 'Issues = "https://github.com/zhuhroscar-tech/rng-leak-audit/issues"' in pyproject
    assert (
        'Changelog = "https://github.com/zhuhroscar-tech/rng-leak-audit/blob/main/CHANGELOG.md"'
        in pyproject
    )


def test_ci_runs_on_release_tags():
    ci = CI.read_text()

    assert "tags:" in ci
    assert '"v*"' in ci
