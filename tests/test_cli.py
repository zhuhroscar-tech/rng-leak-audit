"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is installed."""
from __future__ import annotations

import json

import pytest

import rng_leak_audit.core as core
from rng_leak_audit.cli import main


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "rng-leak-audit" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert code in (0, 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_torch_unavailable_json_output_reports_error_and_exit_2(capsys, monkeypatch):
    """CLI's TorchUnavailableError handling (json branch) was previously
    untested: nothing proved the --json error payload is actually valid
    JSON with an "error" key, or that the exit code is 2 rather than the
    default 0/1 used by the normal report paths."""

    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis and guarding")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis and guarding"}
    assert code == 2


def test_torch_unavailable_text_output_shows_fail_headline_and_exit_2(capsys, monkeypatch):
    """Text-mode counterpart of the above: the human-readable path must
    also surface the error via the shared fail status headline, not
    silently swallow it or print the raw traceback."""

    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis and guarding")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[X] torch unavailable: torch is required for diagnosis and guarding" in out
    assert code == 2


def _fake_report(any_leak_present, guard_fully_effective, per_config_guard_effective):
    return {
        "torch_version": "0.0.0-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/11062"],
        "results": [
            {
                "name": "shuffle=False",
                "leak_present": any_leak_present,
                "guard_effective": per_config_guard_effective,
                "max_abs_diff_raw": 0.1 if any_leak_present else 0.0,
                "max_abs_diff_guarded": 0.0 if per_config_guard_effective else 0.1,
            }
        ],
        "any_leak_present": any_leak_present,
        "guard_fully_effective": guard_fully_effective,
        "limitation": "fake limitation text for CLI display testing",
    }


def test_leak_not_present_shows_info_message_not_warn(capsys, monkeypatch):
    """When any_leak_present is False (e.g. a future torch build fixes the
    upstream bug), the CLI must report that as neutral "info", not the
    "warn" headline used for a reproduced leak -- these are two visually
    and semantically distinct outcomes that were both only ever exercised
    via the currently-installed torch build's actual (leaky) behavior."""

    monkeypatch.setattr(
        core, "diagnose", lambda **kwargs: _fake_report(False, True, True)
    )
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[i] RNG-state leak NOT reproduced" in out
    assert "[!] RNG-state leak reproduced" not in out
    assert code == 0


def test_guard_failed_label_shown_when_guard_ineffective_for_a_config(capsys, monkeypatch):
    """The per-configuration table's "GUARD-FAILED" label (as opposed to
    "guard-ok") was never actually rendered by any test -- every real run
    on this host has isolated_iter() succeed for every probed config, so
    this branch was only reachable by construction, not by observation."""

    monkeypatch.setattr(
        core, "diagnose", lambda **kwargs: _fake_report(True, False, False)
    )
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "GUARD-FAILED" in out
    assert "[X] guard did NOT neutralize the leak" in out
    assert code == 1
