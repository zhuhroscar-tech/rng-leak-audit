"""Tests for rng-leak-audit. Requires the 'torch' extra (skipped otherwise).

Design mirrors this fleet's established discipline: every guard claim is
backed by a real reproduction, not an assumption, and at least one test
proves the test suite itself would have failed before the fix (bug-
injection verification), not just that the fix's own code path returns
success."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rng_leak_audit.core import (  # noqa: E402
    PROBE_CONFIGS,
    TorchUnavailableError,
    _draw_without_loader,
    _make_dataset,
    diagnose,
    isolated_iter,
)


def test_diagnose_runs_and_reports_torch_version():
    report = diagnose(seed=123)
    assert report["torch_version"] == torch.__version__
    assert len(report["results"]) == len(PROBE_CONFIGS)


def test_leak_is_actually_reproduced_on_this_host():
    """This is the core evidentiary claim of the whole tool: prove the
    leak is real on the CURRENTLY installed torch build, not merely
    cited from an old GitHub issue. If this ever goes False because
    upstream fixed the underlying behavior, that is itself important
    news the tool should surface (via any_leak_present), not silently
    pass."""
    report = diagnose(seed=42)
    assert report["any_leak_present"] is True, (
        "Expected the known upstream DataLoader-iterator RNG-state leak "
        "(pytorch/pytorch#11062 et al) to reproduce on torch "
        f"{torch.__version__}; if this now fails, the bug may have been "
        "fixed upstream -- verify against the issue tracker before "
        "assuming a test regression."
    )


def test_guard_neutralizes_leak_for_every_config():
    report = diagnose(seed=7)
    assert report["guard_fully_effective"] is True
    for r in report["results"]:
        assert r["guard_effective"], r


def test_isolated_iter_matches_untouched_baseline_directly():
    """Direct, minimal reproduction of the guard's core claim without
    going through diagnose(): construct a DataLoader, fully consume it
    via isolated_iter(), and confirm the very next global RNG draw is
    bit-identical to a draw taken with no DataLoader ever constructed."""
    from torch.utils.data import DataLoader

    seed = 99
    baseline = _draw_without_loader(torch, seed)

    torch.manual_seed(seed)
    dataset = _make_dataset(torch)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for _batch in isolated_iter(loader):
        pass
    guarded_draw = torch.rand(4)

    assert torch.equal(baseline, guarded_draw)


def test_isolated_iter_is_safe_under_early_break():
    """The guard must restore state even when the caller stops iterating
    before exhausting the loader (a common validation-loop pattern:
    `for batch in isolated_iter(loader): ... ; break`)."""
    from torch.utils.data import DataLoader

    seed = 55
    baseline = _draw_without_loader(torch, seed)

    torch.manual_seed(seed)
    dataset = _make_dataset(torch)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for _batch in isolated_iter(loader):
        break  # deliberately stop after the first batch
    guarded_draw = torch.rand(4)

    assert torch.equal(baseline, guarded_draw)


def test_unguarded_iter_diverges_from_baseline_bug_injection_check():
    """Bug-injection check proving the regression tests above are real:
    deliberately use the RAW (unguarded) loader instead of
    isolated_iter() and confirm the draw now DIFFERS from baseline --
    i.e. if isolated_iter() were a no-op (the bug this tool guards
    against), test_isolated_iter_matches_untouched_baseline_directly
    would correctly fail. This proves that test is not tautological."""
    from torch.utils.data import DataLoader

    seed = 99
    baseline = _draw_without_loader(torch, seed)

    torch.manual_seed(seed)
    dataset = _make_dataset(torch)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for _batch in loader:  # deliberately unguarded
        pass
    unguarded_draw = torch.rand(4)

    assert not torch.equal(baseline, unguarded_draw), (
        "Expected the unguarded loader to perturb the global RNG draw "
        "(that is the whole bug this tool detects); if this assertion "
        "fails, the underlying leak may have disappeared upstream, which "
        "would make the guard tautologically pass for the wrong reason."
    )


def test_isolated_iter_preserves_cuda_rng_state_when_available():
    """Skip cleanly (not xfail-silently) when no CUDA device is present --
    this host (macOS, no CUDA) cannot exercise this path; documented as a
    stated limitation, not silently claimed as tested."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available on this host; CPU-only RNG path is covered by other tests")

    from torch.utils.data import DataLoader

    torch.cuda.manual_seed_all(123)
    baseline_cuda_state = torch.cuda.get_rng_state_all()

    dataset = _make_dataset(torch)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for _batch in isolated_iter(loader):
        pass

    after_cuda_state = torch.cuda.get_rng_state_all()
    assert all(torch.equal(a, b) for a, b in zip(baseline_cuda_state, after_cuda_state))


def test_torch_unavailable_error_is_distinct_type():
    """Sanity check the error type exists and is a RuntimeError subclass,
    independent of whether torch is actually installed in this env."""
    assert issubclass(TorchUnavailableError, RuntimeError)
