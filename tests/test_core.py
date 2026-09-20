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


def test_isolated_iter_with_block_restores_state_for_manual_next_pattern():
    """Regression test for a real bug found by this run's stewardship
    backstop audit: the ORIGINAL isolated_iter() implementation was a
    bare generator whose restore-on-exit logic lived in a `finally`
    block inside the generator's own frame. That `finally` only ran when
    the generator was actually closed, exhausted, or garbage collected --
    NOT synchronously when the caller's code around a manual `next()`
    call raises and the generator object is held in a named local
    variable (a real pattern in prefetch/double-buffering training loops
    that call `next()` directly instead of using a bare `for` loop).
    Before the fix, there was no way to get a synchronous restoration
    guarantee for this exact pattern at all -- this test fails against
    the pre-fix implementation (`TypeError: 'generator' object does not
    support the context manager protocol`) and passes with the fix's
    `with isolated_iter(loader) as batches:` form, which now gives
    manual-`next()` callers the same deterministic guarantee as the
    `for` idiom already had.
    """
    from torch.utils.data import DataLoader

    seed = 42
    baseline = _draw_without_loader(torch, seed)

    torch.manual_seed(seed)
    dataset = _make_dataset(torch)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    try:
        with isolated_iter(loader) as batches:
            while True:
                _batch = next(batches)
                raise RuntimeError("boom mid-manual-iteration")
    except RuntimeError:
        pass
    except StopIteration:
        pass

    # __exit__ has already run synchronously by this point -- the state
    # must be restored regardless of any lingering reference to `batches`.
    draw_after_with_block = torch.rand(4)
    assert torch.equal(baseline, draw_after_with_block), (
        "isolated_iter() used as a context manager must restore RNG "
        "state deterministically via __exit__ for the manual-next() "
        "iteration pattern, not just the bare `for` idiom"
    )


def test_isolated_iter_as_context_manager_restores_on_exception_in_with_block():
    """The `with isolated_iter(loader) as batches:` form must restore
    state deterministically via __exit__ even when the caller's own code
    inside the `with` block raises for a reason unrelated to iteration
    itself (e.g. a training-step exception after a batch was already
    yielded) -- this is the documented, guaranteed-restoration idiom."""
    from torch.utils.data import DataLoader

    seed = 71
    baseline = _draw_without_loader(torch, seed)

    torch.manual_seed(seed)
    dataset = _make_dataset(torch)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    with pytest.raises(RuntimeError, match="boom in training step"):
        with isolated_iter(loader) as batches:
            for _batch in batches:
                raise RuntimeError("boom in training step")

    guarded_draw = torch.rand(4)
    assert torch.equal(baseline, guarded_draw)


def test_isolated_iter_restores_when_iter_construction_itself_raises():
    """If iter(dataloader) itself raises AFTER already drawing from the
    global RNG (the exact class of mutation this tool guards against),
    state must still be restored before the exception propagates --
    the original code's snapshot-then-call-iter() sequence had no
    protection for a raising iter() call at all."""

    class _ExplodingIterable:
        """A minimal iterable whose __iter__ mutates the global RNG (as
        real DataLoader.__init__/_base_seed construction does) and then
        raises, simulating any real-world failure during iterator setup
        (e.g. a worker-process spawn failure)."""

        def __iter__(self):
            torch.rand(1)  # simulate the real leak: a mutating draw...
            raise RuntimeError("boom during iterator construction")

    seed = 13
    baseline = _draw_without_loader(torch, seed)

    torch.manual_seed(seed)
    with pytest.raises(RuntimeError, match="boom during iterator construction"):
        isolated_iter(_ExplodingIterable())

    draw_after = torch.rand(4)
    assert torch.equal(baseline, draw_after), (
        "isolated_iter() must restore RNG state even when the wrapped "
        "iterable's own __iter__ raises after mutating global RNG state"
    )


def test_isolated_iter_close_is_idempotent():
    """Calling close() more than once, or letting __del__ run after an
    explicit close(), must not double-restore or raise."""
    from torch.utils.data import DataLoader

    dataset = _make_dataset(torch)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    guard = isolated_iter(loader)
    for _batch in guard:
        pass
    guard.close()
    guard.close()  # must not raise or double-apply state


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
