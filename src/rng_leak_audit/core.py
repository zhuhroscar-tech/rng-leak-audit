"""rng-leak-audit core: detect and neutralize PyTorch's global-RNG-state
leak from consuming a DataLoader (pytorch/pytorch#11062, open since
2018; #122697, #107443 -- confirmed still-open design behavior, not a
fixed bug, as of torch 2.14).

The leak: iterating a DataLoader -- even with ``shuffle=False``,
``num_workers=0``, and no randomness anywhere in the dataset/collate
pipeline -- draws from the *global* torch RNG both when its internal
iterator is constructed (a ``_base_seed`` draw) and, for
``shuffle=True``, lazily on each ``next()`` call while building the
shuffled index permutation (independently reproduced on this host, not
just cited from the issue tracker -- see ``diagnose()``/the test suite).
Any code that runs after that DataLoader is consumed and expects
reproducible ``torch.rand()``/``torch.randn()``/etc. output silently
gets a different sequence than it would have without the DataLoader
ever being touched. This is most dangerous when an unrelated
validation/eval DataLoader is iterated between training steps: the mere
act of running validation perturbs the *training* run's subsequent
random draws, so two training runs that only differ in "did validation
run this epoch" silently diverge -- with no error, warning, or visible
symptom.

This module provides:

  - ``PROBE_CONFIGS``: DataLoader construction configs (shuffle x
    num_workers) used to both diagnose the leak on the currently
    installed torch build and regression-test the guard.
  - ``diagnose()``: reproduces the leak from scratch against whatever
    torch version is actually installed on this host -- never trusts the
    upstream issue tracker's version report alone.
  - ``isolated_iter()``: a guard that snapshots global RNG state once
    before consuming a DataLoader and restores it exactly once when
    iteration ends (normally, via ``break``, or via an exception), so
    consuming a DataLoader is side-effect-free with respect to unrelated
    code's subsequent random draws.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List


class TorchUnavailableError(RuntimeError):
    """Raised when torch cannot be imported. Kept as a distinct type so
    callers can distinguish "torch isn't installed" from an actual
    diagnostic failure."""


def _import_torch():
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader, Dataset  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise TorchUnavailableError(
            "torch (with torch.utils.data) is required for diagnosis and "
            "guarding; install the 'torch' extra."
        ) from exc
    return torch, DataLoader, Dataset


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    name: str
    shuffle: bool
    num_workers: int


# num_workers > 0 spawns subprocesses and cannot be exercised inside a
# plain pytest/CI run without multiprocessing start-method friction, so
# the packaged probe set only covers num_workers=0. The README documents
# that the leak is a torch.utils.data.DataLoader.__init__/_base_seed
# implementation detail independent of num_workers, and diagnose() notes
# this explicitly as a stated limitation.
PROBE_CONFIGS: List[ProbeConfig] = [
    ProbeConfig("shuffle=False", shuffle=False, num_workers=0),
    ProbeConfig("shuffle=True", shuffle=True, num_workers=0),
]


def _make_dataset(torch_module):
    class _PlainDataset(torch_module.utils.data.Dataset):
        """A dataset with NO randomness anywhere: __getitem__ is a pure
        function of the index. If the global RNG state still changes
        after iterating this DataLoader, the leak is unambiguously the
        DataLoader machinery itself, not the dataset/collate pipeline."""

        def __len__(self):
            return 20

        def __getitem__(self, idx):
            return idx

    return _PlainDataset()


def _draw_without_loader(torch_module, seed: int):
    torch_module.manual_seed(seed)
    return torch_module.rand(4).clone()


@dataclasses.dataclass
class LeakDiagnosis:
    name: str
    leak_present: bool           # draw differs from the untouched baseline
    guard_effective: bool        # guarded draw matches the untouched baseline
    max_abs_diff_raw: float
    max_abs_diff_guarded: float


class _IsolatedIterator:
    """Iterator AND context manager returned by :func:`isolated_iter`.

    Restoration is guaranteed *synchronously* when this object is used as
    a context manager -- ``with isolated_iter(loader) as batches: ...``
    -- because ``__exit__`` runs deterministically on any exit from the
    ``with`` block (normal completion, ``break``, ``return``, or any
    exception propagating through it), per the ``with``-statement
    specification. This does NOT depend on reference counting or
    garbage-collection timing, unlike a bare generator.

    Without ``with``, this object still restores state once it is closed
    or garbage collected -- reliable for the common
    ``for batch in isolated_iter(loader): ...`` pattern under CPython
    (the for-loop's own hidden iterator reference is dropped, and thus
    finalized via refcounting, as soon as the loop is exited by any
    means), but NOT reliable if the object is instead stored in a named
    variable and driven with manual ``next()`` calls whose caller-side
    exception handling outlives the loop body (a real pattern in
    prefetch/double-buffering training loops) -- use the ``with`` form
    whenever the restoration guarantee matters, not just the bare
    iterator form.

    Two additional gaps closed versus the original generator-only
    implementation (found by this fleet's backstop audit, independently
    reproduced before fixing):

    1. If ``iter(dataloader)`` itself raises AFTER already drawing from
       the global RNG (the exact mutation this tool exists to guard
       against), state is now restored before the exception propagates,
       instead of being silently left dirty (the original code only
       snapshotted, then let a raising ``iter()`` call skip restoration
       entirely).
    2. If the underlying iterator's own ``__next__`` raises mid-consumption
       (e.g. a worker crash), state is restored immediately rather than
       only when this object is eventually closed/collected.
    """

    def __init__(self, dataloader, torch_module):
        self._torch = torch_module
        self._restored = False
        self._cpu_state = torch_module.get_rng_state()
        self._cuda_states = (
            torch_module.cuda.get_rng_state_all()
            if torch_module.cuda.is_available()
            else None
        )
        try:
            self._iterator = iter(dataloader)
        except BaseException:
            self._restore()
            raise

    def _restore(self):
        if self._restored:
            return
        self._restored = True
        self._torch.set_rng_state(self._cpu_state)
        if self._cuda_states is not None:
            self._torch.cuda.set_rng_state_all(self._cuda_states)

    def close(self):
        """Explicitly restore now; safe to call more than once."""
        self._restore()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except BaseException:
            # Covers both normal StopIteration (exhaustion) and any
            # other exception raised by the wrapped loader's own
            # __next__ (e.g. a worker crash) -- either way this
            # iterator's useful life is over, so restore now rather
            # than waiting for close()/__del__.
            self._restore()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._restore()
        return False

    def __del__(self):
        # Best-effort fallback for the bare for-loop pattern and for
        # callers who neither exhaust the iterator nor use `with`; not a
        # substitute for `with` when the restoration guarantee matters
        # (see class docstring).
        self._restore()


def isolated_iter(dataloader):
    """Iterate ``dataloader`` end to end without perturbing the caller's
    global torch RNG state, as observed immediately BEFORE this call and
    immediately AFTER consumption ends.

    Root-cause note: the leak is not confined to constructing the
    iterator (``iter(dataloader)``) -- ``next()`` calls on a
    ``shuffle=True`` loader also draw from the global RNG lazily (to
    build the shuffled index permutation), and this was independently
    reproduced on this host before writing this guard (see the test
    suite / README). A guard that only brackets ``iter()`` construction
    is therefore NOT sufficient -- it must bracket the entire consumption
    of the loader.

    Returns an :class:`_IsolatedIterator`, which is both a plain iterator
    (for the ``for batch in isolated_iter(loader): ...`` idiom) and a
    context manager giving a deterministic restoration guarantee (for the
    ``with isolated_iter(loader) as batches: ...`` idiom) -- see that
    class's docstring for exactly which usage pattern gets which
    guarantee.

    Any randomness a caller's own model/transform code deliberately draws
    from the global RNG *while consuming a yielded batch* (e.g. inside
    the training step's own forward pass) happens between two calls to
    this iterator and is therefore NOT rolled back -- only the
    DataLoader's own internal bookkeeping draws are neutralized. Use this
    to wrap a validation/eval loop so it cannot perturb the training
    run's subsequent random draws, not to make a training loop's own
    stochastic operations reproducible (that is `torch.manual_seed`'s
    job).
    """
    torch_module, _dataloader_cls, _dataset_cls = _import_torch()
    return _IsolatedIterator(dataloader, torch_module)


def _run_probe(torch_module, dataloader_cls, cfg: ProbeConfig, seed: int, guarded: bool):
    """Returns the RNG draw taken immediately after constructing and
    fully consuming a DataLoader, either raw or through the
    isolated_iter() guard."""
    torch_module.manual_seed(seed)
    dataset = _make_dataset(torch_module)
    loader = dataloader_cls(dataset, batch_size=4, shuffle=cfg.shuffle, num_workers=cfg.num_workers)
    if guarded:
        for _batch in isolated_iter(loader):
            pass
    else:
        for _batch in loader:
            pass
    return torch_module.rand(4).clone()


def diagnose(seed: int = 20260917) -> Dict[str, Any]:
    """Reproduce the RNG-state leak from scratch against the currently
    installed torch build, for every config in PROBE_CONFIGS, and verify
    the isolated_iter() guard neutralizes it in every case. Never trusts
    a cached/prior result -- every call re-runs the actual repro."""
    torch_module, dataloader_cls, _dataset_cls = _import_torch()

    results: List[LeakDiagnosis] = []
    for cfg in PROBE_CONFIGS:
        baseline = _draw_without_loader(torch_module, seed)
        raw = _run_probe(torch_module, dataloader_cls, cfg, seed, guarded=False)
        guarded = _run_probe(torch_module, dataloader_cls, cfg, seed, guarded=True)

        diff_raw = (baseline - raw).abs().max().item()
        diff_guarded = (baseline - guarded).abs().max().item()

        results.append(
            LeakDiagnosis(
                name=cfg.name,
                leak_present=diff_raw > 1e-9,
                guard_effective=diff_guarded <= 1e-9,
                max_abs_diff_raw=diff_raw,
                max_abs_diff_guarded=diff_guarded,
            )
        )

    return {
        "torch_version": torch_module.__version__,
        "issue_urls": [
            "https://github.com/pytorch/pytorch/issues/11062",
            "https://github.com/pytorch/pytorch/issues/122697",
            "https://github.com/pytorch/pytorch/issues/107443",
        ],
        "results": [dataclasses.asdict(r) for r in results],
        "any_leak_present": any(r.leak_present for r in results),
        "guard_fully_effective": all(r.guard_effective for r in results),
        "limitation": (
            "Probes only cover num_workers=0 (in-process). The leak is "
            "documented upstream as independent of num_workers, but "
            "num_workers>0 multiprocessing worker-seed behavior is not "
            "separately re-verified by this tool."
        ),
    }
