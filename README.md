[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# rng-leak-audit

Detect whether constructing a PyTorch `DataLoader` iterator silently consumes global RNG state, and use a wrapper that neutralizes it. This targets the behavior described in [pytorch/pytorch#11062](https://github.com/pytorch/pytorch/issues/11062) (open since 2018), [#122697](https://github.com/pytorch/pytorch/issues/122697), and [#107443](https://github.com/pytorch/pytorch/issues/107443) — not every possible source of training non-reproducibility.

The bug: merely calling `iter(dataloader)` — even with `shuffle=False`, `num_workers=0`, and a dataset with no randomness anywhere in it — draws from the global torch RNG to build an internal `_base_seed`. Any code that runs afterward and expects reproducible `torch.rand()`/`torch.randn()` output silently gets a different sequence than it would have without the DataLoader ever being touched. This is most dangerous when an unrelated validation/eval `DataLoader` is iterated between training steps: running validation perturbs the *training* run's subsequent random draws, so two training runs that only differ in "did validation run this epoch" can silently diverge, with no error, warning, or visible symptom. The CLI reproduces this from scratch on your installed PyTorch build rather than assuming a particular version is affected.

## Install and diagnose

Requires Python 3.9+ and a PyTorch build. From source:

```bash
git clone https://github.com/zhuhroscar-tech/rng-leak-audit.git
cd rng-leak-audit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
rng-leak-audit
rng-leak-audit --json
```

If you already manage a compatible PyTorch installation, install `.` without the extra. Use `--seed` for a reproducible probe and `--no-color` for plain text.

Exit codes: **0** means the guard matched the untouched baseline for every tested configuration, **1** means at least one guard check failed, and **2** means PyTorch could not be imported. A successful guard check does not mean the upstream leak was reproduced on your build; inspect `any_leak_present` separately.

## Python API

Wrap the point where you iterate a DataLoader (e.g. a validation loop
you don't want to perturb subsequent training-loop randomness):

```python
from rng_leak_audit import isolated_iter

for batch in isolated_iter(val_loader):
    ...  # evaluate; the global RNG state is restored once this loop ends
```

`isolated_iter()` snapshots `torch.get_rng_state()` (and every visible CUDA device's RNG state) once, before consuming the loader at all, then restores it exactly once — when the loop ends normally, via `break`, or via an exception — using a `try/finally` around full iteration. Both DataLoader-iterator construction and per-batch `shuffle=True` permutation draws are neutralized, since both were independently confirmed to consume global RNG state on this host (see Scope and limitations).

## Scope and limitations

The probe covers `shuffle=False` and `shuffle=True` with `num_workers=0` (in-process) on CPU tensors. The leak is documented upstream as independent of `num_workers`, but `num_workers>0` multiprocessing worker-seed behavior is not separately re-verified by this tool — treat that as untested, not confirmed-safe. This tool does not audit `IterDataPipe`/`torchdata` pipelines, distributed samplers, or CUDA-specific RNG paths beyond the isolated_iter() CUDA-state save/restore itself. Restoration happens immediately after `iter()` returns, not in a `finally` block wrapping the whole training step — an exception during iterator construction itself (rare) could still leave state perturbed.

## Development

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest -v
```

[MIT license](LICENSE).
