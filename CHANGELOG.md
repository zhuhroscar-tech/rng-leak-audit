# Changelog

## v0.2.4 — 2026-09-26

- Added package resource links for the project homepage, issue tracker, and changelog.
- Made release-tag CI coverage explicit for `v*` tags.
- Added repository-contract coverage so package metadata and release-tag CI wiring stay intact.

## v0.2.3 — 2026-09-24

- Added a Chinese README that the language switcher already advertised.
- Added repository-contract tests for required files, README links, release history, CI artifacts, CodeQL, and version/changelog parity.
- Added `MANIFEST.in` so source distributions include release notes, tests, workflows, and bilingual README files.

## v0.2.2 — 2026-09-24

- Modernized packaging license metadata to the current SPDX-string format.
- Declared `LICENSE` through `license-files`, removed the deprecated MIT license classifier, and raised the setuptools build floor to `>=77`.
- Added regression coverage so deprecated license metadata does not return.

## v0.2.1 — 2026-09-23

- Added a real terminal screenshot of example CLI output to the README.

## v0.2.0 — 2026-09-20

- Reworked `isolated_iter()` so it can be used as a context manager for deterministic, garbage-collection-independent RNG restoration.
- Restored RNG state when `iter(dataloader)` raises after mutating state.
- Restored RNG state when the wrapped iterator's `__next__` raises mid-consumption.
- Documented the remaining limitation of the bare iterator form for manual `next()`-driven callers that keep the iterator alive.

## v0.1.0 — 2026-09-17

- Initial release.
- Added CLI diagnostics for PyTorch DataLoader global-RNG-state leakage.
- Added `isolated_iter()` to preserve torch CPU and visible CUDA RNG state around DataLoader iteration.
