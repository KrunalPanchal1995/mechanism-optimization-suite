# tests/

Unit tests for the MUQ-SAC production modules, runnable both as
`pytest tests/` and directly as `python3.9 tests/test_doptimal.py` (from any
cwd — `conftest.py` / the file's own inline bootstrap both resolve the
MUQ-SAC root from `__file__`, not cwd).

## Files

- `conftest.py` — pytest-only bootstrap: adds the MUQ-SAC root
  (`test_build/` today, `src/MUQ-SAC/` once mirrored) to `sys.path` so
  `import AdaptiveResampler` etc. resolve, no matter where `pytest` is
  invoked from.
- `test_doptimal.py` — 7 synthetic unit tests (9 assertions across them,
  matching the "9/9 tests" figure in the plan's verification gate) for
  `AdaptiveResampler.DOptimalResampler`, ported as-is from
  `test_build/test_doptimal.py`. Uses a `MockExtractor` standing in for the
  real `Uncertainty.UncertaintyExtractor` (see the file's own docstring for
  why, and for a 2026-07-31 note on re-checking that rationale once this
  runs from `src/MUQ-SAC/`, which has the files that make the real
  extractor importable — not re-checked in this task, which does not touch
  `src/MUQ-SAC/`).

## Running

```
python3.9 test_build/tests/test_doptimal.py         # direct
cd / && python3.9 test_build/tests/test_doptimal.py  # from a foreign cwd -- still works
pytest test_build/tests/ -v                          # via pytest
```
