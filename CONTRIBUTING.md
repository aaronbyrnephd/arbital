# Contributing to arbital

Thanks for your interest. Contributions of every size are welcome: bug
reports, documentation fixes, new estimators, and examples.

## Reporting bugs and requesting features

Open a GitHub issue. For bugs, include a minimal reproducible example (a
few lines constructing an array and the call that misbehaves), the arbital
and NumPy versions, and what you expected. For feature requests, describe
the use case before the mechanism — the visual grammar is deliberately
small, and new encodings need to earn their channel.

## Development setup

```bash
git clone https://github.com/aaronbyrnephd/arbital
cd arbital
pip install -e ".[test]"
pytest -q                 # the full suite runs in about a minute
ruff check src/ tests/    # lint
```

The vignette rebuilds with `PYTHONPATH=src python3 demo/make_demo.py`, and
the README's static figures with `PYTHONPATH=src python3 assets/make_figure.py`.

## Design ground rules

- **The core stays pure NumPy.** Estimators are implemented here, readably,
  rather than delegated; new runtime dependencies need a strong case.
  Optional integrations belong in extras.
- **Tests state theory, not snapshots.** Each test asserts something an
  estimator or mapping must do on data with known ground truth (e.g.
  "Linfoot's r_I recovers |rho| on Gaussian data"). Follow that style.
- **Every visual channel documents what it encodes** — and when it
  deliberately encodes nothing (e.g. arc sweep without `uncertainty=True`).
  If you add or change an encoding, update `plot.py`'s docstring, the
  README table, and the vignette together.
- **Cite sources.** New estimators or procedures reference the paper they
  implement, in the docstring and in `README.md`.

## Pull requests

Branch from `main`, keep PRs focused, add or adjust tests for behaviour
changes, and make sure `pytest` and `ruff check` pass. CI runs the suite on
Python 3.9–3.13 plus a minimum-NumPy job. Note user-visible changes under
"Unreleased" in `CHANGELOG.md`.

## Questions

Open a GitHub Discussion (or an issue labelled "question").
