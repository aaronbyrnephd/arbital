# Changelog

All notable changes to arbital are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-07-15

First public release.

- Orbit plots: every variable measured against a target with both correlation
  (Pearson/Spearman) and mutual information, drawn as an orbit whose radius,
  eccentricity, angle, marker size, and colour stand in for strength,
  nonlinearity, redundancy, feature-selection value, and direction.
- Mutual information estimators for continuous, categorical, and mixed pairs
  (KSG, Ross, and a plug-in estimator), put on the same scale as correlation
  via Linfoot's r_I.
- Chance calibration against shuffled copies of the target, so a feature that
  isn't actually associated with anything can't win a feature-selection pick
  on estimator noise alone.
- Greedy mRMR feature selection, usable on its own or as the marker sizes and
  pick order in the figure.
- Optional bootstrap uncertainty bands on each marker.
- A few layout and scale options (`spread`/`embed`/`ordered` angles,
  `info`/`linear` radial scale), plus hover-to-trace-the-full-orbit.
- Four bundled offline datasets, with everything else in seaborn available
  via an optional extra.
- Standalone interactive HTML output, live Plotly figures, and plain
  dict/DataFrame tables of the underlying metrics.
- A usage vignette, tutorial notebook, and API reference.
