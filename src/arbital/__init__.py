"""arbital: orbit plots of general association.

The name: a(ssociation) + (o)rbital, association orbits grounded in
information theory (the "it"), for association learning (the "al").

Quick start::

    import arbital
    from arbital import datasets

    cars = arbital.orbits(datasets.load_mpg(), target="mpg")
    cars.to_html("orbits.html")     # interactive plot, open in browser
    cars.table()                    # metrics as a list of dicts

    arbital.select_features(datasets.load_mpg(), target="mpg")  # ranking only

Each feature's closest approach to the target (its periastron) is set by
its total information-theoretic association: the stronger the association,
the closer the feature swings in.
"""

from __future__ import annotations

import warnings

import numpy as np

from .measures import (profile, association_matrix, pearson, spearman,
                       mutual_information, mutual_information_mixed, linfoot,
                       shuffled_mi)
from .geometry import (orbit_parameters, angular_layout, greedy_selection,
                       select_target)
from .plot import orbit_figure, figure_html

# Note: `arbital.datasets` (optional example data, loaded via seaborn) is
# intentionally NOT imported here, so the core package has no data
# dependency.  Import it explicitly:  from arbital import datasets

__version__ = "0.1.1"
__all__ = ["orbits", "select_features", "OrbitSystem",
           "pearson", "spearman", "mutual_information", "linfoot",
           "profile", "select_target", "shuffled_mi"]


# ---------------------------------------------------------------------------
# Input handling: names, matrix, and which columns are categorical
# ---------------------------------------------------------------------------

def _resolve_input(X, categorical):
    """Return (matrix, names, discrete_flags) from array / DataFrame / Table.

    discrete_flags[j] is True when column j is categorical, so the right
    mutual-information estimator is used for it.  Sources, merged together:
      - a datasets.Table's .categorical set,
      - a pandas DataFrame's object / category / bool dtypes,
      - the explicit `categorical` argument (column names or indices).
    """
    # names
    if hasattr(X, "columns"):
        names = [str(c) for c in X.columns]
    else:
        names = [f"x{i}" for i in range(np.asarray(X).shape[1])]

    discrete = [False] * len(names)

    # a datasets.Table advertises its categorical columns
    if hasattr(X, "categorical"):
        for c in X.categorical:
            if str(c) in names:
                discrete[names.index(str(c))] = True

    # a pandas DataFrame exposes dtypes we can inspect
    dtypes = getattr(X, "dtypes", None)
    if dtypes is not None:
        for j, name in enumerate(names):
            kind = getattr(dtypes.iloc[j] if hasattr(dtypes, "iloc")
                           else dtypes[j], "kind", "f")
            if kind in ("O", "b", "U", "S"):     # object/bool/string
                discrete[j] = True

    # explicit override (names or integer indices)
    for c in (categorical or []):
        j = names.index(str(c)) if isinstance(c, str) else int(c)
        discrete[j] = True

    return _to_float_matrix(X, names, discrete), names, discrete


def _to_float_matrix(X, names, discrete):
    """Convert X to a float matrix, integer-coding string columns in place.

    A pandas DataFrame (or any columnar input) may hold string columns;
    np.asarray(..., float) would fail on those, so each offending column
    is encoded as integer category codes (and flagged in `discrete`).
    Rows containing a missing value (NaN, or a 'nan'/'None'/'' string in
    a coded column) are dropped: the estimators need complete rows.
    """
    try:
        A = np.asarray(X, dtype=float)
    except (ValueError, TypeError):
        cols = []
        for j, name in enumerate(names):
            raw = np.asarray(X[name] if hasattr(X, "columns") else X[:, j]).ravel()
            try:
                cols.append(raw.astype(float))
            except (ValueError, TypeError):
                s = raw.astype(str)
                missing = np.isin(s, ("nan", "None", "NaN", "<NA>", ""))
                labels = sorted(set(s[~missing]))
                code = {lab: float(i) for i, lab in enumerate(labels)}
                col = np.array([code.get(v, np.nan) for v in s])
                cols.append(col)
                discrete[j] = True
        A = np.column_stack(cols)
    keep = np.isfinite(A).all(axis=1)
    return A[keep] if not keep.all() else A


# ---------------------------------------------------------------------------
# The computed system
# ---------------------------------------------------------------------------

class OrbitSystem:
    """Computed association measures + orbit geometry for one target.

    Attributes:
      target_name   name of the target variable (at the centre)
      names         feature names, in input order
      metrics       per-feature dict: pearson, spearman, mi, r_info,
                    r_mono, nonlinearity, categorical, r_peri, r_apo, a, e
                    (e equals nonlinearity: the orbit eccentricity is nu)
      thetas        angular position per feature (radians)
      sizes         per-feature value driving marker size (0..1)
      size_label    what `sizes` means ('gain', 'rinfo', or 'uniform')
      ranks         greedy selection pick order per feature (always computed)
      gains         marginal gain per feature at pick time
      show_picks    whether the plot labels features with their pick number
      se            per-feature bootstrap SE of r_info, or None
    """

    def __init__(self, target_name, names, metrics, thetas, sizes, size_label,
                 ranks, gains, show_picks=False, se=None, scale="info",
                 n_rows=None, assoc=None):
        self.target_name = target_name
        self.names = names
        self.metrics = metrics
        self.thetas = thetas
        self.sizes = sizes
        self.size_label = size_label
        self.ranks = ranks
        self.gains = gains
        self.show_picks = show_picks
        self.se = se
        self.scale = scale
        self.n_rows = n_rows          # rows measured (after any subsampling)
        self.assoc = assoc            # feature-feature r_info matrix

    def figure(self, title=None) -> dict:
        """Plotly figure as a plain dict (data + layout)."""
        # pick-number labels are shown only when requested; marker size is
        # always driven by `sizes` regardless
        ranks = self.ranks if self.show_picks else None
        gains = self.gains if self.show_picks else None
        return orbit_figure(
            self.names, self.metrics, self.thetas, self.sizes,
            self.size_label, target_name=self.target_name, title=title,
            scale=self.scale, ranks=ranks, gains=gains, se=self.se,
            assoc=self.assoc, n_rows=self.n_rows)

    def to_plotly(self, title=None):
        """Live plotly Figure object (requires the optional plotly install)."""
        import plotly.graph_objects as go
        return go.Figure(self.figure(title))

    def to_html(self, path=None, title=None) -> str:
        """Standalone HTML (plotly.js from CDN). Writes to `path` if given."""
        snippet = figure_html(self.figure(title))
        page = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>arbital</title></head>"
                f"<body style='margin:0;background:#0b0e1a'>{snippet}</body></html>")
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(page)
        return page

    def table(self) -> list:
        """Per-feature metrics, sorted by selection pick order."""
        rows = []
        for i, (n, m, t) in enumerate(zip(self.names, self.metrics, self.thetas)):
            row = dict(name=n, theta=float(t), size=float(self.sizes[i]),
                       pick=int(self.ranks[i]), gain=float(self.gains[i]), **m)
            if self.se is not None:
                row["r_info_se"] = float(self.se[i])
            rows.append(row)
        return sorted(rows, key=lambda r: r["pick"])

    def to_df(self):
        """Per-feature metrics as a pandas DataFrame (needs pandas).

        One row per feature, indexed by name, sorted by selection pick
        order; every key from table() becomes a column (r_info, mi,
        pearson, spearman, nonlinearity, chance, below_chance, pick,
        gain, distances, ...).  A convenience for exporting or joining
        with other tables; arbital itself does not require pandas.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_df() needs pandas; install it with 'pip install pandas' "
                "or use table(), which returns plain dicts") from exc
        return pd.DataFrame(self.table()).set_index("name")

    def selection(self) -> list:
        """The greedy mRMR walkthrough: one dict per pick, in order.

        `relevance` is r_info_adj when calibration ran (the quantity
        greedy_selection() actually optimized), falling back to the
        plotted r_info when calibrate=False; `redundancy` is derived as
        relevance - gain to match, so the three columns are always
        mutually consistent with what orbits() computed.
        """
        rows = []
        for r in self.table():
            relevance = r.get("r_info_adj", r["r_info"])
            rows.append({"pick": r["pick"], "name": r["name"],
                         "relevance": relevance,
                         "redundancy": relevance - r["gain"],
                         "gain": r["gain"]})
        return rows


# ---------------------------------------------------------------------------
# Core computation shared by orbits() and select_features()
# ---------------------------------------------------------------------------

def _compute(X, target, n_neighbors, categorical, max_samples, random_state):
    """Shared work: resolve the target, measure every feature vs it, and
    build the feature-feature association matrix.

    Returns (target_name, feat_names, metrics, assoc, feat_discrete).
    """
    A, names, discrete = _resolve_input(X, categorical)
    if A.ndim != 2 or A.shape[1] < 2:
        raise ValueError("X must be 2-D with at least 2 columns")

    if A.shape[0] > max_samples:                 # keep the O(n^2) MI fast
        rng = np.random.default_rng(random_state)
        keep = rng.choice(A.shape[0], max_samples, replace=False)
        A = A[keep]

    if target is None:
        t_idx = select_target(A, k=n_neighbors)
    elif isinstance(target, str):
        t_idx = names.index(target)
    else:
        t_idx = int(target)
    target_name = names[t_idx]
    y_disc = discrete[t_idx]

    y = A[:, t_idx]
    feat_idx = [j for j in range(A.shape[1]) if j != t_idx]
    feat_names = [names[j] for j in feat_idx]
    feat_disc = [discrete[j] for j in feat_idx]
    F = A[:, feat_idx]

    metrics = []
    for j in range(F.shape[1]):
        m = profile(F[:, j], y, k=n_neighbors,
                    x_discrete=feat_disc[j], y_discrete=y_disc)
        metrics.append(m)

    assoc = association_matrix(F, k=n_neighbors, discrete=feat_disc)
    return target_name, feat_names, metrics, assoc, feat_disc, F, y, y_disc


def _chance_levels(F, y, feat_disc, y_disc, n_neighbors, n_shuffles,
                   random_state, confidence):
    """Chance levels for both channels, at the same confidence, from the
    same permutation draws: one shuffle procedure, one quantile, reused
    twice, instead of an information-channel permutation test sitting
    next to an unrelated analytic formula for the monotone channel.

    Returns (mi_levels, pearson_chance, spearman_chance):

      mi_levels        per-feature array, nats.  Continuous features
        share one pooled value (see below); categorical features each
        get their own (the plug-in estimator's chance value scales with
        the feature's level counts, roughly (Kx-1)(Ky-1)/(2n) in
        Miller-Madow terms, so a high-cardinality column has a materially
        higher noise floor than a binary one, and pooling across
        categorical features would be wrong).
      pearson_chance, spearman_chance   single floats (pooled over
        continuous features only; meaningless for categorical ones,
        whose pearson/spearman are always 0).

    Pooling assumption (continuous features, both channels): after the
    KSG estimator's internal standardisation, its null distribution
    depends on the sample size and k and only weakly on a continuous
    feature's marginal shape, and the same is true of a shuffled
    correlation coefficient, so all continuous features share one
    pooled level per channel, with the shuffle draws cycling round-robin
    over the actual features.  This is an asserted approximation, not
    something this function verifies for your specific data: skewed,
    heavy-tailed, or near-duplicate-valued columns can shift an
    individual feature's true null away from the pooled estimate (see
    the vignette's caveats section).

    `confidence` (the same quantile for every channel, e.g. 0.95) is a
    tunable parameter, not a fixed constant, so raise it to be more
    conservative, e.g. when screening many features at once (this
    module applies no multiple-comparison correction: at the default
    0.95 with p features independently tested, the expected number of
    false "clears chance" verdicts grows with p).

    Cost: n_shuffles draws for the continuous pool (each producing an MI,
    a |pearson|, and a |spearman| estimate) plus n_shuffles per
    categorical feature, not n_shuffles * p.  A 1 - confidence quantile
    from only n_shuffles draws is itself a noisy estimate (at the
    default 200 shuffles and 0.95 confidence, only ~10 draws are expected
    beyond it); a warning fires when that expected count is small.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    expected_tail = n_shuffles * (1.0 - confidence)
    if expected_tail < 10:
        warnings.warn(
            f"n_shuffles={n_shuffles} at confidence={confidence} leaves "
            f"only ~{expected_tail:.1f} draws beyond the quantile: the "
            f"chance level is a noisy estimate; raise n_shuffles for a "
            f"more stable boundary.", stacklevel=3)

    rng = np.random.default_rng(random_state + 7)
    p = F.shape[1]
    mi_levels = np.zeros(p)
    cont = [j for j in range(p) if not feat_disc[j]]
    pearson_chance = spearman_chance = 0.0
    if cont:
        mi_draws = np.empty(n_shuffles)
        r_draws = np.empty(n_shuffles)
        rho_draws = np.empty(n_shuffles)
        for b in range(n_shuffles):
            j = cont[b % len(cont)]
            y_shuf = rng.permutation(y)
            mi_draws[b] = mutual_information_mixed(
                F[:, j], y_shuf, False, y_disc, k=n_neighbors)
            r_draws[b] = abs(pearson(F[:, j], y_shuf))
            rho_draws[b] = abs(spearman(F[:, j], y_shuf))
        mi_levels[cont] = float(np.quantile(mi_draws, confidence))
        pearson_chance = float(np.quantile(r_draws, confidence))
        spearman_chance = float(np.quantile(rho_draws, confidence))
    for j in range(p):
        if feat_disc[j]:
            draws = shuffled_mi(F[:, j], y, True, y_disc, k=n_neighbors,
                                n_shuffles=n_shuffles, rng=rng)
            mi_levels[j] = float(np.quantile(draws, confidence))
    return mi_levels, pearson_chance, spearman_chance


def _apply_calibration(metrics, mi_chance, pearson_chance, spearman_chance):
    """Attach chance-calibration fields without moving the marker.

    r_info (and everything geometry.py derives from it: r_peri, the
    orbit, nonlinearity) stays exactly the estimated value profile()
    computed; calibration never rewrites it.  This is deliberate: the
    chance boundary drawn in the figure is a threshold on that same
    estimated r_info, so "beyond the dashed circle" is a direct
    geometric fact you can read by eye.  Calibrating the marker's own
    position as well would double-count chance: a feature sitting
    exactly at the chance level would get pulled in to r_info = 0 (the
    very rim) by the position adjustment, and then look emphatically
    "below" a boundary circle that represents that identical threshold,
    when really it belongs exactly on the line.  One subtraction, one
    place: the comparison, not the coordinates.

    Calibration earns its keep in relevance instead.  The k-NN estimator
    cannot return zero on finite data, so the estimated r_info has a
    positive noise floor (~0.1-0.25 at a few hundred rows) that would
    otherwise let an unassociated feature win a greedy mRMR round on
    estimator noise alone.  So a second quantity, r_info_adj, subtracts
    the estimator's own chance level (measured against shuffled copies
    of the target) in nats before mapping back through Linfoot, never on
    the 0-1 scale, where the map's steepness near zero would make
    `estimate - chance` the wrong arithmetic:

        mi_adj = max(0, I - chance);   r_info_adj = linfoot(mi_adj)

    orbits() and select_features() use r_info_adj (not r_info) as
    relevance for greedy_selection(), so a below-chance feature scores
    (near-)zero relevance and cannot win a pick, without its marker
    moving an inch.  A feature is below_chance when its estimated MI
    does not clear the chance level (mi <= chance).

    Pearson and Spearman are reported separately and are not folded into
    r_info, r_info_adj, or the radius: they never were the total
    association, only its monotone slice (r_mono = max(|pearson|,
    |spearman|), used for the ghost marker / nonlinear share, unaffected
    by this function; profile() already floors r_info at r_mono, so the
    two stay consistent without any clipping here).  Each coefficient is
    checked against pearson_chance / spearman_chance, the same
    confidence, from the same permutation draws as mi_chance, computed
    once by _chance_levels() and passed in here (see its docstring):
    `pearson_sig` / `spearman_sig` are False when the coefficient does
    not clear that level.  This is a separate question from whether
    r_info clears the chance boundary (one channel can pass while the
    other doesn't, see the vignette's caveats section for when that
    happens and why it's not a bug), and deliberately not mixed with it.

    `chance` is mi_chance in r_info units, `chance_nats` the same level
    in nats.  For continuous features mi_chance (and therefore `chance`)
    is one pooled value shared by every continuous feature in the
    dataset; categorical features each carry their own.
    """
    for j, m in enumerate(metrics):
        m["chance"] = linfoot(mi_chance[j])
        m["chance_nats"] = float(mi_chance[j])
        i_cal = max(0.0, m["mi"] - mi_chance[j])
        m["mi_adj"] = i_cal
        m["r_info_adj"] = linfoot(i_cal)
        m["below_chance"] = bool(m["mi"] <= mi_chance[j])
        m["pearson_chance"] = pearson_chance
        m["spearman_chance"] = spearman_chance
        m["pearson_sig"] = bool(abs(m["pearson"]) > pearson_chance)
        m["spearman_sig"] = bool(abs(m["spearman"]) > spearman_chance)
    return metrics


def _bootstrap_se(F, y, y_disc, feat_disc, n_neighbors, n_bootstrap, random_state):
    """Bootstrap standard error of r_info for each feature vs the target.

    Nonparametric bootstrap (Efron 1979, "Bootstrap methods: another look
    at the jackknife", Annals of Statistics 7(1), 1-26): resamples rows
    with replacement n_bootstrap times and recomputes r_info;
    the spread of those estimates becomes the arc's angular sweep, so a
    shaky association reads as a wide, smeared orbit.
    """
    rng = np.random.default_rng(random_state + 1)
    n = F.shape[0]
    ses = []
    for j in range(F.shape[1]):
        vals = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            mi = mutual_information_mixed(F[idx, j], y[idx],
                                          feat_disc[j], y_disc, k=n_neighbors)
            vals.append(linfoot(mi))
        ses.append(float(np.std(vals)))
    return np.array(ses)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def orbits(X, target=None, *, n_neighbors: int = 5,
           angle_layout: str = "spread", scale: str = "info",
           size: str = "gain", selection: bool = False,
           uncertainty: bool = False, n_bootstrap: int = 100,
           categorical=None, calibrate: bool = True, confidence: float = 0.95,
           n_shuffles: int = 200,
           max_samples: int = 2000, random_state: int = 0) -> OrbitSystem:
    """Compute the orbit system for dataset X around a target column.

    The two positional arguments are the data and the target; everything
    else is keyword-only, following scikit-learn conventions
    (n_neighbors, random_state, max_samples).

    Args:
      X        2-D array, DataFrame, or a datasets.Table.
      target   column name or index of the target, or None to auto-pick the
               column most associated with all others.
      n_neighbors  neighbours for the k-NN mutual-information estimators.
      angle_layout  "spread" (default): embedding angles with a minimum
               separation enforced, so angular gaps keep their relative
               meaning (redundant features huddle, unrelated ones sit
               apart) without labels piling up; "embed": the raw
               embedding angles; "ordered": even spacing that keeps only
               the circular order (gaps carry no magnitude).
      scale    "info" (default): distance = exp(-I), logarithmic in mutual
               information; "linear": distance = 1 - r_I.
      size     what marker AREA encodes:
                 "gain" (default): the marginal gain from greedy selection,
                     scaled relative to the top pick (largest marker = the
                     feature selected first; near-duplicates shrink).  This
                     reads together with the radius: inner orbits are the
                     strongest associations and are selected first, so a
                     large inner marker is a strong, non-redundant feature.
                 "rinfo": total association r_I.
                 "uniform": constant (size carries nothing).
      selection  if True, label each feature with its selection pick number
               and sort the table by pick order (default False).  Marker
               size already reflects the selection via size="gain"; this
               flag only adds the explicit numbering.
      uncertainty  if True, bootstrap a standard error of r_info for every
               feature and draw it as a radial band through the marker
               (r_info +/- se): uncertainty acts along the radius, so the
               band shows directly whether the estimate could sit closer
               in or farther out, and a diamond flags any crossing of the
               chance boundary (the association could be noise).
      n_bootstrap  bootstrap resamples used when uncertainty=True.
      categorical  extra column names or indices to treat as categorical
               (string columns and Table categoricals are detected already).
      calibrate  if True (default), measure each estimator's chance level
               (its 95th-percentile output against shuffled copies of the
               target, i.e. under exact independence) and use it for two
               things that do not move any marker: (1) a dashed boundary
               circle in the figure, drawn at that level, and a
               `below_chance` flag per feature; (2) relevance for the
               greedy selection is the chance-subtracted r_info_adj, not
               the plotted r_info, so the k-NN estimator's positive noise
               floor (~0.1-0.25 at a few hundred rows) cannot win an
               unassociated feature a selection round.  r_info itself is
               always the estimated value, at every setting of this flag;
               calibrating the position too would double-count chance
               (see `_apply_calibration`'s docstring).  Pearson and
               Spearman are checked against their own chance level too
               (`pearson_sig`/`spearman_sig`), from the same shuffle
               draws and the same `confidence`, see `_chance_levels`.
               The table carries `chance`, `chance_nats`, `mi_adj`,
               `r_info_adj`, `below_chance`, `pearson_chance`,
               `spearman_chance`, `pearson_sig`, `spearman_sig` when
               this is True. Note: this is a permutation-based screen
               with no multiple-comparison correction across the p
               features tested at once, and the boundary circle in the
               figure reflects only the pooled continuous level; a
               categorical feature's own level (cardinality-dependent)
               can differ from the drawn circle. See the vignette's
               "Assumptions and calibration caveats" section.
      confidence  the quantile used for every chance level (MI, Pearson,
               Spearman alike), default 0.95. Raise it (e.g. to 0.99) to
               be more conservative, particularly when screening many
               features at once, since no automatic correction is
               applied for that multiple-comparisons setting.
      n_shuffles  shuffle draws behind the chance levels.  Continuous
               features share one pooled level per channel so the cost
               is n_shuffles draws total (plus n_shuffles per
               categorical feature), not n_shuffles per feature. A
               warning fires if n_shuffles * (1 - confidence) < 10 (too
               few draws beyond the quantile for a stable estimate).
      max_samples  rows are subsampled beyond this (MI is O(n^2)).
      random_state  seed for subsampling, calibration shuffles, and the
               bootstrap; results are deterministic given the same seed.
    """
    (target_name, feat_names, metrics, assoc,
     feat_disc, F, y, y_disc) = _compute(X, target, n_neighbors, categorical,
                                         max_samples, random_state)

    if calibrate:
        mi_chance, pearson_chance, spearman_chance = _chance_levels(
            F, y, feat_disc, y_disc, n_neighbors, n_shuffles, random_state,
            confidence)
        _apply_calibration(metrics, mi_chance, pearson_chance, spearman_chance)

    for m in metrics:                            # attach orbit geometry
        m.update(orbit_parameters(m["r_info"], m["r_mono"], scale=scale))

    thetas = angular_layout(assoc, layout=angle_layout)

    # greedy selection always runs: it drives the default marker size and
    # the optional pick numbering.  Relevance is the adjusted r_info
    # (r_info_adj) when available, never the plotted r_info: selection
    # and position are deliberately different numbers (see
    # _apply_calibration).
    relevance = np.array([m.get("r_info_adj", m["r_info"]) for m in metrics])
    ranks, gains = greedy_selection(relevance, assoc)

    if size == "gain":
        g = np.clip(gains, 0.0, None)
        top = g.max()
        sizes = g / top if top > 0 else np.zeros_like(g)   # relative to top pick
    elif size == "rinfo":
        sizes = np.array([m["r_info"] for m in metrics])
    elif size == "uniform":
        sizes = np.full(len(metrics), 0.5)
    else:
        raise ValueError(f"unknown size {size!r}, use 'gain', 'rinfo' or 'uniform'")

    se = None
    if uncertainty:
        se = _bootstrap_se(F, y, y_disc, feat_disc, n_neighbors, n_bootstrap,
                           random_state)

    return OrbitSystem(target_name, feat_names, metrics, thetas, sizes, size,
                       ranks, gains, show_picks=selection, se=se, scale=scale,
                       n_rows=F.shape[0], assoc=assoc)


def select_features(X, target=None, *, n_neighbors: int = 5,
                    categorical=None, calibrate: bool = True,
                    confidence: float = 0.95, n_shuffles: int = 200,
                    max_samples: int = 2000, random_state: int = 0) -> list:
    """Greedy mRMR feature selection, independent of any visualization.

    By default (calibrate=True) relevance is chance-adjusted: each
    estimator's output against shuffled copies of the target (its value
    under exact independence) is subtracted in nats before conversion to
    r_I, so unassociated features score ~0 relevance and cannot win a
    selection round on estimator noise alone.  This is the same
    r_info_adj orbits() uses for relevance, see its docstring and
    _apply_calibration for why it is a different number from the
    (uncalibrated) r_info reported elsewhere.  `confidence` (default
    0.95) sets the quantile behind that chance level, and behind
    `below_chance`; no multiple-comparison correction is applied across
    the features screened here (see `_chance_levels`'s docstring and the
    vignette's caveats section).

    Returns a list of dicts in pick order, each with:
      pick          1-based selection order
      name          feature name
      relevance     chance-adjusted r_info_adj(feature, target) (the
                    estimated r_info if calibrate=False)
      redundancy    mean r_info to the already-selected features
      gain          relevance - redundancy (the marginal information added;
                    gain <= 0 marks the natural stopping point)
      below_chance  True when the measured MI does not clear the chance
                    level (only present when calibrate=True)

    Example::

        for row in arbital.select_features(df, target="price"):
            print(row["pick"], row["name"], round(row["gain"], 3))
    """
    (target_name, feat_names, metrics, assoc,
     feat_disc, F, y, y_disc) = _compute(X, target, n_neighbors, categorical,
                                         max_samples, random_state)
    if calibrate:
        mi_chance, pearson_chance, spearman_chance = _chance_levels(
            F, y, feat_disc, y_disc, n_neighbors, n_shuffles, random_state,
            confidence)
        _apply_calibration(metrics, mi_chance, pearson_chance, spearman_chance)
    relevance = np.array([m.get("r_info_adj", m["r_info"]) for m in metrics])
    ranks, gains = greedy_selection(relevance, assoc)
    rows = []
    for i in range(len(feat_names)):
        row = {"pick": int(ranks[i]), "name": feat_names[i],
               "relevance": float(relevance[i]),
               "redundancy": float(relevance[i] - gains[i]),
               "gain": float(gains[i])}
        if calibrate:
            row["below_chance"] = metrics[i]["below_chance"]
        rows.append(row)
    return sorted(rows, key=lambda r: r["pick"])
