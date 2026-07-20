"""Tests: each checks that an estimator or geometric mapping behaves the
way the theory says it must on data where the ground truth is known."""

import numpy as np
import pytest

import arbital
from arbital.measures import _digamma, linfoot
from arbital.geometry import (orbit_parameters, angular_layout,
                              greedy_selection, strength_to_distance)
from arbital import datasets

RNG = np.random.default_rng(42)
N = 600


def _load_or_skip(loader):
    """Load a seaborn-backed dataset, or skip if seaborn or the network
    (seaborn downloads on first use) is unavailable."""
    try:
        return loader()
    except Exception as exc:                     # ImportError, URLError, ...
        pytest.skip(f"dataset unavailable: {exc}")


# ---------------------------------------------------------------- measures

def test_pearson_matches_numpy():
    x, y = RNG.standard_normal(N), RNG.standard_normal(N)
    assert arbital.pearson(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-12)


def test_spearman_perfect_monotone():
    # y = exp(x) is nonlinear but perfectly monotone: rho == 1, r < 1
    x = RNG.standard_normal(N)
    y = np.exp(x)
    assert arbital.spearman(x, y) == pytest.approx(1.0, abs=1e-12)
    assert abs(arbital.pearson(x, y)) < 1.0


def test_digamma_known_values():
    # psi(1) = -euler_gamma, psi(2) = 1 - euler_gamma
    g = 0.5772156649015329
    assert _digamma(np.array([1.0]))[0] == pytest.approx(-g, abs=1e-10)
    assert _digamma(np.array([2.0]))[0] == pytest.approx(1 - g, abs=1e-10)


def test_mi_independent_near_zero():
    x, y = RNG.standard_normal(N), RNG.standard_normal(N)
    assert arbital.mutual_information(x, y) < 0.1


def test_mi_gaussian_matches_theory():
    # bivariate Gaussian with rho=0.8: I = -0.5*ln(1-rho^2) = 0.5108 nats
    rho = 0.8
    x = RNG.standard_normal(N)
    y = rho * x + np.sqrt(1 - rho**2) * RNG.standard_normal(N)
    mi = arbital.mutual_information(x, y)
    assert mi == pytest.approx(-0.5 * np.log(1 - rho**2), abs=0.12)


def test_linfoot_recovers_gaussian_correlation():
    # the whole point of Linfoot's r_I: on Gaussian data it equals |rho|
    rho = 0.7
    x = RNG.standard_normal(N)
    y = rho * x + np.sqrt(1 - rho**2) * RNG.standard_normal(N)
    r_i = linfoot(arbital.mutual_information(x, y))
    assert r_i == pytest.approx(rho, abs=0.08)


# ------------------------------------------- sanity checks: who wins where
# These pin down the intended division of labour between the measures.

def test_rank_beats_linear_on_monotone_curve():
    # y = exp(2x) is monotone but strongly curved: Spearman should clearly
    # outperform Pearson, because ranks are invariant to the curvature.
    x = RNG.standard_normal(N)
    y = np.exp(2 * x) + 0.05 * RNG.standard_normal(N)
    assert abs(arbital.spearman(x, y)) > abs(arbital.pearson(x, y)) + 0.2


def test_info_beats_rank_on_non_monotone():
    # y = sin(2.5x) has no consistent direction: both correlations are
    # blind, but mutual information (via r_I) sees it clearly.
    x = RNG.standard_normal(N)
    y = np.sin(2.5 * x) + 0.1 * RNG.standard_normal(N)
    m = arbital.profile(x, y)
    assert m["r_mono"] < 0.35
    assert m["r_info"] > m["r_mono"] + 0.3


def test_mi_invariant_to_monotone_transform():
    # MI's defining property: reparametrising an axis with any smooth
    # monotone map should leave the estimate (nearly) unchanged.
    x = RNG.standard_normal(N)
    y = x + 0.5 * RNG.standard_normal(N)
    mi_raw = arbital.mutual_information(x, y)
    mi_exp = arbital.mutual_information(np.exp(x), y)   # monotone-transform x
    assert mi_exp == pytest.approx(mi_raw, abs=0.15)


def test_monotone_curve_shows_no_hidden_surplus():
    # A monotone curve (y = exp(x)) must NOT be flagged as nonlinear:
    # rank correlation already captures it, so the nonlinearity fraction
    # stays well below the 10% ghost-marker threshold used by the plot.
    # Note: raw eccentricity is NOT asserted here -- for near-perfect
    # associations both radii shrink toward 0 and e becomes an unstable
    # ratio of two small numbers; the nonlinear share is the quantity
    # that stays meaningful (and is what triggers ghosts).
    x = RNG.standard_normal(N)
    y = np.exp(x) + 0.05 * RNG.standard_normal(N)
    m = arbital.profile(x, y)
    assert m["nonlinearity"] < 0.05


# ---------------------------------------------------------------- geometry

def test_linear_feature_orbit_is_circular():
    x = RNG.standard_normal(N)
    y = x + 0.1 * RNG.standard_normal(N)
    m = arbital.profile(x, y)
    orb = orbit_parameters(m["r_info"], m["r_mono"])
    assert orb["e"] < 0.15            # near-circular
    assert m["nonlinearity"] < 0.25


def test_quadratic_feature_orbit_is_eccentric():
    # y = x^2 on symmetric x: Pearson & Spearman ~ 0, MI large
    x = RNG.standard_normal(N)
    y = x**2 + 0.05 * RNG.standard_normal(N)
    m = arbital.profile(x, y)
    assert m["r_mono"] < 0.2          # invisible to correlation
    assert m["r_info"] > 0.8          # obvious to MI
    orb = orbit_parameters(m["r_info"], m["r_mono"])
    assert orb["e"] > 0.3             # clearly elliptical
    assert m["nonlinearity"] > 0.8


def test_orbit_parameter_algebra_linear_scale():
    orb = orbit_parameters(r_info=0.8, r_mono=0.5, scale="linear")
    assert orb["r_peri"] == pytest.approx(0.2)   # 1 - r_info
    assert orb["r_apo"] == pytest.approx(0.5)     # 1 - r_mono
    assert orb["a"] == pytest.approx(0.35)
    # eccentricity is now the nonlinear share nu, not the geometric ratio
    assert orb["e"] == pytest.approx(1 - (0.5 / 0.8) ** 2)


def test_orbit_parameter_algebra_info_scale():
    # default scale: distances are residual uncertainties sqrt(1 - r^2)
    orb = orbit_parameters(r_info=0.8, r_mono=0.5)
    assert orb["r_peri"] == pytest.approx(np.sqrt(1 - 0.64))
    assert orb["r_apo"] == pytest.approx(np.sqrt(1 - 0.25))


def test_info_distance_is_log_in_information():
    # distance = exp(-I): stepping one nat closer multiplies distance by 1/e
    for mi in (0.5, 1.0, 2.0):
        d = strength_to_distance(linfoot(mi))
        d_next = strength_to_distance(linfoot(mi + 1.0))
        assert d_next / d == pytest.approx(np.exp(-1.0), abs=1e-9)
    # endpoints: no association -> rim, perfect association -> centre
    assert strength_to_distance(0.0) == pytest.approx(1.0)
    assert strength_to_distance(1.0) == pytest.approx(0.0)


def test_info_scale_spreads_strong_associations():
    # the motivating property: 0.90 vs 0.99 are cramped on the linear
    # scale (0.10 vs 0.01) but clearly separated on the info scale
    d90 = strength_to_distance(0.90)
    d99 = strength_to_distance(0.99)
    assert d90 - d99 > 0.25


def test_greedy_selection_defers_the_clone():
    # features 0 and 1 are near-clones (assoc 0.95), feature 2 is
    # independent.  0 is most relevant so it's picked first; then the
    # INDEPENDENT feature 2 should be preferred over the redundant clone 1.
    relevance = np.array([0.85, 0.80, 0.75])
    assoc = np.array([[1.0, 0.95, 0.1],
                      [0.95, 1.0, 0.1],
                      [0.1, 0.1, 1.0]])
    ranks, gains = greedy_selection(relevance, assoc)
    assert ranks[0] == 1               # most relevant picked first
    assert ranks[2] == 2               # independent feature beats the clone
    assert ranks[1] == 3               # redundant clone picked last
    # the clone's gain is small (it duplicates feature 0)
    assert gains[1] < gains[2]


def test_greedy_selection_ranks_everything_once():
    relevance = np.array([0.6, 0.9, 0.3, 0.7])
    assoc = np.eye(4) + 0.2 * (np.ones((4, 4)) - np.eye(4))
    ranks, gains = greedy_selection(relevance, assoc)
    assert sorted(ranks.tolist()) == [1, 2, 3, 4]   # a permutation
    assert ranks[1] == 1                            # highest relevance first


def test_redundant_feature_gets_nonpositive_gain():
    # an exact duplicate of a picked feature adds nothing -> gain <= 0
    relevance = np.array([0.9, 0.9])
    assoc = np.array([[1.0, 1.0], [1.0, 1.0]])
    ranks, gains = greedy_selection(relevance, assoc)
    second = int(np.argmax(ranks))     # the feature picked second
    assert gains[second] <= 0.0


def test_spread_layout_keeps_gap_structure_and_min_separation():
    # two tight pairs, far apart: "spread" must keep the pairs closer to
    # each other than to the other group, while no gap collapses below
    # the enforced minimum separation
    assoc = np.array([[1.0, 0.97, 0.05, 0.05],
                      [0.97, 1.0, 0.05, 0.05],
                      [0.05, 0.05, 1.0, 0.97],
                      [0.05, 0.05, 0.97, 1.0]])
    th = angular_layout(assoc)                      # default = "spread"

    def gap(a, b):
        d = abs(a - b) % (2 * np.pi)
        return min(d, 2 * np.pi - d)

    p = assoc.shape[0]
    g_min = 0.35 * 2 * np.pi / p
    # within-pair gaps respect the minimum but stay smaller than
    # the between-pair gaps
    assert gap(th[0], th[1]) >= g_min - 1e-9
    assert gap(th[2], th[3]) >= g_min - 1e-9
    assert gap(th[0], th[1]) < gap(th[0], th[2])
    assert gap(th[2], th[3]) < gap(th[1], th[3])
    # angles cover one full turn: sorted consecutive gaps sum to 2*pi
    s = np.sort(th % (2 * np.pi))
    total = np.diff(s, append=s[0] + 2 * np.pi).sum()
    assert total == pytest.approx(2 * np.pi, abs=1e-9)


def test_unknown_layout_raises():
    with pytest.raises(ValueError):
        angular_layout(np.eye(3), layout="circular")


def test_angular_layout_groups_associated_features():
    # features 0,1 tightly associated; 2,3 tightly associated; groups apart
    assoc = np.array([[1.0, 0.9, 0.1, 0.1],
                      [0.9, 1.0, 0.1, 0.1],
                      [0.1, 0.1, 1.0, 0.9],
                      [0.1, 0.1, 0.9, 1.0]])
    th = angular_layout(assoc, layout="embed")

    def gap(a, b):  # smallest angle between two directions
        d = abs(a - b) % (2 * np.pi)
        return min(d, 2 * np.pi - d)

    assert gap(th[0], th[1]) < gap(th[0], th[2])
    assert gap(th[2], th[3]) < gap(th[1], th[3])


# ---------------------------------------------------------------- end-to-end

def _demo_data(n=400):
    x = RNG.standard_normal(n)
    data = np.column_stack([
        x,                                    # y (target)
        x + 0.3 * RNG.standard_normal(n),     # linear
        x**2 + 0.2 * RNG.standard_normal(n),  # quadratic
        RNG.standard_normal(n),               # noise
    ])
    return data


def test_orbits_end_to_end():
    space = arbital.orbits(_demo_data(), target=0)
    assert space.target_name == "x0"
    assert len(space.names) == 3
    rows = {r["name"]: r for r in space.table()}
    # both informative features orbit far closer than the noise feature
    # (info scale: r_I ~ 0.95 -> distance ~ 0.3, noise -> distance ~ 1)
    assert rows["x1"]["r_peri"] < 0.5 and rows["x2"]["r_peri"] < 0.5
    assert rows["x3"]["r_peri"] > 0.85
    # quadratic feature has the more eccentric orbit
    assert rows["x2"]["e"] > rows["x1"]["e"] + 0.2


def test_auto_target_picks_hub():
    # x0 drives three noisy copies of itself; x4 is pure noise.
    # The hub (highest mean association with everything) must be x0.
    n = 400
    h = RNG.standard_normal(n)
    data = np.column_stack([h] +
        [h + 0.5 * RNG.standard_normal(n) for _ in range(3)] +
        [RNG.standard_normal(n)])
    assert arbital.select_target(data) == 0


def test_html_output(tmp_path):
    space = arbital.orbits(_demo_data(), target=0)
    f = tmp_path / "orbits.html"
    space.to_html(str(f))
    html = f.read_text()
    assert "Plotly.newPlot" in html and "cdn.plot.ly" in html


def test_to_df_exports_table():
    try:
        import pandas  # noqa: F401
    except ImportError:
        pytest.skip("pandas not installed")
    df = arbital.orbits(_demo_data(), target=0).to_df()
    assert df.shape[0] == 3                          # one row per feature
    for col in ("r_info", "mi", "pearson", "pick", "gain"):
        assert col in df.columns


def test_uncertainty_band_and_crossing_in_figure():
    # radial uncertainty band traces appear with uncertainty=True, and a
    # figure built without it carries none
    space = arbital.orbits(_demo_data(), target=0, uncertainty=True,
                      n_bootstrap=10)
    fig = space.figure()
    hovers = [tr.get("hovertext", "") for tr in fig["data"]
              if isinstance(tr.get("hovertext"), str)]
    assert any("uncertainty band" in h for h in hovers)
    plain = arbital.orbits(_demo_data(), target=0).figure()
    hovers = [tr.get("hovertext", "") for tr in plain["data"]
              if isinstance(tr.get("hovertext"), str)]
    assert not any("uncertainty band" in h for h in hovers)


def test_size_is_relative_gain_by_default():
    space = arbital.orbits(_demo_data(), target=0)
    assert space.size_label == "gain"
    # sizes are relative to the top pick: the first-picked feature is 1.0
    top = max(space.table(), key=lambda r: r["gain"])
    assert top["size"] == pytest.approx(1.0)
    assert all(0.0 <= s <= 1.0 for s in space.sizes)


def test_selection_numbering_is_opt_in():
    default = arbital.orbits(_demo_data(), target=0)
    numbered = arbital.orbits(_demo_data(), target=0, selection=True)
    # gains are always computed (they drive size), so selection() always works
    assert default.selection() and numbered.selection()
    # only the opt-in figure carries pick-number labels
    assert not default.show_picks and numbered.show_picks
    assert "Plotly.newPlot" in default.to_html()


def test_select_features_standalone():
    # feature selection returns picks + gains with NO visualization, on a
    # small synthetic frame (no seaborn needed)
    rng = np.random.default_rng(3)
    h = rng.standard_normal(400)
    X = np.column_stack([h,                                  # y (target)
                         h + 0.4 * rng.standard_normal(400),  # relevant
                         h + 0.5 * rng.standard_normal(400),  # redundant clone
                         rng.standard_normal(400)])           # noise

    class Frame:
        columns = ["y", "signal", "clone", "noise"]
        def __array__(self, dtype=None): return X
    rows = arbital.select_features(Frame(), target="y")
    assert [r["pick"] for r in rows] == sorted(r["pick"] for r in rows)
    assert rows[0]["name"] in ("signal", "clone")       # a real signal first
    assert {"relevance", "redundancy", "gain"} <= set(rows[0])


def test_string_columns_and_missing_rows_are_handled():
    # a columnar input with a string column and a missing value: the
    # string column must be integer-coded as categorical, and the row
    # with the missing value dropped -- this is the pandas-DataFrame path
    n = 200
    grp = np.array(["a", "b", "c"])[RNG.integers(0, 3, n)].astype(object)
    y = RNG.standard_normal(n)
    x = y + 0.3 * RNG.standard_normal(n)
    x[3] = np.nan                                   # one incomplete row

    class Frame:
        columns = ["y", "x", "grp"]
        def __init__(self):
            self._cols = {"y": y, "x": x, "grp": grp}
        def __getitem__(self, name):
            return self._cols[name]
        def __array__(self, dtype=None):            # mimics DataFrame: fails
            return np.column_stack([y, x, grp])     # object dtype -> no float

    space = arbital.orbits(Frame(), target="y")
    rows = {r["name"]: r for r in space.table()}
    assert rows["grp"]["categorical"] is True        # string col detected
    assert rows["x"]["r_info"] > 0.5                 # numeric col intact


# ------------------------------------------------ datasets (need seaborn)

def test_datasets_load_numeric_only():
    cars = _load_or_skip(datasets.load_mpg)
    assert cars.shape[1] == 7                     # 7 numeric columns
    assert "name" not in cars.columns             # string column dropped
    assert np.isfinite(np.asarray(cars)).all()    # no NaNs (rows dropped)
    # the mpg selection behaviour we rely on in the docs
    order = [r["name"] for r in arbital.select_features(cars, target="mpg")]
    assert order[0] == "weight"
    assert order.index("model_year") < order.index("horsepower")


def test_penguins_loads_and_orbits():
    p = _load_or_skip(datasets.load_penguins)
    assert p.shape[0] > 300
    space = arbital.orbits(p, target="body_mass_g")
    assert space.target_name == "body_mass_g"


def test_bundled_datasets_load_offline():
    # the four doc datasets ship inside the package: they must load with
    # no seaborn and no network, and report themselves as bundled
    for loader, target in [(datasets.load_mpg, "mpg"),
                           (datasets.load_penguins, "body_mass_g"),
                           (datasets.load_titanic, "survived"),
                           (datasets.load_tips, "tip")]:
        t = loader()
        assert "bundled" in t.notes
        assert target in t.columns
        assert np.isfinite(np.asarray(t)).all()


def test_tips_types_detected():
    t = datasets.load_tips()
    assert t.categorical == {"sex", "smoker", "day", "time"}
    assert t.shape == (244, 7)
    space = arbital.orbits(t, target="tip")
    rows = {r["name"]: r for r in space.table()}
    # the bill is by far the strongest predictor of the tip
    assert rows["total_bill"]["r_info"] == max(r["r_info"]
                                               for r in space.table())


def test_datasets_error_without_seaborn():
    # bundled datasets never need seaborn; NON-bundled names do.  When
    # seaborn is absent, a non-bundled name must raise a clear
    # ImportError rather than something cryptic.  (No-op when seaborn is
    # installed: we will not hit the network in tests.)
    try:
        import seaborn  # noqa: F401
    except ImportError:
        assert datasets.load_mpg().shape[0] > 0     # bundled: still works
        with pytest.raises(ImportError):
            datasets.load("diamonds")


# ---------------------------------------------------------------- transformed / lagged features

def test_transformed_features_cluster_and_selection_keeps_one():
    # a target and monotone transforms of the same base carry almost the
    # same information: the transform that best matches the target should
    # be selected first, and the remaining transforms should show a large
    # drop in marginal gain (they are redundant with the first pick).
    n = 800
    base = RNG.uniform(1, 60, n)
    y = np.log(base) + 0.1 * RNG.standard_normal(n)
    cols = {
        "y": y,
        "raw": base,
        "log": np.log(base),
        "sqrt": np.sqrt(base),
        "capped": np.minimum(base, 30),
    }
    names = list(cols)
    X = np.column_stack([cols[c] for c in names])

    class Named:
        columns = names
        def __array__(self, dtype=None): return X
    picks = arbital.select_features(Named(), target="y")
    assert picks[0]["name"] == "log"                 # best transform first
    # a clear redundancy drop: the second pick adds far less than the first
    assert picks[1]["gain"] < 0.5 * picks[0]["gain"]


def test_lagged_features_reveal_time_dependence():
    # an AR(1)-like series: y_t depends on y_{t-1} strongly and y_{t-2}
    # weakly.  Lag-1 should be the strongest feature and be selected first.
    n = 600
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.8 * s[t - 1] + RNG.standard_normal()
    lag1 = np.roll(s, 1)
    lag2 = np.roll(s, 2)
    lag5 = np.roll(s, 5)
    data = np.column_stack([s, lag1, lag2, lag5])[6:]   # drop wrap-around rows
    picks = arbital.select_features(data, target=0)
    assert picks[0]["name"] == "x1"                     # lag-1 selected first
    # lag-1 is more associated with the series than lag-5
    rows = {r["name"]: r for r in arbital.orbits(data, target=0).table()}
    assert rows["x1"]["r_info"] > rows["x3"]["r_info"]


# ---------------------------------------------------------------- mixed / categorical MI

def test_mixed_mi_continuous_binary():
    # a continuous feature that perfectly determines a binary label carries
    # about ln(2) nats of information about it
    x = RNG.standard_normal(N)
    label = (x > 0).astype(int)
    from arbital.measures import mutual_information_mixed
    mi = mutual_information_mixed(x, label, x_discrete=False, y_discrete=True)
    assert mi == pytest.approx(np.log(2), abs=0.1)


def test_mixed_mi_discrete_discrete():
    a = RNG.integers(0, 3, N)
    from arbital.measures import mutual_information_mixed
    same = mutual_information_mixed(a, a.copy(), True, True)
    indep = mutual_information_mixed(a, RNG.integers(0, 3, N), True, True)
    assert same == pytest.approx(np.log(3), abs=0.1)   # identical -> ln 3
    assert indep < 0.1                                  # independent -> 0


def test_nominal_feature_has_no_direction_or_nonlinearity():
    # a 3-level nominal variable: magnitude only, circular orbit
    x = RNG.integers(0, 3, N).astype(float)
    y = x + 0.1 * RNG.standard_normal(N)
    m = arbital.profile(x, y, x_discrete=True, y_discrete=False)
    assert m["categorical"] is True
    assert m["pearson"] == 0.0 and m["spearman"] == 0.0
    assert m["nonlinearity"] == 0.0


def test_synthetic_categorical_orbits_end_to_end():
    # full orbits() pipeline with a categorical target and a nominal
    # feature, built synthetically so it runs without seaborn
    n = 500
    grp = RNG.integers(0, 3, n)                    # 3-level nominal feature
    binary = (RNG.standard_normal(n) + grp > 1.5).astype(int)  # binary target
    cont = grp + 0.5 * RNG.standard_normal(n)      # continuous, tied to grp
    X = np.column_stack([binary, grp, cont, RNG.standard_normal(n)])

    class Frame:
        columns = ["target", "grp", "cont", "noise"]
        def __array__(self, dtype=None): return X
    space = arbital.orbits(Frame(), target="target", categorical=["target", "grp"])
    rows = {r["name"]: r for r in space.table()}
    assert rows["grp"]["categorical"] is True      # nominal flagged
    assert rows["grp"]["pearson"] == 0.0           # no direction for nominal
    assert rows["cont"]["r_info"] > rows["noise"]["r_info"]


def test_titanic_categorical_end_to_end():
    t = _load_or_skip(datasets.load_titanic)
    assert "sex" in t.categorical and "embarked" in t.categorical
    space = arbital.orbits(t, target="survived")
    rows = {r["name"]: r for r in space.table()}
    # sex is the strongest single predictor of Titanic survival
    assert rows["sex"]["r_info"] > 0.3
    assert rows["sex"]["categorical"] is True


# ---------------------------------------------------------------- geometry identity

def test_eccentricity_equals_nonlinearity():
    # the drawn orbit's eccentricity is the nonlinear share nu
    for r_info, r_mono in [(0.9, 0.9), (0.9, 0.3), (0.6, 0.5)]:
        orb = orbit_parameters(r_info, r_mono)
        nu = 1 - (min(r_mono, r_info) / r_info) ** 2
        assert orb["e"] == pytest.approx(max(0.0, nu))


def test_uncertainty_flag_populates_se():
    data = _demo_data()
    off = arbital.orbits(data, target=0)
    on = arbital.orbits(data, target=0, uncertainty=True, n_bootstrap=15)
    assert off.se is None                              # off by default
    assert on.se is not None and len(on.se) == len(on.names)
    assert all(s >= 0 for s in on.se)
