"""Mathematical soundness checks, deliberately on NON-Gaussian data.

The Gaussian statement attached to Linfoot's r_I ("equals |rho| for a
bivariate Gaussian") is a calibration, not an assumption: r_I is a fixed
monotone transform of mutual information, which is defined for any
distribution.  These tests pin down the properties that must therefore
hold whatever the true distribution is:

  * symmetry:            I(x; y) == I(y; x)
  * copula invariance:   for continuous pairs, r_I depends only on the
                         copula, not the margins
  * information ordering: more noise -> less information, monotonically
  * independence:        r_I near 0 for independent pairs, for ANY margins
                         (heavy-tailed included), up to the estimator floor
  * determinism:         r_I -> 1 as the relationship becomes deterministic,
                         even when strongly non-Gaussian
  * permutation:         shuffling one side destroys the association
  * hard bounds:         0 <= nu <= 1, r_mono <= r_I <= 1, r_peri <= r_apo,
                         on arbitrary data
  * discrete cap:        plug-in MI <= log(min(#levels))
"""

import numpy as np
import pytest

import arbital
from arbital.measures import (mutual_information, mutual_information_mixed,
                              _mi_discrete_discrete, linfoot, profile)
from arbital.geometry import orbit_parameters

RNG = np.random.default_rng(99)
N = 600


def _dependent_pair(n=N, noise=0.5):
    """A correlated pair to reuse across tests."""
    x = RNG.standard_normal(n)
    return x, x + noise * RNG.standard_normal(n)


# ---------------------------------------------------------------- symmetry

def test_mi_is_symmetric_continuous():
    x, y = _dependent_pair()
    assert mutual_information(x, y) == pytest.approx(
        mutual_information(y, x), abs=0.05)


def test_mi_is_symmetric_mixed_and_discrete():
    x = RNG.standard_normal(N)
    d = (x + 0.5 * RNG.standard_normal(N) > 0).astype(int)
    ab = mutual_information_mixed(x, d, x_discrete=False, y_discrete=True)
    ba = mutual_information_mixed(d, x, x_discrete=True, y_discrete=False)
    assert ab == pytest.approx(ba, abs=1e-12)      # same code path, flipped
    a = RNG.integers(0, 4, N)
    b = (a + RNG.integers(0, 2, N)) % 4
    assert _mi_discrete_discrete(a, b) == pytest.approx(
        _mi_discrete_discrete(b, a), abs=1e-12)


# ------------------------------------------------------- copula invariance

def test_r_info_depends_only_on_the_copula():
    # push each margin through a different strictly monotone map: the
    # copula (and hence MI, and hence r_I) must not move.  This is the
    # precise sense in which "Linfoot assumes Gaussian" is only a
    # calibration: the VALUE is a property of the dependence structure.
    # (Own rng: this test's numbers must not depend on execution order.
    # The tolerance covers finite-sample kNN error; ever more extreme
    # maps degrade the ESTIMATE further, not the underlying identity.)
    rng = np.random.default_rng(5)
    x = rng.standard_normal(N)
    y = x + 0.5 * rng.standard_normal(N)
    base = linfoot(mutual_information(x, y))
    for fx, fy in [(np.exp, np.cbrt),
                   (lambda v: v**3, np.exp),
                   (np.arctan, np.exp)]:
        transformed = linfoot(mutual_information(fx(x), fy(y)))
        assert transformed == pytest.approx(base, abs=0.1)


# ------------------------------------------------- information ordering

def test_more_noise_means_less_information():
    x = RNG.standard_normal(N)
    e = RNG.standard_normal(N)
    r_at = [linfoot(mutual_information(x, x + s * e))
            for s in (0.2, 0.6, 1.5, 4.0)]
    # strictly decreasing with a real margin, not just within noise
    assert all(a > b + 0.05 for a, b in zip(r_at, r_at[1:]))


# ---------------------------------------------- independence, any margins

def test_independence_floor_for_non_gaussian_margins():
    # r_I should be near zero for independent pairs whatever the margins;
    # the kNN estimator has a positive noise floor, so assert "small",
    # not "zero".  Heavy tails (lognormal, Cauchy-ish) included.
    margins = {
        "uniform":   lambda n: RNG.uniform(size=n),
        "exponential": lambda n: RNG.exponential(size=n),
        "lognormal": lambda n: np.exp(RNG.standard_normal(n)),
        "heavy":     lambda n: RNG.standard_t(df=1.5, size=n),
    }
    for name_x, gx in margins.items():
        for name_y, gy in margins.items():
            r = linfoot(mutual_information(gx(N), gy(N)))
            assert r < 0.35, f"independent {name_x} vs {name_y}: r_I={r:.2f}"


# ------------------------------------------------ determinism, non-Gaussian

def test_near_deterministic_non_gaussian_approaches_one():
    # strongly non-Gaussian base, non-monotone deterministic map + tiny
    # noise: r_I must be close to its upper limit
    base = np.exp(RNG.standard_normal(N))            # lognormal margin
    y = np.sin(base) + 0.01 * RNG.standard_normal(N)
    assert linfoot(mutual_information(base, y)) > 0.9


# ------------------------------------------------------------- permutation

def test_permuting_one_side_destroys_association():
    x, y = _dependent_pair(noise=0.3)
    strong = linfoot(mutual_information(x, y))
    shuffled = linfoot(mutual_information(x, RNG.permutation(y)))
    assert strong > 0.85
    assert shuffled < 0.35                            # estimator floor only


# -------------------------------------------------------------- hard bounds

def test_metric_bounds_hold_on_arbitrary_data():
    # a grab-bag of margins and relationships: every profile must satisfy
    # the identities the geometry relies on
    n = 400
    x = RNG.standard_normal(n)
    candidates = [
        (x, x**2),                                    # non-monotone
        (RNG.uniform(size=n), RNG.exponential(size=n)),   # independent
        (np.exp(x), np.floor(3 * RNG.uniform(size=n))),   # cont vs coded
        (RNG.standard_t(1.5, n), RNG.standard_t(1.5, n)), # heavy tails
    ]
    for a, b in candidates:
        m = profile(a, b)
        assert 0.0 <= m["nonlinearity"] <= 1.0
        assert 0.0 <= m["r_mono"] <= m["r_info"] <= 1.0
        assert m["mi"] >= 0.0
        orb = orbit_parameters(m["r_info"], m["r_mono"])
        assert 0.0 <= orb["r_peri"] <= orb["r_apo"] <= 1.0
        assert 0.0 <= orb["e"] <= 1.0


def test_linfoot_transform_properties():
    # the transform itself: fixed points and monotonicity
    assert linfoot(0.0) == 0.0
    mis = np.linspace(0.0, 6.0, 25)
    rs = [linfoot(m) for m in mis]
    assert all(b > a for a, b in zip(rs, rs[1:]))     # strictly increasing
    assert rs[-1] < 1.0                               # never reaches 1


# ------------------------------------------------- chance calibration

def _frame(cols):
    names = list(cols)
    X = np.column_stack([cols[c] for c in names])

    class Frame:
        columns = names
        def __array__(self, dtype=None):
            return X
    return Frame()


def _noisy_system(n=400, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    cols = {"y": x + 0.5 * rng.standard_normal(n), "signal": x}
    for i in range(6):
        cols[f"n{i}"] = rng.standard_normal(n)
    return cols


def test_calibration_zeroes_independent_features_RELEVANCE_not_position():
    # calibration must not move the marker: r_info is identical whether
    # or not calibrate=True (see _apply_calibration: calibrating the
    # position too would double-count chance against the boundary circle
    # drawn at the same threshold).  What it zeroes is r_info_adj /
    # relevance for the greedy selection.  A 95% level legitimately
    # admits the occasional false positive, so allow up to two of six.
    raw = arbital.orbits(_frame(_noisy_system()), target="y", calibrate=False)
    cal = arbital.orbits(_frame(_noisy_system()), target="y")
    r_raw = {r["name"]: r["r_info"] for r in raw.table()}
    rows = {r["name"]: r for r in cal.table()}
    for name in r_raw:
        assert rows[name]["r_info"] == pytest.approx(r_raw[name])
    assert rows["signal"]["below_chance"] is False
    below = [rows[f"n{i}"] for i in range(6) if rows[f"n{i}"]["below_chance"]]
    assert len(below) >= 4
    for r in below:
        assert r["r_info_adj"] == 0.0
        # position is untouched: still the raw (small but nonzero, noise
        # floor) estimate, not pinned to the rim
        assert 0.0 <= r["r_info"] < 1.0


def test_calibration_relevance_zeroes_noise_but_keeps_strong_signal():
    # relevance (r_info_adj, what greedy_selection actually ranks on)
    # drops noise features near zero while barely touching a strong one
    space = arbital.orbits(_frame(_noisy_system()), target="y")
    rows = {r["name"]: r for r in space.table()}
    assert rows["signal"]["r_info_adj"] > 0.8
    assert rows["signal"]["r_info_adj"] > rows["signal"]["r_info"] - 0.06
    below = [rows[f"n{i}"] for i in range(6) if rows[f"n{i}"]["below_chance"]]
    for r in below:
        assert r["r_info_adj"] == 0.0


def test_orbitsystem_selection_reports_r_info_adj_as_relevance():
    # OrbitSystem.selection() must report the same relevance the greedy
    # selection actually ranked on (r_info_adj when calibrated), not the
    # plotted r_info, otherwise relevance/redundancy/gain in the
    # vignette's selection tables would not describe the pick that was
    # actually made.
    space = arbital.orbits(_frame(_noisy_system()), target="y")
    by_name = {r["name"]: r for r in space.table()}
    for s in space.selection():
        r = by_name[s["name"]]
        assert s["relevance"] == pytest.approx(r["r_info_adj"])
        assert s["redundancy"] == pytest.approx(s["relevance"] - s["gain"])
    # a below-chance feature must show (near-)zero relevance here too,
    # not its raw (small but nonzero) r_info
    below = [s for s in space.selection()
             if by_name[s["name"]]["below_chance"]]
    for s in below:
        assert s["relevance"] == 0.0


def test_calibrate_false_reproduces_uncalibrated_profile():
    space = arbital.orbits(_frame(_noisy_system()), target="y", calibrate=False)
    for r in space.table():
        assert "chance" not in r and "below_chance" not in r and "r_info_adj" not in r


def test_calibration_is_deterministic():
    a = arbital.orbits(_frame(_noisy_system()), target="y").table()
    b = arbital.orbits(_frame(_noisy_system()), target="y").table()
    assert a == b


def test_select_features_reports_below_chance():
    rows = arbital.select_features(_frame(_noisy_system()), target="y")
    by_name = {r["name"]: r for r in rows}
    assert by_name["signal"]["pick"] == 1
    assert by_name["signal"]["below_chance"] is False
    # below-chance features contribute (near-)zero relevance
    for r in rows:
        if r["below_chance"]:
            assert r["relevance"] <= 0.15


def _mixed_system(n=500, seed=2, levels=6):
    # continuous drivers/noise plus a higher-cardinality categorical
    # column, so the pooled-continuous vs. per-categorical chance split
    # actually exercises both branches
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    cols = {"y": x + 0.6 * rng.standard_normal(n),
            "signal": x,
            "noise_cont": rng.standard_normal(n),
            "noise_cat": rng.integers(0, levels, n).astype(float)}
    return cols, ["noise_cat"]


def test_continuous_features_share_one_chance_level():
    # the whole point of pooling: every continuous feature's `chance`,
    # `pearson_chance`, and `spearman_chance` must be numerically
    # identical (same permutation draws), so the drawn boundary circle
    # can honestly represent all of them with one number
    cols, cat = _mixed_system()
    space = arbital.orbits(_frame(cols), target="y", categorical=cat)
    cont_rows = [r for r in space.table() if not r["categorical"]]
    assert len(cont_rows) >= 2
    chances = {r["chance"] for r in cont_rows}
    p_chances = {r["pearson_chance"] for r in cont_rows}
    s_chances = {r["spearman_chance"] for r in cont_rows}
    assert len(chances) == 1
    assert len(p_chances) == 1
    assert len(s_chances) == 1


def test_categorical_chance_can_differ_from_continuous_and_from_circle():
    # a categorical column's own level is computed separately (cardinality
    # -dependent) and is NOT forced to match the continuous pooled level;
    # the figure's drawn boundary circle must equal the continuous value,
    # not some max() over continuous and categorical levels together
    cols, cat = _mixed_system()
    space = arbital.orbits(_frame(cols), target="y", categorical=cat)
    rows = {r["name"]: r for r in space.table()}
    cont_chance = rows["signal"]["chance"]
    assert rows["noise_cont"]["chance"] == pytest.approx(cont_chance)

    fig = space.figure()
    circles = [tr for tr in fig["data"]
              if tr.get("hovertext") and "chance boundary for continuous"
              in (tr["hovertext"] if isinstance(tr["hovertext"], str) else "")]
    assert len(circles) == 1
    # the circle's radius must correspond to exactly the continuous
    # chance level, never a max() that also swept in the categorical one.
    # plot.py's _display() is R_FLOOR + (1 - R_FLOOR) * true_distance;
    # recover the true distance the circle was drawn at and compare it to
    # strength_to_distance(cont_chance) directly.
    from arbital.geometry import strength_to_distance
    from arbital.plot import R_FLOOR
    got_r = (circles[0]["x"][0] ** 2 + circles[0]["y"][0] ** 2) ** 0.5
    got_true_distance = (got_r - R_FLOOR) / (1.0 - R_FLOOR)
    expected_true_distance = strength_to_distance(cont_chance, "info")
    assert got_true_distance == pytest.approx(expected_true_distance, abs=1e-6)


def test_confidence_raises_chance_level_monotonically():
    # a stricter (higher) confidence must never LOWER the chance level:
    # the quantile function is monotone non-decreasing in its argument
    cols, cat = _mixed_system()
    lo = arbital.orbits(_frame(cols), target="y", categorical=cat, confidence=0.80)
    hi = arbital.orbits(_frame(cols), target="y", categorical=cat, confidence=0.99)
    lo_chance = {r["name"]: r["chance_nats"] for r in lo.table()}
    hi_chance = {r["name"]: r["chance_nats"] for r in hi.table()}
    for name in lo_chance:
        assert hi_chance[name] >= lo_chance[name] - 1e-12


def test_confidence_out_of_range_raises():
    cols, cat = _mixed_system()
    with pytest.raises(ValueError):
        arbital.orbits(_frame(cols), target="y", categorical=cat, confidence=1.5)
    with pytest.raises(ValueError):
        arbital.orbits(_frame(cols), target="y", categorical=cat, confidence=0.0)


def test_low_n_shuffles_warns_about_quantile_stability():
    import warnings as warnings_mod
    cols, cat = _mixed_system()
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        arbital.orbits(_frame(cols), target="y", categorical=cat, n_shuffles=20)
    msgs = [str(w.message) for w in caught]
    assert any("draws beyond the quantile" in m for m in msgs)


def test_default_n_shuffles_does_not_warn():
    import warnings as warnings_mod
    cols, cat = _mixed_system()
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        arbital.orbits(_frame(cols), target="y", categorical=cat)
    msgs = [str(w.message) for w in caught]
    assert not any("draws beyond the quantile" in m for m in msgs)


def test_pearson_spearman_significance_uses_pooled_permutation_level():
    # pearson_sig/spearman_sig must be computed against the SAME
    # confidence-quantile permutation level for every continuous feature
    # (not a per-feature analytic SE formula), so they share one
    # pearson_chance / spearman_chance value just like `chance` does
    cols, cat = _mixed_system()
    space = arbital.orbits(_frame(cols), target="y", categorical=cat)
    cont_rows = [r for r in space.table() if not r["categorical"]]
    assert len({r["pearson_chance"] for r in cont_rows}) == 1
    assert len({r["spearman_chance"] for r in cont_rows}) == 1
    for r in cont_rows:
        assert r["pearson_sig"] == (abs(r["pearson"]) > r["pearson_chance"])
        assert r["spearman_sig"] == (abs(r["spearman"]) > r["spearman_chance"])


# ---------------------------------------------------- categorical direction

def test_binary_categorical_keeps_direction_nominal_does_not():
    # profile() already computes a real point-biserial correlation for a
    # BINARY categorical column (nominal=False): the hover must show it,
    # not hide it behind the blanket "categorical: no direction" wording
    # that only makes sense for a multi-level nominal column (nominal=True).
    from arbital.plot import _hover
    from arbital.geometry import orbit_parameters
    n = 400
    x = RNG.standard_normal(n)
    sex = (x > 0).astype(float)                    # binary categorical
    grp = RNG.integers(0, 4, n).astype(float)       # 4-level nominal

    m_binary = arbital.profile(sex, x + 0.1 * RNG.standard_normal(n),
                          x_discrete=True, y_discrete=False)
    m_nominal = arbital.profile(grp, x + 0.1 * RNG.standard_normal(n),
                           x_discrete=True, y_discrete=False)
    # _hover() also reads the orbit-geometry fields (r_peri etc.) that
    # orbits() normally merges in; add them here since we are calling
    # profile() directly rather than going through the full pipeline.
    m_binary.update(orbit_parameters(m_binary["r_info"], m_binary["r_mono"]))
    m_nominal.update(orbit_parameters(m_nominal["r_info"], m_nominal["r_mono"]))

    assert m_binary["categorical"] is True and m_binary["nominal"] is False
    assert m_nominal["categorical"] is True and m_nominal["nominal"] is True
    assert m_binary["pearson"] != 0.0            # real direction available

    hover_binary = _hover("sex", m_binary, "gain", 0.5)
    hover_nominal = _hover("grp", m_nominal, "gain", 0.5)
    assert "r_pearson" in hover_binary            # direction shown
    assert "no direction" not in hover_binary
    assert "no direction" in hover_nominal        # still hidden for nominal
    assert "r_pearson" not in hover_nominal


# ------------------------------------------------------------ discrete cap

def test_discrete_mi_bounded_by_log_min_levels():
    # I(x;y) <= min(H(x), H(y)) <= log(min(#levels)); the plug-in
    # estimator must respect the cap even under perfect dependence
    a = RNG.integers(0, 3, N)                         # 3 levels
    b = a * 10 + 7                                    # relabelled copy, 3 levels
    c = RNG.integers(0, 8, N)                         # 8 levels
    assert _mi_discrete_discrete(a, b) <= np.log(3) + 1e-9
    assert _mi_discrete_discrete(a, c) <= np.log(3) + 1e-9
