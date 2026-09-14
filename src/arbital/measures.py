"""Association measures, implemented transparently in pure NumPy.

Intent:
    Measure how strongly two variables are associated -- through a
    straight line, through any monotone curve, and through any
    relationship at all -- and report all three on one comparable scale.

The three estimators here answer three different questions about a pair
of variables (x, y):

  Pearson r    "How well does a straight line describe the relationship?"
  Spearman rho "How well does any monotone (always-increasing or
               always-decreasing) curve describe it?"
  Mutual information I(x;y) "How much does knowing x reduce my
               uncertainty about y, through any relationship at all?"

Mutual information is measured in nats (natural-log bits) and is hard to
compare with correlations directly, so we convert it to Linfoot's
*informational coefficient of correlation*:

    r_I = sqrt(1 - exp(-2 * I))

Why this formula: for a bivariate Gaussian, I = -1/2 * ln(1 - rho^2)
exactly, so inverting gives r_I == |rho|.  In other words, r_I is "the
correlation this MI would correspond to if the relationship were
Gaussian/linear".  That puts total (any-shape) dependence on the same
0..1 scale as ordinary correlation, and the *gap* between r_I and the
best monotone correlation becomes a principled nonlinearity measure.

References
----------
- Pearson, K. (1895). Notes on regression and inheritance in the case of
  two parents. Proceedings of the Royal Society of London, 58, 240-242.
- Spearman, C. (1904). The proof and measurement of association between
  two things. American Journal of Psychology, 15(1), 72-101.
- Shannon, C. E. (1948). A mathematical theory of communication.
  Bell System Technical Journal, 27, 379-423.
- Linfoot, E. H. (1957). An informational measure of correlation.
  Information and Control, 1(1), 85-89.
- Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating
  mutual information. Physical Review E, 69(6), 066138.
- Ross, B. C. (2014). Mutual information between discrete and continuous
  data sets. PLoS ONE, 9(2), e87357.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "association_matrix",
    "linfoot",
    "mutual_information",
    "mutual_information_mixed",
    "pearson",
    "profile",
    "spearman",
]


# ---------------------------------------------------------------------------
# Decidable cores
# ---------------------------------------------------------------------------
# Each estimator below measures a quantity and then draws a conclusion
# from it.  The conclusions are these four functions: they take numbers
# rather than samples, so what this module concludes from an estimate can
# be read, stated and checked without running an estimator.

def _ksg_mi(psi_k: float, psi_n: float, psi_marginals: float) -> float:
    """Assemble the KSG estimate from its three digamma terms, in nats.

    Intent:
        Turn the estimator's three averaged digamma terms into a mutual
        information, held at zero because information cannot be negative.
    """
    return max(0.0, psi_k + psi_n - psi_marginals)


def _monotone_strength(r: float, rho: float) -> float:
    """Best monotone description of a pair: r_mono = max(|r|, |rho|).

    Intent:
        Say how much of an association a monotone curve can account for,
        so the remainder can be attributed to shape correlation misses.
    """
    return max(abs(r), abs(rho))


def _nonlinear_share(r_info: float, r_mono: float) -> float:
    """The share nu of total dependence a monotone description misses.

    nu = 1 - (r_mono / r_info)^2, floored at zero, and zero when there is
    no association at all to take a share of.  r_mono is capped at
    r_info: a monotone fit cannot explain more than the total.

    Intent:
        Quantify how much of an association is invisible to correlation,
        as the fraction of the total dependence a monotone description
        fails to reach.
    """
    if r_info <= 0.0:
        return 0.0
    return max(0.0, 1.0 - (min(r_mono, r_info) / r_info) ** 2)


def _floored_total(r_info_raw: float, r_mono: float) -> float:
    """Total association, floored at what a monotone fit already reaches.

    Total dependence cannot be less than the part a monotone description
    already captures, and the k-NN estimator is noisy enough to report
    less, so the floor is enforced rather than assumed.

    Intent:
        Keep the total and its monotone part consistent with each other,
        so the nonlinear share they define is never negative.
    """
    return max(r_info_raw, r_mono)


def _is_nominal(n_levels: int, is_discrete: bool) -> bool:
    """True when a side's integer codes are arbitrary labels.

    A discrete side with more than two levels has no meaningful sign or
    monotone baseline; a binary one still carries a point-biserial
    direction and is not nominal in this sense.

    Intent:
        Decide whether a variable's codes carry a direction, which is
        what settles whether a correlation is reported for the pair.
    """
    return is_discrete and n_levels > 2


# ---------------------------------------------------------------------------
# Classical correlations
# ---------------------------------------------------------------------------

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Plain Pearson correlation coefficient (signed, -1..1).

    A constant sample has no linear relationship with anything, so a zero
    denominator reports 0.0 instead of dividing.

    Intent:
        Answer how well a straight line describes the relationship
        between two samples, on the conventional -1..1 scale.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    if denom == 0.0:  # a constant variable has no correlation with anything
        return 0.0
    return float((xc * yc).sum() / denom)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties given the average of the ranks they occupy.

    Intent:
        Replace values by their order so a correlation of the ranks sees
        any monotone relationship, whatever its curvature.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")          # stable sort
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average the ranks within each group of tied values
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:  # a tie group from i..j -> give all of them the mean rank
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation = Pearson correlation of the ranks.

    Intent:
        Answer how well any monotone curve describes the relationship
        between two samples, on the same -1..1 scale as Pearson.
    """
    return pearson(_ranks(x), _ranks(y))


# ---------------------------------------------------------------------------
# Mutual information (Kraskov / KSG k-nearest-neighbour estimator)
# ---------------------------------------------------------------------------

def _digamma(x: np.ndarray) -> np.ndarray:
    """Digamma (psi) function, vectorised, pure NumPy.

    Uses the recurrence psi(x) = psi(x+1) - 1/x to push arguments above 10,
    then the standard asymptotic series.  Accurate to ~1e-11 for x > 0,
    which is far more than the MI estimator needs.

    Intent:
        Supply the one special function the k-NN estimators need, so the
        package stays pure NumPy with no SciPy dependency.
    """
    x = np.asarray(x, dtype=float).copy()
    result = np.zeros_like(x)
    # recurrence: accumulate -1/x while x < 10
    while np.any(x < 10):
        small = x < 10
        result[small] -= 1.0 / x[small]
        x[small] += 1.0
    # asymptotic expansion for large x
    inv = 1.0 / x
    inv2 = inv * inv
    result += (
        np.log(x) - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    )
    return result


def mutual_information(
    x: np.ndarray, y: np.ndarray, k: int = 5, rng_seed: int = 0
) -> float:
    """Estimate I(x; y) in nats with the KSG estimator (Kraskov et al. 2004).

    Sketch of the algorithm:
      1. Standardise x and y, add a whisper of noise to break ties.
      2. For each point i, find the distance eps_i to its k-th nearest
         neighbour in the joint (x,y) space using the Chebyshev
         (max-coordinate) metric.
      3. Count n_x(i): how many other points fall strictly within eps_i
         of point i along the x axis alone; same for n_y(i).
      4. Average the digamma formula:
             I = psi(k) + psi(N) - < psi(n_x + 1) + psi(n_y + 1) >
    Intuition: if x and y are strongly dependent, the joint neighbourhood
    is "tight" relative to the marginal neighbourhoods, so the marginal
    counts n_x, n_y are small and the estimate is large.

    O(n^2) memory/time, fine up to a few thousand points; subsample
    beyond that.

    Intent:
        Measure how much knowing one continuous variable reduces
        uncertainty about another, through a relationship of any shape.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n <= k + 1:
        raise ValueError(f"need more than k+1={k+1} samples, got {n}")

    # 1. standardise so the Chebyshev metric treats both axes equally,
    #    and jitter to break ties (KSG assumes continuous distributions)
    rng = np.random.default_rng(rng_seed)
    xs = (x - x.mean()) / (x.std() + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    xs = xs + 1e-10 * rng.standard_normal(n)
    ys = ys + 1e-10 * rng.standard_normal(n)

    # 2. pairwise distances along each axis, and their Chebyshev max
    dx = np.abs(xs[:, None] - xs[None, :])   # (n, n)
    dy = np.abs(ys[:, None] - ys[None, :])
    dj = np.maximum(dx, dy)                  # joint-space distance
    np.fill_diagonal(dj, np.inf)             # a point is not its own neighbour

    # distance to the k-th nearest neighbour of each point
    eps = np.partition(dj, k - 1, axis=1)[:, k - 1]

    # 3. marginal neighbour counts strictly inside eps_i
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    n_x = (dx < eps[:, None]).sum(axis=1)
    n_y = (dy < eps[:, None]).sum(axis=1)

    # 4. KSG formula; _ksg_mi clips at 0 because MI cannot be negative
    return _ksg_mi(
        float(_digamma(np.array([float(k)]))[0]),
        float(_digamma(np.array([float(n)]))[0]),
        float(np.mean(_digamma(n_x + 1.0) + _digamma(n_y + 1.0))),
    )


def linfoot(mi: float) -> float:
    """Linfoot's informational coefficient of correlation, r_I in [0, 1).

    r_I = sqrt(1 - exp(-2*I)).  Equals |rho| exactly when (x, y) is
    bivariate Gaussian, so it is 'MI expressed on the correlation scale'.
    Linfoot (1957), "An informational measure of correlation",
    Information and Control 1(1), 85-89.

    Intent:
        Put mutual information on the correlation scale, so dependence
        of any shape can be compared against an ordinary correlation.
    """
    return float(np.sqrt(1.0 - np.exp(-2.0 * max(0.0, mi))))


# ---------------------------------------------------------------------------
# Mutual information for discrete and mixed variable types
# ---------------------------------------------------------------------------
# A single MI number can join any pair of variables, but the right
# estimator depends on whether each side is continuous or categorical.
# mutual_information_mixed() dispatches to the correct one.

def _mi_discrete_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in MI (nats) between two categorical variables.

    Straight from the definition using the joint frequency table:
        I = sum_{a,b} p(a,b) * log( p(a,b) / (p(a) p(b)) ).
    Both inputs are treated as labels (their numeric value is ignored
    except as an identity), so integer codes are fine.

    Intent:
        Measure association between two categorical variables exactly
        from their joint frequency table, with no estimator in between.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = len(x)
    xs = {v: i for i, v in enumerate(np.unique(x))}
    ys = {v: i for i, v in enumerate(np.unique(y))}
    joint = np.zeros((len(xs), len(ys)))
    for a, b in zip(x, y):
        joint[xs[a], ys[b]] += 1.0
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px * py)
        terms = joint * np.log(ratio)
    return max(0.0, float(np.nansum(terms)))


def _mi_continuous_discrete(c: np.ndarray, d: np.ndarray, k: int = 5,
                            rng_seed: int = 0) -> float:
    """MI (nats) between a continuous variable c and a discrete label d.

    Ross (2014), "Mutual Information between Discrete and Continuous Data
    Sets", PLoS ONE.  Intuition mirrors the KSG estimator: if c is
    informative about the class d, then a point's k nearest neighbours of
    the same class are much closer than its neighbours overall, and the
    digamma formula turns that ratio into nats.

        I = psi(N) + psi(k) - <psi(N_class)> - <psi(m)>
    where for each point i, N_class is its class size, and m is the number
    of points (any class) within the distance to its k-th same-class
    neighbour.

    Intent:
        Measure association between a continuous variable and a class
        label, which neither the KSG nor the plug-in estimator can do.
    """
    c = np.asarray(c, dtype=float).ravel()
    d = np.asarray(d).ravel()
    n = len(c)
    rng = np.random.default_rng(rng_seed)
    c = c + 1e-10 * rng.standard_normal(n)      # jitter to break ties
    psi_N = _digamma(np.array([float(n)]))[0]

    total = 0.0
    for label in np.unique(d):
        idx = np.where(d == label)[0]
        nx = len(idx)
        if nx < 2:                               # singleton class adds ~0 info
            continue
        kk = min(k, nx - 1)                       # shrink k for tiny classes
        cc = c[idx]
        within = np.abs(cc[:, None] - cc[None, :])
        np.fill_diagonal(within, np.inf)
        eps = np.partition(within, kk - 1, axis=1)[:, kk - 1]   # k-th NN dist
        # m_i = neighbours of point i within eps_i across the full sample
        m = (np.abs(c[None, :] - cc[:, None]) <= eps[:, None]).sum(axis=1) - 1
        psi_class = _digamma(np.array([float(kk), float(nx)]))
        total += float(np.sum(psi_N + psi_class[0] - psi_class[1]
                              - _digamma(np.maximum(m, 1).astype(float))))
    return max(0.0, total / n)


def mutual_information_mixed(x, y, x_discrete: bool, y_discrete: bool,
                             k: int = 5) -> float:
    """Estimate I(x; y) in nats, dispatching on each side's type.

      continuous / continuous -> KSG (mutual_information)
      continuous / discrete   -> Ross estimator
      discrete   / discrete   -> plug-in from the contingency table

    Intent:
        Give every pair of variables one comparable information number,
        whatever mix of continuous and categorical types it holds.
    """
    if x_discrete and y_discrete:
        return _mi_discrete_discrete(x, y)
    if x_discrete and not y_discrete:
        return _mi_continuous_discrete(y, x, k=k)
    if y_discrete and not x_discrete:
        return _mi_continuous_discrete(x, y, k=k)
    return mutual_information(x, y, k=k)


# ---------------------------------------------------------------------------
# Chance calibration: what does this estimator report with no association?
# ---------------------------------------------------------------------------

def shuffled_mi(x, y, x_discrete: bool, y_discrete: bool, k: int = 5,
                n_shuffles: int = 16, rng=None) -> np.ndarray:
    """MI estimates between x and shuffled copies of y, in nats.

    Shuffling y destroys any dependence while keeping both marginal
    distributions and the sample size, so these draws are the estimator's
    output under exact independence: its finite-sample "chance level".
    The k-NN estimators cannot return zero on finite data; with a few
    hundred rows an independent pair typically measures 0.01-0.05 nats,
    which Linfoot's steep transform near zero turns into r_I ~ 0.1-0.25.
    Comparing a measured value against these draws separates signal from
    that artefact.

    Returns an array of n_shuffles MI estimates.

    Intent:
        Show what this estimator reports when there is nothing to find,
        so a measured value can be read against its own noise floor.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(x)
    y = np.asarray(y)
    out = np.empty(n_shuffles)
    for b in range(n_shuffles):
        out[b] = mutual_information_mixed(x, rng.permutation(y),
                                          x_discrete, y_discrete, k=k)
    return out


# ---------------------------------------------------------------------------
# Bundled per-pair profile, and the all-pairs matrix
# ---------------------------------------------------------------------------

def profile(x, y, k: int = 5, x_discrete: bool = False,
            y_discrete: bool = False) -> dict:
    """All association measures for one (x, y) pair, as a dict.

    Set x_discrete / y_discrete when a variable is categorical (nominal or
    an integer-coded label) so the correct MI estimator is used.

    Keys:
      pearson, spearman        signed coefficients (0 when a side is nominal)
      mi                       mutual information (nats)
      r_info                   Linfoot r_I, total association, 0..1
      r_mono                   best monotone correlation = max(|r|, |rho|)
      nonlinearity             package-defined composite ('nu'): the fraction
                               of total dependence a monotone description
                               misses, 1 - (r_mono/r_info)^2
      categorical              True if either side is discrete (binary or
                               nominal): used to pick the calibration
                               granularity and the orbit shape, not to
                               decide whether direction is shown (see
                               `nominal` below)
      nominal                 True only when a discrete side has more than
                               two categories, so its integer codes are
                               arbitrary labels with no meaningful sign or
                               monotone baseline (then pearson/spearman are
                               0 and the orbit is circular). A binary
                               discrete side (nominal=False) still gets a
                               real point-biserial correlation, so it keeps
                               a direction like a continuous variable.

    Intent:
        Report every association measure for one pair at once, on one
        scale, so the total and its monotone part can be compared.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mi = mutual_information_mixed(x, y, x_discrete, y_discrete, k=k)
    r_info_raw = linfoot(mi)
    xf = x.astype(float)
    yf = y.astype(float)

    # A nominal variable with more than two categories has no meaningful
    # sign or monotone baseline (its integer codes are arbitrary labels).
    # A binary variable is fine: point-biserial correlation still gives a
    # consistent direction.  So we only drop the monotone framing for
    # multi-category nominal sides.  Level counts are taken only for a
    # side already known to be discrete, and the decision itself is
    # _is_nominal().
    x_levels = len(np.unique(x)) if x_discrete else 0
    y_levels = len(np.unique(y)) if y_discrete else 0

    if _is_nominal(x_levels, x_discrete) or _is_nominal(y_levels, y_discrete):
        return {
            "pearson": 0.0, "spearman": 0.0, "mi": mi,
            "r_info": r_info_raw, "r_mono": r_info_raw,
            "nonlinearity": 0.0, "categorical": True, "nominal": True,
        }

    r = pearson(xf, yf)
    rho = spearman(xf, yf)
    r_mono = _monotone_strength(r, rho)
    r_info = _floored_total(r_info_raw, r_mono)
    nonlin = _nonlinear_share(r_info, r_mono)
    return {
        "pearson": r, "spearman": rho, "mi": mi,
        "r_info": r_info, "r_mono": r_mono,
        "nonlinearity": nonlin,
        "categorical": bool(x_discrete or y_discrete), "nominal": False,
    }


def association_matrix(X: np.ndarray, k: int = 5, discrete=None) -> np.ndarray:
    """Symmetric matrix of r_info between every pair of columns of X.

    discrete: optional boolean sequence, one flag per column, marking
    categorical columns so the right MI estimator is used.

    Intent:
        Give the redundancy structure among features one number per
        pair, which the angular layout and the selection both read.
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    if discrete is None:
        discrete = [False] * p
    M = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            pr = profile(X[:, i], X[:, j], k=k,
                         x_discrete=discrete[i], y_discrete=discrete[j])
            M[i, j] = M[j, i] = pr["r_info"]
    return M
