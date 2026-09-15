"""Mapping association measures onto Keplerian orbit geometry.

Intent:
    Turn each feature's association numbers into a position and a shape,
    so strength, nonlinearity and redundancy can all be read off one
    picture at once.

The target variable is the central feature, sitting at the focus of
every orbit; each feature is a body on its own ellipse.  In plain terms,
each orbit has a closest point and a farthest point from the centre
(in orbital mechanics: the periastron and apastron), and those two
distances carry the two views of the same feature:

  closest point (periastron) = distance(r_info)   <- total association
  farthest point (apastron)  = distance(r_mono)   <- monotone-only view

where distance() is strength_to_distance() below.  The default "info"
scale sets

    distance = sqrt(1 - r^2) = exp(-I)     (I = mutual information)

i.e. distance from the centre is the residual uncertainty left after
observing the feature (for a bivariate Gaussian, sqrt(1 - rho^2) is
exactly the unexplained fraction of standard deviation).  Because
distance = exp(-I), the radial axis is logarithmic in information:
every factor of e closer to the centre is one more nat of shared
information.  This spreads out the crowded strong-association zone
(r = 0.90 and r = 0.99 land at radius 0.44 and 0.14 instead of 0.10
and 0.01).  scale="linear" keeps the naive distance = 1 - r mapping.

So a feature swings between "how close it gets when you account for any
kind of dependence" and "how far away it looks if you only trust
linear/monotone correlation".  Two consequences fall straight out of the
ellipse algebra:

  * If linear tells the whole story (r_mono == r_info), the closest and
    farthest points coincide and the orbit is a circle.
  * The bigger the nonlinear surplus, the more eccentric the orbit:
        eccentricity e = (r_apo - r_peri) / (r_apo + r_peri)

Angular position is chosen so that features associated with each other
sit at similar angles: we run classical MDS on the feature-feature
distance matrix d = 1 - r_info and read the angle off the 2-D embedding.
The default "spread" layout keeps the embedding's gap structure
(clusters of mutually-redundant features visibly huddle, unrelated
features sit far apart) while enforcing a minimum angular separation so
labels stay readable; see angular_layout() for the alternatives.

Marker size and the pick numbers on the labels come from an explicit
greedy mRMR forward selection (greedy_selection below): size is the
marginal gain a feature contributed at the moment it was selected, so
the plot shows an actual feature-selection order, not a score computed
against features you might never keep.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from mathema.types import Mat, Vec

from .measures import _nonlinear_share, association_matrix

__all__ = [
    "angular_layout",
    "greedy_selection",
    "orbit_parameters",
    "select_target",
    "strength_to_distance",
]


def _distance_from_strength(r: float,
                            scale: Literal["info", "linear"] = "info") -> float:
    """Distance from the centre for an association strength r in [0, 1].

    The radial mapping itself, with no coercion: callers hand it a value
    already known to be in [0, 1].

    scale="info" (default): distance = sqrt(1 - r^2), the residual
      uncertainty after observing the feature.  Since Linfoot's
      r = sqrt(1 - exp(-2I)), this simplifies to exp(-I): the radial
      axis is logarithmic in mutual information (one factor of e per
      nat), which spreads out strong associations near the centre.
    scale="linear": distance = 1 - r, the naive mapping.

    Intent:
        Decide how far from the centre a given association strength
        sits, which is the one choice the whole radial axis rests on.
    """
    if scale == "info":
        return np.sqrt(1.0 - r**2)
    if scale == "linear":
        return 1.0 - r
    raise ValueError(f"unknown scale {scale!r}, use 'info' or 'linear'")


def strength_to_distance(r, scale: Literal["info", "linear"] = "info"):
    """Map association strength r in [0, 1] to distance from the centre.

    Accepts a scalar or an array and clamps it into [0, 1] before
    applying _distance_from_strength(), which is the mapping itself.

    Intent:
        Offer the radial mapping over whatever numeric shape a caller
        has, so estimates that stray outside [0, 1] still plot.
    """
    r = np.clip(np.asarray(r, dtype=float), 0.0, 1.0)
    return _distance_from_strength(r, scale)


def orbit_parameters(r_info: float, r_mono: float,
                     scale: Literal["info", "linear"] = "info") -> dict:
    """Ellipse parameters (focus at origin) for one feature.

    The orbit's eccentricity is set to equal the nonlinearity share

        nu = 1 - (r_mono / r_info)^2

    so shape and the reported statistic 'nu' are the same quantity.  A
    circle (e = nu = 0) means a monotone description is complete; e -> 1
    means the association is almost entirely non-monotone.

    Returns a dict with:
      r_peri, r_apo   closest / farthest point of the orbit
                      (periastron / apastron), via strength_to_distance()
                      on r_info / r_mono; these fix the solid marker and
                      the ghost/tether
      a               semi-major axis  = (r_peri + r_apo) / 2
      e               eccentricity = nu (the nonlinear share)

    Intent:
        Give one feature the ellipse that carries both its views at
        once: how close it gets, and how far correlation alone puts it.
    """
    r_info = min(max(r_info, 0.0), 1.0)
    r_mono = min(max(min(r_mono, r_info), 0.0), 1.0)
    r_peri = float(_distance_from_strength(r_info, scale))
    r_apo = float(_distance_from_strength(r_mono, scale))
    a = 0.5 * (r_peri + r_apo)
    nu = _nonlinear_share(r_info, r_mono)
    return {"r_peri": r_peri, "r_apo": r_apo, "a": a, "e": float(nu)}


def _eccentricity(r_peri: float, r_apo: float) -> float:
    """Geometric eccentricity of an ellipse from its two apsides.

    e = (r_apo - r_peri) / (r_apo + r_peri), and 0 for the degenerate
    orbit whose apsides both sit at the origin.

    Intent:
        Recover the shape of an orbit from the two distances that
        define it, without needing the semi-major axis as well.
    """
    a = 0.5 * (r_peri + r_apo)
    if a == 0.0:
        return 0.0
    return (r_apo - r_peri) / (r_apo + r_peri)


def ellipse_path(theta: float, r_peri: float, r_apo: float, n: int = 120):
    """(x, y) points tracing the orbit, focus at the origin.

    Standard focal polar form of an ellipse:  r(phi) = p / (1 + e*cos(phi))
    where p = a(1 - e^2) is the semi-latus rectum and phi is measured from
    the periastron direction.  We point the periastron along `theta`, so
    the marker (drawn at periastron) sits exactly on its own orbit.

    Intent:
        Trace the full ellipse a feature would sweep, so its shape can
        be drawn rather than inferred from two marker positions.
    """
    a = 0.5 * (r_peri + r_apo)
    e = _eccentricity(r_peri, r_apo)
    p = a * (1.0 - e**2)
    phi = np.linspace(0.0, 2.0 * np.pi, n)
    r = p / (1.0 + e * np.cos(phi))
    ang = theta + phi
    return r * np.cos(ang), r * np.sin(ang)


def _classical_mds_2d(D: Mat("n", "n")) -> np.ndarray:
    """Classical (Torgerson) MDS into 2 dimensions, pure NumPy.

    Double-centre the squared distance matrix to recover an inner-product
    matrix B, then use its top-2 eigenvectors scaled by sqrt(eigenvalue).
    Torgerson (1952), "Multidimensional scaling: I. Theory and method",
    Psychometrika 17(4), 401-419.

    Intent:
        Place features in a plane so that how far apart they sit
        reflects how little information they share.
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n          # centring matrix
    B = -0.5 * J @ (D**2) @ J
    vals, vecs = np.linalg.eigh(B)               # ascending order
    idx = np.argsort(vals)[::-1][:2]             # top-2 eigenpairs
    L = np.sqrt(np.clip(vals[idx], 0.0, None))
    return vecs[:, idx] * L[None, :]


def _min_gap(p: int, min_gap_frac: float) -> float:
    """The smallest angular gap the "spread" layout will allow.

    A fraction of the even share 2*pi/p that p features would get.

    Intent:
        Fix a separation below which two labels would collide, as a
        share of the spacing an even layout would have given them.
    """
    return min_gap_frac * 2.0 * np.pi / p


def _widened_gap(gap: float, g_min: float, scale: float) -> float:
    """One angular gap after the "spread" rescaling.

    Every gap is raised to at least g_min, and whatever it had above
    g_min is kept in proportion `scale`, so the gaps that were large
    stay large relative to each other while the pile-ups are opened.

    Intent:
        Open a crowded gap to a readable minimum without discarding how
        much larger the roomy gaps were.
    """
    return g_min + max(gap - g_min, 0.0) * scale


def angular_layout(assoc: Mat("n", "n"),
                   layout: Literal["spread", "embed", "ordered"] = "spread",
                   min_gap_frac: float = 0.35) -> np.ndarray:
    """Angle (radians) for each feature from the feature-feature matrix.

    assoc: symmetric matrix of r_info between features (diagonal 1).  We
    run classical MDS (Torgerson 1952) on the distance matrix d = 1 - assoc
    and read each feature's angle from the 2-D embedding, so
    mutually-associated features land near each other on the circle.

    layout controls what the angular gaps mean:
      "spread" (default): the embedding angles, with every gap widened to
          at least min_gap_frac of an even share (0.35 * 2*pi/p). Related
          features still cluster and unrelated ones still sit far apart:
          the gaps keep their relative magnitudes, but no two labels can
          pile onto each other.
      "embed": the raw embedding angles.  Gaps approximate the
          correlation-structure distances exactly as embedded, at the cost
          of possible label crowding.
      "ordered": keep only the circular order from the embedding and space
          features evenly.  Neighbours are still associated, but a gap
          carries no magnitude: the conservative choice when you only
          trust the ordering.

    `min_gap_frac` is a fraction of an even share and is meaningful in
    [0, 1]; at 1 every gap is the even share and the layout carries no
    structure at all.

    Intent:
        Put associated features near each other on the circle, so a
        redundant group is visible as a huddle rather than a table.
    """
    p = assoc.shape[0]
    if p == 0:
        raise ValueError("assoc must have at least one row")
    if p == 1:
        return np.array([0.0])
    D = 1.0 - assoc                              # similarity -> distance
    coords = _classical_mds_2d(D)
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    if layout == "embed":
        return theta
    order = np.argsort(theta)
    if layout == "ordered":
        # even spacing that preserves the embedding's circular order
        spaced = np.empty(p)
        spaced[order] = np.linspace(0.0, 2.0 * np.pi, p, endpoint=False)
        return spaced
    if layout != "spread":
        raise ValueError(
            f"unknown layout {layout!r}, use 'spread', 'embed' or 'ordered'")
    # "spread": raise every consecutive gap to a readable minimum, then
    # shrink the remaining (large) gaps by a common factor so the total is
    # still one full turn.  Relative differences between the large gaps
    # survive; only the pile-ups are opened.
    g_min = _min_gap(p, min_gap_frac)
    gaps = np.diff(theta[order], append=theta[order][0] + 2.0 * np.pi)
    excess = np.clip(gaps - g_min, 0.0, None)     # room above the minimum
    budget = 2.0 * np.pi - p * g_min              # what the excesses may sum to
    scale = budget / excess.sum() if excess.sum() > 0 else 0.0
    new_gaps = np.array([_widened_gap(float(g), g_min, scale) for g in gaps])
    spread = np.empty(p)
    spread[order] = theta[order][0] + np.concatenate(
        ([0.0], np.cumsum(new_gaps[:-1])))
    return spread


def _marginal_gain(relevance: float, redundancy: float) -> float:
    """What a feature adds on top of the already-selected set.

    Intent:
        Charge a feature for the information it duplicates, so a
        near-copy of an earlier pick cannot win a round on relevance.
    """
    return relevance - redundancy


def greedy_selection(relevance: Vec("n"), assoc: Mat("n", "n")):
    """Greedy mRMR forward selection: an explicit feature-selection order.

    Minimum-redundancy maximum-relevance selection follows Peng, Long &
    Ding (2005), "Feature selection based on mutual information: criteria
    of max-dependency, max-relevance, and min-redundancy", IEEE TPAMI
    27(8), 1226-1238, with r_I as the relevance/redundancy measure.

    Why not a static "relevance minus redundancy against everything"
    score?  Because that charges each feature for overlapping with
    features you may never keep.  Redundancy only costs you against the
    features you have actually selected, so the score must be computed
    incrementally:

      step 1:  pick the most relevant feature.
      step t:  for each unpicked feature j, compute the marginal gain
                   gain_j = relevance_j - mean_{s in selected} assoc[j, s]
               and pick the largest.

    Every feature is ranked (the loop runs to the end), and the gain
    recorded at selection time is the honest "what this adds on top of
    what you already have".  A gain <= 0 means the feature duplicates
    the selected set more than it informs about the target: the natural
    stopping point.

    Returns:
      ranks  int array, ranks[j] = 1-based pick order of feature j
      gains  float array, gains[j] = marginal gain when feature j was picked

    Intent:
        Produce an honest feature-selection order, where every feature's
        score is what it added at the moment it was chosen.
    """
    relevance = np.asarray(relevance, dtype=float)
    p = len(relevance)
    if not np.all(np.isfinite(relevance)):
        # every gain would compare false against -inf and no feature
        # could ever be picked
        raise ValueError("relevance must be finite, got a NaN or infinity")
    ranks = np.zeros(p, dtype=int)
    gains = np.zeros(p)
    selected: list = []
    remaining = list(range(p))
    for step in range(1, p + 1):
        best_j, best_gain = None, -np.inf
        for j in remaining:
            if selected:
                redundancy = float(np.mean([assoc[j, s] for s in selected]))
            else:
                redundancy = 0.0
            gain = _marginal_gain(relevance[j], redundancy)
            if gain > best_gain:
                best_j, best_gain = j, gain
        ranks[best_j] = step
        gains[best_j] = best_gain
        selected.append(best_j)
        remaining.remove(best_j)
    return ranks, gains


def _mean_association(row_sum: float, p: int) -> float:
    """Mean association of one column to the other p - 1 columns.

    `row_sum` is that column's row of the association matrix summed,
    including its own diagonal 1, which is subtracted off here.

    Intent:
        Score how central one column is to the rest of the data, on the
        same 0..1 scale as any single association.
    """
    return (row_sum - 1.0) / (p - 1)


def select_target(X: Mat("n", "m"), k: int = 5) -> int:
    """Default target choice: the column most associated with all others.

    Returns the index of the column with the highest mean r_info to the
    remaining columns: the variable the rest of the data 'revolves
    around' the most.  A single-column input has no other columns to be
    associated with, so its only column is the answer.

    Intent:
        Choose what the plot should revolve around when the caller has
        not decided yet, without asking them to guess.
    """
    M = association_matrix(np.asarray(X, dtype=float), k=k)
    p = M.shape[0]
    if p == 0:
        raise ValueError("X must have at least one column")
    if p == 1:
        return 0
    mean_assoc = [_mean_association(float(s), p) for s in M.sum(axis=1)]
    return int(np.argmax(mean_assoc))
