"""Build the orbit plot as a Plotly figure.

To stay dependency-light the figure is assembled as a plain dict in the
Plotly JSON schema.  You can:

  * pass it to plotly.graph_objects.Figure(...) if plotly is installed, or
  * call figure_html(...) which embeds the JSON in a self-contained HTML
    snippet that loads plotly.js from a CDN, no plotly install needed.

Visual encoding (see geometry.py for the maths):
  distance from centre  -> residual uncertainty exp(-I): the radial axis
                           is logarithmic in shared information, so every
                           factor of e closer to the centre is one more
                           nat of mutual information (closer = stronger)
  solid marker          -> the feature at its orbit's closest point
                           (periastron), sized by relative selection gain;
                           large markers get a small centre dot marking
                           their exact position
  ghost marker (hollow) -> where the same feature would sit judged by
                           correlation alone (the orbit's farthest point);
                           the dotted tether shows the gap = nonlinearity.
                           Drawn only when at least 10% of the dependence
                           is invisible to correlation, so estimator noise
                           does not conjure ghosts.
  orbit arc             -> a segment of the ellipse anchored at the marker.
                           Its curvature encodes eccentricity = nu (the
                           nonlinear share). The angular reach of the arc
                           toward each angular neighbour scales with the
                           feature-to-feature association: arcs of related
                           features reach toward each other and overlap,
                           arcs of unrelated features leave a clear gap
                           (qualitative; supplied via `assoc`).
  uncertainty band      -> with a bootstrap SE, a radial segment through
                           the marker spanning r_info +/- se: uncertainty
                           acts along the radius, so the band shows
                           directly whether the estimate could sit
                           elsewhere; a diamond marks any crossing of the
                           chance boundary (the estimate could be noise)
  chance boundary       -> dashed circle + shaded outer band, drawn at the
                           level shared by every continuous feature: beyond
                           it, association is indistinguishable from noise
                           at this sample size. Categorical features carry
                           their own, generally different (cardinality-
                           dependent) level instead: read theirs from the
                           hover text, not this circle. If every feature is
                           categorical no circle is drawn at all.
  marker colour         -> direction: signed Spearman rho (blue -, red +,
                           near-white = no monotone direction or nominal)
  hover                 -> highlights the marker with a halo and traces
                           the feature's full elliptical orbit (clipped
                           where it would leave the plot)

Readability notes: all radii pass through a display transform that
reserves a small exclusion zone around the centre so the strongest
features do not pile onto the target marker.
"""

from __future__ import annotations

import json

import numpy as np

from .geometry import strength_to_distance

__all__ = ["figure_html", "orbit_figure"]

# --- palette (dark theme) ---------------------------------------------------
BG = "#0b0e1a"
GRID = "rgba(255,255,255,0.10)"
TEXT = "#d6d9e0"
ORBIT = "rgba(160,180,255,0.45)"
ORBIT_FULL = "#c3d0ff"                 # hover-revealed full orbit
GHOST = "rgba(214,217,224,0.55)"
CENTER = "#ffd166"
BAND = "rgba(214,217,224,0.65)"        # radial uncertainty band

# Display transform: reserve the innermost 6% of the plot as an exclusion
# zone so features with near-perfect association stay legible instead of
# piling onto the centre.  Applied uniformly to rings, arcs and markers,
# so relative positions and the ring calibration stay truthful.
R_FLOOR = 0.06
R_CLIP = 1.06          # true distance beyond which orbit lines are clipped


def _display(r):
    """Map a true distance (0..1) to its displayed radius."""
    return R_FLOOR + (1.0 - R_FLOOR) * np.asarray(r, dtype=float)


def _clipped_line(r, ang):
    """(x, y) lists for a polar line, with None where it leaves the plot.

    Points whose true distance exceeds R_CLIP are dropped (plotly breaks
    the line at nulls), so an eccentric orbit visibly exits the system
    instead of smearing along the rim.
    """
    rd = _display(np.minimum(r, R_CLIP))
    xs = rd * np.cos(ang)
    ys = rd * np.sin(ang)
    keep = r <= R_CLIP
    x = [float(v) if k else None for v, k in zip(xs, keep)]
    y = [float(v) if k else None for v, k in zip(ys, keep)]
    return x, y


CHANCE_NOTE_MARGIN = 0.2  # r_info units: how close to the chance boundary
                          # is worth flagging in the hover text (below this,
                          # "clears chance" is obvious and not worth a line)


def _hover(name, m, size_label, size_val, rank=None, gain=None,
           se=None) -> str:
    """Hover label: one short line per channel, defined in the vignette
    glossary ("What the numbers mean").  No symbols that need decoding:
    r_info (total association), r_pearson, r_spearman."""
    lines = [f"<b>{name}</b>"]
    # 1. strength, with its uncertainty on the same line.  r_info is the
    #    estimated value (never chance-adjusted, see _apply_calibration
    #    for why), so this nats figure and r_info always agree through
    #    the one Linfoot formula, and the marker sits on the same
    #    coordinate system as the chance boundary below.
    pm = f" &plusmn; {se:.2f}" if se is not None else ""
    lines.append(f"r_info = {m['r_info']:.2f}{pm}   "
                 f"({m['mi']:.2f} nats, distance {m['r_peri']:.2f})")
    # 2. signal vs chance: short, but always names the level so the
    #    number behind "near"/"below" is visible.  Continuous features
    #    share the exact level the drawn circle sits at, so this says
    #    "chance boundary"; categorical features carry their own,
    #    cardinality-dependent level, which is generally not the drawn
    #    circle, so this says "its own chance level" instead. Reading
    #    a categorical marker's position against the circle would be
    #    comparing it to someone else's threshold. Above the boundary
    #    by a wide margin needs no line: repeating chance on every
    #    marker was the clutter the margin below avoids.
    if "chance" in m:
        label = "its own chance level" if m.get("categorical") else "chance boundary"
        if m.get("below_chance"):
            lines.append(f"<i>below {label} ({m['chance']:.2f}): "
                         f"likely noise</i>")
        elif m["r_info"] - m["chance"] <= CHANCE_NOTE_MARGIN:
            lines.append(f"near {label} ({m['chance']:.2f})")
        if (se is not None and not m.get("below_chance")
                and m["r_info"] - se < m["chance"]):
            lines.append(f"<i>uncertainty band crosses {label} "
                         f"({m['chance']:.2f})</i>")
    # 3. direction (colour) and shape (curvature).  Pearson/Spearman are
    #    reported as-is (they never set the radius); a coefficient not
    #    above its own permutation chance level (same confidence, same
    #    shuffle draws as the MI channel, see _chance_levels) is flagged
    #    as not significant, since it could be noise regardless of r_info.
    #    Gated on `nominal`, not `categorical`: a nominal column (more
    #    than two levels, e.g. embarked) has no meaningful sign, but a
    #    binary categorical column (e.g. sex) gets a real point-biserial
    #    correlation from profile() and keeps a direction like a
    #    continuous variable, matching the colour already shown on its
    #    marker.
    if m.get("nominal"):
        lines.append("categorical, multiple levels: strength only, no direction")
    else:
        lines.append(f"r_pearson = {m['pearson']:+.2f}, "
                     f"r_spearman = {m['spearman']:+.2f}")
        not_sig = []
        if not m.get("pearson_sig", True):
            not_sig.append("r_pearson")
        if not m.get("spearman_sig", True):
            not_sig.append("r_spearman")
        if not_sig:
            lines.append(f"<i>{' and '.join(not_sig)} not significant "
                         f"(below its own chance level)</i>")
        if m["nonlinearity"] >= 0.005:
            lines.append(f"nonlinear share = "
                         f"{100 * m['nonlinearity']:.0f}% "
                         f"(association that correlation misses)")
    # 4. selection
    if rank is not None and gain is not None:
        lines.append(f"pick {rank}, marginal gain {gain:+.2f}" if gain > 0
                     else f"adds nothing beyond earlier picks "
                          f"(gain {gain:+.2f})")
    elif size_label == "gain":
        lines.append(f"relative gain {size_val:.2f}")
    elif size_label == "rinfo":
        lines.append(f"relative r_info {size_val:.2f}")
    return "<br>".join(lines)


def _arc_spans(thetas, assoc, arc_length, r_display):
    """Per-feature (ccw_span, cw_span) in radians for the orbit arcs.

    With an association matrix, each arc reaches toward its angular
    neighbour in proportion to how associated the two features are:
    strongly associated neighbours' arcs overlap, unrelated neighbours'
    arcs leave a clear gap.  Qualitative by design.  Without `assoc`,
    falls back to a fixed readable length.
    """
    p = len(thetas)
    if assoc is None or p < 3:
        half = [float(np.clip(arc_length / (2.0 * rd), 0.35, 1.6))
                for rd in r_display]
        return list(zip(half, half))
    order = np.argsort(thetas)
    pos = np.empty(p, dtype=int)
    pos[order] = np.arange(p)
    spans = [None] * p
    for i in range(p):
        nxt = order[(pos[i] + 1) % p]          # ccw neighbour
        prv = order[(pos[i] - 1) % p]          # cw neighbour
        gap_ccw = (thetas[nxt] - thetas[i]) % (2 * np.pi)
        gap_cw = (thetas[i] - thetas[prv]) % (2 * np.pi)
        # reach = 12% of the gap as a floor, plus up to 55% more with
        # association: two sides at a > ~0.7 overlap in the middle
        ccw = gap_ccw * (0.12 + 0.55 * float(assoc[i, nxt]))
        cw = gap_cw * (0.12 + 0.55 * float(assoc[i, prv]))
        spans[i] = (float(np.clip(cw, 0.10, 0.9 * gap_cw + 0.10)),
                    float(np.clip(ccw, 0.10, 0.9 * gap_ccw + 0.10)))
    return spans


def orbit_figure(
    names: list,
    metrics: list,          # list of dicts from measures.profile + orbit params merged
    thetas: np.ndarray,     # angle per feature (radians)
    sizes: np.ndarray,      # per-feature value driving marker size (0..1)
    size_label: str = "gain",  # what `sizes` means
    target_name: str = "target",
    title: str | None = None,
    show_colorbar: bool = True,
    scale: str = "info",     # radial scale; must match how metrics were built
    ranks=None,              # greedy selection pick order (1-based), or None
    gains=None,              # marginal gain at pick time, or None
    se=None,                 # per-feature bootstrap SE of r_info, or None
    assoc=None,              # feature-feature r_info matrix (arc reach)
    arc_length: float = 0.5,  # fallback arc display-length (no assoc given)
    n_rows=None,             # rows measured (labels the chance boundary)
) -> dict:
    """Return the orbit plot as a Plotly figure dict (data + layout)."""
    data = []
    se = None if se is None else np.asarray(se, dtype=float)
    thetas = np.asarray(thetas, dtype=float)

    # chance boundary: one circle, at the pooled level shared by every
    # continuous feature (see _chance_levels: they all carry the exact
    # same `chance`, so any one of them is "the" value). Categorical
    # features are deliberately excluded here: each gets its own level
    # (cardinality-dependent), so a single shared circle would either be
    # too strict for a low-cardinality column or too lax for a
    # high-cardinality one. Drawing it anyway would let a marker's
    # numeric below_chance verdict disagree with which side of the
    # circle it visually sits on. Categorical markers still carry their
    # own chance value, shown in their hover text instead of read off
    # this circle. If every feature is categorical there is no single
    # honest threshold to draw, so no circle is drawn at all.
    chance_vals = [m["chance"] for m in metrics
                  if "chance" in m and not m.get("categorical")]
    chance_level = chance_vals[0] if chance_vals else None
    if chance_level is not None:
        c_r = float(_display(strength_to_distance(chance_level, scale)))
        t = np.linspace(0, 2 * np.pi, 100)
        outer, inner = 1.02, c_r
        data.append({                       # shaded annulus (outer band)
            "type": "scatter", "mode": "lines",
            "x": np.concatenate([outer * np.cos(t), inner * np.cos(t[::-1])]).tolist(),
            "y": np.concatenate([outer * np.sin(t), inner * np.sin(t[::-1])]).tolist(),
            "fill": "toself", "fillcolor": "rgba(214,217,224,0.05)",
            "line": {"width": 0}, "hoverinfo": "skip", "showlegend": False,
        })
        n_txt = f" at n = {n_rows}" if n_rows else " at this sample size"
        data.append({                       # dashed boundary circle
            "type": "scatter", "mode": "lines",
            "x": (c_r * np.cos(t)).tolist(),
            "y": (c_r * np.sin(t)).tolist(),
            "line": {"color": "rgba(214,217,224,0.35)", "width": 1,
                      "dash": "dash"},
            "hovertext": (f"chance boundary for continuous features:<br>beyond "
                          f"this line, association is indistinguishable<br>"
                          f"from noise{n_txt}.<br>Categorical features are "
                          f"compared against<br>their own level instead, "
                          f"see their hover text."),
            "hoverinfo": "text", "showlegend": False,
        })
        data.append({                       # boundary label (bottom), kept
                                             # short so it stays legible when
                                             # several plots sit side by side
                                             # at reduced width
            "type": "scatter", "mode": "text",
            "x": [0.0], "y": [-c_r + 0.035],
            "text": [f"chance boundary{n_txt.replace(' at ', ', ')}"],
            "textfont": {"color": "rgba(214,217,224,0.45)", "size": 10},
            "hoverinfo": "skip", "showlegend": False,
        })

    # guide rings: dotted circles of constant total association.  On the
    # info scale the interesting rings crowd toward the centre, so label
    # levels are denser at the strong end.  Labelled "assoc" rather than
    # "association" so the text stays short at reduced plot widths.
    levels = (0.5, 0.8, 0.9, 0.95, 0.99) if scale == "info" else (0.25, 0.5, 0.75)
    ring_angles = np.linspace(0, 2 * np.pi, 100)
    for lvl in levels:
        radius = float(_display(strength_to_distance(lvl, scale)))
        data.append({
            "type": "scatter", "mode": "lines",
            "x": (radius * np.cos(ring_angles)).tolist(),
            "y": (radius * np.sin(ring_angles)).tolist(),
            "line": {"color": GRID, "width": 1, "dash": "dot"},
            "hoverinfo": "skip", "showlegend": False,
        })
        data.append({
            "type": "scatter", "mode": "text",
            "x": [0.0], "y": [radius + 0.015],
            "text": [f"assoc {lvl:g}"],
            "textfont": {"color": "rgba(214,217,224,0.45)", "size": 10},
            "hoverinfo": "skip", "showlegend": False,
        })

    r_disp = [float(_display(m["r_peri"])) for m in metrics]
    spans = _arc_spans(thetas, assoc, arc_length, r_disp)

    ghost_x, ghost_y, ghost_hover = [], [], []
    ellipse_trace_idx = []      # per feature: trace index of its full orbit
    for i, (name, m, th) in enumerate(zip(names, metrics, thetas)):
        ecc = m["e"]
        p_focal = m["r_peri"] * (1.0 + ecc)

        # 0. the full orbit, hidden until the feature's marker is hovered
        #    (figure_html / the vignette attach the hover handler); same
        #    periapsis focal form as the arc, swept through a whole turn,
        #    clipped where it would leave the plot.
        phi_full = np.linspace(0.0, 2.0 * np.pi, 181)
        r_full = p_focal / (1.0 + ecc * np.cos(phi_full))
        fx, fy = _clipped_line(r_full, th + phi_full)
        ellipse_trace_idx.append(len(data))
        data.append({
            "type": "scatter", "mode": "lines",
            "x": fx, "y": fy,
            "line": {"color": ORBIT_FULL, "width": 1.6},
            "opacity": 0.0,                  # revealed on hover
            "hoverinfo": "skip", "showlegend": False,
            "meta": "full-orbit",
        })

        # 1. orbit arc anchored at the marker. Its curvature encodes the
        #    nonlinear share (ecc = nu); its reach toward each angular
        #    neighbour encodes how associated the two features are, so
        #    related features' arcs overlap (see _arc_spans).
        cw_span, ccw_span = spans[i]
        phi = np.linspace(-cw_span, ccw_span, 91)
        r = p_focal / (1.0 + ecc * np.cos(phi))
        ax, ay = _clipped_line(r, th + phi)
        data.append({
            "type": "scatter", "mode": "lines",
            "x": ax, "y": ay,
            "line": {"color": ORBIT, "width": 1.3},
            "hoverinfo": "skip", "showlegend": False,
        })

        # 2. tether + ghost marker: only when a meaningful share of the
        #    dependence is invisible to correlation (nonlinear share > 10%).
        gap = m["r_apo"] - m["r_peri"]
        if m["nonlinearity"] > 0.10 and gap > 0.02:
            p0, p1 = float(_display(m["r_peri"])), float(_display(m["r_apo"]))
            data.append({
                "type": "scatter", "mode": "lines",
                "x": [p0 * np.cos(th), p1 * np.cos(th)],
                "y": [p0 * np.sin(th), p1 * np.sin(th)],
                "line": {"color": GHOST, "width": 1, "dash": "dot"},
                "hoverinfo": "skip", "showlegend": False,
            })
            ghost_x.append(p1 * np.cos(th))
            ghost_y.append(p1 * np.sin(th))
            ghost_hover.append(
                f"<b>{name}</b>: where correlation alone would put it<br>"
                f"best monotone fit {m['r_mono']:.2f}, "
                f"distance {m['r_apo']:.2f}<br>"
                f"the tether is the association correlation misses "
                f"({100 * m['nonlinearity']:.0f}%)")

        # 3. radial uncertainty band: r_info +/- se along the feature's
        #    angle.  Uncertainty acts on the radius, so it is drawn there;
        #    a diamond flags a crossing of the chance boundary.
        if se is not None:
            r_hi = min(m["r_info"] + float(se[i]), 0.999)
            r_lo = max(m["r_info"] - float(se[i]), 0.0)
            d_in = float(_display(strength_to_distance(r_hi, scale)))
            d_out = float(_display(strength_to_distance(r_lo, scale)))
            data.append({
                "type": "scatter", "mode": "lines",
                "x": [d_in * np.cos(th), d_out * np.cos(th)],
                "y": [d_in * np.sin(th), d_out * np.sin(th)],
                "line": {"color": BAND, "width": 2.5},
                "hovertext": (f"<b>{name}</b> uncertainty band: r_info "
                              f"{m['r_info']:.2f} &plusmn; {se[i]:.2f}"),
                "hoverinfo": "text", "showlegend": False,
            })
            if chance_level is not None and r_lo < chance_level < r_hi:
                c_r_local = float(_display(
                    strength_to_distance(chance_level, scale)))
                data.append({
                    "type": "scatter", "mode": "markers",
                    "x": [c_r_local * np.cos(th)],
                    "y": [c_r_local * np.sin(th)],
                    "marker": {"size": 8, "symbol": "diamond",
                                "color": CENTER,
                                "line": {"color": BG, "width": 1}},
                    "hovertext": (f"<b>{name}</b>: uncertainty band crosses "
                                  f"the chance boundary, this association "
                                  f"could be noise"),
                    "hoverinfo": "text", "showlegend": False,
                })

    if ghost_x:
        data.append({
            "type": "scatter", "mode": "markers",
            "x": ghost_x, "y": ghost_y,
            "marker": {"size": 9, "symbol": "circle-open",
                        "color": GHOST, "line": {"width": 1.5}},
            "hovertext": ghost_hover, "hoverinfo": "text",
            "showlegend": False,
        })

    # 4. feature markers at the closest point of their orbit. Marker area
    #    encodes `sizes` on an absolute scale (px ~ sqrt(size)), so a given
    #    size means the same thing across every arbital figure. When a
    #    greedy selection ran, labels also carry the pick number (features
    #    with non-positive gain get no number).
    px = [rd * np.cos(th) for rd, th in zip(r_disp, thetas)]
    py = [rd * np.sin(th) for rd, th in zip(r_disp, thetas)]
    sz = np.asarray(sizes, dtype=float)
    if size_label == "uniform":
        marker_px = np.full(len(names), 13.0)     # size carries no variable
    else:
        marker_px = 9.0 + 30.0 * np.sqrt(np.clip(sz, 0.0, 1.0))
    if ranks is not None and gains is not None:
        g = np.asarray(gains, dtype=float)
        labels = [f"{rk}. {n}" if gg > 0 else n
                  for n, rk, gg in zip(names, ranks, g)]
        hovers = [_hover(n, m, size_label, s, rank=int(rk), gain=float(gg),
                         se=(None if se is None else float(se[i])))
                  for i, (n, m, s, rk, gg) in
                  enumerate(zip(names, metrics, sz, ranks, g))]
    else:
        labels = list(names)
        hovers = [_hover(n, m, size_label, s,
                         se=(None if se is None else float(se[i])))
                  for i, (n, m, s) in enumerate(zip(names, metrics, sz))]
    # alternate label positions to reduce collisions in crowded regions
    positions = ["top center" if i % 2 == 0 else "bottom center"
                 for i in range(len(names))]
    # Explicit diverging scale so the mapping cannot drift with plotly's
    # named-colorscale internals: negative -> blue, positive -> red,
    # matching the intuitive cool/warm convention and the legend text.
    marker = {
        "size": marker_px.tolist(),
        "color": [m["spearman"] for m in metrics],   # signed direction
        "colorscale": [[0.0, "#5a8fe6"], [0.5, "#c9cdd6"], [1.0, "#e06a6a"]],
        "cmin": -1, "cmax": 1,
        "line": {"color": "rgba(255,255,255,0.6)", "width": 1},
    }
    if show_colorbar:
        marker["colorbar"] = {
            "title": {"text": "direction<br>(Spearman rho)",
                      "font": {"color": TEXT, "size": 11}},
            "tickfont": {"color": TEXT, "size": 10},
            "thickness": 12, "len": 0.5,
        }
    data.append({
        "type": "scatter", "mode": "markers+text",
        "x": px, "y": py,
        "text": labels, "textposition": positions,
        "textfont": {"color": TEXT, "size": 12},
        "marker": marker,
        "hovertext": hovers,
        "hoverinfo": "text", "showlegend": False,
        # per-point index of the feature's hidden full-orbit trace, used
        # by the hover handler to fade the full ellipse in and out
        "customdata": ellipse_trace_idx,
        "meta": "features",
    })

    # 5. centre dots inside large markers: the exact measured position
    big = [(x, y) for x, y, mp in zip(px, py, marker_px) if mp >= 18.0]
    if big:
        data.append({
            "type": "scatter", "mode": "markers",
            "x": [b[0] for b in big], "y": [b[1] for b in big],
            "marker": {"size": 3, "color": "rgba(255,255,255,0.85)"},
            "hoverinfo": "skip", "showlegend": False,
        })

    # 6. hover halo: a single hidden open circle moved onto whichever
    #    marker is hovered (handled by the JS in figure_html/vignette)
    halo_idx = len(data)
    data.append({
        "type": "scatter", "mode": "markers",
        "x": [0.0], "y": [0.0],
        "marker": {"size": 34, "symbol": "circle-open",
                    "color": ORBIT_FULL, "line": {"width": 2}},
        "opacity": 0.0,
        "hoverinfo": "skip", "showlegend": False,
        "meta": "halo",
    })

    # 7. the central feature (the target)
    data.append({
        "type": "scatter", "mode": "markers+text",
        "x": [0.0], "y": [0.0],
        "marker": {"size": 22, "color": CENTER, "symbol": "circle",
                    "line": {"color": "#fff3c4", "width": 2}},
        "text": [f"<b>{target_name}</b>"], "textposition": "bottom center",
        "textfont": {"color": CENTER, "size": 13},
        "hovertext": [f"central feature: <b>{target_name}</b>"],
        "hoverinfo": "text", "showlegend": False,
    })

    layout = {
        "title": {"text": title or f"arbital, association orbits around '{target_name}'",
                   "font": {"color": TEXT, "size": 17}, "x": 0.5},
        "paper_bgcolor": BG, "plot_bgcolor": BG,
        "xaxis": {"visible": False, "range": [-1.28, 1.28],
                   "scaleanchor": "y", "scaleratio": 1},
        "yaxis": {"visible": False, "range": [-1.2, 1.2]},
        "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
        "hoverlabel": {"bgcolor": "rgba(26,31,53,0.85)",
                        "font": {"color": TEXT, "size": 12}},
        "meta": {"halo": halo_idx},
    }
    return {"data": data, "layout": layout}


# JavaScript hover behaviour shared by figure_html() and the vignette
# template (demo/template.html keeps a copy in sync): hovering a feature
# marker reveals its full orbit and rings it with a halo.
HOVER_JS = """
      function featurePoint(ev) {
        var p = ev.points && ev.points[0];
        return (p && p.data.meta === "features" &&
                typeof p.customdata === "number") ? p : null;
      }
      gd.on("plotly_hover", function(ev) {
        var p = featurePoint(ev);
        if (!p) return;
        Plotly.restyle(gd, {opacity: 0.55}, [p.customdata]);
        var halo = gd.layout.meta && gd.layout.meta.halo;
        if (typeof halo === "number")
          Plotly.restyle(gd, {x: [[p.x]], y: [[p.y]], opacity: 0.9,
                              "marker.size": (p["marker.size"] || 13) + 12},
                         [halo]);
      });
      gd.on("plotly_unhover", function(ev) {
        var p = featurePoint(ev);
        if (!p) return;
        Plotly.restyle(gd, {opacity: 0.0}, [p.customdata]);
        var halo = gd.layout.meta && gd.layout.meta.halo;
        if (typeof halo === "number")
          Plotly.restyle(gd, {opacity: 0.0}, [halo]);
      });
"""


def figure_html(fig: dict, div_id: str = "arbital-plot",
                height: str = "720px", include_cdn: bool = True) -> str:
    """Self-contained HTML snippet rendering the figure via plotly.js CDN."""
    cdn = ('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
           if include_cdn else "")
    payload = json.dumps(fig)
    return f"""{cdn}
<div id="{div_id}" style="width:100%;height:{height};"></div>
<script>
  (function() {{
    var fig = {payload};
    Plotly.newPlot("{div_id}", fig.data, fig.layout,
                   {{responsive: true, displaylogo: false}}).then(function(gd) {{
{HOVER_JS}
    }});
  }})();
</script>"""
