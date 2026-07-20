"""Generate the vignette page (arbital_vignette.html).

Builds every figure and table used in the vignette from live computation,
so the numbers in the text cannot drift from the code.

Run from the repo root:  PYTHONPATH=src python3 demo/make_demo.py
"""

import functools
import json
import os
import warnings

import numpy as np

import arbital
from arbital import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(7)
PANEL = "#12172b"

# The library default n_shuffles=200 is chosen so a *user's* call doesn't
# trip the "too few draws" stability warning (see arbital's docstrings).
# This demo rebuilds ~20 systems from scratch on every run, so it trades
# that stability margin for build speed: n_shuffles=64 everywhere unless a
# call overrides it (the sample-size and uncertainty panels do, on purpose,
# to *demonstrate* the tradeoff). functools.partial keeps every call site
# below identical to a plain arbital.orbits()/select_features() call.
orbits = functools.partial(arbital.orbits, n_shuffles=64)
select_features = functools.partial(arbital.select_features, n_shuffles=64)


class Named:
    """DataFrame-like wrapper so orbits() sees column names."""
    def __init__(self, X, names):
        self._X, self.columns = X, names
    def __array__(self, dtype=None):
        return self._X if dtype is None else self._X.astype(dtype)


def system(cols, target, **kw):
    names = list(cols)
    X = np.column_stack([cols[c] for c in names])
    return orbits(Named(X, names), target=target, **kw)


def fig(space, title, colorbar=True):
    f = space.figure(title=title)
    if not colorbar:
        for tr in f["data"]:
            if tr.get("mode") == "markers+text" and "colorbar" in tr.get("marker", {}):
                del tr["marker"]["colorbar"]
    f["layout"]["paper_bgcolor"] = PANEL
    f["layout"]["plot_bgcolor"] = PANEL
    return f


def scatter_grid(cols, feats, target_name):
    y = cols[target_name]
    ncol = min(4, len(feats))
    nrow = int(np.ceil(len(feats) / ncol))
    gap = 0.06
    w = (1 - gap * (ncol - 1)) / ncol
    h = (1 - 0.24) / nrow
    data, axes, notes = [], {}, []
    for i, f in enumerate(feats):
        r, c = divmod(i, ncol)
        ax = "" if i == 0 else str(i + 1)
        x0, y0 = c * (w + gap), (nrow - 1 - r) * (h + 0.24 / max(nrow, 1))
        axes[f"xaxis{ax}"] = {"domain": [x0, x0 + w], "anchor": f"y{ax}", "visible": False}
        axes[f"yaxis{ax}"] = {"domain": [y0, y0 + h], "anchor": f"x{ax}", "visible": False}
        data.append({
            "type": "scattergl", "mode": "markers",
            "x": cols[f].tolist(), "y": y.tolist(),
            "xaxis": f"x{ax}" if ax else "x", "yaxis": f"y{ax}" if ax else "y",
            "marker": {"size": 3, "color": "#7aa2ff", "opacity": 0.5},
            "hoverinfo": "skip", "showlegend": False,
        })
        notes.append({
            "x": x0 + w / 2, "y": y0 + h + 0.015, "xref": "paper", "yref": "paper",
            "xanchor": "center", "yanchor": "bottom", "showarrow": False,
            "text": f"<b>{f}</b>", "font": {"color": "#d6d9e0", "size": 13},
        })
    return {"data": data,
            "layout": {"paper_bgcolor": PANEL, "plot_bgcolor": PANEL,
                        "margin": {"l": 10, "r": 10, "t": 24, "b": 10},
                        "annotations": notes, **axes}}


def metrics_rows(space):
    out = []
    for m in space.table():
        cat = "✓" if m.get("categorical") else ""
        out.append(
            f"<tr><td>{m['name']}</td><td class='num'>{cat}</td>"
            f"<td class='num'>{m['pearson']:+.2f}</td>"
            f"<td class='num'>{m['spearman']:+.2f}</td>"
            f"<td class='num'>{m['mi']:.3f}</td>"
            f"<td class='num'>{m['r_info']:.2f}</td>"
            f"<td class='num'>{m['nonlinearity']:.0%}</td>"
            f"<td class='num'>{m['pick']}</td>"
            f"<td class='num'>{m['gain']:+.2f}</td></tr>")
    return "\n  ".join(out)


def selection_rows(picks):
    out = []
    for s in picks:
        chosen = s["gain"] > 0
        num = str(s["pick"]) if chosen else "&ndash;"
        style = "" if chosen else " style='opacity:0.55'"
        out.append(
            f"<tr{style}><td class='num'>{num}</td><td>{s['name']}</td>"
            f"<td class='num'>{s['relevance']:.2f}</td>"
            f"<td class='num'>{s['redundancy']:.2f}</td>"
            f"<td class='num'><b>{s['gain']:+.2f}</b></td></tr>")
    return "\n  ".join(out)


# ---------------------------------------------------------------- datasets

def synthetic():
    t = RNG.standard_normal(500)
    e = RNG.standard_normal
    return {
        "target":      t,
        "linear":      t + 0.35 * e(500),
        "linear_copy": t + 0.45 * e(500),
        "parabola":    t**2 + 0.30 * e(500),
        "sine":        np.sin(2.5 * t) + 0.25 * e(500),
        "monotone":    np.exp(t) + 0.25 * e(500),
        "weak":        0.45 * t + e(500),
        "noise":       e(500),
    }


def transformed():
    n = 800
    base = RNG.uniform(1, 60, n)
    y = np.log(base) + 0.10 * RNG.standard_normal(n)
    return {
        "y":      y,
        "raw":    base,
        "log":    np.log(base),
        "sqrt":   np.sqrt(base),
        "capped": np.minimum(base, 30),
    }


def grouped():
    """Two latent factors each driving a group of proxies, one lone
    informative variable, two noise variables: the angle-axis demo."""
    n = 500
    e = RNG.standard_normal
    f1, f2, g = e(n), e(n), e(n)
    return {
        "target": f1 + 0.8 * f2 + 0.6 * g + 0.4 * e(n),
        "a1":     f1 + 0.30 * e(n),
        "a2":     f1 + 0.35 * e(n),
        "a3":    -f1 + 0.30 * e(n),
        "b1":     f2 + 0.30 * e(n),
        "b2":     f2 + 0.35 * e(n),
        "lone":   g + 0.30 * e(n),
        "noise1": e(n),
        "noise2": e(n),
    }


def interaction():
    """y = x1*x2: each parent is uncorrelated with y (r ~ rho ~ 0) yet
    strongly associated through the sign-flipping interaction."""
    n = 500
    e = RNG.standard_normal
    x1, x2 = e(n), e(n)
    y = x1 * x2 + 0.3 * e(n)
    return {"y": y, "x1": x1, "x2": x2,
            "product": x1 * x2 + 0.1 * e(n), "noise": e(n)}


def simpson_scatter(pen):
    """bill_depth vs body_mass coloured by species: the pooled slope is
    negative while every within-species slope is positive."""
    depth = pen.column("bill_depth_mm")
    mass = pen.column("body_mass_g")
    species = pen.column("species")
    labels = pen.levels["species"]
    colors = ["#7aa2ff", "#ffd166", "#e06a6a"]
    data = []
    for i, lab in enumerate(labels):
        m = species == i
        data.append({
            "type": "scattergl", "mode": "markers", "name": lab,
            "x": depth[m].tolist(), "y": mass[m].tolist(),
            "marker": {"size": 5, "color": colors[i % 3], "opacity": 0.7},
        })
    return {"data": data,
            "layout": {"paper_bgcolor": PANEL, "plot_bgcolor": PANEL,
                        "margin": {"l": 55, "r": 10, "t": 10, "b": 40},
                        "font": {"color": "#d6d9e0", "size": 11},
                        "xaxis": {"title": {"text": "bill_depth_mm"},
                                   "gridcolor": "rgba(255,255,255,0.08)"},
                        "yaxis": {"title": {"text": "body_mass_g"},
                                   "gridcolor": "rgba(255,255,255,0.08)"},
                        "legend": {"orientation": "h", "y": 1.08}}}


def weak_signals(n=700):
    """Four genuine-but-weak drivers plus noise: nothing clears r_I ~ 0.5,
    so the info radial scale piles everything at the rim. This is the
    dataset that motivates scale="linear", and (drawn at several n) the
    chance boundary's dependence on sample size."""
    rng = np.random.default_rng(11)          # own seed: numbers quoted in text
    e = rng.standard_normal
    u1, u2, u3, u4 = e(n), e(n), e(n), e(n)
    return {
        "outcome": 0.6 * u1 + 0.5 * u2 + 0.4 * u3 + 0.3 * u4 + e(n),
        "driver_a": u1 + 0.4 * e(n),
        "driver_b": u2 + 0.4 * e(n),
        "driver_c": u3 + 0.4 * e(n),
        "driver_d": u4 + 0.4 * e(n),
        "noise1": e(n),
        "noise2": e(n),
    }


def lagged():
    n = 600
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = 0.8 * s[i - 1] + RNG.standard_normal()
    cols = {"series_t": s, "lag_1": np.roll(s, 1), "lag_2": np.roll(s, 2),
            "lag_3": np.roll(s, 3), "lag_8": np.roll(s, 8)}
    return {k: v[10:] for k, v in cols.items()}   # drop wrap-around rows


def main():
    syn = synthetic()
    sys_syn = system(syn, "target")

    grp = grouped()
    sys_grp = system(grp, "target", selection=True)
    lay_figs = {lay: system(grp, "target", angle_layout=lay)
                for lay in ("spread", "embed", "ordered")}

    tf = transformed()
    sys_tf = system(tf, "y")
    tf_sel = select_features(Named(np.column_stack([tf[c] for c in tf]),
                                           list(tf)), target="y")
    # whichever of raw/log/sqrt/capped wins pick 1: since they are monotone
    # transforms of one another, their r_info values are nearly tied (see
    # section 13's copula-invariance caveat), so the winner is decided by
    # estimator noise and the vignette prose should name it rather than
    # assume it is always "log".
    tf_winner = tf_sel[0]["name"]

    lg = lagged()
    sys_lg = system(lg, "series_t")

    ia = interaction()
    sys_ia = system(ia, "y")
    ia_rows = {r["name"]: r for r in sys_ia.table()}

    wk = weak_signals()
    sys_wk_info = system(wk, "outcome")
    sys_wk_lin = system(wk, "outcome", scale="linear")
    wk_top = max(sys_wk_info.table(), key=lambda r: r["r_info"])
    wk_top_lin = [r for r in sys_wk_lin.table()
                  if r["name"] == wk_top["name"]][0]
    # the weakest of the four genuine (non-noise) drivers, named and
    # measured live rather than quoted from the generating coefficients:
    # sampling noise at n=700 means the realised r/r_info for a given
    # driver can differ noticeably from the coefficient it was built with
    # (see weak_signals() above), so the text should read off this driver's
    # actual measured numbers, not the nominal 0.3.
    wk_true_coef = {"driver_a": 0.6, "driver_b": 0.5,
                    "driver_c": 0.4, "driver_d": 0.3}
    wk_worst = min((r for r in sys_wk_info.table()
                    if r["name"] in wk_true_coef),
                   key=lambda r: r["r_info"])

    # the same weak-driver process at three sample sizes: the chance
    # boundary recedes toward the rim as n grows.  The 95th-percentile
    # level needs enough shuffle draws to be stable across panels, so use
    # more draws where they are cheap (small n).
    sys_by_n = {n: system(weak_signals(n), "outcome", max_samples=n,
                          scale="linear",       # weak signals: linear lens
                          n_shuffles=(160 if n <= 1000 else 48))
                for n in (250, 1000, 2000)}
    cb = {n: s.table()[0]["chance"] for n, s in sys_by_n.items()}

    sys_mpg = orbits(datasets.load_mpg(), target="mpg", selection=True)
    sys_tit = orbits(datasets.load_titanic(), target="survived")

    # uncertainty tutorial: the weak-driver system (true r 0.5/0.4/0.3/0.25)
    # actually shows bands crossing the boundary, unlike Palmer penguins
    # (where every physical measurement clears chance comfortably and the
    # diamond never appears).  Two panels, small n vs. large n, pair with
    # the "more data pushes the boundary out" story below.
    sys_unc_lo = system(weak_signals(250), "outcome", max_samples=250,
                        scale="linear", uncertainty=True, n_bootstrap=200,
                        n_shuffles=160)
    sys_unc_hi = system(weak_signals(2000), "outcome", max_samples=2000,
                        scale="linear", uncertainty=True, n_bootstrap=60,
                        n_shuffles=48)
    unc_lo_rows = {r["name"]: r for r in sys_unc_lo.table()}
    unc_hi_rows = {r["name"]: r for r in sys_unc_hi.table()}

    def _crossing(rows):
        out = []
        for r in rows.values():
            se = r.get("r_info_se")
            if se is None or "chance" not in r:
                continue
            if r["r_info"] - se < r["chance"] < r["r_info"] + se:
                out.append(r["name"])
        return sorted(out)

    unc_lo_crossing = _crossing(unc_lo_rows)
    unc_hi_crossing = _crossing(unc_hi_rows)

    def _describe_hi(rows, crossing):
        # position is always the estimated value (see _apply_calibration),
        # so which variable straddles the boundary at n = 2000 is not fixed
        # in advance: describe whatever actually happened this run rather
        # than naming a driver ahead of time.
        drivers = [n for n in crossing if n.startswith("driver")]
        noise = [n for n in crossing if n.startswith("noise")]
        if drivers:
            n, r = drivers[0], rows[drivers[0]]
            return (f"only {n} ({r['r_info']:.2f} &plusmn; {r['r_info_se']:.2f}, "
                    f"chance {r['chance']:.2f}) still straddles it, the last "
                    f"of the four genuine drivers to resolve as n grows")
        if noise:
            n, r = noise[0], rows[noise[0]]
            return (f"all four genuine drivers have cleared it; {n} "
                    f"({r['r_info']:.2f} &plusmn; {r['r_info_se']:.2f}, "
                    f"chance {r['chance']:.2f}) sits right at it instead, "
                    f"a reminder that the boundary is a 95th-percentile "
                    f"threshold, not a guarantee, so a genuinely independent "
                    f"variable will occasionally land this close by chance")
        return "all six variables have resolved cleanly, clear of the boundary"

    unc_hi_describe = _describe_hi(unc_hi_rows, unc_hi_crossing)

    tips = datasets.load_tips()
    sys_tips = orbits(tips, target="tip")
    tips_rows = {r["name"]: r for r in sys_tips.table()}

    pen_full = datasets.load("penguins")
    sys_pf = orbits(pen_full, target="body_mass_g", selection=True)
    pf_rows = {r["name"]: r for r in sys_pf.table()}

    mpg_sel = select_features(datasets.load_mpg(), target="mpg")
    tit_rows = {r["name"]: r for r in sys_tit.table()}
    tit_sel = select_features(datasets.load_titanic(), target="survived")

    # a real to_df() output for the "every column" section: three rows,
    # a representative slice of columns so it fits on the page, with a
    # note that the rest of the columns (listed in the glossary below)
    # are there too, just not printed here.
    df_cols = ["r_info", "mi", "chance", "below_chance", "mi_adj",
              "r_info_adj", "pearson", "pearson_sig", "pick", "gain"]
    df_example = sys_mpg.to_df()[df_cols].round(3).to_string()
    df_all_columns = ", ".join(sys_mpg.to_df().columns.tolist())

    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as fh:
        page = fh.read()

    repl = {
        "{VERSION}": arbital.__version__,
        "{FIG_SYNTH}": json.dumps(fig(sys_syn, "Synthetic features around a target")),
        "{FIG_SCATTER}": json.dumps(scatter_grid(syn, sys_syn.names, "target")),
        "{FIG_GROUPS}": json.dumps(fig(sys_grp, "Two redundancy groups, a lone variable, and noise")),
        "{FIG_LAY_SPREAD}": json.dumps(fig(lay_figs["spread"], "spread (default)", colorbar=False)),
        "{FIG_LAY_EMBED}": json.dumps(fig(lay_figs["embed"], "embed", colorbar=False)),
        "{FIG_LAY_ORDERED}": json.dumps(fig(lay_figs["ordered"], "ordered", colorbar=False)),
        "{FIG_TRANSFORM}": json.dumps(fig(sys_tf, "A variable and monotone transforms of it")),
        "{FIG_LAG}": json.dumps(fig(sys_lg, "An autoregressive series against its own lags")),
        "{FIG_INTERACT}": json.dumps(fig(sys_ia, "An interaction: y = x1·x2, invisible to correlation")),
        "{FIG_TITANIC}": json.dumps(fig(sys_tit, "Titanic survival, with categorical fields")),
        "{FIG_TIPS}": json.dumps(fig(sys_tips, "What does the tip depend on?")),
        "{FIG_PENFULL}": json.dumps(fig(sys_pf, "Penguins with all categorical fields kept")),
        "{FIG_SIMPSON}": json.dumps(simpson_scatter(pen_full)),
        "{FIG_SELECT}": json.dumps(fig(sys_mpg, "Auto MPG with selection numbering")),
        "{FIG_UNC_LO}": json.dumps(fig(sys_unc_lo, "n = 250", colorbar=False)),
        "{FIG_UNC_HI}": json.dumps(fig(sys_unc_hi, "n = 2,000", colorbar=False)),
        "{UNC_LO_CROSSING}": ", ".join(unc_lo_crossing) if unc_lo_crossing else "none",
        "{UNC_HI_DESCRIBE}": unc_hi_describe,
        "{FIG_SCALE_INFO}": json.dumps(fig(sys_wk_info, "scale=\"info\" (default)", colorbar=False)),
        "{FIG_SCALE_LINEAR}": json.dumps(fig(sys_wk_lin, "scale=\"linear\"", colorbar=False)),
        "{WEAK_TOP_RI}": f"{wk_top['r_info']:.2f}",
        "{WEAK_TOP_D_INFO}": f"{wk_top['r_peri']:.2f}",
        "{WEAK_TOP_D_LIN}": f"{wk_top_lin['r_peri']:.2f}",
        "{WEAK_WORST_NAME}": wk_worst["name"],
        "{WEAK_WORST_COEF}": f"{wk_true_coef[wk_worst['name']]:g}",
        "{WEAK_WORST_R}": f"{wk_worst['pearson']:+.2f}",
        "{WEAK_WORST_RI}": f"{wk_worst['r_info']:.2f}",
        "{FIG_N250}": json.dumps(fig(sys_by_n[250], "n = 250", colorbar=False)),
        "{FIG_N1000}": json.dumps(fig(sys_by_n[1000], "n = 1,000", colorbar=False)),
        "{FIG_N2000}": json.dumps(fig(sys_by_n[2000], "n = 2,000", colorbar=False)),
        "{CB_250}": f"{cb[250]:.2f}",
        "{CB_1000}": f"{cb[1000]:.2f}",
        "{CB_2000}": f"{cb[2000]:.2f}",
        "{ROWS_SYN}": metrics_rows(sys_syn),
        "{ROWS_PENFULL}": metrics_rows(sys_pf),
        "{INT_X1_R}": f"{max(abs(ia_rows['x1']['pearson']), abs(ia_rows['x1']['spearman'])):.2f}",
        "{INT_X1_RI}": f"{ia_rows['x1']['r_info']:.2f}",
        "{INT_X1_NU}": f"{ia_rows['x1']['nonlinearity']:.0%}",
        "{TIPS_BILL_RI}": f"{tips_rows['total_bill']['r_info']:.2f}",
        "{TIPS_NOTE}": tips.notes,
        "{PEN_BD_R}": f"{pf_rows['bill_depth_mm']['pearson']:+.2f}",
        "{PEN_BD_RI}": f"{pf_rows['bill_depth_mm']['r_info']:.2f}",
        "{PEN_BD_NU}": f"{pf_rows['bill_depth_mm']['nonlinearity']:.0%}",
        "{PEN_SP_RI}": f"{pf_rows['species']['r_info']:.2f}",
        "{SEL_TRANSFORM}": selection_rows(tf_sel),
        "{TF_WINNER}": tf_winner,
        "{SEL_MPG}": selection_rows(mpg_sel),
        "{SEL_TITANIC}": selection_rows(tit_sel),
        "{S2_R}": f"{[r for r in sys_syn.table() if r['name']=='monotone'][0]['pearson']:.2f}",
        "{S2_RHO}": f"{[r for r in sys_syn.table() if r['name']=='monotone'][0]['spearman']:.2f}",
        "{S2_RI}": f"{[r for r in sys_syn.table() if r['name']=='monotone'][0]['r_info']:.2f}",
        "{TIT_SEX_RI}": f"{tit_rows['sex']['r_info']:.2f}",
        "{TIT_SEX_R}": f"{tit_rows['sex']['pearson']:+.2f}",
        "{TIT_FARE_GAIN}": f"{[s for s in tit_sel if s['name']=='fare'][0]['gain']:+.2f}",
        "{DF_EXAMPLE}": df_example,
        "{DF_ALL_COLUMNS}": df_all_columns,
        "{MPG_NOTE}": datasets.load_mpg().notes,
        "{PEN_NOTE}": datasets.load_penguins().notes,
        "{TIT_NOTE}": datasets.load_titanic().notes,
    }
    for key, val in repl.items():
        page = page.replace(key, val)

    out = os.path.join(HERE, "arbital_vignette.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote", out)
    for label, s in [("transformed", sys_tf), ("lagged", sys_lg)]:
        print(f"--- {label}")
        for r in s.table():
            print(f"  {r['name']:<12} r_I={r['r_info']:.2f} pick={r['pick']} "
                  f"gain={r['gain']:+.2f} size={r['size']:.2f}")


if __name__ == "__main__":
    # this file deliberately trades quantile stability for build speed
    # (see the comment on `orbits`/`select_features` above); the warning
    # that trade-off triggers is expected here, not a real problem
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*draws beyond the quantile.*")
        main()
