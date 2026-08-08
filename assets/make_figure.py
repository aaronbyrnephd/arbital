"""Render the static figures used in the README.

Draws orbit systems with matplotlib so no browser screenshot is needed.
The geometry mirrors arbital.plot (display floor, periapsis-anchored
arcs, ghost tethers).  Outputs:

  assets/hero.png        the README hero image (penguins with categoricals)
  assets/quickstart.png  the README quick-start output (Auto MPG)

Run from the repo root:  PYTHONPATH=src python3 assets/make_figure.py
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import arbital
from arbital import datasets
from arbital.geometry import strength_to_distance

HERE = os.path.dirname(os.path.abspath(__file__))

BG, TEXT, CENTER = "#0b0e1a", "#d6d9e0", "#ffd166"
ORBIT, GHOST, GRID = "#a0b4ff", "#d6d9e0", "#ffffff"
R_FLOOR = 0.06                       # same display floor as arbital.plot
CMAP = LinearSegmentedColormap.from_list(
    "direction", ["#5a8fe6", "#c9cdd6", "#e06a6a"])


def display(r):
    """True distance (0..1) -> displayed radius (same rule as the plot)."""
    return R_FLOOR + (1.0 - R_FLOOR) * np.asarray(r, dtype=float)


def render(sys, title, out):
    """Draw one OrbitSystem as a static PNG at `out`."""
    fig, ax = plt.subplots(figsize=(8.4, 7.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # guide rings of constant association
    ring_phi = np.linspace(0, 2 * np.pi, 200)
    for lvl in (0.5, 0.8, 0.9, 0.95, 0.99):
        rad = float(display(strength_to_distance(lvl)))
        ax.plot(rad * np.cos(ring_phi), rad * np.sin(ring_phi),
                color=GRID, alpha=0.10, lw=0.8, ls=":")
        ax.text(0, rad + 0.02, f"association {lvl:g}", color=TEXT,
                alpha=0.4, fontsize=7, ha="center")

    # chance boundary: dashed circle + shaded outer band (same rule as
    # arbital.plot -- anything beyond is indistinguishable from noise)
    chance_vals = [m["chance"] for m in sys.metrics if "chance" in m]
    if chance_vals:
        c_r = float(display(strength_to_distance(max(chance_vals))))
        ax.fill_between(np.cos(ring_phi), 0, 0, color="none")  # no-op keeps mpl happy
        band = plt.matplotlib.patches.Annulus(
            (0, 0), 1.02, 1.02 - c_r, color="#d6d9e0", alpha=0.05)
        ax.add_patch(band)
        ax.plot(c_r * np.cos(ring_phi), c_r * np.sin(ring_phi),
                color="#d6d9e0", alpha=0.35, lw=1.0, ls="--")
        n_txt = f", n = {sys.n_rows}" if sys.n_rows else ""
        ax.text(0, -c_r + 0.035, f"chance boundary{n_txt}", color=TEXT,
                alpha=0.45, fontsize=7, ha="center")

    for i, (name, m, th) in enumerate(zip(sys.names, sys.metrics, sys.thetas)):
        rd_marker = float(display(m["r_peri"]))
        # orbit arc: periapsis focal form, vertex on the marker
        half = float(np.clip(0.5 / (2.0 * rd_marker), 0.35, 1.6))
        phi = np.linspace(-half, half, 91)
        ecc = m["e"]
        r = np.minimum(m["r_peri"] * (1 + ecc) / (1 + ecc * np.cos(phi)), 1.15)
        rd = display(r)
        ax.plot(rd * np.cos(th + phi), rd * np.sin(th + phi),
                color=ORBIT, alpha=0.45, lw=1.1)
        # ghost + tether for the correlation-only view
        gap = m["r_apo"] - m["r_peri"]
        if m["nonlinearity"] > 0.10 and gap > 0.02:
            p0, p1 = float(display(m["r_peri"])), float(display(m["r_apo"]))
            ax.plot([p0 * np.cos(th), p1 * np.cos(th)],
                    [p0 * np.sin(th), p1 * np.sin(th)],
                    color=GHOST, alpha=0.55, lw=0.9, ls=":")
            ax.scatter([p1 * np.cos(th)], [p1 * np.sin(th)], s=60,
                       facecolors="none", edgecolors=GHOST, alpha=0.7,
                       linewidths=1.4, zorder=4)
        # feature marker, area ~ selection gain, colour = signed Spearman
        px, py = rd_marker * np.cos(th), rd_marker * np.sin(th)
        size = (9.0 + 30.0 * np.sqrt(max(sys.sizes[i], 0.0))) ** 2 / 6.0
        ax.scatter([px], [py], s=size, c=[m["spearman"]], cmap=CMAP,
                   vmin=-1, vmax=1, edgecolors="white", linewidths=0.8,
                   zorder=5)
        va = "bottom" if i % 2 == 0 else "top"
        ax.annotate(name, (px, py), textcoords="offset points",
                    xytext=(0, 11 if va == "bottom" else -11),
                    color=TEXT, fontsize=9.5, ha="center", va=va, zorder=6)

    # the target at the focus
    ax.scatter([0], [0], s=260, color=CENTER, edgecolors="#fff3c4",
               linewidths=1.6, zorder=6)
    ax.annotate(sys.target_name, (0, 0), textcoords="offset points",
                xytext=(0, -16), color=CENTER, fontsize=10.5,
                fontweight="bold", ha="center", va="top", zorder=6)

    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.22, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    os.makedirs(HERE, exist_ok=True)

    penguins = arbital.orbits(datasets.load("penguins"), target="body_mass_g")
    render(penguins, "arbital: association orbits around penguin body mass",
           os.path.join(HERE, "hero.png"))

    mpg = arbital.orbits(datasets.load_mpg(), target="mpg")
    render(mpg, "arbital: what does fuel economy depend on?",
           os.path.join(HERE, "quickstart.png"))


if __name__ == "__main__":
    main()
