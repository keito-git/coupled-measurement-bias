"""
Figures for the TKDE manuscript, drawn in code.
fig_pipeline.pdf    : observation pipeline that produces the apparent negative modulation (schematic, no data)
fig_calibration.pdf : true gamma vs. naive estimate (mean and 95% band over simulations) and the real-data estimate
Data for fig_calibration: RESULTS_ROOT/E32/E32_results.json (calibration_map) and
RESULTS_ROOT/E27e/E27e_results.json (real-data gamma). Figures are written to analysis/figures/.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MPath

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                     "font.size": 8, "pdf.fonttype": 42})
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

X = config.RESULTS_ROOT
EDGE, LW, PAD = "#333333", 0.8, 0.02
EDGE_OFF = PAD + 0.8 / 72 / 2


def box(ax, x, y, w, h, fc, text, size=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD},rounding_size=0.05", fc=fc, ec=EDGE, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.25,
            fontweight="bold" if bold else "normal")


def conn(ax, pts, style="-|>", ls="-"):
    path = MPath(pts, [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle=style, mutation_scale=7, lw=LW, color=EDGE, ls=ls,
                                 shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt"))


def pipeline():
    W, H = 3.45, 2.15
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    LAT, OBS, FIT = "#EEEEEE", "#D0E4F0", "#F8E0C8"
    bh = 0.42
    x1, w1 = 0.05, 0.98          # left column: true categories / majority label
    x2, w2 = 1.40, 0.98          # middle column: ambiguity / annotators / entropy
    x3, w3 = 2.58, 0.82          # right column: model fit
    y_lat, y_ann, y_obs = 1.62, 0.92, 0.22
    box(ax, x1, y_lat, w1, bh, LAT, "true categories $c_m$\n(excitation process)")
    box(ax, x2, y_lat, w2, bh, LAT, "latent ambiguity\n$u_m$")
    box(ax, x2, y_ann, w2, bh, OBS, "$A$ annotators\naccuracy falls with $u_m$")
    box(ax, x1, y_obs, w1, bh, OBS, "majority label $\\hat{c}_m$\n(errors at high $u_m$)")
    box(ax, x2, y_obs, w2, bh, OBS, "vote entropy $H_m$\n(high at high $u_m$)")
    fit_y, fit_h = y_obs, y_ann + bh - y_obs
    box(ax, x3, fit_y, w3, fit_h, FIT,
        "DT-AMHP fit\n\nmark: $\\hat{c}_m$\nmodifier: $H_m$\ngain $e^{\\gamma H_m}$\n\n$\\Rightarrow\\ \\hat{\\gamma} < 0$\neven if $\\gamma = 0$")
    R = lambda x: x + EDGE_OFF
    L = lambda x: x - EDGE_OFF
    T = lambda y: y + EDGE_OFF
    B = lambda y: y - EDGE_OFF
    c1, c2 = x1 + w1 / 2, x2 + w2 / 2
    ym_lat, ym_obs = y_lat + bh / 2, y_obs + bh / 2
    conn(ax, [(c2, B(y_lat)), (c2, T(y_ann + bh))])                         # u -> annotators
    conn(ax, [(L(x2), ym_lat), (R(x1 + w1), ym_lat)])                       # u modulates excitation (true gamma)
    ax.text((x1 + w1 + x2) / 2, ym_lat + 0.03, "$\\gamma$", ha="center", va="bottom", fontsize=7.6)
    conn(ax, [(c1, B(y_lat)), (c1, y_ann + bh / 2), (L(x2), y_ann + bh / 2)])  # c -> annotators
    yb = 0.80                                                               # branch point of the vote arrow
    conn(ax, [(c2, B(y_ann)), (c2, yb)], style="-")                         # votes (common trunk)
    conn(ax, [(c2, yb), (c1, yb), (c1, T(y_obs + bh))])                     # votes -> majority label
    conn(ax, [(c2, yb), (c2, T(y_obs + bh))])                               # votes -> entropy
    ax.plot([c2], [yb], marker="o", ms=2.2, color=EDGE, zorder=5)          # branch dot
    conn(ax, [(R(x1 + w1), ym_obs), (L(x2), ym_obs)], style="-", ls=(0, (2, 1.5)))  # coupling (dashed)
    ax.text((x1 + w1 + x2) / 2, ym_obs + 0.03, "coupled", ha="center", va="bottom", fontsize=6.0, style="italic")
    conn(ax, [(R(x2 + w2), ym_obs), (L(x3), ym_obs)])                       # entropy -> fit (modifier)
    xm = x3 + w3 / 2
    conn(ax, [(c1, B(y_obs)), (c1, 0.08), (xm, 0.08), (xm, B(fit_y))])     # label -> fit (mark)
    fig.savefig(FIG_DIR / "fig_pipeline.pdf")
    plt.close(fig)


def calibration():
    cm = json.loads((X / "E32/E32_results.json").read_text())["calibration_map"]
    real = json.loads((X / "E27e/E27e_results.json").read_text())["real_gamma"]["full"]
    g, m, lo, hi = cm["g_arr"], cm["m_arr"], cm["lo_arr"], cm["hi_arr"]
    fig = plt.figure(figsize=(3.45, 1.9))
    ax = fig.add_axes([0.15, 0.22, 0.82, 0.74])
    ax.fill_between(g, lo, hi, color="#9EC3E0", alpha=0.6, lw=0, label="95% range of $\\hat{\\gamma}$")
    ax.plot(g, m, "-o", color="#1F4E79", ms=3, lw=1.0, label="mean of $\\hat{\\gamma}$")
    ax.plot([-1.3, 0.5], [-1.3, 0.5], ls=(0, (3, 2)), color="#777777", lw=0.8, label="$\\hat{\\gamma} = \\gamma$")
    ax.axhline(real, color="#C0504D", lw=1.0, label=f"real data ($\\hat{{\\gamma}}$)")
    ax.set_xlim(-1.3, 0.5); ax.set_ylim(-2.5, 0.4)
    ax.set_xlabel("true modification $\\gamma$"); ax.set_ylabel("naive estimate $\\hat{\\gamma}$")
    ax.legend(fontsize=6.6, frameon=False, loc="lower left", ncol=2, columnspacing=1.2, handlelength=1.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / "fig_calibration.pdf")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    calibration()
    print("wrote", FIG_DIR / "fig_pipeline.pdf", FIG_DIR / "fig_calibration.pdf")
