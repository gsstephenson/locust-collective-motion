"""
Regenerate Figure 1 using Table 1 values from the paper text.

Rework goals (per coauthor comment):
  - Remove the "AI summary" rounded-corner text rectangle.
  - Rename "Hybrid" to "Anisotropic Vicsek" everywhere.
  - Use the numbers reported in Table 1 of the current draft (they differ
    from the earlier figure rendering).
  - Drop the "Week 4 -- Full Model Comparison" class-assignment title.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

MODELS = ["Field", "Vicsek", "Pull", "Anisotropic\nVicsek"]
COLORS = ["#2d2d2d", "#e07a3c", "#4a8cd8", "#3fa66a"]

POL_MEAN = [0.820, 0.822, 0.037, 0.885]
POL_STD  = [0.062, 0.019, 0.001, 0.025]

TURN_MEAN = [0.276, 0.603, np.nan, 0.519]
TURN_STD  = [0.000, 0.005, 0.000, 0.008]

NND_MEAN_MED = [3.893, 2.554, np.nan, 2.147]
NND_STD_MED  = [0.000, 0.036, 0.000, 0.030]

NND_MEAN_MEAN = [4.473, 2.940, np.nan, 2.405]
NND_STD_MEAN  = [0.000, 0.047, 0.000, 0.046]

ANISO = [1, 0, 0, 1]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle(
    "Four-metric comparison: Vicsek, Pull, and Anisotropic Vicsek vs. field data (Weinburd et al. 2024)",
    fontsize=12, y=0.995,
)

def bar_panel(ax, means, stds, title, ylabel, field_ref=None, miss_label="n/a"):
    xs = np.arange(len(MODELS))
    vals = np.array([0 if np.isnan(m) else m for m in means])
    errs = np.array([0 if np.isnan(s) else s for s in stds])
    ax.bar(xs, vals, yerr=errs, color=COLORS, capsize=4,
           edgecolor="black", linewidth=0.6)
    if field_ref is not None:
        ax.axhline(field_ref, linestyle="--", color="gray", linewidth=0.8, zorder=0)
    for i, m in enumerate(means):
        if np.isnan(m):
            ax.text(i, ax.get_ylim()[1] * 0.04, miss_label,
                    ha="center", va="bottom", style="italic",
                    color="#888", fontsize=9)
        else:
            ax.text(i, m + (errs[i] if not np.isnan(errs[i]) else 0) + 0.015 * max(vals),
                    f"{m:.3f}" if m < 10 else f"{m:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(MODELS)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

bar_panel(axes[0,0], POL_MEAN, POL_STD,
          "Polarization $\\Phi$", "$\\Phi$", field_ref=POL_MEAN[0])
axes[0,0].set_ylim(0, 1.05)

bar_panel(axes[0,1], TURN_MEAN, TURN_STD,
          "Turning-angle std", "rad", field_ref=TURN_MEAN[0])
axes[0,1].set_ylim(0, 0.75)

bar_panel(axes[1,0], NND_MEAN_MED, NND_STD_MED,
          "Nearest-neighbor distance (median)", "cm", field_ref=NND_MEAN_MED[0])
axes[1,0].set_ylim(0, 4.5)

ax = axes[1,1]
xs = np.arange(len(MODELS))
ax.bar(xs, ANISO, color=COLORS, edgecolor="black", linewidth=0.6)
ax.set_xticks(xs)
ax.set_xticklabels(MODELS)
ax.set_yticks([0, 1])
ax.set_yticklabels(["isotropic", "anisotropic\n(forward void)"])
ax.set_title("Neighbor density map structure")
ax.set_ylim(-0.05, 1.25)
for i, v in enumerate(ANISO):
    label = "void" if v == 1 else "no void"
    ax.text(i, v + 0.05, label, ha="center", va="bottom", fontsize=9,
            color="black" if v else "#555")

legend_elements = [
    Patch(facecolor=COLORS[0], edgecolor="black", label="Field (Weinburd 2024)"),
    Patch(facecolor=COLORS[1], edgecolor="black", label=r"Vicsek  $\eta=1.3$ rad"),
    Patch(facecolor=COLORS[2], edgecolor="black", label=r"Pull  $\eta=0.2$ rad"),
    Patch(facecolor=COLORS[3], edgecolor="black",
          label=r"Anisotropic Vicsek  $\eta_{\mathrm{base}}=1.1,\ \lambda=0.8,\ \alpha=0.6$"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=(0, 0.05, 1, 0.97))
fig.savefig("figure1.png", dpi=200, bbox_inches="tight")
fig.savefig("figure1.pdf", bbox_inches="tight")
print("Saved figure1.png and figure1.pdf")
