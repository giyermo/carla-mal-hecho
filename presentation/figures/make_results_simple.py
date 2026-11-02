"""Plain version of the results-per-round chart for the slide (no title: the slide has one)."""
from pathlib import Path

import matplotlib.pyplot as plt

INK, SECONDARY, GRID, AXIS = "#1E2B26", "#52514e", "#e1e0d9", "#c3c2b7"
SETUPS = {  # label, colour, marker, mean test mAP50 per round (3 seeds)
    "Cold": ("#2a78d6", "o", [0.768, 0.796, 0.821]),
    "Warm": ("#eb6834", "s", [0.768, 0.793, 0.815]),
    "Shrink & Perturb": ("#1baf7a", "D", [0.768, 0.799, 0.829]),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "text.color": INK,
                     "axes.edgecolor": AXIS, "xtick.color": SECONDARY, "ytick.color": SECONDARY})

fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
fig.subplots_adjust(left=0.09, right=0.9, top=0.88, bottom=0.12)
for label, (color, marker, values) in SETUPS.items():
    ax.plot([1, 2, 3], values, color=color, marker=marker, markersize=10, linewidth=2.5, label=label)
    ax.text(3.06, values[-1], f"{values[-1]:.3f}", va="center", fontsize=19, color=INK)

ax.set_xticks([1, 2, 3], ["Round 1", "Round 2", "Round 3"])
ax.set_xlim(0.85, 3.15)
ax.set_ylim(0.76, 0.835)
ax.set_ylabel("test mAP50", color=SECONDARY, labelpad=12)
ax.grid(axis="y", color=GRID, linewidth=1)
ax.set_axisbelow(True)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.tick_params(length=0, pad=10)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=19)

fig.savefig(Path(__file__).parent / "results-by-round-simple.png", facecolor="white")
print("saved")
