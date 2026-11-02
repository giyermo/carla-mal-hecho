"""Slide diagram: what Shrink & Perturb does to the weights (illustrative values)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, SHRINK, PERTURB, LIGHT = "#1E2B26", "#5E6963", "#A8481F", "#2F6B6E", "#C9C2B1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "text.color": INK})

weights = np.array([1.0, 0.55, 0.85, 0.35, 0.7])  # an example "trained" pattern
shrunk = 0.6 * weights
noise = np.array([0.05, -0.04, 0.03, 0.05, -0.03])  # enlarged so it is visible
x = np.arange(len(weights))

fig = plt.figure(figsize=(16, 9), dpi=120)
fig.patch.set_facecolor("white")

# Formula, with the two parts coloured like the panels below. Pieces are
# placed one after another using their measured width, so nothing overlaps.
pieces = [("new weights  =  ", INK, "normal", None), ("0.6 × old weights", SHRINK, "bold", "shrink"),
          ("   +   ", INK, "normal", None), ("0.01 × random weights", PERTURB, "bold", "perturb")]
renderer = fig.canvas.get_renderer()
texts = [fig.text(0, 0.9, t, fontsize=30, color=c, weight=w, va="center") for t, c, w, _ in pieces]
widths = [t.get_window_extent(renderer).width / fig.bbox.width for t in texts]
xpos = 0.5 - sum(widths) / 2
for t, wdt, (_, c, _, label) in zip(texts, widths, pieces):
    t.set_x(xpos)
    if label:
        fig.text(xpos + wdt / 2, 0.835, label, ha="center", fontsize=20, color=c)
    xpos += wdt

panels = [
    ("1. Trained model", "same pattern,\nvery confident", INK),
    ("2. Shrink  (× 0.6)", "same pattern,\nless confident", SHRINK),
    ("3. Perturb  (+ noise)", "tiny random nudges\n(enlarged here to be visible)", PERTURB),
]
left, width, gap = 0.06, 0.25, 0.065
for i, (title, caption, color) in enumerate(panels):
    ax = fig.add_axes([left + i * (width + gap), 0.40, width, 0.33])
    if i == 0:
        ax.bar(x, weights, color=INK, width=0.65)
    else:
        ax.bar(x, weights, color="none", edgecolor=LIGHT, linestyle="--", linewidth=1.5, width=0.65)
        ax.bar(x, shrunk, color=SHRINK, width=0.65)
    if i == 2:
        ax.bar(x, noise, bottom=shrunk, color=PERTURB, width=0.65)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.6, len(weights) - 0.4)
    ax.axis("off")
    ax.axhline(0, color=MUTED, linewidth=1.5)
    ax.set_title(title, fontsize=24, weight="bold", color=color, pad=14)
    ax.text(0.5, -0.1, caption, transform=ax.transAxes, ha="center", va="top", fontsize=19, color=MUTED)
    if i < 2:
        fig.text(left + (i + 1) * (width + gap) - gap / 2, 0.565, "→", ha="center", va="center",
                 fontsize=44, color=MUTED)


# Bottom: the effect on training (callback to the "why it hurts" slide).
fig.add_artist(plt.Line2D([0.06, 0.94], [0.255, 0.255], color=LIGHT, linewidth=1))
fig.text(0.06, 0.13, "Effect on\ntraining", fontsize=21, weight="bold", va="center")
for label, old, new, x0, note, ncolor in [
        ("warm start", 0.12, 1.0, 0.27, "old data barely\nshapes the model", SHRINK),
        ("after shrink & perturb", 0.8, 0.95, 0.68, "old and new data\nboth count again", PERTURB)]:
    ax = fig.add_axes([x0, 0.05, 0.13, 0.13])
    ax.barh([1, 0], [old, new], color=[MUTED, INK], height=0.62)
    ax.set_xlim(0, 1.05)
    ax.set_yticks([1, 0], ["old data", "new data"], fontsize=15)
    ax.tick_params(length=0)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0, 1.2, f"{label}: gradient size", transform=ax.transAxes, fontsize=16, color=MUTED)
    fig.text(x0 + 0.145, 0.115, note, fontsize=15, color=ncolor, va="center")
fig.text(0.565, 0.115, "→", ha="center", va="center", fontsize=40, color=MUTED)

out = Path(__file__).parent
fig.savefig(out / "shrink-perturb.png", facecolor="white")
fig.savefig(out / "shrink-perturb.svg", facecolor="white")
print("saved", out / "shrink-perturb.png")
