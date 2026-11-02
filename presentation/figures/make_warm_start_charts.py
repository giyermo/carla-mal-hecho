"""
Slide charts (16:9, 1920x1080, white background) for the warm start experiment.

    epochs-choice.png   why 30 epochs per round (original thesis training curve)
    results-by-round.png test mAP50 per round, one line per setup
    round3-gap.png       round 3 close-up: mean and spread over seeds
    train-time.png       training time per round: all setups cost the same
    warm-start-results.csv  the numbers behind the charts (table view)

Colours: validated categorical slots 1-3 (blue, orange, aqua), all-pairs safe
for colour-vision deficiency; every series also has a direct label and its own
marker shape, so identity never depends on colour alone.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

OUT = Path(__file__).parent
ORIGINAL_CURVE = (Path.home() / "university/TFG/carla/metrics/yolo/t/average/model_yolo26"
                  / "yolo26_tiled_training_mean_results.csv")

# Text and chrome
INK, SECONDARY, MUTED = "#1E2B26", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

SETUPS = {  # name: (label, colour, marker)
    "cold": ("Cold (COCO weights)", "#2a78d6", "o"),
    "warm": ("Warm (last round's weights)", "#eb6834", "s"),
    "sp": ("Shrink & Perturb", "#1baf7a", "D"),
}
ROUNDS = [1, 2, 3]
ROUND_LABELS = ["Round 1\n17 Nov · 704 tiles", "Round 2\n+ 5 Dec · 1600 tiles", "Round 3\n+ 11 Dec · 1864 tiles"]

# Test mAP50 of last.pt, mean and std over seeds 42/43/44 (summarize.py output).
MAP50 = {
    "cold": [(0.768, 0.005), (0.796, 0.004), (0.821, 0.003)],
    "warm": [(0.768, 0.005), (0.793, 0.005), (0.815, 0.003)],
    "sp": [(0.768, 0.005), (0.799, 0.002), (0.829, 0.002)],
}
TRAIN_MIN = {"cold": [10.5, 18.8, 20.9], "warm": [10.5, 18.5, 21.0], "sp": [10.5, 18.2, 21.2]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 20, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": SECONDARY, "axes.linewidth": 1,
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
})


def new_figure(title: str, subtitle: str):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.06, 0.93, title, fontsize=30, weight="bold", va="center")
    fig.text(0.06, 0.875, subtitle, fontsize=19, color=SECONDARY, va="center")
    return fig


def style_axes(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=10)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("saved", OUT / name)


def epochs_choice():
    rows = list(csv.DictReader(ORIGINAL_CURVE.open()))
    epochs = [int(float(r["epoch"])) for r in rows]
    val = [float(r["metrics/mAP50(B)"]) for r in rows]
    best = max(val)

    fig = new_figure("Why 30 epochs per round",
                     "Validation mAP50 while training YOLO26n in the original thesis (mean of 5 seeds)")
    ax = fig.add_axes([0.08, 0.13, 0.86, 0.66])
    style_axes(ax)
    ax.plot(epochs, val, color="#2a78d6", linewidth=2.5, solid_capstyle="round")
    ax.axvline(30, color=AXIS, linewidth=1.2)
    v30 = val[epochs.index(30)]
    ax.plot([30], [v30], "o", markersize=12, color="#2a78d6", markeredgecolor=SURFACE, markeredgewidth=2.5)
    ax.annotate(f"epoch 30: {v30:.3f}\n≈ {v30 / best:.0%} of the best ({best:.3f})",
                (30, v30), xytext=(38, 0.70), fontsize=19, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))
    ax.text(100, 0.80, "after ~50 epochs it slowly gets worse (overfitting)", fontsize=17,
            color=SECONDARY, ha="center")
    ax.set_xlim(0, max(epochs) + 2)
    ax.set_ylim(0.55, 0.83)
    ax.set_xlabel("epoch", fontsize=18, labelpad=8)
    ax.set_ylabel("validation mAP50", fontsize=18, labelpad=10)
    save(fig, "epochs-choice.png")


def results_by_round():
    fig = new_figure("Shrink & Perturb ends best, warm start ends worst",
                     "Test mAP50 after each round · mean of 3 seeds, band = ± 1 std · same 30 epochs for all")
    ax = fig.add_axes([0.08, 0.17, 0.64, 0.62])
    style_axes(ax)
    for name, (label, color, marker) in SETUPS.items():
        mean = [m for m, _ in MAP50[name]]
        std = [s for _, s in MAP50[name]]
        ax.fill_between(ROUNDS, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                        color=color, alpha=0.10, linewidth=0)
        ax.plot(ROUNDS, mean, color=color, linewidth=2.5, marker=marker, markersize=11,
                markeredgecolor=SURFACE, markeredgewidth=2.5, label=label, solid_capstyle="round")
        # Direct label at the end of each line (text in ink, a coloured marker carries identity).
        ax.plot([3.13], [mean[-1]], marker=marker, markersize=10, color=color, clip_on=False)
        ax.text(3.2, mean[-1], f"{mean[-1]:.3f}  {label.split(' (')[0]}", va="center", fontsize=18, color=INK)
    ax.text(1.0, 0.7585, "round 1: one shared run for all three", fontsize=15, color=SECONDARY, va="center")
    ax.set_xticks(ROUNDS, ROUND_LABELS, fontsize=16)
    ax.set_xlim(0.9, 3.1)
    ax.set_ylim(0.755, 0.835)
    ax.set_ylabel("test mAP50", fontsize=18, labelpad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=16, handlelength=2.2)
    save(fig, "results-by-round.png")


def round3_gap():
    fig = new_figure("Round 3 close-up",
                     "Test mAP50 after round 3 · dot = mean of 3 seeds, whisker = ± 1 std")
    ax = fig.add_axes([0.32, 0.24, 0.62, 0.54])
    style_axes(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    order = ["warm", "cold", "sp"]
    for y, name in enumerate(order):
        label, color, marker = SETUPS[name]
        mean, std = MAP50[name][2]
        ax.plot([mean - std, mean + std], [y, y], color=color, linewidth=2.5, solid_capstyle="round")
        ax.plot([mean], [y], marker=marker, markersize=16, color=color, markeredgecolor=SURFACE, markeredgewidth=2.5)
        ax.text(mean, y + 0.24, f"{mean:.3f}", ha="center", fontsize=18, color=INK)
    ax.set_yticks(range(len(order)), [SETUPS[n][0] for n in order], fontsize=19)
    ax.tick_params(axis="y", colors=INK)
    ax.spines["bottom"].set_visible(True)
    ax.set_xlim(0.806, 0.836)
    ax.set_ylim(-0.6, 2.75)
    ax.set_xlabel("test mAP50 (axis zoomed in: 0.806 – 0.836)", fontsize=17, labelpad=10)
    fig.text(0.32, 0.06, "Shrink & Perturb: +0.014 vs. warm, +0.008 vs. cold   ·   cold vs. warm: +0.006",
             fontsize=18, color=SECONDARY)
    save(fig, "round3-gap.png")


def rounded_bar(ax, x, height, width, color, radius_px=6):
    """Column with rounded top corners (the data end) and a square base, as one shape."""
    fig = ax.figure
    fig.canvas.draw()  # make sure the data-to-pixel transform is final
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    rx = radius_px / (x1 - x0)  # corner radius in data units, so it is round on screen
    ry = radius_px / (y1 - y0)
    k = 0.5523  # Bezier constant for a quarter circle
    left, right, top = x - width / 2, x + width / 2, height
    verts = [(left, 0), (left, top - ry),
             (left, top - ry + k * ry), (left + rx - k * rx, top), (left + rx, top),
             (right - rx, top),
             (right - rx + k * rx, top), (right, top - ry + k * ry), (right, top - ry),
             (right, 0), (left, 0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none"))


def train_time():
    fig = new_figure("Same cost for all three",
                     "Training time per round on a Colab Tesla T4 (30 epochs) · mean of 3 seeds")
    ax = fig.add_axes([0.08, 0.17, 0.86, 0.6])
    style_axes(ax)
    width, step = 0.1, 0.14
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0, 25)
    for i, name in enumerate(SETUPS):
        for r, minutes in zip(ROUNDS, TRAIN_MIN[name]):
            rounded_bar(ax, r + (i - 1) * step, minutes, width, SETUPS[name][1])
    for r in ROUNDS:
        top = max(TRAIN_MIN[n][r - 1] for n in SETUPS)
        ax.text(r, top + 0.8, f"≈ {sum(TRAIN_MIN[n][r - 1] for n in SETUPS) / 3:.0f} min", ha="center",
                fontsize=19, color=INK)
    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for _, c, _ in SETUPS.values()]
    ax.legend(handles, [l for l, _, _ in SETUPS.values()], loc="upper left", frameon=False, fontsize=16)
    ax.set_xticks(ROUNDS, ROUND_LABELS, fontsize=16)
    ax.set_ylabel("minutes", fontsize=18, labelpad=10)
    fig.text(0.08, 0.06, "Round 1 is one shared run, so its three columns are identical.", fontsize=16,
             color=SECONDARY)
    save(fig, "train-time.png")


def table_view():
    with (OUT / "warm-start-results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setup", "round", "test_map50_mean", "test_map50_std", "train_minutes_mean"])
        for name in SETUPS:
            for r in ROUNDS:
                m, s = MAP50[name][r - 1]
                w.writerow([name, r, m, s, TRAIN_MIN[name][r - 1]])
    print("saved", OUT / "warm-start-results.csv")


if __name__ == "__main__":
    epochs_choice()
    results_by_round()
    round3_gap()
    train_time()
    table_view()
