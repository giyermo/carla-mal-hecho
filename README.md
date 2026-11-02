# TFG — learning from new drone flights (CARLA, UC3M)

Bachelor's thesis extending **CARLA** ([jmartinpizarro/carla](https://github.com/jmartinpizarro/carla)),
a detector for **cardilla** (*Eryngium horridum*) in DJI drone video. The original
project trains once on a fixed dataset; this one asks what to do as **new flights
keep arriving**: where each retrain should start from, and which tiles are worth a
person's time to label.

Everything here is research code: scripts meant to be read and rerun, not a library.

## What is in here

| Path | |
|---|---|
| `HANDOVER.md` | **if someone handed you this repo, read that first**: the three places the project lives and what is in each |
| `research/` | the thinking: state of the art, the plan, one handoff doc per experiment. **Start at `research/00-context.md`** |
| `experiments/warm_start/` | experiment 02: warm start vs. cold restart vs. Shrink & Perturb |
| `experiments/high_altitude/` | tiling and evaluating the high-altitude flight |
| `presentation/` | Typst handouts and the figures for them |

### The documents worth reading first
- `research/00-context.md` — the project in one page: goals, constraints, plan, open questions.
- `research/experiment-02-warm-start.md` — the experiment that is actually running, with results so far.
- `research/shrink-perturb-explained.md` — how Shrink & Perturb works, from a full read of Ash & Adams (NeurIPS 2020), and where their setting differs from ours.
- `research/experiment-04-tailor-handoff.md` — the TAILOR bandit, the core RL contribution. Designed, not started.

## What is deliberately **not** in here

| Missing | Why / how to get it |
|---|---|
| `data/` (~3 GB of tiles) | too big for git. Roboflow `carla-4gysv/carla-low-flight-tiling-rgb-fudwn` v1 for the low flights; the zips also sit in `MyDrive/TFG/` |
| `carla/` | the previous student's repo, cloned separately: `git clone https://github.com/jmartinpizarro/carla` next to this one |
| `papers/` (4 PDFs) | publisher copyright. `research/papers.md` has the titles and links |
| `*.pt` checkpoints | ~5 MB each, rewritten every run. Only the metrics are versioned (see below) |
| `experiments/*/rounds/`, `eval_sets/` | generated files holding **absolute** paths, so they are only valid on the machine that made them. `make_rounds.py` and `evaluate_matrix.py` rebuild them in seconds |

## Experiment 02 in one paragraph

Every new flight means a retrain. Rounds are flights, arriving in order and keeping
the archive. Three setups: `cold` restarts from the COCO weights every round, `warm`
continues from last round's `last.pt`, `sp` applies Shrink & Perturb to it
(θ ← 0.6·θ + 0.01·θ_random). Rounds 1–3 are three low flights over the same field,
so there is almost no distribution shift and the three setups land within 1 mAP50
point of each other (test mAP50 at round 3: sp 0.828, cold 0.820, warm 0.815).
**Round 4 adds a high-altitude flight**, a condition the model scores mAP50 0.011 on
before ever training on it — that is the round where retention and plasticity can
actually pull apart.

### How to rerun it

The whole thing runs in `experiments/warm_start/warm_start_colab.ipynb` on a free
Colab T4 (~7 GPU hours for the main run). Outside Colab, the same four steps:

```bash
cd experiments/warm_start
python make_rounds.py --low /path/to/carla-v1 --high /path/to/high-altitude-tiles-1080p --out rounds
python run_experiment.py --rounds-dir rounds --out results --seeds 42 43 44 --epochs 30
python evaluate_matrix.py --results results --low ... --high-1080p ... --eval-sets eval_sets
python summarize.py --results results
```

`run_experiment.py` gives every (seed, setup, round) job its own folder and treats
`results.json` as the "finished" marker, so it can be interrupted and restarted at
will — which is what makes a Colab session that disconnects survivable.

## Getting the results out of Google Drive

Training writes each job to `MyDrive/TFG/results/`, checkpoints included. Only the
metrics are versioned here: `results.json`, `eval_matrix.json`, `results.csv` and
`args.yaml`, a few hundred kB in total, under `experiments/warm_start/results/`.

In the notebook (last two cells), after a run:

```python
!python {TFG}/export_results.py --src {RESULTS} {RESULTS_HIGH_FIRST} \
    --zip {TFG}/results_export.zip
```

Then download that single file from Drive and, in this repo:

```bash
python experiments/warm_start/export_results.py \
    --unzip ~/Downloads/results_export.zip --dest experiments/warm_start/results
git add experiments/warm_start/results && git commit -m "results: round 4"
```

`summarize.py --results experiments/warm_start/results/results` then reproduces every
table in the docs from what is committed, with no GPU and no checkpoints.

## Environment

Ultralytics **8.4.14** throughout (the version the original thesis used), YOLO26n,
640 px tiles, batch 16, `box=5.0`, seeds 42/43/44, `deterministic=True`. Colab ran
torch 2.11.0+cu130 on a Tesla T4; the laptop venv is torch 2.13.0 CPU and is only
good for debugging (3.5 min/epoch).

```bash
python -m venv .venv && . .venv/bin/activate
pip install ultralytics==8.4.14
```

## Status

| Experiment | |
|---|---|
| 01 — generic active-learning design | superseded by 04 |
| 02 — warm start vs. Shrink & Perturb | rounds 1–3 done, **round 4 + variants running** |
| 03 — LUF-net colour correction | blocked: needs the DJI `.SRT` telemetry, which we do not have |
| 04 — TAILOR bandit (core RL contribution) | designed, not started |
