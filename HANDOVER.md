# Handover: reproduce the experiment, then pick something up

This is a bachelor's thesis extending **CARLA**
([jmartinpizarro/carla](https://github.com/jmartinpizarro/carla)), a detector for
cardilla (*Eryngium horridum*) in DJI drone video. The original project trains once
on a fixed dataset. This one asks what to do as **new flights keep arriving**:
where each retrain should start from, and (later) which tiles are worth labelling.

You need two things from me that are not in this repo:

| | What | Size |
|---|---|---|
| **A** | three dataset zips | ~1.0 GB |
| **B** | the metric files from the runs so far — *already in the repo* under `experiments/warm_start/results/` | ~300 kB |

Nothing else. No server, no database.

---

## Step 1 — check you can read my results (2 minutes, no GPU)

```bash
git clone <this repo> && cd TFG
python -m venv .venv && . .venv/bin/activate
pip install ultralytics==8.4.14          # pin it; see "gotchas"
python experiments/warm_start/summarize.py --results experiments/warm_start/results/results
```

That prints the tables from `research/experiment-02-warm-start.md` out of committed
text files. If your numbers match mine, we agree on what has happened so far — and
that is the cheapest possible reproduction, because it skips training entirely.

## Step 2 — check the pipeline runs on your machine (~6 min on any GPU)

Put the three zips somewhere and unzip them, then:

```bash
cd experiments/warm_start
python make_rounds.py --low /path/to/carla-v1 \
    --high /path/to/high-altitude-tiles-1080p --out rounds
# expect: 704 / 1600 / 1864 / 2688 tiles

python run_experiment.py --rounds-dir rounds --out /tmp/quick_test \
    --seeds 42 --epochs 1 --fraction 0.1
```

`--fraction 0.1 --epochs 1` is a throwaway smoke test: one job per setup on 10% of
the tiles. It proves the rounds, the Shrink & Perturb step and the Ultralytics call
all work before you spend GPU hours.

## Step 3 — the real thing

The full run is **3 seeds × [shared round 1 + 3 setups × rounds 2–4] = 30 jobs,
~7 GPU hours** on a Colab T4. You almost certainly do not want to redo all of it,
because my results are in the repo already. What is worth your GPU is one of the
open variants in the next section.

If you do want to run it: `experiments/warm_start/warm_start_colab.ipynb` does the
whole thing on free Colab. It survives disconnects — each job's `results.json` is
its "finished" marker and finished jobs are skipped, so you just rerun the cell.

**Colab setup, the part that trips people up:** the notebook reads and writes
`/content/drive/MyDrive/TFG/`. A folder shared *with* you lands in "Shared with me",
which is not inside `MyDrive`, and you need **write** access anyway because results
go there. So copy the three zips plus the `.py` files and the notebook into **your
own** `MyDrive/TFG/`. Then the notebook works unmodified.

---

## Where you can actually help

The experiment is built so variants parallelise across people with no coordination.
`--tag` gives a run its own folder, and `--rounds 4` makes it continue from my
existing round 3 instead of retraining the chain. So each line below is an
independent job someone can run on their own GPU and send back as a ~100 kB zip.

Rounds are flights, cumulative: round 1 = 17 Nov 2025, round 2 = +5 Dec, round 3 =
+11 Dec, **round 4 = + a high-altitude flight**. Rounds 1–3 are the same field from
the same altitude, so there is barely any shift and the three setups land within
1 mAP50 point of each other. Round 4 is the first round where retention and
plasticity can pull apart: a model trained on the low flights scores mAP50 **0.011**
on the high flight before ever training on it.

**1. Epoch budget — the one that decides whether any of this is useful.**

```bash
python run_experiment.py --rounds-dir rounds --out results \
    --rounds 4 --setups warm sp --epochs 10 --tag ep10 --seeds 42 43 44
```

Equal-epoch comparisons can only ever conclude "just retrain from scratch". Warm
and Shrink & Perturb are only *useful* if they get there in fewer epochs. 6 jobs, ~1 h.

**2. How hard to shrink.** λ is the retention/plasticity dial (λ→1 is warm, λ→0 is
cold), and it only has something to trade off at a shift round.

```bash
python run_experiment.py --rounds-dir rounds --out results \
    --rounds 4 --setups sp --lam 0.3 --tag lam0.3 --seeds 42 43 44
python run_experiment.py --rounds-dir rounds --out results \
    --rounds 4 --setups sp --lam 0.9 --tag lam0.9 --seeds 42 43 44
```

6 jobs, ~3 h. Ash & Adams used λ = 0.6 in their headline figure but λ = 0.3 in the
domain-shift section, which is the section closest to our round 4.

**3. The reverse stream.** `make_rounds.py --order high-first` puts the high flight
first, which gives a second, independent shift event (low flights arriving after a
high one) out of the same data. Round 1 of it is also the only way to read round 4's
high score at all — without it there is no telling whether 0.011 → 0.55 is good.

```bash
python make_rounds.py --low ... --high ... --order high-first --out rounds_high_first
python run_experiment.py --rounds-dir rounds_high_first --out results_high_first \
    --rounds 1 --epochs 30 --seeds 42 43 44
```

**4. More seeds.** The seed-to-seed spread of one setup is ~0.004 mAP50, about the
size of the gap between setups. Seeds 45–47 on round 4 would help more than any
new variant. `--seeds 45 46 47 --rounds 4`.

**5. Not GPU work at all:** read `research/shrink-perturb-explained.md` §3 and tell
me whether the BatchNorm / effective-learning-rate argument holds up. It is flagged
in the file as my own reasoning with no citation, and it is the weakest link in the
write-up.

### Sending results back

```bash
python experiments/warm_start/export_results.py --src results --zip ~/my_results.zip
```

That takes only `results.json`, `eval_matrix.json`, `results.csv` and `args.yaml` —
a few hundred kB, no checkpoints. Jobs are keyed by seed/setup/tag/round, so two
people's zips merge by unzipping into the same folder. Nothing collides as long as
you picked a `--tag` nobody else used.

---

## Gotchas, in the order they will bite you

1. **Pin `ultralytics==8.4.14`.** Every number in the docs comes from that version,
   and it is also the version the original thesis used. A newer one changes defaults
   and the comparison stops being a comparison.
2. **`carla-v1-clean.zip` is not the raw Roboflow export.** 72 train tiles (6 frames
   that were uploaded twice with *conflicting* labels, 3 of them leaking across
   splits) were removed by hand. The reasoning is in `CHANGES.md` inside the zip. Use
   my zip, or your tile counts and every number will differ. Frame 9 is still
   duplicated inside `test/` on purpose, so the test set matches the original thesis.
3. **`rounds/` and `eval_sets/` hold absolute paths**, so they are not committed and
   are not portable. `make_rounds.py` and `evaluate_matrix.py` rebuild them in
   seconds. On Colab they must be rebuilt every session.
4. **The next round starts from `last.pt`, not `best.pt`.** `best.pt` is chosen with
   the validation set, which would smuggle validation information into the next
   round's starting point. `results.json` records both; the docs report `last`.
5. **The high-altitude tiles are generated**, not downloaded, by
   `experiments/high_altitude/tile_high_flight.py`: 4K → 1920×1080 with
   `cv2.INTER_AREA`, then 640 px tiles with 160 px overlap, replicating
   `carla/scripts/dataset/dataset_tiling.py` down to the ≤2 px fragment rule.
   Verified against the pre-existing test tiles: same 264 filenames, byte-identical
   labels.
6. **Round 4 confounds three changes at once** — altitude, season, and 4K→1080p
   resolution. It has to be described as "a new condition", not "altitude". The clean
   altitude comparison needs a same-day low/high pair, which is not labelled yet.
7. **Read the paired table, not the error bars.** Seed spread ≈ the inter-setup gap,
   so `paired.csv` subtracts two setups *within* the same seed to remove the noise
   they share. Three seeds pointing the same way is weak but real; two up and one
   down is nothing.

## What to read

1. `README.md` — repo layout and the commands.
2. `research/00-context.md` — the project in one page: goals, constraints, plan, open questions.
3. `research/experiment-02-warm-start.md` — this experiment, with results so far.
4. `research/shrink-perturb-explained.md` — the mechanism, from a full read of
   Ash & Adams, *On Warm-Starting Neural Network Training* (NeurIPS 2020), including
   where their setting differs from ours (no augmentation, no LR schedule,
   classification not detection).
5. `research/experiment-04-tailor-handoff.md` — the TAILOR bandit, the intended core
   contribution. Designed, not started. If you want a bigger piece than a variant,
   this is it.

The four papers behind these docs are PDFs under publisher copyright, so they are not
in the repo — `research/papers.md` has the titles and links, and I can send them.
