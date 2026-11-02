# Experiment 04: TAILOR bandit for choosing which tiles to label (handoff, 18 Oct 2026)

*Paste this together with `00-context.md` and `experiment-02-warm-start.md` to start a new chat. Nothing has been built yet for this experiment. This is the **core RL contribution** of the thesis.*

## Question
When a person can only label a few tiles per round, does a TAILOR-style bandit, which learns *which labeling strategy to trust*, reach a given test mAP50 with **fewer labeled tiles** than random selection and than each single strategy?

## The paper (`papers/NeurIPS-2023-algorithm-selection-for-deep-active-learning-with-imbalanced-datasets-Paper-Conference.pdf`)
Zhang, Shao, Verma & Nowak, *Algorithm Selection for Deep Active Learning with Imbalanced Datasets* (TAILOR), NeurIPS 2023. Code: https://github.com/jifanz/TAILOR

Verified in the PDF (sections 3–4, 6.1):
- **Framework (Algorithm 1):** M candidate active-learning algorithms (the "arms"). Each round t, a batch of B slots; for each slot the meta-algorithm picks an algorithm, which selects one not-yet-selected example. The batch is labeled, rewards are observed **from the new labels only**, the model is retrained on everything labeled so far.
- **TAILOR (Algorithm 2):** Thompson sampling. Each arm i keeps a Beta posterior per label entry (multi-label case), starting from a uniform prior a_i = b_i = 1. For each slot: sample θ̂_i ~ Beta(a_i, b_i) for every arm, choose the arm maximizing ⟨v_t, θ̂_i⟩. After labeling: a_i += sum of the label vectors y from arm i's picks, b_i += sum of (1 − y).
- **Reward:** r = ⟨v_t, y⟩. "Class diversity" weights each class by 1/max(1, count so far), so examples of still-rare classes pay more. A "domain-specific" weight vector v_t, set by the user and allowed to change over time, is also allowed.
- **Non-stationarity trick:** discount old observations, a_i ← γ·a_i + …, b_i ← γ·b_i + …, with **γ = 0.9** in all experiments.
- **Setup:** 10 datasets (CIFAR, SVHN, ImageNet, CelebA, COCO, VOC, …), batch sizes 500–10,000, **10 or 20 rounds**. Baselines: each single strategy, **Random Meta** (pick arms uniformly at random) and **ALBL** (bandit rewarded by accuracy).
- **Results (also in `papers.md`):** as good as the best single strategy on 9 of 10 datasets; ALBL did no better than random arm choice.
- **Caveat:** classification only, never object detection. Adapting it to detection is part of the contribution.

## Adapting it to CARLA (my proposal, from `carla-general-model-rl-plan.md` §4.2, adjusted to the data I actually have)
- **Pool:** the 1864 train tiles (233 frames) of `data/carla-v1/`, labels hidden and revealed only when a tile is picked (simulated annotator). Valid and test unchanged (704 / 696 tiles).
- **Label vector per tile** (TAILOR's multi-label case, K entries in {0,1}, known as soon as the tile is labeled):
  1. contains cardilla;
  2. the current detector **missed** a plant on it (false negative vs. the revealed labels);
  3. the current detector produced a **false positive** on it;
  4. flight day one-hot (17 Nov / 5 Dec / 11 Dec), standing in for the altitude × light cell, which I can't get without the `.SRT` files.
- **Reward:** TAILOR's inverse-count weighting over those entries. Entries 2–3 need the detector's predictions on the picked tiles, which are computed anyway for the uncertainty arm.
- **Arms** (from the plan, minus the ones the missing videos rule out):
  1. Random
  2. Least confident (aggregate per tile, e.g. mean of the most uncertain box scores)
  3. Most different from the already-labeled tiles (core-set / k-center on backbone features)
  4. Under-covered flight day
  5. Optional: detector vs. T-Rex2 disagreement (`carla/metrics/trex/trex_predictions_v2.json` may already hold T-Rex2 predictions; check which images it covers)
  - Dropped: **temporal inconsistency** (frames are 3 s apart, no videos) and **"every 3 s"** (the pool already *is* every-3-s frames).
- **Training each round:** YOLO26n, retrained on all labeled tiles so far. Use the warm-start experiment's answer: Shrink & Perturb from the previous round (or cold, if seed 44 changes the picture). Then the RTSM idea (pseudo-labels on the unlabeled tiles + the real labels) can be added later as a second step.
- **Evaluation:** test mAP50 vs. number of labeled tiles (learning curve), labels needed to reach e.g. 95% of the full-data mAP50 (≈ 0.82 from experiment 02), area under the curve. Also mAP50 per flight day. 3 seeds, same seed set for all methods.
- **Methods compared:** each single arm, Random Meta, TAILOR. Optional: ALBL-style bandit with Δ mAP reward (the plan suggests comparing label-based vs. accuracy-based rewards).

## Facts about the data that matter here
- Boxes per train tile: mean 8.5, median 4, max 58; **318 of 1864 train tiles are empty** (no cardilla). Cardilla is common, so "contains cardilla" alone gives the bandit little to learn from; the error entries (missed / false positive) are probably the informative ones.
- Labeling cost differs a lot per tile (0 to 58 boxes). Budget by tiles is the TAILOR way; also report boxes labeled, since that is closer to a person's real effort.
- Tiles of the same frame overlap by 160 px, so two tiles from one frame are correlated. Worth deciding whether an arm may pick several tiles from the same frame in one batch.
- The labels themselves have some inconsistency (duplicate frames had different box counts; see `data/carla-v1/CHANGES.md`).

## Compute (my estimate, to be measured)
- Each round needs a retrain. On a Colab T4, 30 epochs on 704 tiles took ~10.5 min (experiment 02). With, say, 10 rounds of 50 tiles after a 50-tile seed set, one run might be roughly 1–2.5 h; × (4–5 arms + Random Meta + TAILOR) × 3 seeds = **tens of GPU hours**. Reduce with fewer epochs per round (warm/SP start helps here), fewer rounds, or a smaller pool.
- **No GPU right now:** the free Colab quota ran out. Plan: build and debug the whole loop on the laptop CPU at tiny scale (few tiles, 1–2 epochs, like experiment 02's quick test), then rent a GPU (RunPod / Vast.ai, Kaggle's free weekly quota, or ask the supervisor / UC3M) only for the real runs.

## Proposed plan
1. Read sections 3–4 and 6.1 of the paper; read the official code's bandit part (https://github.com/jifanz/TAILOR).
2. Design the simulation: seed set size, batch size B, number of rounds T, epochs per round. Write down why.
3. Write the loop on CPU with **only the random arm** first (pool, hidden labels, reveal, retrain, evaluate, resume-on-disconnect like `run_experiment.py`).
4. Add the other arms one by one, each tested alone.
5. Add Random Meta and TAILOR (Thompson sampling with Beta posteriors, γ = 0.9, the label-vector reward).
6. Tiny CPU run end to end; then rent a GPU for the real runs.

## Reusable pieces from experiment 02 (`experiments/warm_start/`)
- `shrink_perturb.py` (retraining between rounds), the job-folder + `results.json` resume pattern in `run_experiment.py`, `summarize.py`, the Colab notebook layout, `make_rounds.py`'s filename parsing (`get_date`, `get_frame`).
- Laptop venv: `experiments/warm_start/.venv` (torch 2.13.0 CPU, ultralytics 8.4.14). Put the new code in `experiments/tailor/`.

## Working style
I want to understand everything to defend the thesis: explanations first, the "why" behind each decision. I sometimes ask Claude to write the code; then I need to be able to explain every line.
