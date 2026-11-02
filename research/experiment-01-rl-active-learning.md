# Experiment 01: RL-driven label selection for shrub detection

*Draft, September 2026. Builds on option A in `rl-state-of-the-art.md`.*

## Research question

> On drone imagery of the target species, does a learned (RL) selection policy reach a given detection accuracy with **fewer labeled tiles** than random sampling and standard active-learning heuristics?

**Hypothesis:** the RL policy needs at least 20% fewer labeled tiles than random selection to reach 90% of the full-supervision mAP@50, and does at least as well as the best heuristic.

## Key design point: simulate the human with a pre-labeled pool

Active-learning experiments are run **offline**. You label a pool of tiles in advance and hide the labels. When a strategy "asks" for a tile, you reveal its label. This makes runs repeatable and fair across strategies. It does mean labeling the pool up front, so budget annotation time for it.

## Data

| Split | Content | Used for |
|---|---|---|
| Pool | Sites/blocks A + B, ~1,000–2,000 tiles, fully labeled but hidden | Tiles the strategies choose from |
| Validation | Site/block C, ~150 tiles | Reward signal for the RL policy, model selection |
| Test | Site/block D, ~200 tiles, **never touched until the end** | Final numbers in the thesis |

- Split by **geographic block or flight**, never by random tile. Leave a buffer strip between blocks so overlapping tiles don't leak across splits.
- Tiles: 640 px (fast) or 1024 px, 20% overlap, fixed GSD. Record flight date, altitude and GSD per tile.
- If your own data isn't ready yet, build and debug the pipeline on a public aerial vegetation dataset first (e.g. the DeepForest / NeonTreeEvaluation tree-crown data, or a public UAV weed dataset). Then switch to your footage.

## Detector (the "learner")

- **YOLO26-n or -s** as the model retrained inside the loop. It is small and fast, and each reward needs a retrain, so speed matters more than peak accuracy here.
- Retrain per round by **fine-tuning from the previous round's weights** for a fixed small number of epochs (e.g. 10–20). Fix the random seed per run.
- Optional final check: re-run the winning strategy's selected tiles with **RF-DETR** to show the result holds for a stronger detector.

## Active-learning protocol

- **Seed set:** 50 tiles, chosen randomly (same seed set for every strategy within a run). These can be bootstrapped with Grounded SAM 2 / DeepForest and hand-corrected.
- **Rounds:** T = 10 rounds × k = 25 tiles, for a final budget of 300 labeled tiles.
- **Repeats:** 5 seeds per strategy; report mean ± std.
- **Upper bound:** a detector trained on the whole pool (full supervision).

## Strategies compared

| # | Strategy | Type |
|---|---|---|
| S0 | Random | Baseline |
| S1 | Least confidence / entropy (aggregate per tile: mean of the top-N box uncertainties) | Uncertainty heuristic |
| S2 | Core-set (k-center greedy on backbone embeddings) | Diversity heuristic |
| S3 | Uncertainty + diversity (cluster embeddings into k groups, take the most uncertain tile per cluster) | Hybrid heuristic (strong baseline) |
| **S4** | **Bandit over S0–S3** (pick which heuristic to use each round) | **RL, level 1** |
| **S5** | **Learned tile-scoring policy** | **RL, level 2** |

### S4: contextual bandit (do this first)

Each round, the agent picks one of the heuristic strategies, uses it to select k tiles, and receives reward r = Δ mAP@50 on validation. Algorithm: EXP3 or a Thompson-sampling bandit, in the spirit of "Active Learning by Learning" (ALBL, Hsu & Lin 2015). It needs very few episodes, so it's cheap and will almost certainly finish. **Treat this as the guaranteed contribution.**

### S5: learned selection policy (the ambitious version)

- **Per-tile features:** mean/max detection confidence, entropy, number of predicted boxes, predicted plant cover (%), distance to the nearest labeled tile in embedding space, and cluster id.
- **Global state:** round t, budget left, current validation mAP.
- **Policy:** a small MLP that scores each candidate tile. Pick k tiles by sampling without replacement from softmax(scores) (Plackett-Luce), which keeps the log-probability tractable.
- **Episode:** one full AL run (T rounds). **Reward:** per-round Δ mAP@50, or the area under the learning curve at the end.
- **Update:** REINFORCE with a **group baseline**, GRPO-style. Run G episodes (e.g. G = 4) from the same seed set, and set advantage = episode return − group mean. This connects the thesis to the current RL state of the art.
- **Generalization test:** train the policy on pool A, then deploy it frozen on pool B. A policy that only works where it was trained is a weak result.

```text
for iteration in 1..N:
    for g in 1..G:                          # group of episodes
        labeled = seed_set; model = train(labeled)
        for t in 1..T:
            feats = features(model, pool \ labeled)
            batch, logp = policy.sample_k(feats, k)
            labeled += batch                # reveal hidden labels
            model = finetune(model, labeled)
            r_t = mAP(model, val) - mAP_prev
        R[g] = sum(r_t);  LOGP[g] = sum(logp)
    A = R - mean(R)                         # group-relative advantage
    loss = -sum(A * LOGP);  policy.step(loss)
```

## Compute budget: the main risk

One episode = T retrains. At about 3 min per fine-tune, that's ~30 min per episode, and G = 4 × 25 iterations would take ~50 GPU-hours. Ways to cut it:

1. **Cache backbone features** and retrain only the detection head inside the loop.
2. Use smaller tiles or input size, and fewer epochs per round.
3. Train the policy on a smaller sub-pool (e.g. 400 tiles, T = 6) and evaluate on the full setting.
4. If it's still too slow, S4 (bandit) plus a lighter S5 is still a complete thesis.

Measure the time of one real retrain in week 1, then fix the scale of S5 based on it.

## Metrics and figures

- **Main figure:** learning curve of test mAP@50 against the number of labeled tiles, one line per strategy, with a shaded std band.
- **Labels to target:** tiles needed to reach 90% of the full-supervision mAP@50.
- **ALC:** area under the learning curve (normalized).
- Also report: mAP@50-95, F1 at a fixed confidence, **count error per hectare** (what an ecologist actually uses), and wall-clock or GPU cost per strategy.
- Significance: paired comparison across seeds (e.g. Wilcoxon signed-rank on ALC).

## Threats to validity

- **Reward noise:** retraining variance can be larger than the differences between strategies. Fix seeds, and average Δ mAP over a couple of repeats if needed.
- **Validation overfitting:** the policy is tuned on site C, so only report the numbers from site D.
- **Class imbalance:** most tiles may contain no target plants. Report how many empty tiles each strategy picks.
- **Simulated annotator:** real annotation cost differs per tile (dense tiles take longer). Optionally weight the budget by box count.

## Rough timeline

| Weeks | Work |
|---|---|
| 1–2 | Tiling pipeline, spatial splits, pool labeling (bootstrapped with Grounded SAM 2 / DeepForest), time one retrain |
| 3 | AL loop + S0–S3 baselines, 5 seeds |
| 4 | S4 bandit |
| 5–7 | S5 policy, cross-pool generalization test |
| 8 | Final test-set runs, figures, RF-DETR check |
