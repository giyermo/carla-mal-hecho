# RL for CARLA: learning from every new flight

*Research brief, updated 18 September 2026. Scope: how reinforcement learning could make [CARLA](https://github.com/jmartinpizarro/carla) (Cardilla Recognition and Localization Analyzer, *Eryngium horridum* detection in DJI video) keep improving as new flights arrive. Companion: `carla-generalization-state-of-the-art.md` (weather and altitude).*

> Every claim links to its source where it is made. The brief uses only the five most relevant sources, listed with the reason for each at the end. Figures come from the papers' abstracts and summaries, so check them against the full text before citing them in the thesis. Anything without a link is my own reasoning or comes from the CARLA repo.

**Assumptions from your answers:** the goal is learning from new footage; flights are manual only; new footage arrives regularly; one desktop GPU; the TFG continues a previous student's project, so the contribution must be clearly new.

---

## 0. Where CARLA is today

| Area | Current state (from the repo) |
|---|---|
| Data | DJI video, 1920×1080. One frame extracted every 3 s (`src/preprocessing/process_video.py:18`), labeled in Roboflow. Training flights are from 15 May 2025; the Jetson demo video is from 13 Feb 2026 |
| Models | YOLOv8n / YOLO11n / YOLO26n, trained from pretrained weights for 200 epochs × 5 seeds (`src/yolo/config.yaml`) |
| Tiling | 640 px training tiles with 160 px overlap (`scripts/dataset/dataset_tiling.py:14`) |
| Results (test mean) | mAP50 0.81–0.85, mAP50-95 0.42–0.45; tiled does not beat non-tiled |
| Baseline | T-Rex2 visual-prompt API (many false positives and misses) |
| Outputs | Coverage %, pixel→GPS conversion, density heatmaps, TensorRT on Jetson Orin Nano |

**What is new for your TFG:** the original thesis is a *static* comparison: one dataset, trained once. Your contribution would be a *closed loop across flights*. Each new flight triggers a decision about what to label and how to update the model, and RL learns that decision.

## 1. Framing it as RL

RL does **not** replace the YOLO detector; that stays supervised. RL sits one level up, in the data and training loop:

- **Time step:** a new flight (or a batch of footage) arrives.
- **Decision:** which frames or tiles to send to a human, and how to update the model.
- **Reward:** detector improvement on a held-out flight (Δ mAP50, and Δ coverage error, the ecological output) per unit of labeling effort.

This framing already exists in the literature. The labeling decision can be learned as a policy, whether data arrives as a stream ([Fang et al., EMNLP 2017](https://arxiv.org/abs/1708.02383)) or sits in a pool of candidates ([Casanova et al., ICLR 2020](https://arxiv.org/abs/2002.06583); [MGRAL, ICASSP 2026](https://arxiv.org/abs/2601.15688)). How to choose the labeling strategy can also be learned as a bandit ([TAILOR, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/1e77af93008ee6cd248a31723ce357d8-Paper-Conference.pdf)).

## 2. State of the art

### 2.1 Learning when to label as data streams in
Fang et al. model active learning as a decision made for each new item as it arrives: label it or skip it. A deep Q-network learns this from a state made of the current model and the incoming instance ([Fang et al., EMNLP 2017](https://arxiv.org/abs/1708.02383)). This is the origin of RL-learned active learning, and the streaming setting matches CARLA, where footage arrives flight after flight rather than as one fixed pool.

### 2.2 Learning which image regions to label
Casanova et al. train a DQN that selects small image **regions** to label instead of whole images, using the segmentation model's own predictions and uncertainties as the state. On Cityscapes it needed about 30% less extra labeled data than the best baseline to reach the same performance. It also asked for more labels of under-represented classes, which reduces class imbalance ([Casanova et al., ICLR 2020](https://arxiv.org/abs/2002.06583)). **Direct analogue for CARLA:** frames are already tiled (`scripts/dataset/dataset_tiling.py`), so tiles can be the labeling unit. Imbalance also matters here, since many tiles contain no cardilla.

### 2.3 RL active learning built for object detection
MGRAL is the closest method to CARLA's problem. An LSTM policy, trained with policy gradient, selects batches of images for a detector, and the **reward is the resulting mAP improvement**. To avoid retraining the detector for every reward, it estimates the mAP change with fast unsupervised look-up tables. It was evaluated on PASCAL VOC and COCO ([MGRAL, Liang et al., ICASSP 2026](https://arxiv.org/abs/2601.15688)).

**Gap you can fill:** this, like the two methods above, was validated on standard benchmarks with heavy models. There is little evidence for small single-class detectors (YOLO nano), aerial video, or data that arrives flight by flight.

### 2.4 Bandits over labeling strategies: the cheapest form of RL
Instead of learning a full policy, TAILOR treats each existing active-learning strategy as an arm of a multi-armed bandit. For each slot in a labeling batch, Thompson sampling picks an arm. The reward is computed **from the labels of the picked examples**: examples of still-rare classes score higher. No retraining is needed to get a reward. TAILOR beats earlier bandit meta-methods (ALBL, whose accuracy-based reward did no better than random), and matches or beats the best single strategy on 9 of 10 datasets within 10–20 rounds ([TAILOR, Zhang et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/1e77af93008ee6cd248a31723ce357d8-Paper-Conference.pdf)). **Fit for CARLA:**
- One round per flight, and rewards come straight from the new labels, so the bandit itself adds no retraining cost.
- Its imbalance-aware reward suits tiles that are mostly empty.
- **But** it was tested on classification only. For single-class detection the reward must be redefined, e.g. rewarding tiles that reveal detector misses or false positives in rare conditions (see `carla-general-model-rl-plan.md`, section 4.2).

### 2.5 Updating the model each round
A loop that retrains after every flight needs to fine-tune from the previous model, since 200 epochs × 5 seeds from scratch is too slow. But fine-tuning on a growing dataset ("warm-starting") generalizes **worse** than retraining from scratch. **Shrink & Perturb** closes the gap: scale the weights down and add a little noise before continuing ([Ash & Adams, NeurIPS 2020](https://arxiv.org/abs/1910.08475)). The update rule can itself be a bandit arm: warm fine-tune, Shrink & Perturb, or from scratch.

## 3. Candidate RL contributions for CARLA

| | Contribution | RL decision | State / context | Reward | Based on | 1-GPU cost |
|---|---|---|---|---|---|---|
| **A** ★ | Bandit choosing the labeling strategy per flight | Which strategy picks this flight's k tiles | Flight date, altitude and lighting from SRT telemetry; mean detector confidence; predicted empty-tile fraction | Label-based, TAILOR-style: rarity-weighted count of detector misses, false positives and cardilla-positive tiles in the new labels. Δ mAP50 is used for evaluation, not as the reward | [TAILOR](https://proceedings.neurips.cc/paper_files/paper/2023/file/1e77af93008ee6cd248a31723ce357d8-Paper-Conference.pdf) | Low: one retrain per round |
| **B** | Learned tile-selection policy | Score every candidate tile, sample top-k | Per-tile uncertainty, embedding novelty, time distance to labeled frames | Δ mAP, MGRAL-style | [Casanova et al.](https://arxiv.org/abs/2002.06583), [MGRAL](https://arxiv.org/abs/2601.15688) | High: many episodes |
| **C** | Stream triage: label, pseudo-label, or skip each tile as a flight arrives | Per-tile decision | Confidence distribution, box count | Validation gain − labeling cost | [Fang et al.](https://arxiv.org/abs/1708.02383) | Medium |
| **D** | Bandit over the update rule | Warm fine-tune / Shrink & Perturb / from scratch, number of epochs | Round, data growth, validation trend | Validation gain per GPU-minute | [Ash & Adams](https://arxiv.org/abs/1910.08475) | Low |

**Suggested arms for A:**
1. Uniform in time (the current 3 s sampling, i.e. what the original thesis did)
2. Random
3. YOLO uncertainty (confidence entropy per tile)
4. Diversity (k-center on embeddings, skipping frames next to ones already labeled)
5. Uncertainty + diversity combined
6. *YOLO–T-Rex2 disagreement*, reusing the repo's T-Rex2 integration (watch API cost)

**Recommended thesis shape:** A as the guaranteed core, then D as an extension, then B if time allows. For C, first build the simple baseline: labeled tiles mixed directly into pseudo-label self-training. Learned triage of pseudo-labels is only worth keeping if it beats that ([Zhang, Xu & Agrawal, 2026](https://arxiv.org/abs/2606.30795) found label-guided pseudo-label tweaks rarely do).

**Not a fit, given your answers:** RL flight-path planning (flights are manual) and RL "where to zoom" for Jetson speed (not the goal).

## 4. Practical constraints on one GPU

- **Measure first:** time one warm-started fine-tune (e.g. yolo11n, tiled, 20–30 epochs). That number sets how many bandit rounds or policy episodes are affordable. MGRAL's look-up-table trick ([MGRAL](https://arxiv.org/abs/2601.15688)) is the fallback if retraining is too slow.
- **Reward noise:** the repo's 5-seed runs give a noise floor. If round-to-round Δ mAP50 is below the seed standard deviation, the reward is noise. Fix the seed, average over repeats, or use larger label batches.
- **Simulate before going live:** fully label one or two existing flights, hide the labels, and reveal them only when a strategy picks a tile. This gives repeatable comparisons. Then run the loop on genuinely new flights.
- **Splits:** leave-one-flight-out, and leave-one-date-out once seasons differ. Keep the spatial-leakage fix from commit `54f6d25`. Condition-based splits for weather and altitude are covered in the companion brief.
- **Roboflow exports:** augmented copies of a frame must not count as separate candidates, and must not end up in different splits.

## 5. Things noticed in the code that affect this plan (not RL)

- **Tiling mismatch:** inference tiles have no overlap (`src/yolo/utils/tiling_utils.py:141`), but training tiles overlap by 160 px (`scripts/dataset/dataset_tiling.py:14`). Plants cut at tile borders depend on the adjacency merge. This may partly explain why tiled doesn't beat non-tiled. Uncertainty scores come from inference, so check it before building the loop.
- **Coverage in inference logs:** `src/yolo/yolo_model.py:109` and `:144` append a coverage value *per box* while the mask is still being filled. "Mean Coverage" therefore averages partial masks and is weighted by box count. `coverage_utils.py`, used in testing, computes it correctly per image, so the reported test metrics are unaffected. If coverage enters the reward, compute it per image as in `coverage_utils.py`.

## 6. Open questions for the experiment design

1. How many frames/tiles are labeled now, and how many unlabeled flights (and dates) exist?
2. How long does one training run take on your GPU?
3. Labeling tool and time per frame (Roboflow?). This sets the realistic k per flight.
4. Does the thesis need the loop to run on the Jetson, or only offline on the GPU?

---

## The five sources

1. **[Liang et al., Performance-guided Reinforced Active Learning for Object Detection (MGRAL, ICASSP 2026)](https://arxiv.org/abs/2601.15688).** The closest method to CARLA's problem: an RL policy picks images for a *detector*, rewarded directly by mAP gain, with a trick to avoid retraining each time.
2. **[Zhang et al., Algorithm Selection for Deep Active Learning with Imbalanced Datasets (TAILOR, NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/1e77af93008ee6cd248a31723ce357d8-Paper-Conference.pdf).** The basis for the recommended core (option A): a cheap bandit over labeling strategies that handles class imbalance.
3. **[Casanova et al., Reinforced Active Learning for Image Segmentation (ICLR 2020)](https://arxiv.org/abs/2002.06583).** Labels regions rather than whole images, like CARLA's tiles. It shows a measured saving (~30% fewer labels) and better handling of rare classes.
4. **[Fang et al., Learning how to Active Learn: A Deep Reinforcement Learning Approach (EMNLP 2017)](https://arxiv.org/abs/1708.02383).** The origin of learned active learning, in the streaming setting that matches footage arriving flight by flight.
5. **[Ash & Adams, On Warm-Starting Neural Network Training (NeurIPS 2020)](https://arxiv.org/abs/1910.08475).** Any loop that retrains after every flight hits this problem. Shrink & Perturb is a cheap fix, and a natural bandit arm.
