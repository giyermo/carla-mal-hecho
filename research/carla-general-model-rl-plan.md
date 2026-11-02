# CARLA: a general detector with RL in the loop, processed offline

*Plan, updated 18 September 2026. Combines `carla-rl-state-of-the-art.md` (learning from every flight) and `carla-generalization-state-of-the-art.md` (sun, cloud, altitude) into one system.*

> **Deployment assumption:** no real-time requirement. The drone records; after landing, the handler uploads the video to the group's servers, where it is processed as a batch job. The Jetson/TensorRT code in the repo belongs to the original thesis and is not a constraint here.
>
> Articles are linked inline where they are used and collected at the end. Most were checked during this research. The ones marked † are well-known papers added from background knowledge and not re-checked, so confirm the details before citing them. Figures come from abstracts and summaries; verify them against the full papers.

---

## 1. The idea in one paragraph

Today CARLA trains one YOLO model once on a fixed dataset. The proposal has three parts:
- A **general detector**, built on a foundation-model backbone, is the production model and carries the generalization goal (weather, altitude). Processing is offline, so speed is no longer a reason to keep YOLO.
- A **bandit (RL)** decides, for every uploaded flight, which few tiles a person should label.
- Optionally, **RL fine-tunes the detector** directly on the ecological output (coverage or count).

Every upload is one round of the loop. The detector also pseudo-labels the unlabeled footage, and because the whole video is available, detections of the same plant can be combined across frames.

```
 handler uploads video + .SRT ──→ server batch job
                                     │
                                     ▼
          frame sampling + colour/scale normalization (SRT: altitude, ISO, shutter)
                                     │
                                     ▼
          GENERAL DETECTOR (DINOv3-YOLO26 / RF-DETR), sliced inference over tiles
                                     │
                                     ▼
          multi-frame aggregation + georeferenced de-duplication
                                     │
            ┌────────────────────────┼──────────────────────────────┐
            ▼                        ▼                              ▼
     maps / counts /       RL #1: bandit picks k tiles     pseudo-labels for the
     coverage report       for a person to label           rest of the flight
                                     │
                                     ▼
          retrain detector ──→ reward: Δ worst-condition mAP / coverage error
                                     │
                                     └── RL #2 (optional): fine-tune on coverage/count reward
```

## 2. Choosing the "general model"

| Option | Example | How general | Where RL comes in | Training on 1 GPU? | Offline processing of a flight |
|---|---|---|---|---|---|
| **1. Foundation-backbone detector** (recommended) | YOLO26 with a DINOv3 backbone ([Deng & Lu, 2026](https://arxiv.org/abs/2603.00160)); [RF-DETR, ICLR 2026](https://github.com/roboflow/rf-detr) (DINOv2 backbone) | Robust features: the ViT backbone added +7.6 and +10.2 mAP50 on out-of-domain years vs. +2 in-domain ([Deng & Lu](https://arxiv.org/abs/2603.00160)) | The bandit chooses what to label; optional reward fine-tuning | Yes | Fast enough: about 35 ms per image for ViT-S on a desktop GPU ([Deng & Lu](https://arxiv.org/abs/2603.00160)) |
| **2. Open-vocabulary detector** | [Grounding DINO](https://arxiv.org/abs/2303.05499)†, [T-Rex2](https://arxiv.org/abs/2403.14610)† (already in the repo), [YOLO-World](https://arxiv.org/abs/2401.17270)† | Detects from a text or image prompt | Same as option 1 | Yes | Fine |
| **3. Vision-language model trained with RL** | Small Qwen-VL-style model fine-tuned with GRPO and an IoU reward ([Visual-RFT, 2025](https://arxiv.org/abs/2503.01785)) | Most general, and strong with few labels ([Visual-RFT](https://arxiv.org/abs/2503.01785)) | **RL trains the model itself** | Borderline: 2–3B model with LoRA, slow | Slow, but acceptable on sampled frames |

**Why option 1 is the main model:**
- **Evidence:** self-supervised ViT backbones (DINOv2/v3) generalized better than domain-adapted CNNs on weed detection from drones ([Schrag et al., 2026](https://arxiv.org/abs/2604.25316)).
- **Where the gain appears:** the DINOv3 backbone's gain appears mostly *out of domain* ([Deng & Lu, 2026](https://arxiv.org/abs/2603.00160)), which is exactly what CARLA needs.
- **Room to scale up:** with offline processing, a *larger* backbone (ViT-B instead of ViT-S) and larger input resolution are options, limited only by training memory.
- **Warning on open-vocabulary models:** T-Rex2's poor results in the repo (many false positives and misses) show that general "detect anything" models don't automatically handle a niche plant seen from above.

**Why option 3 is only a side experiment:**
- VLMs are weak at *dense, small* objects (dozens of 10–50 px cardillas per frame).
- They output boxes as text, which is slow.
- GRPO training ([DeepSeekMath, 2024](https://arxiv.org/abs/2402.03300)†) on one GPU is tight.
- Offline processing removes the speed objection at inference, but not the other two.
- It is still the most "RL-native" option; see the survey [Reinforcement Learning for Large Model: A Survey (2025)](https://arxiv.org/abs/2508.08189) and its [paper list](https://github.com/weijiawu/Awesome-Visual-Reinforcement-Learning).

## 3. What offline processing makes possible

These weren't options with a real-time Jetson. They are now.

- **Heavier, slower-but-better inference:**
  - Tiles with overlap at inference as well as in training. Today inference tiles don't overlap while training tiles do (`src/yolo/utils/tiling_utils.py:141` vs. `scripts/dataset/dataset_tiling.py:14`). Sliced inference as in [SAHI](https://arxiv.org/abs/2202.06934)† handles this.
  - Test-time augmentation (flips, scales) and small model ensembles.
- **Pre-processing per frame:** colour correction conditioned on camera exposure data ([Wang et al., LUF-net, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13452194/)). At about 0.4 s per 1080p frame it was too slow for live video but is fine in a batch job. Also resize each frame to a fixed ground resolution using its SRT altitude ([Velumani et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8404552/)).
- **Multi-frame aggregation:** each plant appears in many consecutive frames.
  - Track detections across frames ([ByteTrack](https://arxiv.org/abs/2110.06864)†), or match them by GPS position. The repo already converts pixels to GPS (`src/yolo/utils/unit_conversor.py`).
  - Then vote or average confidence per plant. This cuts one-frame false positives and misses, and gives **unique plant counts per area** instead of per-frame boxes.
- **Smarter frame sampling:** at 30 FPS consecutive frames are near-duplicates. Process at 1–2 frames per second, or sample by distance flown (from SRT GPS), instead of fixed 3-second steps.
- **A free uncertainty signal for RL:** plants that flicker in and out of detection across adjacent frames are exactly the cases the model finds hard. That becomes a new bandit arm.

## 4. The components

### 4.1 General detector
- Replace YOLO26's backbone with a DINOv3 ViT, as in [Deng & Lu (2026)](https://arxiv.org/abs/2603.00160), or fine-tune [RF-DETR](https://github.com/roboflow/rf-detr). Start from the official [DINOv3](https://arxiv.org/abs/2508.10104) weights. Adapting DINOv3 to plant images helped only out of domain, and it was expensive ([Deng & Lu](https://arxiv.org/abs/2603.00160)).
- Train on the human-labeled tiles. Use mixed-resolution training for altitude, following [Velumani et al. (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8404552/).
- Optional robustness add-on: an adversarial head that stops the features from encoding altitude or weather, using the SRT metadata as labels ([NDFT, Wu et al., ICCV 2019](https://arxiv.org/abs/1908.03856)).

### 4.2 RL #1: the bandit that picks what to label (core contribution)
- **Based on TAILOR** ([Zhang et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/1e77af93008ee6cd248a31723ce357d8-Paper-Conference.pdf); code: [github.com/jifanz/TAILOR](https://github.com/jifanz/TAILOR)). Each labeling strategy is an arm. For **every slot in the labeling batch**, Thompson sampling picks an arm, so one batch mixes strategies. The **reward comes from the labels of the picked examples themselves**, not from retraining, and rewards examples of classes that are still rare. TAILOR matched the best single strategy on 9 of 10 datasets in only 10–20 rounds. Accuracy-based rewards (ALBL) did no better than choosing strategies at random. Caveat: TAILOR was tested on image *classification* only, never on detection.
- **One round per uploaded flight.** Pick k tiles via TAILOR, label them, compute rewards immediately from the new labels, then retrain the detector.
- **Arms (labeling strategies):**
  1. Every 3 s (the original thesis's baseline)
  2. Random
  3. Where the detector is least sure
  4. Most different from the already-labeled tiles
  5. **Temporal inconsistency:** plants that flicker across adjacent frames (new, from multi-frame processing)
  6. Detector vs. T-Rex2 disagreement
  7. Under-covered altitude × light conditions
- **Reward, adapted to CARLA (my proposal):** CARLA has only one class, so TAILOR's class-balance reward has to be redefined. Give each labeled tile a small label vector known right after annotation:
  - contains cardilla;
  - the detector **missed** a plant;
  - the detector produced a **false positive**;
  - plus its altitude × light cell from the `.SRT` file.

  The reward weights these entries by how rare they still are in the labeled set. That is TAILOR's inverse-count weighting, and its "domain-specific" weights let you, for example, favour errors in under-covered conditions.
- **Why not Δ mAP as the reward:** it needs a retrain per reward and is noisy. TAILOR's results suggest performance-based rewards work poorly. Keep Δ worst-condition mAP50 as the **evaluation** metric. Comparing a label-based reward against a Δ mAP reward is itself a good thesis experiment.
- **Context (extension beyond TAILOR, which uses none):** altitude, ISO/shutter and date from the `.SRT` file could make it a contextual bandit, e.g. [CAAL (2026)](https://arxiv.org/abs/2606.07910).
- **Ambitious extension:** replace the bandit with a learned tile-scoring policy trained with policy gradient and an mAP reward ([MGRAL, ICASSP 2026](https://arxiv.org/abs/2601.15688)), selecting regions rather than whole images ([Casanova et al., ICLR 2020](https://arxiv.org/abs/2002.06583)).

### 4.3 Pseudo-labels on every flight
- The detector labels the unlabeled frames, and high-confidence, multi-frame-consistent detections are added to training: teacher–student self-training ([Mean Teacher, 2017](https://arxiv.org/abs/1703.01780)†; [Unbiased Teacher for detection, 2021](https://arxiv.org/abs/2102.09480)†). Multi-frame agreement (section 3) *may* be a better filter than a single confidence threshold. That's my hypothesis, and it has to beat the simple baseline below. In the closest test, agreement across augmented views of one image, gains were small but the most stable of the ten methods ([Zhang, Xu & Agrawal, 2026](https://arxiv.org/abs/2606.30795)).
- **Use the human labels directly; don't over-engineer the pseudo-labeling.** Adapting detectors to a new domain (including fog and new weather/scene conditions) works best when a small *random* set of labeled target images is simply added as a supervised loss *alongside* the usual pseudo-label self-training (RTSM). With 1–10% of target images labeled, this beat pure self-training by +1.7 to +18.3 AP50. Ten methods that instead used the same labels to *steer* the pseudo-labels (calibrating thresholds, recovering missed objects, balancing losses) gave inconsistent, method-dependent gains, and on average none beyond 1% labeled ([Zhang, Xu & Agrawal, 2026](https://arxiv.org/abs/2606.30795)). For CARLA this means:
  - Mix every round's labeled tiles straight into training together with the pseudo-labels.
  - Treat "random tiles + pseudo-labels" (RTSM) as the baseline that the bandit's chosen tiles, and any pseudo-label filtering trick, must beat.
  - Caveat: transformer detectors (DINO, DETA) on driving scenes; YOLO and aerial imagery were not tested.

### 4.4 Retraining each round
- Fine-tune rather than retrain from scratch, but watch the generalization cost of warm-starting. Shrink & Perturb is a cheap fix ([Ash & Adams, NeurIPS 2020](https://arxiv.org/abs/1910.08475)). A second bandit could choose between fine-tuning, Shrink & Perturb and training from scratch.

### 4.5 RL #2: reward fine-tuning of the detector (stretch goal)
- After supervised training, fine-tune the detector with REINFORCE on the metric ecologists use: coverage error or plant count. For detection this approach raised mAP from 40.2% to 52.3% in 20k steps ([Pinto et al., ICML 2023](https://arxiv.org/abs/2302.08242)). A group baseline as in GRPO ([DeepSeekMath](https://arxiv.org/abs/2402.03300)†) can reduce variance.
- **Side experiment (option 3):** fine-tune a small VLM with GRPO and IoU rewards on a few hundred tiles ([Visual-RFT](https://arxiv.org/abs/2503.01785)). Compare it with the main detector on the cloudy and high-altitude test splits.

### 4.6 Is YOLO still needed?
Not for production. Keep YOLO-nano as a **baseline** (the original thesis's model), and as a **cheap proxy** inside the RL loop if retraining the ViT detector every round is too slow on one GPU. Distilling the general detector into it ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)†) only matters if server throughput becomes a problem.

## 5. One round, concretely

1. The handler uploads a flight: 40 m, overcast, 12 minutes of video, plus its `.SRT` file.
2. The server samples frames, corrects colour and scale using the SRT data, runs the detector with overlapping tiles, and aggregates detections per plant across frames.
3. A report is produced: map, unique count and coverage per area.
4. The bandit sees high altitude, high ISO (overcast), and many plants flickering between frames. It picks the *temporal inconsistency* arm, which has paid off in similar conditions before.
5. That strategy selects 25 tiles, which you label in Roboflow. Consistent high-confidence detections from the rest of the flight become pseudo-labels.
6. The detector is retrained. Worst-condition mAP50 on the validation flights rises 1.8 points, which becomes the bandit's reward.
7. Next upload, repeat.

## 6. Build order

1. **Parse SRT telemetry** and build test sets split by condition (altitude × sun/overcast, held-out flights).
2. **Offline inference pipeline:** frame sampling, overlapping tiles, multi-frame aggregation and GPS de-duplication. Measure how much aggregation improves the current YOLO before changing the model.
3. **Simulate:** fully label one or two existing flights and hide the labels. A strategy "asks" for a tile and its label is revealed, which gives repeatable comparisons.
4. **Baselines:** the current YOLO-nano vs. the general detector, on the out-of-condition splits.
5. **Time one retrain** of the general detector (and of YOLO-nano as a proxy). That sets how many rounds are affordable.
6. **Single-strategy loops** (arms 1–5), then **the bandit** on top.
7. **Pseudo-labels:** with vs. without, filtered by multi-frame agreement.
8. **Go live** on real uploads.
9. **Stretch:** RL #2 reward fine-tuning, or the VLM side experiment.

## 7. What the thesis shows

| Comparison | Question answered |
|---|---|
| YOLO-nano vs. general detector, on out-of-condition splits | Does a general model generalize across sun, cloud and altitude? |
| Single-frame vs. multi-frame aggregated results | How much does using the whole video help counts and coverage? |
| Bandit vs. each single labeling strategy (labels needed to reach a target mAP) | Does RL cut labeling effort per flight? |
| With vs. without pseudo-labels | Does unlabeled footage help, or does simple labeling win? |
| Accuracy vs. server processing time per flight | Is it practical for routine uploads? |

**One-line contribution:** *a plant-mapping pipeline that keeps improving from routine drone uploads with minimal labeling, and doesn't break when weather or altitude change.*

## 8. Open questions

1. **Server GPU(s):** is training done on the same single GPU as processing? How much memory? About 12 GB rules out the VLM side experiment; 24 GB or more makes it possible with LoRA.
2. How many frames or tiles are labeled now, and how many uploaded flights and dates exist?
3. Which drone model? It decides whether full LUF-net colour correction applies (it needs a light sensor) or only the exposure-only version.
4. Does the handler upload the `.SRT` file along with the video? The pipeline depends on it.
5. What should the final output be for the ecologists: per-flight maps, counts per hectare, coverage per plot?

---

## Articles

**General / foundation models**
- Deng & Lu, *DINOv3 Meets YOLO26 for Weed Detection in Vegetable Crops* (2026) — https://arxiv.org/abs/2603.00160
- Siméoni et al., *DINOv3* (2025) — https://arxiv.org/abs/2508.10104
- *RF-DETR* (ICLR 2026) — https://github.com/roboflow/rf-detr
- Schrag et al., *Towards Robust Rumex Obtusifolius Detection from Drone Images* (2026) — https://arxiv.org/abs/2604.25316
- Liu et al., *Grounding DINO* (2023)† — https://arxiv.org/abs/2303.05499
- Jiang et al., *T-Rex2* (2024)† — https://arxiv.org/abs/2403.14610
- Cheng et al., *YOLO-World* (2024)† — https://arxiv.org/abs/2401.17270

**RL for learning what to label**
- Zhang et al., *TAILOR: Algorithm Selection for Deep Active Learning with Imbalanced Datasets* (NeurIPS 2023) — https://proceedings.neurips.cc/paper_files/paper/2023/file/1e77af93008ee6cd248a31723ce357d8-Paper-Conference.pdf
- Liang et al., *MGRAL: Performance-guided Reinforced Active Learning for Object Detection* (ICASSP 2026) — https://arxiv.org/abs/2601.15688
- Casanova et al., *Reinforced Active Learning for Image Segmentation* (ICLR 2020) — https://arxiv.org/abs/2002.06583

**RL for training the model itself**
- Pinto et al., *Tuning Computer Vision Models with Task Rewards* (ICML 2023) — https://arxiv.org/abs/2302.08242
- Liu et al., *Visual-RFT: Visual Reinforcement Fine-Tuning* (2025) — https://arxiv.org/abs/2503.01785
- Shao et al., *DeepSeekMath* (introduces GRPO, 2024)† — https://arxiv.org/abs/2402.03300
- Wu et al., *Reinforcement Learning for Large Model: A Survey* (2025) — https://arxiv.org/abs/2508.08189 · paper list: https://github.com/weijiawu/Awesome-Visual-Reinforcement-Learning

**Offline video processing**
- Akyon et al., *SAHI: Slicing Aided Hyper Inference for Small Object Detection* (2022)† — https://arxiv.org/abs/2202.06934
- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* (2022)† — https://arxiv.org/abs/2110.06864

**Pseudo-labels and retraining**
- Tarvainen & Valpola, *Mean Teacher* (2017)† — https://arxiv.org/abs/1703.01780
- Liu et al., *Unbiased Teacher for Semi-Supervised Object Detection* (2021)† — https://arxiv.org/abs/2102.09480
- Zhang, Xu & Agrawal, *Simple Supervision Is Hard to Beat: A Bitter Lesson from Sparse Target Labels in Domain-Adaptive Object Detection* (2026) — https://arxiv.org/abs/2606.30795
- Hinton et al., *Distilling the Knowledge in a Neural Network* (2015)† — https://arxiv.org/abs/1503.02531
- Ash & Adams, *On Warm-Starting Neural Network Training* (NeurIPS 2020) — https://arxiv.org/abs/1910.08475

**Weather and altitude**
- Velumani et al., *Maize plant density from UAV RGB: impact of spatial resolution* (2021) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8404552/
- Wang et al., *LUF-net: colour calibration for UAV RGB images* (Plant Phenomics, 2026) — https://pmc.ncbi.nlm.nih.gov/articles/PMC13452194/
- Wu et al., *NDFT: Robust Object Detection from UAVs via Nuisance Disentanglement* (ICCV 2019) — https://arxiv.org/abs/1908.03856
