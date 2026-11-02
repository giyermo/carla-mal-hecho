#import "style.typ": *

#show: doc.with(
  title: "Four papers for RL in CARLA",
  subtitle: "Bullet-point summary",
)

#note[Numbers come from the PDFs in `papers/`. Items marked _(my proposal)_ are ideas for CARLA, not results from the papers.]

= Why these four papers
- CARLA detects cardilla in DJI drone video; the previous thesis trained YOLO-nano once (mAP50 0.81–0.85).
- This TFG: after every uploaded flight, a person labels a few tiles and the detector is updated.
- The loop needs four answers, one per paper:
  - *Which tiles to label?* → TAILOR (the RL component).
  - *How to use the labels + unlabeled footage?* → Simple Supervision (RTSM).
  - *How to retrain every round?* → On Warm-Starting (Shrink & Perturb).
  - *How to compare sunny and cloudy flights?* → LUF-net (colour correction).
- Only TAILOR is RL (a multi-armed bandit). The other three make the loop work.

= 1. TAILOR — NeurIPS 2023
*Problem*
- Many active-learning strategies exist; which one is best changes per dataset and cannot be known in advance.
- Earlier online selectors used training-accuracy rewards, which are useless for deep networks (always near 100%).

*Method*
- Each labeling strategy is an arm of a multi-armed bandit.
- For each slot in a batch, Thompson sampling (Beta posteriors) picks an arm; that strategy picks one example.
- Reward comes from the new labels only, no retraining: $r = 1/K sum_i y_i \/ max(1, "COUNT"_t (i))$, so rare classes score more.
- Posteriors are discounted ($gamma = 0.9$) to follow rewards that change over time.

*Results (10 imbalanced classification datasets, 10–20 rounds of 500–10,000 labels)*
- Better than other meta-selectors on all datasets.
- On par with the best single strategy in accuracy on 9 of 10 (not Caltech256); better class diversity on 4 of 10; clearly better on CelebA.
- ALBL (accuracy-based reward) = random selection on all datasets.

*Limits:* classification only; large batches; non-stationarity left as future work.

*For CARLA*
- One round per flight; arms: every 3 s, random, least confident, most different, flickering plants, YOLO vs. T-Rex2 disagreement, under-covered altitude × light.
- _(my proposal)_ Describe each labeled tile by binary attributes (contains cardilla, missed plant, false positive, altitude × light cell) and use TAILOR's multi-label reward unchanged.
- Measure: labels needed to reach a target mAP50, bandit vs. each arm vs. random.

= 2. Simple Supervision Is Hard to Beat — arXiv 2026
*Problem*
- Source-free adaptation of a detector with Mean Teacher pseudo-labels; a small random subset of target images is labeled. How to use those labels?

*Method*
- RTSM: keep self-training unchanged and add the ordinary detection loss on the labeled images.
- Compare against 10 plugins that use labels to steer pseudo-labels (selection, object completion, optimization control).

*Results (Cityscapes → Foggy / BDD100K; DINO and DETA; 4 SFDA methods; 1–10% labels)*
- RTSM: +1.7 to +18.3 AP50 over pure self-training, in every setting (e.g. DDT 53.7 → 58.7 with 5% labels).
- No plugin consistently beats RTSM; the best one changes with the method.
- Worst case: multi-view hard recovery −13.6 AP50 (PETS, BDD100K).
- More labels → plugins matter less (average effect +0.10 at 1%, −0.86 at 5%).
- Most stable: soft multi-view foreground revival (+0.2 to +0.8).

*Limits:* preprint; transformer detectors on street scenes; labels chosen at random, not actively.

*For CARLA*
- Training recipe each round: pseudo-labels + new labeled tiles as plain examples.
- Baseline the bandit must beat: random tiles + RTSM.
- Multi-frame filtering is close to "multi-view" plugins → _(my proposal)_ use it as a soft signal, not as hard pseudo-labels.

= 3. On Warm-Starting — NeurIPS 2020
*Problem*
- Continuing training from the previous model is faster but generalizes worse, with the same training accuracy.

*Results (50% → 100% of data; no augmentation, no LR schedule)*
- ResNet-18 CIFAR-10: 56.2% → 51.7% (SGD), 78.0% → 74.4% (Adam).
- ResNet-18 CIFAR-100 (Adam): 41.4% → 35.0%. Logistic regression (convex): barely affected.
- Online (1,000 images per round): the gap grows as data arrive.
- Tuning LR, batch size or regularization closes the gap only by losing the speed benefit.

*Fix: Shrink & Perturb*
- Before each round: $theta <- lambda theta + p$, with $0 < lambda < 1$ and small noise $p$.
- $lambda = 0.6$: same accuracy as training from scratch, much faster. Smaller $lambda$ can beat it.
- From pre-trained weights ($lambda = 0.3$): at least as good as the better of warm start and scratch.

*Limits:* classification; no augmentation or schedules, which YOLO uses.

*For CARLA*
- Every flight = one retrain; from-scratch training every round is too slow on one GPU.
- CARLA's YOLO starts from COCO weights (pre-training case) and then gets flights (online case).
- _(my proposal)_ Compare warm start, S&P ($lambda$ = 0.3, 0.6) and restart from COCO: mAP50 and GPU time per round.

= 4. LUF-net — Plant Phenomics 2026
*Problem*
- Drone colours shift with sun, clouds and the camera's automatic exposure, so flights are not comparable.

*Method*
- Light U-net (0.95M parameters) in Lab colour space.
- Metadata (aperture, exposure time, ISO + light-sensor irradiance) modulates features with FiLM: $F' = F (1 + gamma(V)) + beta(V)$.
- Ground truth from a 24-patch colour chart; 230 images, 58,292 patches, DJI Mavic 3M, 6–40 m, four light conditions.

*Results (MAPE)*
- Gray-world 14.6% · image-only U-net 9.31% · exposure only 4.95% · irradiance only 4.10% · full 3.89%.
- Sunny is hardest (5.30%), overcast easiest (3.53%).
- Error no longer depends on acquisition settings (r from 0.20–0.28 down to 0.07–0.08).

*Limits:* colour accuracy only, never detection; needs a colour chart; crops, still images, one drone.

*For CARLA*
- Exposure-only version needs no extra hardware if the DJI `.SRT` logs ISO, shutter and aperture (to check).
- The same telemetry gives each tile its altitude × light cell for the TAILOR reward.
- About 0.4 s per 1080p frame: fine offline.
- _(my proposal)_ Train on sunny, test on overcast: none vs. photometric augmentation vs. LUF-net.

= Putting it together
- One round: upload + telemetry → colour correction → detect on tiles → TAILOR picks $k$ tiles → person labels → reward updates bandit → Shrink & Perturb → RTSM fine-tune → evaluate on held-out flights (worst condition).
- Shared lesson: simple signals win (label reward beats accuracy reward; plain supervision beats pseudo-label tricks; a rescaling beats hyperparameter tuning). Always report the simple baselines.

= Questions for the supervisor
- Is a bandit enough RL, or is a learned policy (e.g. MGRAL, ICASSP 2026) expected?
- Training GPU and VRAM?
- How many flights and labeled tiles exist? Can 1–2 flights be fully labeled to simulate the loop?
- Which drone, are `.SRT` files uploaded, and can a colour chart be placed in a few flights?
