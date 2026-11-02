# Papers in `papers/`

*Short guide to the PDFs in `../papers/`: what each one is about, how its method works, and where it fits in the CARLA project. Updated 18 September 2026. The full plan is in `carla-general-model-rl-plan.md`.*

| File | Paper | Role in CARLA |
|---|---|---|
| `../papers/NeurIPS-2023-algorithm-selection-for-deep-active-learning-with-imbalanced-datasets-Paper-Conference.pdf` | TAILOR (NeurIPS 2023) | **Core RL method:** the bandit that picks which tiles to label |
| `../papers/simple_supervision.pdf` | Simple Supervision Is Hard to Beat (arXiv 2026) | **Training recipe** for each round, and the baseline the bandit must beat |
| `../papers/warm_start.pdf` | On Warm-Starting Neural Network Training (NeurIPS 2020) | **How to retrain** after every flight without degrading the model |
| `../papers/plant_phenomics.pdf` | LUF-net (Plant Phenomics 2026) | **Colour correction** for sunny vs. cloudy flights |

---

## 1. TAILOR: choosing which labeling strategy to use

**File:** `../papers/NeurIPS-2023-algorithm-selection-for-deep-active-learning-with-imbalanced-datasets-Paper-Conference.pdf`
**Citation:** Zhang, Shao, Verma & Nowak, *Algorithm Selection for Deep Active Learning with Imbalanced Datasets*, NeurIPS 2023. Code: https://github.com/jifanz/TAILOR

**What it's about.** There are many strategies for picking which images a person should label next: most uncertain, most diverse, and so on. Which one works best changes from dataset to dataset, and you can't know in advance. TAILOR learns which strategies to use *while* you label.

**How it works.**
- Each labeling strategy is an "arm" of a **multi-armed bandit** (a simple form of RL).
- For each slot in a labeling batch, TAILOR guesses which strategy is most promising (Thompson sampling), and that strategy picks one image.
- After a person labels the batch, each strategy gets a **reward computed from the new labels themselves**: images of classes that are still rare score more. No retraining is needed to compute the reward.
- The model is retrained, and the next round starts.

**Key results.**
- As good as the best single strategy on 9 of 10 imbalanced datasets.
- Better than the best single strategy in collecting labels for rare classes on 4 of 10.
- Works in only 10–20 labeling rounds.
- A bandit rewarded by model accuracy instead (ALBL) did no better than choosing strategies at random.

**In CARLA.**
- It is the basis of **RL #1**: each uploaded flight is one round, and the bandit picks which tiles you label.
- CARLA has only one class, so the reward must be adapted. Reward tiles where the detector **missed a plant** or produced a **false positive**, weighted by how rare that case still is in each altitude × light condition.
- Use mAP only to *evaluate* the loop, not as the reward.

**Caveat:** it was tested only on image classification, not object detection.

---

## 2. Simple Supervision Is Hard to Beat: how to use a few labels

**File:** `../papers/simple_supervision.pdf`
**Citation:** Zhang, Xu & Agrawal (Amazon Robotics), *Simple Supervision Is Hard to Beat: A Bitter Lesson from Sparse Target Labels in Domain-Adaptive Object Detection*, arXiv 2606.30795, June 2026 (preprint, not peer-reviewed).

**What it's about.** A detector is moved to a new setting (fog, other cities and cameras). Most new images are unlabeled, so the model trains on its own guesses (pseudo-labels). If a person labels a small random share of the new images (1–10%), what is the best way to use those labels?

**How it works.**
- **Simple option (RTSM):** keep the pseudo-label training as it is, and add the few real labels as normal training examples.
- **10 "clever" options:** use the real labels to *fix* the pseudo-labels, for example better confidence cut-offs, recovering missed objects, or balancing the losses.
- Tested with 4 adaptation methods and 2 transformer detectors (DINO, DETA), on Cityscapes → Foggy Cityscapes and Cityscapes → BDD100K.

**Key results.**
- The simple option gives **+1.7 to +18.3 AP50** every time.
- The clever options don't reliably add anything on top. They help in one setup and hurt in another (one lost 13.6 AP50).
- The more labels you have, the less the tricks matter.

**In CARLA.**
- **Training recipe for every round:** pseudo-labels for the unlabeled footage, plus the tiles a person labeled, used directly as training examples.
- **The baseline for the RL bandit** is random tiles with this same recipe. The bandit must beat that.
- **Treat pseudo-label filtering as a hypothesis.** Trusting only detections that appear in several video frames has to be shown to beat the simple recipe.

**Caveat:** it covers transformer detectors on street scenes, not YOLO or drone imagery.

---

## 3. On Warm-Starting Neural Network Training: how to retrain each round

**File:** `../papers/warm_start.pdf`
**Citation:** Ash & Adams, *On Warm-Starting Neural Network Training*, NeurIPS 2020.

**What it's about.** When data keeps arriving, it is tempting to continue training from last round's model ("warm start") instead of starting over. It's faster, but the model then does **worse on new, unseen data**, even though it looks just as good on training data. The gap grows with every round.

**How it works (the fix).** Before each new round, **"Shrink & Perturb"** the weights:
1. **Shrink:** multiply all weights by a number below 1 (e.g. 0.3–0.6).
2. **Perturb:** add a little random noise (a scaled-down freshly initialized network).

The model keeps what it learned but becomes less sure of itself, so old and new data both get learned properly again.

**Key results.**
- Warm-starting cost several points of accuracy, e.g. CIFAR-10 ResNet-18: 56.2% → 51.7%; CIFAR-100: 41.4% → 35.0%.
- Tuning the learning rate or batch size, or adding regularization, did not fix it without losing the speed advantage.
- Shrink & Perturb (0.6) **matched training from scratch while training much faster**, and stronger shrinking could even beat it.

**In CARLA.**
- Every uploaded flight means a retrain. Simply continuing from last round's weights may slowly make the detector worse.
- Use Shrink & Perturb before each round. It's only a few lines of code.
- It is the basis of **option D**: compare, or let a small bandit choose between, "continue", "shrink & perturb" and "restart from pretrained weights".

**Caveat:** it covers image classification only, with no data augmentation and no learning-rate schedule. YOLO training uses both, so the effect on CARLA must be measured.

---

## 4. LUF-net: correcting colour for sunny vs. cloudy flights

**File:** `../papers/plant_phenomics.pdf`
**Citation:** Wang, Liu, Zhang, Jin & Yuan, *LUF-net: A physically informed color calibration method for UAV RGB images based on exposure and irradiance information*, Plant Phenomics 8 (2026) 100262. Code: https://github.com/wangchufeng3652/color-correction

**What it's about.** Drone photos don't record plants' true colours. Colours shift with the sun, the clouds and the camera's automatic settings, so the same field looks different from flight to flight.

**How it works.**
- A small neural network (under 1M parameters) takes an image patch and outputs a colour-corrected patch.
- It is also given how the photo was taken: ISO, shutter speed, aperture and, if available, the drone's light-sensor readings. These adjust the network's internal features (FiLM).
- It is trained against true colours measured from a colour chart placed in the field.

**Key results.**
- Colour error drops to **3.89%** (MAPE), vs. 9.31% for the same network without the photo settings and 14.6% for the classic gray-world method.
- **With camera settings only (no light sensor): 4.95%.**
- Remaining error by light: overcast 3.53%, early morning 3.63%, cloudy 4.19%, **sunny 5.30%** (hardest).
- Tested at 6–40 m over soybean, rice and maize.
- The paper reports a colour error of 6.96 both overall and for overcast. This was checked in the PDF.

**In CARLA.**
- A preprocessing step on the server that makes sunny and cloudy flights look more alike before detection.
- The camera settings it needs are already in DJI's `.SRT` files, so the settings-only version needs no extra hardware.
- It is a **hypothesis to test**: does colour correction reduce the detector's sunny-vs-cloudy gap? Compare it against simple colour augmentation.

**Caveats:**
- It measures colour accuracy only; it was never tested on detection.
- Training needs a colour chart in some flights.
- Tested on crops with a DJI Mavic 3 Multispectral, not on pasture or on CARLA's drone.
