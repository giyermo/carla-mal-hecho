# Reinforcement Learning: State of the Art and Fit for the TFG

*Research brief, updated 18 September 2026. Also published as a web page (older format): https://claude.ai/code/artifact/0445e785-8706-4ed0-80ef-a1530a39e8a3. CARLA-specific follow-ups: `carla-rl-state-of-the-art.md` and `carla-generalization-state-of-the-art.md`.*

Thesis problem: detect a specific plant or bush species across large areas of drone footage, with a model that keeps improving as more footage arrives.

> Every claim links to its source where it is made. The brief uses only the five most relevant sources, listed with the reason for each at the end. Figures come from the papers' abstracts and summaries, so check them against the full text before citing them in the thesis. Anything without a link is textbook background or my own reasoning.

---

## Short answer

"The model gets better as we get more footage" is **mostly not an RL problem**. RL is for an agent that takes *actions* and receives *rewards* over time. A detector that improves as labeled footage accumulates is supervised learning. The fields that cover it are:

- **Active learning:** which frames or tiles to label next.
- **Semi-supervised learning / self-training:** getting value from the unlabeled footage.
- **Continual learning:** retraining on new sites without forgetting old ones.

RL has three legitimate roles in the thesis, each with a direct precedent (section 3):

1. Choosing what to label ([MGRAL, ICASSP 2026](https://arxiv.org/abs/2601.15688)). **Recommended.**
2. Planning where the drone flies ([UAV weed path planning, 2024–25](https://arxiv.org/abs/2412.11717)).
3. Fine-tuning the detector on a task reward ([Pinto et al., ICML 2023](https://arxiv.org/abs/2302.08242)).

Suggested framing: *a supervised detector inside a human-in-the-loop data flywheel, with one RL component as the novel contribution.*

---

## 1. Foundations (background chapter)

Textbook material. Cite a standard reference such as Sutton & Barto's *Reinforcement Learning: An Introduction* or each algorithm's original paper.

| Family | Key methods | Idea |
|---|---|---|
| Value-based | DQN, Double/Dueling DQN, Rainbow | Learn Q(s,a), act greedily |
| Policy gradient | REINFORCE, PPO, TRPO | Raise the probability of actions that did well |
| Actor-critic, off-policy | SAC, TD3 | Policy plus a learned critic, reusing old data |
| Critic-free group methods | GRPO and variants | Group-average baseline instead of a value network |
| Model-based | Dreamer, MuZero | Learn a world model, train in imagination |
| Offline RL | CQL, IQL, Decision Transformer | Learn from a fixed log, no interaction |
| Preference-based | RLHF, DPO | Reward from human comparisons |

## 2. Frontier, 2025–26

### 2.1 RL for reasoning models and agents
Since DeepSeek-R1, RL has become a standard post-training stage for large language models. Reasoning RL has moved away from expensive PPO-style pipelines, which need a separate value network, toward cheaper **critic-free, group-relative** methods such as **GRPO** and **RL with verifiable rewards (RLVR)**. In RLVR the reward is an automatic check, such as a correct maths answer. Preference-based methods (DPO variants) and **agentic policy optimization** for multi-step, tool-using models are also part of the 2026 landscape ([Reasoning RL in 2026, Turing Post](https://www.turingpost.com/p/reasoning-rl-in-2026)). **Relevance to you:** the recipe of a verifiable reward plus a group baseline carries over to vision. Detection has a natural verifiable reward (box overlap / mAP), which is what makes option C below possible.

### 2.2 World models
Dreamer 4 trains its agent entirely by *imagination* inside a fast, accurate learned world model. It is the first agent to obtain diamonds in Minecraft **purely from a fixed offline dataset**, without interacting with the environment. The same paper notes that controllable video models such as Genie 3 still struggle with precise physics and object interactions, and often need many GPUs to simulate one scene in real time ([Dreamer 4, Hafner et al., 2025](https://arxiv.org/abs/2509.24527)). **Relevance to you:** mostly background. It shows where RL is heading when real interaction is expensive, as it is for drone flights.

### 2.3 RL for perception models
Pinto et al. first train a vision model normally with maximum likelihood, then fine-tune it with the REINFORCE algorithm to maximize a **task-specific reward**. They apply this to object detection, panoptic segmentation, colorization and image captioning. For detection, mAP rose from **40.2% to 52.3% within 20k steps** of reward tuning, reaching 53.2% at 60k ([Pinto et al., ICML 2023](https://arxiv.org/abs/2302.08242)). **Relevance to you:** a detector can be tuned directly on the metric an ecologist cares about, such as recall of the target species or count error.

*Also active in 2025–26 but outside the five sources:* RL fine-tuning of robot vision-language-action models, and loss of plasticity in continual RL, where networks gradually lose the ability to learn. The second matters in practice. After many retraining rounds, check against a model trained from scratch.

## 3. Where RL fits the thesis

### A. RL-driven active learning (recommended)
MGRAL frames labeling as an RL problem for **object detectors**. An LSTM agent, trained with policy gradient, selects the next batch of images to label. The **reward is the mAP improvement**, and fast unsupervised look-up tables estimate it without retraining for every reward. It was evaluated on PASCAL VOC and COCO ([MGRAL, Liang et al., ICASSP 2026](https://arxiv.org/abs/2601.15688)). This matches "improves with more footage" exactly: the agent decides which new tiles a person should label.

- **State:** detector uncertainty and embeddings of candidate tiles.
- **Action:** pick a batch to label.
- **Reward:** Δ mAP after retraining.
- **Main risk:** each true reward needs a retrain. Use a small detector, or MGRAL's look-up-table approximation.

### B. RL path planning for the survey
A deep Q-learning policy plans UAV flights that localize weeds using two inputs: a low-resolution, uncertain prior map of where weeds are, and the detector's live output during the flight. On real-world image data it achieved a **66% shorter flight path** than row-by-row coverage. The learned policy was robust to detection errors and did not need detailed prior knowledge ([UAV weed path planning, 2024–25](https://arxiv.org/abs/2412.11717)).

- **State:** belief map of plant locations, plus a prior map.
- **Action:** next waypoint or altitude.
- **Reward:** new plants found, minus distance flown.
- **Main risk:** needs programmable flights. Otherwise it stays a simulation on existing orthomosaics.

### C. Reward fine-tuning of the detector
After supervised training, fine-tune with REINFORCE (or a group-baseline method such as GRPO) on the metric you care about ([Pinto et al., ICML 2023](https://arxiv.org/abs/2302.08242)).

- **Reward:** IoU, F1, or counting error.
- **Main risk:** it can be unstable, and a well-tuned supervised baseline is hard to beat. Keep it as an extension.

## 4. The non-RL core

The core of a detector that improves with footage is supervised. I'd consider (my recommendations; the companion CARLA briefs cite sources for several of them):

- **A strong detector from few labels:** a modern real-time detector (YOLO26, or a transformer detector with a self-supervised backbone such as DINOv2/v3).
- **Cheap initial labels:** open-vocabulary models (e.g. Grounding DINO + SAM) to pre-label tiles, then correct them by hand.
- **Choosing what to label:** uncertainty + diversity sampling. This is the baseline an RL selector (option A) must beat.
- **Using unlabeled footage:** teacher–student pseudo-labeling, with the human-labeled tiles mixed directly into training as ordinary supervised examples. Avoid elaborate label-guided pseudo-label tricks unless they beat that simple mix.
- **Retraining without forgetting old sites:** a replay buffer of earlier examples plus distillation from the previous model.

## 5. Proposed system (data flywheel)

1. **Fly and stitch:** orthomosaic or video frames at a known ground sampling distance.
2. **Tile:** e.g. 640–1024 px with overlap, keeping the georeferencing.
3. **Bootstrap:** pre-labels from an open-vocabulary model, corrected by a human.
4. **Train:** the detector; pseudo-label the unlabeled tiles.
5. **Select:** uncertainty + diversity, *or the RL selector (option A)*.
6. **Annotate:** a human labels the top-k tiles.
7. **Retrain:** replay buffer + distillation.
8. **Evaluate:** a fixed, spatially separate test site, with per-round learning curves.

Steps 5–8 repeat for each batch of new footage. The headline figure is mAP vs. number of labeled tiles for random, uncertainty-based and RL-based selection.

## 6. Pitfalls

- **Spatial leakage:** never split by random tiles. Split by flight, site or geographic block.
- **Phenology drift:** species appearance changes with season. Record dates and treat season as a domain shift.
- **GSD mismatch:** altitude changes object scale. Normalize resolution or use multi-scale augmentation.
- **RL for its own sake:** always report the non-RL baselines.
- **Loss of plasticity:** after many rounds, compare against a model trained from scratch on all the data.

---

## The five sources

1. **[Liang et al., Performance-guided Reinforced Active Learning for Object Detection (MGRAL, ICASSP 2026)](https://arxiv.org/abs/2601.15688).** The most direct fit: RL chooses what to label for a detector, rewarded by mAP. It is the basis for the recommended option A.
2. **[Pinto et al., Tuning Computer Vision Models with Task Rewards (ICML 2023)](https://arxiv.org/abs/2302.08242).** The reference for applying RL to a detector itself, with a large measured mAP gain. It is the basis for option C.
3. **[UAV-based path planning for efficient localization of non-uniformly distributed weeds using prior knowledge: a reinforcement-learning approach (2024–25)](https://arxiv.org/abs/2412.11717).** RL on drones for plant detection in agriculture. It is the basis for option B.
4. **[Reasoning RL in 2026: GRPO, DPO, RLVR, Agentic PO & Beyond (Turing Post)](https://www.turingpost.com/p/reasoning-rl-in-2026).** A single up-to-date overview of where most RL research effort is going in 2026. It is a secondary source; for the thesis, follow it to the primary papers it covers.
5. **[Hafner et al., Training Agents Inside of Scalable World Models (Dreamer 4, 2025)](https://arxiv.org/abs/2509.24527).** The leading model-based RL result, trained from offline data only. It represents the other major 2025–26 frontier.
