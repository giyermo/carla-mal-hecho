# TFG context (paste this to start a new chat)

## Project
My bachelor's thesis (UC3M) extends **CARLA** (https://github.com/jmartinpizarro/carla), a previous student's project that detects **cardilla** (*Eryngium horridum*) in DJI drone video. Existing repo: YOLOv8/11/26-nano, tiled (640 px, 160 px overlap) vs non-tiled, 5 seeds, test mAP50 ≈ 0.81–0.85 (tiled does **not** beat non-tiled); T-Rex2 zero-shot baseline (poor); TensorRT on Jetson Orin Nano; pixel→GPS conversion, coverage %, density heatmaps. Frames are extracted every 3 s and labeled in Roboflow. Training flights: 15 May 2025. There is also a 13 Feb 2026 video.

## My goals
1. The model keeps improving as new footage arrives, with as little labeling as possible → **reinforcement learning in the data loop**.
2. It generalizes across **sunny/overcast weather and different flight altitudes**.
3. The contribution must be clearly new vs. the original thesis (which trained once on a fixed dataset).

## Constraints
- Flights are **manual**; I can't control the drone programmatically (but I can choose the altitude).
- **No real-time requirement:** the drone records, the handler uploads the video to our servers after landing, processing is offline/batch. Jetson deployment is **not** a constraint.
- **One desktop GPU**; new flights arrive regularly.

## The plan (in `research/carla-general-model-rl-plan.md`)
- **General detector** (DINOv3 backbone in YOLO26, or RF-DETR) as the production model; YOLO-nano only as a baseline / cheap proxy.
- **RL #1 (core): a TAILOR-style bandit** picks which tiles a person labels each uploaded flight. Arms: every-3-s, random, least-sure, most-different, frame-flicker (temporal inconsistency), T-Rex2 disagreement, under-covered condition. **Reward is label-based** (rarity-weighted count of detector misses / false positives / positives per altitude × light cell), not Δ mAP — mAP is for evaluation.
- **Training recipe per round:** pseudo-labels on unlabeled footage + the newly labeled tiles as plain supervised examples (RTSM). Baseline to beat: random tiles, same recipe.
- **Retraining:** Shrink & Perturb before each round (warm-starting degrades generalization).
- **Offline extras:** overlapping tiles at inference (current code doesn't overlap), multi-frame aggregation + GPS de-duplication for unique plant counts, colour/scale normalization from DJI `.SRT` telemetry (altitude, ISO, shutter).
- **Stretch:** RL #2 (reward fine-tuning on coverage/count), or a small VLM fine-tuned with GRPO as a side experiment.

## Files
- `README.md` — repo front page for other people: layout, what is deliberately not committed (data, checkpoints, `carla/`, papers), how to rerun experiment 02, how the Drive results get in
- `research/00-context.md` — this file
- `research/carla-general-model-rl-plan.md` — **the main plan**
- `research/carla-rl-state-of-the-art.md` — RL for learning from new flights (5 sources, cited inline)
- `research/carla-generalization-state-of-the-art.md` — weather/altitude (5 sources, cited inline)
- `research/rl-state-of-the-art.md` — general RL state of the art (5 sources)
- `research/papers.md` — guide to the 4 PDFs in `papers/`
- `research/experiment-01-rl-active-learning.md` — older, generic experiment design (pre-CARLA; needs updating)
- `research/experiment-02-warm-start.md` — **warm start vs. Shrink & Perturb experiment: data, code, results so far** (Oct 2026)
- `research/shrink-perturb-explained.md` — **how Shrink & Perturb works and why, from a full read of the paper**, with the differences from our setup and what to look for in the results (Nov 2026)
- `research/experiment-03-luf-net-handoff.md` — LUF-net colour correction: **not feasible for now** (no videos, so no `.SRT` camera settings)
- `research/experiment-04-tailor-handoff.md` — **TAILOR bandit (core RL contribution): design adapted to CARLA, plan, compute** (Oct 2026, not started)
- `papers/` — TAILOR (NeurIPS 2023), Simple Supervision Is Hard to Beat (2026), On Warm-Starting (NeurIPS 2020), LUF-net (Plant Phenomics 2026)

## Style I want
- Inline citations: every claim links to its source in the paragraph.
- Say what's verified vs. my own reasoning; flag caveats (e.g. a paper tested classification, not detection).
- Plain language.

## Open questions (still unanswered)
1. Server GPU: same machine for training? How much VRAM? (~12 GB rules out the VLM experiment; ≥24 GB allows LoRA.)
2. How many tiles/frames are labeled now, and how many uploaded flights/dates exist?
3. Which drone model (decides whether LUF-net's full colour correction applies or only the exposure-only version)? Does the handler upload the `.SRT` file?
4. How long does one training run take on that GPU?
5. What output do the ecologists need: maps, counts per hectare, coverage per plot?

## Next steps
1. Parse `.SRT` telemetry; build test sets split by condition (altitude × sun/overcast, leave-one-flight-out).
2. Build the offline inference pipeline (overlapping tiles, multi-frame aggregation); measure the gain on the current YOLO first.
3. Simulate the labeling loop: fully label 1–2 existing flights, hide the labels, reveal on request.
4. Baselines: YOLO-nano vs. general detector on out-of-condition splits; time one retrain.
5. Single strategies, then the bandit; then pseudo-labels with/without.
