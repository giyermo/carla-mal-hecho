# Experiment 02: warm start vs. Shrink & Perturb (status, 18 Oct 2026)

*Handoff summary. Paste this together with `00-context.md` to start a new chat.*

## Question
Every uploaded flight means a retrain. Where should each training round start from? This tests Ash & Adams, *On Warm-Starting Neural Network Training* (NeurIPS 2020, `papers/warm_start.pdf`): continuing from last round's weights ("warm") is claimed to generalize worse than starting over, and **Shrink & Perturb** (θ ← λθ + noise_scale · θ_random) is claimed to fix it. The paper only tested classification (ResNets, CIFAR), without augmentation or a learning-rate schedule; this is the first test on YOLO and drone imagery.

## Data
- Roboflow `carla-4gysv/carla-low-flight-tiling-rgb-fudwn`, **version 1** (no augmentation, no preprocessing), YOLOv11 format. Unzipped in `data/carla-v1/`.
- 1920×1080 video frames (one every 3 s) cut into 8 tiles of 640 px (160 px overlap) by `carla/scripts/dataset/dataset_tiling.py`. The split was made on whole frames before tiling.
- **3 flight days, 13 videos:** 17 Nov 2025, 5 Dec 2025, 11 Dec 2025. Filename: `DJI_<YYYYMMDDhhmmss>_<video>_D_MP4_<frame>_png_tile<n>_jpg.rf.<hash>.jpg`.
- **Cleaned:** 6 duplicated frames (72 tiles, uploaded twice as `D-MP4`/`D_MP4`, with *conflicting labels*; 3 leaked into valid/test) removed from train only. Documented in `data/carla-v1/CHANGES.md`. Frame 9 of the same video is still duplicated inside test (kept so test matches the original thesis).
- After cleaning: **train 1864 tiles / 233 frames, valid 704 / 88, test 696 / 87.** Clean zip: `data/carla-v1-clean.zip`.
- Open question: my earlier notes say the thesis flights were 15 May 2025, but this dataset is Nov–Dec 2025. Check with the previous student which data the thesis results used.

## Design
- **Rounds = flight days, chronological, cumulative:** round 1 = 17 Nov (88 frames, 704 tiles), round 2 = +5 Dec (200 / 1600), round 3 = +11 Dec (233 / 1864). Valid and test are the same for every round.
- **Setups:** *cold* (COCO weights every round), *warm* (round k−1's `last.pt`), *sp* (Shrink & Perturb on round k−1's `last.pt`, λ = 0.6, noise scale 0.01, noise = freshly initialized network, BatchNorm running stats untouched; paper's main setting).
- Round 1 is identical for all setups (all start from COCO), so it is trained once per seed and shared.
- YOLO26n, Ultralytics 8.4.14, **30 epochs per round** (the original thesis curve reaches ~98% of peak val mAP50 by epoch 30 and overfits after ~50), batch 16, imgsz 640, `box=5.0` (as in the original thesis), seeds 42/43/44, `deterministic=True`.
- Next round starts from `last.pt` (not `best.pt`, which is chosen with the validation set). Test metrics are saved for both.
- Note: the cold start is not fully pretrained. YOLO replaces the 80-class COCO head with a new 1-class one ("Overriding nc=80 with nc=1").

## Code (`experiments/warm_start/`)
| File | Does |
|---|---|
| `make_rounds.py` | Groups train tiles by date (regex on the filename), writes `rounds/round_k.txt` + `round_k.yaml`. `--data`, `--out` |
| `shrink_perturb.py` | Shrink & Perturb on a YOLO checkpoint. Verified: weights ×0.600, BN stats unchanged, Ultralytics loads 708/708 items |
| `run_experiment.py` | Runs all 21 jobs (3 seeds × [shared round 1 + 3 setups × rounds 2–3]). Each job has its own folder; `results.json` = done marker; finished jobs are skipped (Colab disconnects) |
| `summarize.py` | Collects all `results.json` into `all_runs.csv` and `summary.csv` (mean ± std over seeds) |
| `warm_start_colab.ipynb` | Colab notebook: GPU check, mount Drive, install, unzip + build rounds, quick test, real run, summary |
| `.venv/` | Laptop environment (torch 2.13.0 CPU, ultralytics 8.4.14) |

On Colab everything lives in `MyDrive/TFG/`; results go to `MyDrive/TFG/results/`. Colab ran torch 2.11.0+cu130 on a Tesla T4.

## Compute
- Laptop (i7-13700H, no NVIDIA GPU): 3.5 min per epoch on round 1. Only for debugging.
- Colab T4: about 10.5 / 18.5 / 21 min per job for rounds 1 / 2 / 3 (30 epochs). The full run is about 7 GPU hours.
- The free Colab GPU quota ran out with **4 jobs left** (seed 44: sp round 2, and cold/warm/sp round 3; about 1.5 h). Rerun all notebook cells when the quota is back; it continues by itself.

## Results so far (test mAP50 of `last.pt`, mean ± std over seeds)
| Round | Cold | Warm | Shrink & Perturb |
|---|---|---|---|
| 1 | 0.768 ± 0.005 (3) | same run | same run |
| 2 | 0.796 ± 0.004 (3) | 0.793 ± 0.005 (3) | 0.798 ± 0.001 (2) |
| 3 | 0.820 ± 0.003 (2) | 0.815 ± 0.004 (2) | 0.828 ± 0.003 (2) |

- Round 3 shows the paper's order (sp > cold > warm), but the effect is small (0.5–1 point of mAP50, vs. 4–6 points in the paper) and round 3 has only 2 seeds. **Provisional.** Round 2: no clear difference.
- Plausible reason for the small effect (my reasoning, not verified): YOLO's augmentation and per-run learning-rate warmup already counter part of the warm-start problem.
- Round 3 cold (0.820) is slightly above the original thesis's YOLO26 tiled test mAP50 (0.810, up to 200 epochs), with 30 epochs. Caveat: duplicates were removed from train.
- Training time is the same for all setups (fixed 30 epochs), as designed.

## To do
1. Finish the 4 remaining jobs, rerun `summarize.py`.
2. **Paired comparison per seed** (sp − warm, sp − cold, warm − cold within each seed; same round 1). Same sign in all 3 seeds = more convincing than overlapping ± values.
3. **Speed:** from each job's `results.csv`, epochs needed to reach ~95% of final val mAP50 per setup. That's the paper's "warm start trains faster" claim.
4. Optional: warm and sp with fewer epochs (e.g. 10) vs. cold at 30; λ ∈ {0.3, 0.6, 0.9}.
5. Plots for the thesis: test mAP50 per round, one line per setup, shaded band over seeds.

## Other files from this work
- `presentation/warm-start-guide.typ` / `.pdf`: step-by-step guide, including a Colab section.
- Typst handouts share `presentation/style.typ`.

## Working style
I want to understand everything to defend the thesis: explanations first, and the "why" behind each decision. I sometimes ask Claude to write the code; then I need to be able to explain every line.
