# Experiment 03: LUF-net colour correction (handoff, 18 Oct 2026)

*Paste this together with `00-context.md` to start a new chat. Nothing has been run yet for this experiment.*

## Question
Does colour-correcting the drone frames with LUF-net make the cardilla detector more robust to lighting (sunny vs. overcast)? Concretely: does test mAP50 of an **already trained** YOLO26n change when the test tiles are colour-corrected first?

This version is **inference only** (no detector training), so it runs on my laptop's CPU. I have no GPU right now (free Colab quota ran out; see `experiment-02-warm-start.md`).

## The paper (`papers/plant_phenomics.pdf`)
Wang, Liu, Zhang, Jin & Yuan, *LUF-net: A physically informed color calibration method for UAV RGB images based on exposure and irradiance information*, Plant Phenomics 8 (2026) 100262. Code: https://github.com/wangchufeng3652/color-correction

Verified in the PDF:
- **Architecture:** Lite U-net in **Lab colour space**. Encoder 32/64/128 channels + 256-channel bottleneck; 0.946 M parameters. Works on **256×256 patches**.
- **Metadata:** a **7-dimensional vector**, passed through an MLP and injected after each encoder block with FiLM: F′ = F·(1 + γ(V)) + β(V). The 7 values are camera exposure (**ISO, exposure time, aperture**) + multispectral **irradiance of 4 bands (R, G, RE, NIR)** from the drone's light sensor (section 2.4).
- **Training data:** 230 still photos (DNG + JPEG, 5472×3648) from a **DJI Mavic 3 Multispectral**, over soybean, rice and maize, at 6/12/25/40 m, under overcast, cloudy, sunny and early-morning light; 58,292 patches. Ground truth from a 24-patch colour chart measured with a colorimeter.
- **Results (colour error / MAPE):** full LUF-net 6.96 / 3.89%; **exposure-only** 8.86 / 4.95%; irradiance-only 7.34 / 4.10%; no FiLM 8.09 / 4.52%; U-net without metadata 16.7 / 9.31%; gray-world 26.1 / 14.6%.
- Hardest light: sunny (MAPE 5.30%). Only colour accuracy was measured, **never detection**.

From the GitHub page (not yet checked in detail):
- Files: `inference.py`, `eval.py`, `train.py`, `config.json`, `requirements.txt`; folders `models/`, `utils/`, `weights/`, `data/`, `outputs/`.
- Pretrained weights: **one file, `weights/best_model.pth`**. Probably the full 7-input model; no exposure-only weights are listed.
- Metadata goes in `vectors.json`; example data in `data/example/`. Field order and normalization are not documented on the page; read `inference.py` and `vectors.json`.
- No license stated.

## Main obstacle: metadata
- My frames come from **video**, so they have no per-frame EXIF (ISO, shutter, aperture).
- DJI drones write a `.SRT` subtitle file next to each video with per-frame telemetry, often including ISO, shutter, f-number and altitude. I don't know yet whether our handler kept/uploads the `.SRT` files. The CARLA code doesn't read them (altitude is passed manually in `carla/src/yolo/utils/unit_conversor.py`).
- **No irradiance values:** our drone is not a Mavic 3 Multispectral (exact model still unknown). The 4 irradiance inputs are missing, and the published weights probably expect them.
- Options to discuss: (a) find the `.SRT` files and fill exposure, set irradiance to a fixed/average value from the example data (a hack, must be stated as such); (b) use the no-metadata baseline idea instead (gray-world, simple colour normalization) as the CPU experiment; (c) retrain the exposure-only variant (needs colour-chart data we don't have, so probably not possible).

## Our data (details in `experiment-02-warm-start.md`)
- `data/carla-v1/`: cleaned Roboflow v1, YOLO format. Test = 696 tiles of 640×640 (87 frames), from 3 flight days: **17 Nov 2025, 5 Dec 2025, 11 Dec 2025**. The date is in the filename (`DJI_<YYYYMMDDhhmmss>_...`).
- Tiles are crops of 1920×1080 video frames (8 tiles per frame, 160 px overlap). LUF-net works on 256×256 patches, so tiles need splitting and stitching (or correct whole frames first).
- **Weather per day is unknown.** To test "sunny vs. overcast", I need to find out (look at the frames, ask the handler, or check historical weather for the site and date).

## Trained detectors available
From the warm start experiment on Google Drive (`MyDrive/TFG/results/`), e.g. `seed42/cold/round3/weights/last.pt` (YOLO26n trained on all 3 days, test mAP50 ≈ 0.82), and `seed42/round1/weights/last.pt` (trained on 17 Nov only, test mAP50 ≈ 0.77). Download one to the laptop.

The round-1 model is interesting here: trained on one day only, so the other two days are "new conditions" for it, where colour correction should help most if it helps at all.

## Proposed plan
1. Clone the LUF-net repo into its own folder; read `inference.py`, `config.json`, `vectors.json`; run it on `data/example/` on CPU.
2. Find out what metadata we can get (`.SRT` files? drone model?). Decide option (a)/(b)/(c) above.
3. Make colour-corrected copies of the test set (same labels, only pixels change). Also make a **gray-world** corrected copy as the simple baseline the paper uses.
4. Run `model.val(split="test")` with the trained detector on original, LUF-net and gray-world test sets. Report mAP50 **per flight day** as well as overall.
5. Look at a few corrected tiles side by side. Colour correction can also break things (e.g. shift green plants toward grey).

Expected outcomes, all valid: LUF-net helps (supports colour correction as preprocessing), does nothing (the detector is already robust; YOLO's HSV augmentation may explain it), or hurts (the model doesn't transfer to our drone and pasture).

## Environment
- Laptop: i7-13700H, 20 threads, 31 GB RAM, **no NVIDIA GPU**. Testing 696 tiles with YOLO26n took ~50 s on CPU.
- Existing venv: `experiments/warm_start/.venv` (Python 3.13, torch 2.13.0 CPU, ultralytics 8.4.14). LUF-net's `requirements.txt` may need its own venv; the paper used Python 3.10.
- Put code in `experiments/luf_net/`. Keep the cloned repo separate from my own code.

## Working style
I want to understand everything to defend the thesis: explanations first, the "why" behind each decision. I sometimes ask Claude to write the code; then I need to be able to explain every line.
