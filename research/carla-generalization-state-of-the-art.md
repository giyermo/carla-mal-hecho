# CARLA across weather and altitude: state of the art in generalization

*Research brief, updated 18 September 2026. Companion to `carla-rl-state-of-the-art.md`. Goal: one cardilla detector that works under sunny and overcast skies and at different flight altitudes.*

> Every claim links to its source where it is made. The brief uses only the five most relevant sources, listed with the reason for each at the end. Figures come from the papers' abstracts and summaries, so check them against the full text before citing them in the thesis. Anything without a link is my own reasoning or a calculation from the CARLA repo.

---

## 0. What changes, and why it hurts a detector

**Sunny vs. overcast.** Direct sun adds hard shadows and directional light, and overcast light is flat and even. A colour-calibration study on drone RGB images of crops found that, even after correction, the remaining colour error was lowest under overcast skies (3.53%) and highest under sun (5.30%). The authors attribute the sunny-day difficulty to strong sunlight, specular reflections and shadows ([Wang et al., LUF-net, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13452194/)). For CARLA, the risk is that a model trained on sunny flights learns the cardilla's *shadow* as a cue, then misses plants on cloudy days.

**Altitude.** Flying higher increases the ground sampling distance (GSD), so each plant covers fewer pixels. For maize seedlings, a Faster R-CNN trained on images at GSD ≈ 0.3 cm and tested at ≈ 0.6 cm saw its counting error (rRMSE) rise from 0.08 to 0.48 ([Velumani et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8404552/)). That's a sixfold increase from just doubling the pixel size.

**GSD in CARLA's terms** (calculated from the repo's 84° horizontal FOV and 1920 px width in `src/yolo/utils/unit_conversor.py`; GSD ≈ 1.80·h / 1920):

| Altitude | GSD | A 0.5 m plant spans (example size) |
|---|---|---|
| 10 m | ~0.94 cm/px | ~53 px |
| 20 m | ~1.9 cm/px | ~27 px |
| 40 m | ~3.8 cm/px | ~13 px |
| 60 m | ~5.6 cm/px | ~9 px (very hard to detect) |

*(Verify the FOV: DJI often quotes the diagonal FOV, and the 16:9 video crop makes the horizontal FOV narrower.)*

## 1. Free metadata: DJI SRT telemetry

DJI video comes with a `.SRT` subtitle track holding **per-frame** relative altitude, GPS, ISO, shutter speed and EV. From it, CARLA gets without manual labeling:
- **GSD per frame**, from the altitude.
- **An illumination proxy**, since overcast frames need higher ISO or longer shutter at the same aperture.
- **Nuisance labels:** altitude and lighting condition for every frame.

Those labels are useful in their own right. NDFT showed that freely available UAV metadata (flying altitude, weather, view angle) can be used during training to make detectors more robust to those same factors ([Wu et al., ICCV 2019](https://arxiv.org/abs/1908.03856)). Today `UnitConversor` takes one constant altitude and nothing in the repo parses SRT files, so keep every flight's `.SRT` file from now on.

## 2. Evaluation first: you can't claim generalization without it

- **Condition matrix:** fly the same plots at, e.g., {15, 30, 45 m} × {sunny, overcast}. Manual flights still let you choose the altitude.
- **Out-of-condition splits:** train on sunny and test on overcast; train on 15 + 30 m and test on 45 m. Keep the flight/area split used against spatial leakage.
- **Report per condition,** as the UAV robustness literature does. NDFT evaluates detection separately by flying altitude, weather and camera view ([Wu et al., ICCV 2019](https://arxiv.org/abs/1908.03856)), and the Rumex study tests on 15 separate UAV flights that the model never saw during training ([Schrag et al., 2026](https://arxiv.org/abs/2604.25316)).
- **Metrics:** mAP50 per condition cell, **worst-condition mAP**, the gap between in- and out-of-condition results, mAP by object size in pixels, coverage error per cell, and Jetson latency whenever the model changes.

## 3. State of the art

### 3.1 Scale and altitude
Training at one resolution and testing at another fails badly. Maize counting error went from 0.08 to 0.48 ([Velumani et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8404552/)). The same study tested three fixes:
- **Mixing native high-resolution images with synthetic low-resolution ones** (blur + downsample) worked best: rRMSE 0.06 on high-res and 0.10 on synthetic low-res.
- **Real low-resolution images stayed harder** (0.29), so synthetic downsampling does not replace real high-altitude footage.
- **GAN super-resolution of low-res test images** helped only partly (0.48 → 0.22).

For CARLA, two options follow from this:
- **GSD-normalized tiling:** resize each frame to a canonical GSD using its SRT altitude, then tile at 640 px.
- **Multi-scale training:** widen YOLO's `scale` augmentation and add blurred, downsampled copies of low-altitude frames. CARLA currently trains with default augmentation (`src/yolo/yolo_run.py`).

### 3.2 Illumination and weather
LUF-net is a small U-net (~0.95M parameters) that corrects drone RGB colour. It is conditioned on the camera's exposure settings (ISO, shutter, aperture) and on the drone's downwelling light (irradiance) sensor. Across overcast, cloudy, sunny and early-morning flights at 6–40 m, it cut the overall colour error to 3.89% MAPE. An image-only U-net managed 9.31%, and the classic gray-world correction 14.6% ([Wang et al., LUF-net, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13452194/)). **The key result for CARLA:** using *exposure data alone*, with no light sensor, still reached 4.95%, about half the image-only error. That data is in DJI's SRT files. Caveats:
- Training needs a 24-patch colour chart photographed in the field.
- It was tested on soybean, rice and maize from a DJI Mavic 3 Multispectral, not on pasture or on CARLA's camera.
- It measures colour accuracy only; whether it improves *detection* is untested.

For CARLA, that suggests two things:
- Use the SRT exposure values to normalize colour before detection.
- Add photometric augmentation (brightness, gamma, HSV, colour temperature) and synthetic cast shadows during training. This is standard practice, and it is the cheap baseline every other method must beat.

### 3.3 Foundation-model backbones: the biggest recent jump
- **DINOv3 inside YOLO26 for weed detection:** replacing YOLO26-L's backbone with a DINOv3 ViT-S raised mAP50 over plain YOLO26 by up to 5.4 points on the training season (2025), 11.9 points on 2024 data and 14.0 points on 2021–23 data ([Deng & Lu, 2026](https://arxiv.org/abs/2603.00160)). Part of this comes from a modified detection head. Measured against that stronger YOLO26 baseline, the ViT itself adds only about +2 points in-domain but +7.6 and +10.2 points on the other years. **The backbone helps mostly where the data differs from training**, which is the property CARLA needs. Caveats:
  - The images are ground-level (smartphone and cart cameras), not drone footage.
  - The "other years" also differ in camera and platform.
  - The model is about 15× larger than YOLO26-nano and about 3× slower than YOLO26-L.
- **Rumex (a weed) from drones:** self-supervised ViTs (DINOv2/v3) beat CNNs with explicit domain-adaptation techniques when moving from ground-vehicle images to UAV images, reaching F1 = 0.8 across 15 flights ([Schrag et al., 2026](https://arxiv.org/abs/2604.25316)). Their UAV dataset, AGSMultiRumex, is public and could serve as extra test data.
- **Trade-off for CARLA:** a ViT backbone is heavy for the Jetson Orin Nano. Distillation keeps deployment fast: a robust DINOv3-based teacher trains the YOLO-nano student on footage from all conditions.

### 3.4 Using the metadata in training
NDFT adds adversarial heads that try to predict altitude, weather and view angle from the detector's features, and trains the features so those heads fail. The detector keeps what identifies the object and drops what only describes the conditions ([Wu et al., ICCV 2019](https://arxiv.org/abs/1908.03856)). The code is public (TAMU-VITA/UAV-NDFT). With SRT files supplying altitude and a lighting label for every frame, CARLA gets these nuisance labels for free. Applying this to aerial *plant* detection would be a novel contribution.

## 4. What I'd suggest for CARLA

| Priority | Step | Cost | Based on |
|---|---|---|---|
| 1 | Parse SRT; log altitude, GSD, ISO/shutter and weather per frame; build the condition matrix and out-of-condition test sets | Low | Evaluation practice in [NDFT](https://arxiv.org/abs/1908.03856) and [Rumex](https://arxiv.org/abs/2604.25316) |
| 2 | Ablation: default YOLO vs. photometric + shadow augmentation vs. exposure-based colour normalization | Low | [LUF-net](https://pmc.ncbi.nlm.nih.gov/articles/PMC13452194/) |
| 3 | Mixed-resolution training and GSD-normalized tiling vs. the current fixed-pixel tiling | Low–medium | [Velumani et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8404552/) |
| 4 | DINOv3 teacher → distilled YOLO-nano student | Medium | [DINOv3 Meets YOLO26](https://arxiv.org/abs/2603.00160), [Rumex](https://arxiv.org/abs/2604.25316) |
| 5 | NDFT-style adversarial head using SRT altitude and lighting labels | Medium | [NDFT](https://arxiv.org/abs/1908.03856) |

**Tie-in with the RL loop** (`carla-rl-state-of-the-art.md`):
- Feed the SRT condition metadata (altitude/GSD, illumination proxy, date) into the **bandit context**.
- Add a **"condition coverage" arm** that labels tiles from under-represented altitude × light cells.
- Compute the reward on a **condition-balanced** validation set, ideally worst-condition mAP.

The loop then directly optimizes generalization, giving the thesis one story: *a detector that improves with every flight and doesn't break when conditions change.*

## 5. Open questions

1. What altitudes and weather do the existing flights cover? Do you still have their `.SRT` files?
2. Can you schedule flights at several altitudes and in different weather over the same plots?
3. Is Jetson deployment still a requirement? It decides whether a ViT backbone is allowed at inference or only as a teacher.
4. What is the realistic altitude range? It sets the canonical GSD and the smallest detectable plant.

---

## The five sources

1. **[Deng & Lu, DINOv3 Meets YOLO26 for Weed Detection in Vegetable Crops (2026)](https://arxiv.org/abs/2603.00160).** The closest match to CARLA's stack (YOLO26, plant detection). It shows that a foundation backbone's gains are largest on out-of-domain data. Note: the images are ground-level, not aerial.
2. **[Velumani et al., Estimates of Maize Plant Density from UAV RGB Images Using Faster-RCNN: Impact of the Spatial Resolution (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8404552/).** The only source that quantifies what happens to UAV plant detection when resolution (altitude) changes, and which fixes work.
3. **[Wu et al., Delving into Robust Object Detection from UAVs: A Deep Nuisance Disentanglement Approach (NDFT, ICCV 2019)](https://arxiv.org/abs/1908.03856).** Uses altitude and weather metadata directly for robustness, which matches the free DJI SRT data. Code is public.
4. **[Schrag et al., Towards Robust Deep Learning-based Rumex Obtusifolius Detection from Drone Images (2026)](https://arxiv.org/abs/2604.25316).** Weed detection from drones across 15 flights. Shows foundation backbones beat classic domain adaptation, and its dataset is public.
5. **[Wang et al., LUF-net: A physically informed color calibration method for UAV RGB images based on exposure and irradiance information (Plant Phenomics, 2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13452194/).** Compares overcast, cloudy, sunny and early-morning drone imagery at 6–40 m. It shows camera exposure data alone halves colour error. Code and weights are public.
