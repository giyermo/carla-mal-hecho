"""
Test one trained model on the normal test set and on the 15 May 2025 high flight.

    reference   normal CARLA test set (Nov-Dec 2025 low flights), 696 tiles
    high_1080p  high flight, 4K frames downscaled to 1920x1080, then tiled (plant size close to training)
    high_4k     high flight, 4K frames tiled directly (plants ~1.7x bigger than in training)

All three run on the same machine with the same settings, so the numbers are comparable.
Writes results.json and a printed table.
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

TFG = Path.home() / "university/TFG"
DATASETS = {
    "reference": TFG / "experiments/warm_start/rounds/round_3.yaml",
    "high_1080p": TFG / "data/high-altitude-tiles-1080p/data.yaml",
    "high_4k": TFG / "data/high-altitude-tiles-4k/data.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on the high-altitude flight.")
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "sp_round3_seed42.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    results = {}
    for name, data in DATASETS.items():
        m = model.val(data=str(data), split="test", imgsz=640, batch=16, device=args.device,
                      project=str(args.out.resolve()), name=name, exist_ok=True, plots=True, verbose=False)
        results[name] = {
            "map50": round(float(m.box.map50), 4),
            "map50_95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({"model": str(args.model), **results}, indent=2))
    print(f"\n{'dataset':<12} {'mAP50':>7} {'mAP50-95':>9} {'precision':>10} {'recall':>7}")
    for name, r in results.items():
        print(f"{name:<12} {r['map50']:>7.3f} {r['map50_95']:>9.3f} {r['precision']:>10.3f} {r['recall']:>7.3f}")


if __name__ == "__main__":
    main()
