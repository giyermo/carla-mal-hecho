"""
Evaluate every finished round on every test set separately.

run_experiment.py reports one pooled number per job: mAP50 on the 696 low-flight
test tiles. That number cannot separate the two things that matter when a new
flight arrives:

    plasticity  did the model learn the flight that just arrived?
    retention   did it stay good on the flights it already knew?

Warm starting, cold restarts and Shrink & Perturb trade those against each
other, so they have to be measured apart. This script loads each job's
checkpoint and evaluates it on each test set on its own, giving a
round x test-set matrix per setup. The diagonal is plasticity, everything to the
left of it is retention.

It reads the finished jobs and writes eval_matrix.json next to each
results.json, so the jobs already trained do not have to be run again. Sets
already in a job's eval_matrix.json are skipped, so the script can be stopped
and restarted.

Test sets (none of them is ever trained on):
    low_all       696 tiles, the three low flights pooled: the original number
    low_<date>    the same tiles split by flight day (264 / 320 / 112)
    high_1080p    264 tiles, the high flight downscaled to 1080p and tiled,
                  the same pipeline the high training tiles go through
    high_4k      1320 tiles, the high flight tiled at native 4K: tests whether
                  training on downscaled tiles transfers to full resolution

Usage:
    python evaluate_matrix.py --results <folder with seed*/> --device 0
"""

import argparse
import json
import re
from pathlib import Path

from ultralytics import YOLO

TFG = Path.home() / "university/TFG"
DEFAULT_LOW = TFG / "data/carla-v1"
DEFAULT_HIGH_1080P = TFG / "data/high-altitude-tiles-1080p"
DEFAULT_HIGH_4K = TFG / "data/high-altitude-tiles-4k"
LOW_DATES = ["20251117", "20251205", "20251211"]


def build_eval_sets(out: Path, low: Path, high_1080p: Path, high_4k: Path) -> dict[str, Path]:
    """
    Write one Ultralytics dataset file per test set and return {name: yaml}.

    The per-day low sets are file lists filtered by the date in the tile name;
    the others point at a folder. Every file has the same train/val/test value
    because only split="test" is ever used. A dataset that is not on this
    machine is skipped with a warning, so the 4K tiles (only used for one
    optional set) do not have to be uploaded to Colab.
    """
    out.mkdir(parents=True, exist_ok=True)
    candidates = {
        "low_all": low / "test" / "images",
        "high_1080p": high_1080p / "test" / "images",
        "high_4k": high_4k / "test" / "images",
    }
    targets: dict[str, Path] = {}
    for name, folder in candidates.items():
        if folder.is_dir():
            targets[name] = folder.resolve()
        else:
            print(f"[warn] test set '{name}' skipped, no images at {folder}")

    low_test = sorted((low / "test" / "images").glob("*.jpg"))
    for date in LOW_DATES:
        tiles = [t.resolve() for t in low_test if re.search(rf"DJI_{date}\d{{6}}", t.name)]
        if not tiles:
            continue
        txt = out / f"low_{date}.txt"
        txt.write_text("".join(f"{tile}\n" for tile in tiles))
        targets[f"low_{date}"] = txt.resolve()

    sets = {}
    for name, target in targets.items():
        yaml_path = out / f"{name}.yaml"
        yaml_path.write_text(
            f"# test set '{name}', built by evaluate_matrix.py\n"
            f"train: {target}\n"
            f"val: {target}\n"
            f"test: {target}\n"
            "\n"
            "nc: 1\n"
            "names: ['0']\n"
        )
        sets[name] = yaml_path
    return sets


def count_images(yaml_path: Path) -> int:
    """How many tiles a test set holds, for the printed summary."""
    target = Path(next(
        line.split("test:", 1)[1].strip()
        for line in yaml_path.read_text().splitlines() if line.startswith("test:")
    ))
    return len(list(target.glob("*.jpg"))) if target.is_dir() else len(target.read_text().splitlines())


def find_jobs(results: Path) -> list[Path]:
    """Folders of finished jobs, i.e. the ones that contain results.json."""
    return sorted(path.parent for path in results.glob("seed*/**/results.json"))


def evaluate(weights: Path, data: Path, save_dir: Path, args: argparse.Namespace) -> dict:
    metrics = YOLO(str(weights)).val(
        data=str(data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    return {
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def evaluate_job(job_dir: Path, sets: dict[str, Path], args: argparse.Namespace) -> dict:
    """Fill in whatever this job is still missing and save it."""
    result = json.loads((job_dir / "results.json").read_text())
    matrix_path = job_dir / "eval_matrix.json"
    matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() and not args.force else {}

    matrix.update({
        "setup": result["setup"],
        "tag": result.get("tag", ""),
        "seed": result["seed"],
        "round": result["round"],
        "job_dir": str(job_dir),
    })
    matrix.setdefault("evals", {})

    for which in args.weights:
        weights = job_dir / "weights" / f"{which}.pt"
        if not weights.exists():
            print(f"  [skip] no {which}.pt")
            continue
        done = matrix["evals"].setdefault(which, {})
        for name, data in sets.items():
            if name in done:
                continue
            done[name] = evaluate(weights, data, job_dir / "eval" / which / name, args)
            print(f"  {which}.pt on {name:<14} mAP50 {done[name]['map50']:.3f}", flush=True)
            matrix_path.write_text(json.dumps(matrix, indent=2))  # save as we go

    matrix_path.write_text(json.dumps(matrix, indent=2))
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every finished round on every test set.")
    parser.add_argument("--results", type=Path, required=True, help="results folder (holds seed42/, seed43/, ...)")
    parser.add_argument("--low", type=Path, default=DEFAULT_LOW, help="low-flight dataset")
    parser.add_argument("--high-1080p", type=Path, default=DEFAULT_HIGH_1080P, help="tiled high flight (1080p)")
    parser.add_argument("--high-4k", type=Path, default=DEFAULT_HIGH_4K, help="tiled high flight (native 4K)")
    parser.add_argument("--eval-sets", type=Path, default=Path(__file__).parent / "eval_sets",
                        help="where the test-set dataset files are written")
    parser.add_argument("--weights", nargs="+", default=["last"], choices=["last", "best"],
                        help="which checkpoint of each job to evaluate (last.pt is the reported one)")
    parser.add_argument("--only-sets", nargs="+", default=None, help="evaluate only these test sets")
    parser.add_argument("--force", action="store_true", help="re-evaluate sets that are already done")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for the first GPU, 'cpu' for CPU")
    args = parser.parse_args()

    sets = build_eval_sets(args.eval_sets.resolve(), args.low, args.high_1080p, args.high_4k)
    if args.only_sets:
        unknown = set(args.only_sets) - set(sets)
        if unknown:
            raise SystemExit(f"unknown test set(s): {sorted(unknown)}. Available: {sorted(sets)}")
        sets = {name: sets[name] for name in args.only_sets}

    print("test sets:")
    for name, yaml_path in sets.items():
        print(f"  {name:<14} {count_images(yaml_path):>5} tiles")

    jobs = find_jobs(args.results.resolve())
    if not jobs:
        raise SystemExit(f"no results.json found under {args.results}")
    print(f"\n{len(jobs)} finished jobs under {args.results}")

    for job_dir in jobs:
        print(f"\n=== {job_dir.relative_to(args.results.resolve())}", flush=True)
        evaluate_job(job_dir, sets, args)

    print("\nDone. Run summarize.py to build the tables.")


if __name__ == "__main__":
    main()
