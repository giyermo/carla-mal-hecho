"""
Warm start experiment: compare three ways of starting each training round.

    cold  every round starts from the COCO-pretrained weights
    warm  round k starts from round k-1's final weights (last.pt)
    sp    round k starts from Shrink & Perturb applied to round k-1's last.pt

Round 1 is the same for all three setups (there is no previous round yet, so
all start from COCO), so it is trained once per seed and shared.

Every job (seed, setup, round) gets its own folder. A job is finished when its
folder contains results.json. Finished jobs are skipped, so after a Colab
disconnect you just run the script again and it continues where it stopped.

A --tag renames the setup folder, so a variant can be run next to the main
experiment without overwriting it: --setups sp --lam 0.3 --tag lam0.3 --rounds 4
trains only round 4 with a different shrink factor, continuing from the untagged
round 3. --rounds picks which rounds to run at all.

Output layout:
    <out>/seed42/round1/                 shared round 1
    <out>/seed42/<setup>/round<k>/       rounds 2..n
    <out>/seed42/<setup>_<tag>/round<k>/ the same with --tag <tag>
        weights/last.pt, weights/best.pt  Ultralytics checkpoints
        results.csv                       per-epoch training curves
        results.json                      test metrics + timing (the "done" marker)
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from shrink_perturb import shrink_perturb

SETUPS = ["cold", "warm", "sp"]


@dataclass
class Job:
    setup: str  # "shared" for round 1, otherwise one of SETUPS
    seed: int
    round: int
    tag: str = ""  # variant name; "" is the main experiment


def job_dir(out: Path, job: Job) -> Path:
    suffix = f"_{job.tag}" if job.tag else ""
    if job.round == 1:
        return out / f"seed{job.seed}" / f"round1{suffix}"
    return out / f"seed{job.seed}" / f"{job.setup}{suffix}" / f"round{job.round}"


def previous_dir(out: Path, job: Job) -> Path:
    """
    Folder of the round this job continues from (same seed and setup).

    A tagged job prefers the tagged previous round, so a variant can be a chain
    of several rounds, but falls back to the untagged one. That fallback is what
    lets a single tagged round (e.g. a different shrink factor applied only at
    round 4) continue from the main experiment's round 3.
    """
    seed_dir = out / f"seed{job.seed}"
    candidates = []
    if job.tag:
        candidates.append(seed_dir / f"round1_{job.tag}" if job.round == 2
                          else seed_dir / f"{job.setup}_{job.tag}" / f"round{job.round - 1}")
    candidates.append(seed_dir / "round1" if job.round == 2
                      else seed_dir / job.setup / f"round{job.round - 1}")

    for candidate in candidates:
        if (candidate / "weights" / "last.pt").exists():
            return candidate
    return candidates[-1]  # missing; run_job reports it


def plan_jobs(setups: list[str], seeds: list[int], rounds: list[int], tag: str) -> list[Job]:
    """Order matters: round k-1 of a setup must finish before round k."""
    jobs = []
    for seed in seeds:
        if 1 in rounds:
            jobs.append(Job("shared", seed, 1, tag))
        for k in [r for r in rounds if r > 1]:
            for setup in setups:
                jobs.append(Job(setup, seed, k, tag))
    return jobs


def start_weights(job: Job, out: Path, args: argparse.Namespace) -> Path | str:
    """Checkpoint this job starts training from."""
    if job.round == 1 or job.setup == "cold":
        return args.model  # COCO-pretrained, Ultralytics downloads it if needed

    prev_last = previous_dir(out, job) / "weights" / "last.pt"
    if not prev_last.exists():
        raise FileNotFoundError(f"{job}: previous round not found at {prev_last}")

    if job.setup == "warm":
        return prev_last

    # sp: the noise seed depends on seed and round, so every run is reproducible
    # but different rounds get different noise.
    return shrink_perturb(
        prev_last,
        job_dir(out, job) / "start_sp.pt",
        lam=args.lam,
        noise_scale=args.noise_scale,
        model_cfg=Path(args.model).with_suffix(".yaml").name,
        seed=job.seed * 100 + job.round,
    )


def test(weights: Path, data: Path, save_dir: Path, args: argparse.Namespace) -> dict:
    """Evaluate a checkpoint on the test split."""
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
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def run_job(job: Job, out: Path, rounds_dir: Path, args: argparse.Namespace) -> None:
    jd = job_dir(out, job)
    jd.mkdir(parents=True, exist_ok=True)
    data = rounds_dir / f"round_{job.round}.yaml"

    start = start_weights(job, out, args)
    print(f"\n=== {job.setup} | seed {job.seed} | round {job.round} | start: {start}", flush=True)

    t0 = time.time()
    YOLO(str(start)).train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        box=args.box,
        device=args.device,
        seed=job.seed,
        deterministic=True,
        fraction=args.fraction,
        project=str(jd.parent),
        name=jd.name,
        exist_ok=True,
        plots=False,
    )
    train_seconds = time.time() - t0

    weights = jd / "weights"
    result = {
        "setup": job.setup,
        "tag": job.tag,
        "seed": job.seed,
        "round": job.round,
        "start_from": str(start),
        "epochs": args.epochs,
        "lam": args.lam if job.setup == "sp" else None,
        "noise_scale": args.noise_scale if job.setup == "sp" else None,
        "train_seconds": round(train_seconds, 1),
        # last.pt is what the next round starts from, so it is the main result.
        "test_last": test(weights / "last.pt", data, jd / "test_last", args),
        "test_best": test(weights / "best.pt", data, jd / "test_best", args),
    }
    # Written last: its existence means the job finished.
    (jd / "results.json").write_text(json.dumps(result, indent=2))
    print(f"=== done: test mAP50 (last.pt) = {result['test_last']['map50']:.3f}, "
          f"{train_seconds / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the warm start experiment.")
    parser.add_argument("--rounds-dir", type=Path, required=True, help="folder with round_k.yaml files")
    parser.add_argument("--out", type=Path, required=True, help="results folder (on Drive for Colab)")
    parser.add_argument("--setups", nargs="+", default=SETUPS, choices=SETUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rounds", nargs="+", type=int, default=None,
                        help="which rounds to run (default: every round_k.yaml found)")
    parser.add_argument("--tag", default="", help="variant name, appended to the setup folder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model", default="yolo26n.pt", help="COCO-pretrained starting weights")
    parser.add_argument("--lam", type=float, default=0.6, help="Shrink & Perturb shrink factor")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Shrink & Perturb noise size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--box", type=float, default=5.0, help="box loss weight (original thesis: 5.0)")
    parser.add_argument("--device", default="0", help="'0' for the first GPU, 'cpu' for CPU")
    parser.add_argument("--fraction", type=float, default=1.0, help="share of training tiles; <1 only for quick tests")
    args = parser.parse_args()

    out = args.out.resolve()
    rounds_dir = args.rounds_dir.resolve()
    n_rounds = len(list(rounds_dir.glob("round_*.yaml")))
    if n_rounds < 2:
        raise FileNotFoundError(f"need at least 2 round_k.yaml files in {rounds_dir}")

    rounds = args.rounds or list(range(1, n_rounds + 1))
    missing = [k for k in rounds if not (rounds_dir / f"round_{k}.yaml").exists()]
    if missing:
        raise FileNotFoundError(f"no round file for round(s) {missing} in {rounds_dir}")

    jobs = plan_jobs(args.setups, args.seeds, rounds, args.tag)
    todo = [j for j in jobs if not (job_dir(out, j) / "results.json").exists()]
    print(f"{len(jobs)} jobs in total, {len(jobs) - len(todo)} already done, {len(todo)} to run")

    for job in todo:
        run_job(job, out, rounds_dir, args)

    print("\nAll jobs finished.")


if __name__ == "__main__":
    main()
