"""
Pull the small metric files out of a results tree and leave the checkpoints behind.

Training runs on Colab and writes every job straight to Google Drive, including
weights/last.pt and weights/best.pt (~5 MB each). Those do not belong in git:
30 jobs is ~300 MB of binaries that change on every rerun. Everything needed to
*read* the experiment is text and weighs a few hundred kB:

    results.json      test metrics + timing (also the "job finished" marker)
    eval_matrix.json  per-flight / per-condition scores (evaluate_matrix.py)
    results.csv       per-epoch training curves
    args.yaml         the exact arguments Ultralytics ran the job with

On Colab, after a run has finished:

    !python export_results.py --src {RESULTS} {RESULTS_HIGH_FIRST} \
        --zip /content/drive/MyDrive/TFG/results_export.zip

Download that one file from Drive, then on the laptop:

    python export_results.py --unzip ~/Downloads/results_export.zip \
        --dest experiments/warm_start/results

Each --src keeps its own top-level folder inside the zip, named after the source
directory, so results/ and results_high_first/ do not collide.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

KEEP = ["results.json", "eval_matrix.json", "results.csv", "args.yaml"]


def collect(src: Path, keep: list[str]) -> list[tuple[Path, Path]]:
    """(absolute file, path relative to src) for every file worth exporting."""
    found = []
    for name in keep:
        for path in sorted(src.rglob(name)):
            found.append((path, path.relative_to(src)))
    return found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", nargs="+", type=Path,
                   help="results tree(s) to export, e.g. the Drive results folder")
    p.add_argument("--dest", type=Path, help="copy into this folder")
    p.add_argument("--zip", type=Path, help="write a zip here instead of copying")
    p.add_argument("--unzip", type=Path, help="unpack a zip made by this script into --dest")
    p.add_argument("--keep", nargs="+", default=KEEP, help=f"file names to export (default: {KEEP})")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.unzip:
        if not args.dest:
            p.error("--unzip needs --dest")
        with zipfile.ZipFile(args.unzip) as z:
            names = z.namelist()
            print(f"{len(names)} files -> {args.dest}")
            if not args.dry_run:
                args.dest.mkdir(parents=True, exist_ok=True)
                z.extractall(args.dest)
        return

    if not args.src:
        p.error("give --src (to export) or --unzip (to unpack)")
    if not args.dry_run and bool(args.dest) == bool(args.zip):
        p.error("give exactly one of --dest or --zip")

    jobs = []
    for src in args.src:
        if not src.exists():
            print(f"[warn] {src} does not exist, skipped")
            continue
        files = collect(src, args.keep)
        # keep each source under its own name so two trees never collide
        jobs += [(path, Path(src.name) / rel) for path, rel in files]
        print(f"{src}: {len(files)} files")

    total = sum(path.stat().st_size for path, _ in jobs)
    print(f"total {len(jobs)} files, {total / 1024:.0f} kB")
    if args.dry_run:
        for _, rel in jobs:
            print("  ", rel)
        return

    if args.zip:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for path, rel in jobs:
                z.write(path, rel.as_posix())
        print(f"wrote {args.zip} ({args.zip.stat().st_size / 1024:.0f} kB)")
    else:
        for path, rel in jobs:
            out = args.dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
        print(f"copied into {args.dest}")


if __name__ == "__main__":
    main()
