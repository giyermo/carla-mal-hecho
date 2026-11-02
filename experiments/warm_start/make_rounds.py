"""
Split the data into rounds, one per flight, to simulate new flights arriving
over time. Round k trains on every flight up to k (the archive is kept, which
is what makes warm starting tempting in the first place).

Two streams can be built:

    low-first  (default)  round 1 17 Nov, 2 +5 Dec, 3 +11 Dec, 4 +high flight
    high-first            round 1 high flight, 2 +17 Nov, 3 +5 Dec, 4 +11 Dec

The last round of the low-first stream is the interesting one: the high flight
is a condition the model has never seen (a low-flight model scores mAP50 0.011
on it), so it is where warm starting, cold restarts and Shrink & Perturb should
actually differ. The high-first stream gives a second, independent shift event
from the same data, and its round 1 doubles as the "high flight only" baseline
that says how good the high test set can get without any low-flight data.

Output (in the round folder):
    round_k.txt       absolute paths of the training tiles
    round_k_val.txt   validation tiles, only for rounds that mix both datasets
    round_k.yaml      Ultralytics dataset file

Validation: the low dataset contributes its whole valid/ folder as soon as any
low flight has arrived, and the high flight contributes its 272 valid tiles when
it arrives. The low valid folder holds all three low dates from round 1 onwards,
which is a property of the original experiment kept here on purpose, so the
rounds 1-3 files stay identical to the ones already trained on. It is also why
the reported checkpoint is last.pt and not best.pt: nothing is selected on the
validation set.

Test: always the low test folder, so results.json stays comparable across every
round. The real evaluation (per flight, per condition) is evaluate_matrix.py.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

TFG = Path.home() / "university/TFG"
DEFAULT_LOW = TFG / "data/carla-v1"
DEFAULT_HIGH = TFG / "data/high-altitude-tiles-1080p"

# Low flights, in the order they were flown.
LOW_DATES = ["20251117", "20251205", "20251211"]
HIGH_NAME = "high"


def get_date(image_name: str) -> str:
    """
    Return the flight date (YYYYMMDD) of a tile from its filename.

    Low flight:   DJI_20251117141435_0002_D_MP4_0_png_tile0_jpg.rf.6c4f...jpg
    High flight:  YDRAY-DJI_20250515122715_0039_D_MP4_117_png.rf.4cd2..._tile0.jpg
    """
    match = re.search(r"DJI_(\d{8})\d{6}", image_name)
    if match is None:
        raise ValueError(f"Unexpected filename: {image_name}")
    return match.group(1)


def get_frame(image_name: str) -> str:
    """Return the frame a tile belongs to (everything before "_tile")."""
    if "_tile" not in image_name:
        raise ValueError(f"Not a tile: {image_name}")
    return image_name.split("_tile")[0]


def sorted_images(images_dir: Path) -> list[Path]:
    """
    All tiles of a folder, sorted. glob() returns files in no fixed order, and
    the order ends up inside round_k.txt, so sorting keeps runs reproducible.
    """
    return sorted(image.resolve() for image in images_dir.glob("*.jpg"))


def group_low_by_date(images_dir: Path) -> dict[str, list[Path]]:
    """Map each low-flight date to its tiles."""
    groups = defaultdict(list)
    for image in sorted_images(images_dir):
        groups[get_date(image.name)].append(image)

    found, expected = set(groups), set(LOW_DATES)
    if found != expected:
        raise ValueError(
            f"Dates don't match. Unexpected: {sorted(found - expected)}, "
            f"missing: {sorted(expected - found)}"
        )
    return groups


def build_parts(low: Path, high: Path | None, order: str) -> list[tuple[str, list[Path]]]:
    """
    One (name, training tiles) pair per flight, in the order they arrive.
    """
    low_groups = group_low_by_date(low / "train" / "images")
    parts = [(date, low_groups[date]) for date in LOW_DATES]

    if high is not None:
        high_part = (HIGH_NAME, sorted_images(high / "train" / "images"))
        parts = [high_part] + parts if order == "high-first" else parts + [high_part]
    return parts


def write_rounds(parts: list[tuple[str, list[Path]]], low: Path, high: Path | None, out: Path) -> None:
    """Write round_k.txt / round_k_val.txt / round_k.yaml for every round."""
    out.mkdir(parents=True, exist_ok=True)
    low_valid = (low / "valid" / "images").resolve()
    low_test = (low / "test" / "images").resolve()
    low_valid_images = sorted_images(low / "valid" / "images")
    high_valid_images = sorted_images(high / "valid" / "images") if high is not None else []

    for k in range(1, len(parts) + 1):
        arrived = parts[:k]
        names = [name for name, _ in arrived]
        images = [image for _, tiles in arrived for image in tiles]

        train_txt = out / f"round_{k}.txt"
        train_txt.write_text("".join(f"{image}\n" for image in images))

        # Validation: the low folder as a whole, plus the high tiles once the
        # high flight has arrived. Only the mixed case needs a file list; a
        # low-only round points straight at the folder, exactly as before.
        if HIGH_NAME in names:
            low_arrived = [name for name in names if name != HIGH_NAME]
            val_images = (low_valid_images if low_arrived else []) + high_valid_images
            val_txt = out / f"round_{k}_val.txt"
            val_txt.write_text("".join(f"{image}\n" for image in val_images))
            val_target = val_txt.resolve()
        else:
            # A low-only round points straight at the folder, so the files of
            # rounds 1-3 stay identical to the ones already trained on.
            val_target = low_valid

        (out / f"round_{k}.yaml").write_text(
            f"train: {train_txt.resolve()}\n"
            f"val: {val_target}\n"
            f"test: {low_test}\n"
            "\n"
            "nc: 1\n"
            "names: ['0']\n"
        )

        frames = {get_frame(image.name) for image in images}
        print(f"round {k}: {', '.join(names)} -> {len(frames)} frames, {len(images)} tiles")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the flights into cumulative training rounds.")
    parser.add_argument("--low", type=Path, default=DEFAULT_LOW, help="low-flight dataset (train/valid/test)")
    parser.add_argument("--high", type=Path, default=DEFAULT_HIGH, help="tiled high-flight dataset")
    parser.add_argument("--no-high", action="store_true", help="only the three low flights (the original 3 rounds)")
    parser.add_argument("--order", choices=["low-first", "high-first"], default="low-first")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default: rounds/ for low-first, rounds_high_first/ for high-first)")
    args = parser.parse_args()

    high = None if args.no_high else args.high
    out = args.out or Path(__file__).parent / ("rounds" if args.order == "low-first" else "rounds_high_first")

    parts = build_parts(args.low, high, args.order)
    write_rounds(parts, args.low, high, out)
    print(f"wrote {len(parts)} rounds to {out}")


if __name__ == "__main__":
    main()
