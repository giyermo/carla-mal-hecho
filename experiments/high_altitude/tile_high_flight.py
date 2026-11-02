"""
Tile the 15 May 2025 high flight the same way the low-flight training data was
tiled, so it can be used as a new round of the warm start experiment.

Per frame:
    3840x2160 (Roboflow export)  --INTER_AREA-->  1920x1080  --tile-->  8 x 640x640

Why downscale to 1080p first: the low-flight data comes from 1920x1080 video
frames cut into 8 tiles of 640 px with 160 px overlap
(carla/scripts/dataset/dataset_tiling.py). Tiling the 4K frames directly gives
40 tiles per frame *and* makes the plants about 1.7x bigger than in training, so
two things would change at once and the round-4 result could not be attributed
to the flight. Downscaling first keeps the pixel pipeline identical to the
training data, leaving the flight itself as the only difference.

The tile geometry, the border clamping and the box rule (drop fragments of 2 px
or less) are taken from carla/scripts/dataset/dataset_tiling.py, so the tiles
written here match the ones already in data/high-altitude-tiles-1080p/test/,
which were made that way.

Labels are YOLO format and normalized, so the downscale does not change them;
only the tiling does.

Usage (writes train/ and valid/ next to the existing test/):
    python tile_high_flight.py
"""

import argparse
from pathlib import Path

import cv2

TILE_SIZE = 640
OVERLAP = 160
STRIDE = TILE_SIZE - OVERLAP  # 480 px of new ground per step
TARGET_SIZE = (1920, 1080)    # (width, height) of the low-flight video frames

TFG = Path.home() / "university/TFG"
DEFAULT_SRC = TFG / "data/high-altitude"                # 4K Roboflow export
DEFAULT_OUT = TFG / "data/high-altitude-tiles-1080p"    # already holds test/


def read_yolo_labels(label_path: Path) -> list[list[float]]:
    """
    Read a YOLO label file: one "class x_center y_center width height" per line,
    all coordinates normalized to [0, 1]. A missing file means no plants.
    """
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
        boxes.append([cls, x, y, w, h])
    return boxes


def clip_box_to_tile(
    box: list[float], img_w: int, img_h: int, x0: int, y0: int, tile_w: int, tile_h: int
) -> list[float] | None:
    """
    Convert a box normalized to the whole frame into one normalized to the tile
    [x0, x0+tile_w) x [y0, y0+tile_h). Returns None if the box hardly touches it.

    A plant on a tile border ends up clipped, so the tile gets the visible part
    of it. Fragments of 2 px or less are dropped: they teach the detector almost
    nothing and would count as missed objects at evaluation time.
    """
    cls, xc, yc, w, h = box

    # normalized -> absolute pixels in the frame
    abs_xc, abs_yc = xc * img_w, yc * img_h
    abs_w, abs_h = w * img_w, h * img_h
    x_min, x_max = abs_xc - abs_w / 2, abs_xc + abs_w / 2
    y_min, y_max = abs_yc - abs_h / 2, abs_yc + abs_h / 2

    # the part of the box that falls inside the tile
    cx_min, cx_max = max(x_min, x0), min(x_max, x0 + tile_w)
    cy_min, cy_max = max(y_min, y0), min(y_max, y0 + tile_h)
    new_w, new_h = cx_max - cx_min, cy_max - cy_min
    if new_w <= 2 or new_h <= 2:
        return None

    # back to normalized, now relative to the tile
    return [
        cls,
        ((cx_min + cx_max) / 2 - x0) / tile_w,
        ((cy_min + cy_max) / 2 - y0) / tile_h,
        new_w / tile_w,
        new_h / tile_h,
    ]


def tile_frame(image_path: Path, label_path: Path, out_images: Path, out_labels: Path) -> tuple[int, int]:
    """
    Downscale one 4K frame to 1080p, cut it into 640 px tiles and write each tile
    with its own label file. Returns (tiles written, boxes written).
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"could not read {image_path}")

    # INTER_AREA averages the pixels it merges, which is the right choice when
    # shrinking an image; it also reproduces the existing test frames.
    frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    img_h, img_w = frame.shape[:2]
    boxes = read_yolo_labels(label_path)
    base = image_path.stem

    tile_idx, n_boxes = 0, 0
    for y0 in range(0, max(img_h - OVERLAP, 1), STRIDE):
        for x0 in range(0, max(img_w - OVERLAP, 1), STRIDE):
            # Clamp the tile so it never runs past the frame. The last tile of a
            # row or column therefore overlaps its neighbour by more than 160 px
            # instead of being smaller than 640 px.
            x_end, y_end = min(x0 + TILE_SIZE, img_w), min(y0 + TILE_SIZE, img_h)
            x_start, y_start = max(x_end - TILE_SIZE, 0), max(y_end - TILE_SIZE, 0)
            tile_w, tile_h = x_end - x_start, y_end - y_start

            tile_boxes = [
                b for b in (
                    clip_box_to_tile(box, img_w, img_h, x_start, y_start, tile_w, tile_h)
                    for box in boxes
                ) if b is not None
            ]

            name = f"{base}_tile{tile_idx}"
            cv2.imwrite(str(out_images / f"{name}.jpg"), frame[y_start:y_end, x_start:x_end])
            (out_labels / f"{name}.txt").write_text(
                "".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in tile_boxes)
            )

            tile_idx += 1
            n_boxes += len(tile_boxes)
            if x_end == img_w:
                break
        if y_end == img_h:
            break

    return tile_idx, n_boxes


def process_split(src: Path, out: Path, split: str) -> None:
    """Tile every frame of one split (train / valid / test)."""
    images_dir = src / split / "images"
    if not images_dir.is_dir():
        print(f"[skip] {split}: no images at {images_dir}")
        return

    out_images, out_labels = out / split / "images", out / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    frames = sorted(images_dir.glob("*.jpg"))
    tiles = boxes = 0
    for image_path in frames:
        label_path = src / split / "labels" / f"{image_path.stem}.txt"
        t, b = tile_frame(image_path, label_path, out_images, out_labels)
        tiles += t
        boxes += b

    empty = sum(1 for p in out_labels.glob("*.txt") if p.stat().st_size == 0)
    print(f"{split:<6} {len(frames):>4} frames -> {tiles:>5} tiles, {boxes:>5} boxes, "
          f"{empty:>4} tiles without cardilla")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile the high flight like the low-flight training data.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="4K dataset (train/valid/test with images+labels)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the tiles")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"],
                        help="splits to tile; test/ already exists in the default output folder")
    args = parser.parse_args()

    print(f"{args.src} -> {args.out}  ({TILE_SIZE} px tiles, {OVERLAP} px overlap, frames resized to {TARGET_SIZE})")
    for split in args.splits:
        process_split(args.src, args.out, split)


if __name__ == "__main__":
    main()
