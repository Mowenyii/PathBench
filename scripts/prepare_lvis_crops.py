#!/usr/bin/env python3
"""Rebuild the LVIS/COCO annotation-instance crops of the frozen confirmation manifest.

COCO pixels are not redistributed. The manifest stores each COCO image ID,
LVIS instance ID, box, and category. Provide COCO val2017 pixels either as the
two parquet shards of the Hugging Face dataset ``detection-datasets/coco``
(revision in configs/pathbench.json; these are the files used for the paper) or
as the official ``val2017/`` JPEG directory:

    python scripts/prepare_lvis_crops.py --coco-parquet val-00000-of-00002.parquet val-00001-of-00002.parquet
    python scripts/prepare_lvis_crops.py --coco-dir /path/to/val2017

Every rebuilt crop is checked against the SHA-256 stored in the manifest.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathbench.common import load_config, read_jsonl, resolve, sha256  # noqa: E402

PAD = 0.125
CROP_SIZE = 336
GRAY = (127, 127, 127)


def annotation_crop(image: Image.Image, box: list[float]) -> Image.Image:
    """Pad the box by 12.5%, letterbox to a square, and resize to 336x336."""
    x, y, width, height = box
    left, top = max(0.0, x - width * PAD), max(0.0, y - height * PAD)
    right = min(float(image.width), x + width * (1.0 + PAD))
    bottom = min(float(image.height), y + height * (1.0 + PAD))
    crop = image.crop((round(left), round(top), round(right), round(bottom)))
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), GRAY)
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--coco-parquet", type=Path, nargs="+",
                        help="COCO val parquet shards from detection-datasets/coco")
    source.add_argument("--coco-dir", type=Path, help="official COCO val2017 JPEG directory")
    args = parser.parse_args()
    cases = read_jsonl(load_config()["protocols"]["lvis_confirmation"]["manifest"])
    wanted = {int(case["image_id"]) for case in cases}
    images = {}
    if args.coco_parquet:
        import pyarrow.parquet as pq
        for path in args.coco_parquet:
            for row in pq.read_table(path, columns=["image_id", "image"]).to_pylist():
                if int(row["image_id"]) in wanted:
                    images[int(row["image_id"])] = row["image"]["bytes"]
    else:
        for image_id in wanted:
            path = args.coco_dir / f"{image_id:012d}.jpg"
            if path.is_file():
                images[image_id] = path.read_bytes()
    if wanted - set(images):
        raise SystemExit(f"{len(wanted - set(images))} COCO images not found")
    mismatches = written = 0
    for case in cases:
        image = Image.open(io.BytesIO(images[int(case["image_id"])])).convert("RGB")
        for row in case["observations"]:
            path = resolve(row["crop_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            annotation_crop(image, row["box_xywh"]).save(path)
            written += 1
            mismatches += sha256(path) != row["crop_sha256"]
    print(f"wrote {written} crops; {mismatches} differ from the frozen SHA-256")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
