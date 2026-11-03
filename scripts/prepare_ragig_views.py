#!/usr/bin/env python3
"""Rebuild the RAG-IGBench full/crop views of the frozen confirmation manifest.

RAG-IGBench rasters are not redistributed. Download ``images.tar.gz`` from the
Hugging Face dataset ``Muyi13/RAG-IGBench`` (revision in configs/pathbench.json),
then run:

    python scripts/prepare_ragig_views.py --images-archive /path/to/images.tar.gz

Every rebuilt view is checked against the SHA-256 stored in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import sys
import tarfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathbench.common import load_config, read_jsonl, resolve, sha256  # noqa: E402

VIEW_SIZE = 336
GRAY = (127, 127, 127)
LOCAL_CROP_FRACTION = 0.7
SELECTION_SALT = "pathbench-ragig-cross-view-identity-v1-20260824"


def rank(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def letterbox(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize((max(1, round(image.width * scale)),
                            max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), GRAY)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def local_view(image: Image.Image, source_hash: str) -> tuple[Image.Image, list[int]]:
    """Deterministic 70% crop seeded by the source-photo hash."""
    image = image.convert("RGB")
    crop_w = max(1, round(image.width * LOCAL_CROP_FRACTION))
    crop_h = max(1, round(image.height * LOCAL_CROP_FRACTION))
    rng = random.Random(int(rank(f"{SELECTION_SALT}|crop|{source_hash}")[:16], 16))
    max_x, max_y = image.width - crop_w, image.height - crop_h
    x = rng.randint(0, max_x) if max_x else 0
    y = rng.randint(0, max_y) if max_y else 0
    return letterbox(image.crop((x, y, x + crop_w, y + crop_h)), VIEW_SIZE), [x, y, crop_w, crop_h]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images-archive", type=Path, required=True,
                        help="images.tar.gz downloaded from the Muyi13/RAG-IGBench dataset")
    args = parser.parse_args()
    cases = read_jsonl(load_config()["protocols"]["ragig_confirmation"]["manifest"])
    wanted = {row["source_member"] for case in cases for row in case["observations"]}
    payload = {}
    with tarfile.open(args.images_archive, "r:gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name in wanted and member.isfile():
                payload[name] = archive.extractfile(member).read()
    if wanted - set(payload):
        raise SystemExit(f"{len(wanted - set(payload))} source images missing from archive")
    mismatches = written = 0
    for case in cases:
        for row in case["observations"]:
            data = payload[row["source_member"]]
            if hashlib.sha256(data).hexdigest() != row["source_sha256"]:
                raise SystemExit(f"source image hash differs: {row['source_member']}")
            image = Image.open(io.BytesIO(data)).convert("RGB")
            if row["role"] == "anchor":
                view = letterbox(image, VIEW_SIZE)
            else:
                view, crop = local_view(image, row["source_sha256"])
                if crop != row["crop_xywh"]:
                    raise SystemExit(f"crop box differs for {row['observation_id']}")
            path = resolve(row["view_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            view.save(path)
            written += 1
            mismatches += sha256(path) != row["view_sha256"]
    print(f"wrote {written} views; {mismatches} differ from the frozen SHA-256")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
