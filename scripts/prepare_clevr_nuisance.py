#!/usr/bin/env python3
"""Rebuild (and verify) the one-pixel nuisance crops from the CLEVR confirmation crops.

The pixel-shifted crops are already shipped in ``data/clevr_nuisance/crops``;
this script documents how they were made and checks them byte-for-byte.
Each crop is translated by one pixel along a diagonal chosen by
sha256(salt|case_id|observation_id) mod 4, with edge replication. The
rank-aligned opaque-ID relabels are stored in the manifest (``relabel_map``).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathbench.common import load_config, read_jsonl, resolve, sha256  # noqa: E402

DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
PIXEL_SALT = "pathbench-nuisance-pixel-v1"


def shift_direction(case_id: str, observation_id: str) -> tuple[int, int]:
    digest = hashlib.sha256(f"{PIXEL_SALT}|{case_id}|{observation_id}".encode()).hexdigest()
    return DIAGONALS[int(digest[:8], 16) % len(DIAGONALS)]


def shift_one_pixel(image: Image.Image, direction: tuple[int, int]) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    dx, dy = direction
    shifted = np.roll(array, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        shifted[0, :, :] = shifted[1, :, :]
    elif dy < 0:
        shifted[-1, :, :] = shifted[-2, :, :]
    if dx > 0:
        shifted[:, 0, :] = shifted[:, 1, :]
    elif dx < 0:
        shifted[:, -1, :] = shifted[:, -2, :]
    return Image.fromarray(shifted)


def main() -> int:
    cases = read_jsonl(load_config()["protocols"]["clevr_nuisance"]["manifest"])
    mismatches = total = 0
    for case in cases:
        canonical = {item["observation_id"]: item
                     for item in case["conditions"]["canonical"]["observations"]}
        for item in case["conditions"]["nuisance_pixel"]["observations"]:
            source = resolve(canonical[item["observation_id"]]["crop_path"])
            direction = shift_direction(case["case_id"], item["observation_id"])
            if list(direction) != item["shift_direction"]:
                raise SystemExit(f"direction drift: {case['case_id']}/{item['observation_id']}")
            target = resolve(item["crop_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shift_one_pixel(Image.open(source), direction).save(target, format="PNG")
            total += 1
            mismatches += sha256(target) != item["crop_sha256"]
    print(f"rebuilt {total} one-pixel crops; {mismatches} differ from the frozen SHA-256")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
