"""Small shared helpers: paths, JSONL I/O, and interval estimators."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    """Resolve a repository-relative path; absolute paths are returned unchanged."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in resolve(path).read_text().splitlines()
            if line.strip()]


def load_config(path: str | Path = "configs/pathbench.json") -> dict:
    return json.loads(resolve(path).read_text())


def quantile(values: list[float], probability: float) -> float:
    """Percentile by floor index, as used by the per-protocol bootstraps."""
    values = sorted(values)
    return values[min(len(values) - 1, int(probability * len(values)))]


def percentile(values: list[float], probability: float) -> float | None:
    """Percentile by rounded index, as used by the post-hoc exposure analyses."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def all_case_sensitivity(events: int, valid: int, total: int) -> list[float]:
    """Bound a valid-only event rate by treating invalid cases as non-events/events."""
    return [events / total, (events + total - valid) / total]
