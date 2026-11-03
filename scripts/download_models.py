#!/usr/bin/env python3
"""Download and/or verify the pinned public VLM checkpoints.

    python scripts/download_models.py                    # Qwen, InternVL, LLaVA
    python scripts/download_models.py --model idefics2   # development-only model
    python scripts/download_models.py --verify-only

Checkpoints go to ``models/<name>`` (see configs/pathbench.json). Downloading a
checkpoint means accepting its upstream license.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathbench.common import load_config, resolve, sha256  # noqa: E402


def verify(spec: dict) -> list[str]:
    local = resolve(spec["local_dir"])
    if not local.is_dir():
        return [f"missing directory {spec['local_dir']}"]
    manifest = json.loads(resolve(spec["checkpoint_manifest"]).read_text())
    issues = []
    for item in manifest["files"]:
        path = local / item["path"]
        if not path.is_file():
            issues.append(f"missing {item['path']}")
        elif path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            issues.append(f"hash mismatch {item['path']}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", choices=("qwen", "internvl", "llava", "idefics2"),
                        help="model to fetch; may be repeated (default: qwen, internvl, llava)")
    parser.add_argument("--verify-only", action="store_true",
                        help="only check local files against configs/checkpoints/*.json")
    args = parser.parse_args()
    models = load_config()["models"]
    names = args.model or ["qwen", "internvl", "llava"]
    failed = False
    for name in names:
        spec = models[name]
        if not args.verify_only:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=spec["repository"], revision=spec["revision"],
                              local_dir=resolve(spec["local_dir"]))
        issues = verify(spec)
        failed |= bool(issues)
        print(f"{name}: {'OK' if not issues else 'FAIL ' + '; '.join(issues[:5])}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
