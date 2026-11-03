#!/usr/bin/env python3
"""Regenerate raw PathBench records with a frozen VLM (greedy, single attempt).

    python scripts/run_inference.py --protocol clevr_confirmation --model qwen --device cuda:0

Protocols: clevr_confirmation, clevr_nuisance, ragig_confirmation, lvis_confirmation.
Output defaults to ``results/rerun/<protocol>/<model>.jsonl``; an existing file is
resumed by skipping its completed case IDs. Score with
``python scripts/reproduce_main.py --results results/rerun --no-check``
once every protocol/model file exists (copy any missing ones from results/frozen).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathbench.common import load_config, read_jsonl, resolve  # noqa: E402
from pathbench.protocols import PROTOCOLS, STATE_KEY  # noqa: E402


def image_paths(case: dict) -> list[str]:
    if "conditions" in case:  # nuisance views
        return [item["crop_path"] for view in case["conditions"].values()
                for item in view["observations"]]
    return [item.get("crop_path") or item["view_path"] for item in case["observations"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    parser.add_argument("--model", required=True, choices=("qwen", "internvl", "llava", "idefics2"))
    parser.add_argument("--device", default="cuda:0", help="torch device (default: %(default)s)")
    parser.add_argument("--output", type=Path,
                        help="output JSONL (default: results/rerun/<protocol>/<model>.jsonl); "
                             "an existing file is resumed")
    parser.add_argument("--limit", type=int, help="run only the first N cases (smoke test)")
    parser.add_argument("--case-id", action="append", default=[],
                        help="run only these case IDs; may be repeated")
    args = parser.parse_args()

    config = load_config()
    spec = config["models"][args.model]
    protocol = PROTOCOLS[args.protocol](config["protocols"][args.protocol], args.model)
    cases = protocol.cases()
    if args.case_id:
        cases = [case for case in cases if case["case_id"] in set(args.case_id)]
    if args.limit:
        cases = cases[:args.limit]
    missing = [path for case in cases for path in image_paths(case) if not resolve(path).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} observation images are missing (e.g. {missing[0]}). "
                         "Build them first; see README 'Data preparation'.")

    output = resolve(args.output or Path("results/rerun") / args.protocol / f"{args.model}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {row["case_id"] for row in read_jsonl(output)} if output.exists() else set()
    todo = [case for case in cases if case["case_id"] not in done]
    print(f"{args.protocol}/{args.model}: {len(todo)} cases to run, {len(done)} already done",
          flush=True)
    if not todo:
        return 0

    from pathbench.models import VLM
    model = VLM(spec, args.device)
    state_key = STATE_KEY.get(args.protocol, "state_entries")
    with output.open("a", encoding="utf-8") as handle:
        for index, case in enumerate(todo, 1):
            started = time.perf_counter()
            responses, states, errors, predictions = {}, {}, {}, {}
            for condition in protocol.conditions:
                try:
                    raw, state, error, prediction = protocol.call(model, case, condition)
                except Exception as exc:  # single attempt; failures are retained as invalid
                    raw, state, prediction = "", ({} if state_key == "state_mappings" else []), None
                    error = f"{type(exc).__name__}: {str(exc)[:500]}"
                responses[condition], states[condition] = raw, state
                errors[condition], predictions[condition] = error, prediction
            valid = all(value is None for value in errors.values())
            record = {**protocol.record(case), "protocol": args.protocol,
                      state_key: states, "raw_responses": responses,
                      "condition_errors": errors, "executable_predictions": predictions,
                      "status": "complete_valid" if valid else "invalid_single_attempt_state",
                      "elapsed_seconds": time.perf_counter() - started,
                      "model_repository": spec["repository"], "model_revision": spec["revision"]}
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(f"[{index}/{len(todo)}] {case['case_id']} {record['status']} "
                  f"exec={predictions} gt={case['answer']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
