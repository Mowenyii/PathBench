#!/usr/bin/env python3
"""Recompute the paper's main results (Tables 1-3 and linked in-text numbers).

No model inference is needed: every number is recomputed from raw per-case
records. By default the frozen records in ``results/frozen`` are scored and each
number is checked against the value printed in the paper. Point ``--results`` at
a directory produced by ``scripts/run_inference.py`` to score a fresh run (use
``--no-check`` because greedy decoding is not guaranteed to be bit-identical
across hardware and library versions).

    python scripts/reproduce_main.py
    python scripts/reproduce_main.py --results results/rerun --no-check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathbench import clevr, metrics  # noqa: E402
from pathbench.common import (all_case_sensitivity, load_config, read_jsonl,  # noqa: E402
                              resolve, wilson)

# Values as printed in the paper (percent unless noted), rounded to one decimal.
EXPECTED = {
    # Table 1: order-preserving nuisance control (state-table flip rates, common valid cases)
    "T1.qwen.common_valid": 175, "T1.internvl.common_valid": 177, "T1.llava.common_valid": 173,
    "T1.qwen.order": 76.6, "T1.qwen.pixel": 46.9, "T1.qwen.relabel": 54.9,
    "T1.qwen.order_silent": 56.6, "T1.qwen.relabel_silent": 44.0,
    "T1.qwen.delta": 21.7, "T1.qwen.delta_lo": 13.1, "T1.qwen.delta_hi": 30.3,
    "T1.internvl.order": 65.0, "T1.internvl.pixel": 29.4, "T1.internvl.relabel": 32.2,
    "T1.internvl.order_silent": 45.8, "T1.internvl.relabel_silent": 22.0,
    "T1.internvl.delta": 32.8, "T1.internvl.delta_lo": 25.4, "T1.internvl.delta_hi": 40.1,
    "T1.llava.order": 100.0, "T1.llava.pixel": 51.4, "T1.llava.relabel": 86.1,
    "T1.llava.order_silent": 54.9, "T1.llava.relabel_silent": 60.1,
    "T1.llava.delta": 13.9, "T1.llava.delta_lo": 9.2, "T1.llava.delta_hi": 19.1,
    "T1.canonical_rerun_identical_all_models": 100.0,
    # Table 2: core state-integrity results
    "T2.clevr.qwen.valid": 176, "T2.clevr.internvl.valid": 177, "T2.clevr.llava.valid": 175,
    "T2.clevr.qwen.state": 76.1, "T2.clevr.qwen.answer": 19.9, "T2.clevr.qwen.silent": 56.2,
    "T2.clevr.internvl.state": 65.0, "T2.clevr.internvl.answer": 19.2, "T2.clevr.internvl.silent": 45.8,
    "T2.clevr.llava.state": 100.0, "T2.clevr.llava.answer": 45.1, "T2.clevr.llava.silent": 54.9,
    "T2.ragig.qwen.valid": 119, "T2.ragig.internvl.valid": 119,
    "T2.ragig.qwen.state": 58.8, "T2.ragig.qwen.answer": 41.2, "T2.ragig.qwen.silent": 17.6,
    "T2.ragig.internvl.state": 40.3, "T2.ragig.internvl.answer": 27.7, "T2.ragig.internvl.silent": 12.6,
    "T2.lvis.qwen.valid": 116, "T2.lvis.internvl.valid": 119,
    "T2.lvis.qwen.state": 50.0, "T2.lvis.qwen.answer": 37.1, "T2.lvis.qwen.silent": 12.9,
    "T2.lvis.internvl.state": 52.9, "T2.lvis.internvl.answer": 38.7, "T2.lvis.internvl.silent": 14.3,
    "T2.max_replay_disagreement_all_cells": 0.0,
    # Section 5.2 in-text numbers tied to Table 2
    "S52.clevr.qwen.state_lo": 69.9, "S52.clevr.qwen.state_hi": 82.4,
    "S52.clevr.internvl.state_lo": 57.6, "S52.clevr.internvl.state_hi": 71.8,
    "S52.clevr.llava.state_wilson_lo": 97.9, "S52.clevr.llava.state_wilson_hi": 100.0,
    "S52.clevr.qwen.answer_lo": 14.2, "S52.clevr.qwen.answer_hi": 26.1,
    "S52.clevr.internvl.answer_lo": 13.6, "S52.clevr.internvl.answer_hi": 24.9,
    "S52.clevr.llava.answer_lo": 37.7, "S52.clevr.llava.answer_hi": 52.6,
    "S52.clevr.qwen.silent_cases": 99, "S52.clevr.internvl.silent_cases": 81,
    "S52.clevr.llava.silent_cases": 96,
    "S52.clevr.qwen.silent_allcase_lo": 55.0, "S52.clevr.qwen.silent_allcase_hi": 57.2,
    "S52.clevr.internvl.silent_allcase_lo": 45.0, "S52.clevr.internvl.silent_allcase_hi": 46.7,
    "S52.clevr.llava.silent_allcase_lo": 53.3, "S52.clevr.llava.silent_allcase_hi": 56.1,
    "S52.clevr.qwen.field_change": 12.5, "S52.clevr.internvl.field_change": 9.6,
    "S52.clevr.llava.field_change": 36.4,
    "S52.clevr.qwen.row_change": 32.2, "S52.clevr.internvl.row_change": 27.1,
    "S52.clevr.llava.row_change": 77.7,
    "S52.ese.unused_attribute": 137, "S52.ese.predicate_preserving": 94,
    "S52.ese.arithmetic_cancellation": 45,
    "S52.dev.qwen.state": 84.3, "S52.dev.internvl.state": 66.7, "S52.dev.llava.state": 100.0,
    "S52.dev.idefics2.state": 100.0, "S52.dev.qwen.answer": 25.7, "S52.dev.internvl.answer": 20.3,
    "S52.dev.llava.answer": 37.7, "S52.dev.idefics2.answer": 40.4,
    "S52.dev.idefics2.valid_rate": 72.2,
    "S52.ragig.silent_cases": 36, "S52.ragig.alt_answer_change": 90.3,
    "S52.ragig.alt_correctness_change": 83.3,
    "S52.lvis.qwen.changed_instance": 19.5, "S52.lvis.internvl.changed_instance": 18.5,
    "S52.lvis.qwen.all_eight_silent": 3.4, "S52.lvis.internvl.all_eight_silent": 5.9,
    # Table 3 and Section 5.3: counterfactual 323-query future exposure
    "T3.query_bank_size": 323,
    "T3.qwen.silent": 99, "T3.qwen.exposed": 92, "T3.qwen.exposure": 92.9,
    "T3.qwen.lo": 86.9, "T3.qwen.hi": 98.0, "T3.qwen.median_diff": 8.7,
    "T3.internvl.silent": 81, "T3.internvl.exposed": 78, "T3.internvl.exposure": 96.3,
    "T3.internvl.lo": 91.4, "T3.internvl.hi": 100.0, "T3.internvl.median_diff": 8.7,
    "T3.llava.silent": 96, "T3.llava.exposed": 96, "T3.llava.exposure": 100.0,
    "T3.llava.wilson_lo": 96.2, "T3.llava.wilson_hi": 100.0, "T3.llava.median_diff": 13.6,
    "T3.pooled.silent": 276, "T3.pooled.exposed": 266, "T3.pooled.exposure": 96.4,
    "T3.pooled.lo": 94.2, "T3.pooled.hi": 98.6, "T3.pooled.median_diff": 9.9,
    "S53.exposed.unused_attribute": 133, "S53.exposed.predicate_preserving": 91,
    "S53.exposed.arithmetic_cancellation": 42,
    "S53.canonical_lower_error": 119, "S53.reverse_lower_error": 131, "S53.ties": 26,
}

MECHANISMS = {"irrelevant_attribute_only": "unused_attribute",
              "predicate_membership_unchanged": "predicate_preserving",
              "arithmetic_cancellation": "arithmetic_cancellation"}


def pct(value: float) -> float:
    return 100.0 * value


def compute(results_dir: Path, config: dict) -> tuple[dict, dict]:
    protocols = config["protocols"]
    values, details = {}, {}

    def records(protocol: str, model: str) -> list[dict]:
        rows = read_jsonl(results_dir / protocol / f"{model}.jsonl")
        return sorted(rows, key=lambda row: row["case_id"])

    # --- CLEVR confirmation (Table 2 rows 1-3, Section 5.2) ---
    conf = protocols["clevr_confirmation"]
    manifest = {row["case_id"]: row for row in read_jsonl(conf["manifest"])}
    conf_records, max_replay = {}, 0.0
    ese_total = {key: 0 for key in MECHANISMS}
    for model in conf["models"]:
        rows = conf_records[model] = records("clevr_confirmation", model)
        m = metrics.order_metrics(rows, metrics.clevr_state,
                                  seed=conf["bootstrap_seed"][model])
        m.update(metrics.clevr_change_rates(rows))
        m["ese"] = metrics.ese_decomposition(rows)
        m["state_wilson_95"] = wilson(m["state_flip_cases"], m["valid_cases"])
        m["silent_all_case_range"] = all_case_sensitivity(
            m["silent_flip_cases"], m["valid_cases"], m["cases"])
        details[f"clevr_confirmation.{model}"] = m
        max_replay = max(max_replay, m["replay_state_disagreement"])
        p = f"T2.clevr.{model}"
        values[f"{p}.valid"] = m["valid_cases"]
        values[f"{p}.state"] = pct(m["state_table_flip"])
        values[f"{p}.answer"] = pct(m["executable_answer_flip"])
        values[f"{p}.silent"] = pct(m["silent_state_flip"])
        s = f"S52.clevr.{model}"
        if model == "llava":
            values[f"{s}.state_wilson_lo"], values[f"{s}.state_wilson_hi"] = map(pct, m["state_wilson_95"])
        else:
            values[f"{s}.state_lo"], values[f"{s}.state_hi"] = map(pct, m["state_table_flip_95"])
        values[f"{s}.answer_lo"], values[f"{s}.answer_hi"] = map(pct, m["executable_answer_flip_95"])
        values[f"{s}.silent_cases"] = m["silent_flip_cases"]
        values[f"{s}.silent_allcase_lo"], values[f"{s}.silent_allcase_hi"] = map(
            pct, m["silent_all_case_range"])
        values[f"{s}.field_change"] = pct(m["attribute_field_change_rate"])
        values[f"{s}.row_change"] = pct(m["object_row_change_rate"])
        for key in MECHANISMS:
            ese_total[key] += m["ese"][key]
    for key, short in MECHANISMS.items():
        values[f"S52.ese.{short}"] = ese_total[key]

    # --- CLEVR development (Section 5.2, Figure 2A) ---
    dev = protocols["clevr_development"]
    for model in dev["models"]:
        rows = records("clevr_development", model)
        m = metrics.order_metrics(rows, metrics.clevr_state, seed=dev["bootstrap_seed"][model])
        details[f"clevr_development.{model}"] = m
        values[f"S52.dev.{model}.state"] = pct(m["state_table_flip"])
        values[f"S52.dev.{model}.answer"] = pct(m["executable_answer_flip"])
        if model == "idefics2":
            values["S52.dev.idefics2.valid_rate"] = pct(m["valid_cases"] / m["cases"])

    # --- Table 1: order-preserving nuisance control ---
    nuis = protocols["clevr_nuisance"]
    anchors = []
    for model in nuis["models"]:
        m = metrics.nuisance_control(conf_records[model], records("clevr_nuisance", model),
                                     seed=nuis["bootstrap_seed"][model])
        details[f"clevr_nuisance.{model}"] = m
        anchors.append(m["canonical_anchor_identical"] / m["common_valid"])
        p = f"T1.{model}"
        values[f"{p}.common_valid"] = m["common_valid"]
        values[f"{p}.order"] = pct(m["order_state_flip"])
        values[f"{p}.pixel"] = pct(m["pixel_state_flip"])
        values[f"{p}.relabel"] = pct(m["relabel_state_flip"])
        values[f"{p}.order_silent"] = pct(m["order_silent_flip"])
        values[f"{p}.relabel_silent"] = pct(m["relabel_silent_flip"])
        values[f"{p}.delta"] = pct(m["order_minus_worst_state"])
        values[f"{p}.delta_lo"], values[f"{p}.delta_hi"] = map(pct, m["order_minus_worst_state_95"])
    values["T1.canonical_rerun_identical_all_models"] = pct(min(anchors))

    # --- Table 3: counterfactual future-query exposure ---
    fq = protocols["clevr_future_queries"]
    bank = clevr.build_query_bank()
    values["T3.query_bank_size"] = len(bank)
    pooled = []
    for model in conf["models"]:
        cases = metrics.future_query_exposure(conf_records[model], manifest, bank)
        pooled.extend(cases)
        s = metrics.summarize_exposure(cases, fq["bootstrap_seed"], fq["bootstrap_samples"])
        s["exposure_wilson_95"] = wilson(s["exposed_cases"], s["silent_cases"])
        details[f"future_queries.{model}"] = s
        p = f"T3.{model}"
        values[f"{p}.silent"], values[f"{p}.exposed"] = s["silent_cases"], s["exposed_cases"]
        values[f"{p}.exposure"] = pct(s["exposure_rate"])
        values[f"{p}.median_diff"] = pct(s["median_differing_query_fraction"])
        if model == "llava":  # saturated row: Wilson interval
            values[f"{p}.wilson_lo"], values[f"{p}.wilson_hi"] = map(pct, s["exposure_wilson_95"])
        else:
            values[f"{p}.lo"], values[f"{p}.hi"] = map(pct, s["exposure_95_bootstrap"])
    s = metrics.summarize_exposure(pooled, fq["bootstrap_seed"] + 1000, fq["bootstrap_samples"])
    details["future_queries.pooled"] = s
    values["T3.pooled.silent"], values["T3.pooled.exposed"] = s["silent_cases"], s["exposed_cases"]
    values["T3.pooled.exposure"] = pct(s["exposure_rate"])
    values["T3.pooled.lo"], values["T3.pooled.hi"] = map(pct, s["exposure_95_bootstrap"])
    values["T3.pooled.median_diff"] = pct(s["median_differing_query_fraction"])
    for key, short in MECHANISMS.items():
        values[f"S53.exposed.{short}"] = sum(c["exposed"] for c in pooled if c["mechanism"] == key)
    values["S53.canonical_lower_error"] = s["canonical_lower_error"]
    values["S53.reverse_lower_error"] = s["reverse_lower_error"]
    values["S53.ties"] = len(pooled) - s["canonical_lower_error"] - s["reverse_lower_error"]

    # --- RAG-IGBench (Table 2 rows 4-5, Section 5.2) ---
    rag = protocols["ragig_confirmation"]
    rag_manifest = {row["case_id"]: row for row in read_jsonl(rag["manifest"])}
    alternates = []
    for model in rag["models"]:
        rows = records("ragig_confirmation", model)
        m = metrics.order_metrics(rows, dict, state_key="state_mappings",
                                  seed=rag["bootstrap_seed"][model])
        details[f"ragig_confirmation.{model}"] = m
        max_replay = max(max_replay, m["replay_state_disagreement"])
        p = f"T2.ragig.{model}"
        values[f"{p}.valid"] = m["valid_cases"]
        values[f"{p}.state"] = pct(m["state_table_flip"])
        values[f"{p}.answer"] = pct(m["executable_answer_flip"])
        values[f"{p}.silent"] = pct(m["silent_state_flip"])
        alternates.extend(metrics.ragig_alternate_targets(rows, rag_manifest))
    values["S52.ragig.silent_cases"] = len(alternates)
    values["S52.ragig.alt_answer_change"] = pct(
        sum(c["answer_change_fraction"] for c in alternates) / len(alternates))
    values["S52.ragig.alt_correctness_change"] = pct(
        sum(c["correctness_change_fraction"] for c in alternates) / len(alternates))

    # --- LVIS (Table 2 rows 6-7, Section 5.2) ---
    lvis = protocols["lvis_confirmation"]
    lvis_cases = read_jsonl(lvis["manifest"])
    for model in lvis["models"]:
        rows = records("lvis_confirmation", model)
        m = metrics.order_metrics(rows, metrics.category_state, seed=lvis["bootstrap_seed"][model])
        m.update(metrics.lvis_reuse(lvis_cases, rows, seed=lvis["reuse_bootstrap_seed"]))
        details[f"lvis_confirmation.{model}"] = m
        max_replay = max(max_replay, m["replay_state_disagreement"])
        p = f"T2.lvis.{model}"
        values[f"{p}.valid"] = m["valid_cases"]
        values[f"{p}.state"] = pct(m["state_table_flip"])
        values[f"{p}.answer"] = pct(m["executable_answer_flip"])
        values[f"{p}.silent"] = pct(m["silent_state_flip"])
        values[f"S52.lvis.{model}.changed_instance"] = pct(m["macro_scene_instance_change_rate"])
        values[f"S52.lvis.{model}.all_eight_silent"] = pct(
            m["all_eight_programs_silent_state_flip_rate"])
    values["T2.max_replay_disagreement_all_cells"] = pct(max_replay)
    return values, details


def markdown(values: dict, config: dict) -> str:
    v = lambda key: values[key]  # noqa: E731
    names = {k: spec["display"] for k, spec in config["models"].items()}
    lines = ["# PathBench main results (recomputed from raw records)", "",
             "## Table 1. Order-preserving nuisance control (state-table flip %, common valid cases)", "",
             "| Model | Common valid | Order | One pixel | ID relabel | Order silent | ID silent | Order - worst nuisance [95%] |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in ("qwen", "internvl", "llava"):
        p = f"T1.{m}"
        lines.append(f"| {names[m]} | {v(p + '.common_valid')} | {v(p + '.order'):.1f} | "
                     f"{v(p + '.pixel'):.1f} | {v(p + '.relabel'):.1f} | {v(p + '.order_silent'):.1f} | "
                     f"{v(p + '.relabel_silent'):.1f} | {v(p + '.delta'):+.1f} "
                     f"[{v(p + '.delta_lo'):.1f},{v(p + '.delta_hi'):.1f}] |")
    lines += ["", "## Table 2. Core state-integrity results (complete-valid pairs)", "",
              "| Protocol | Model | Valid | State flip (%) | Answer flip (%) | Silent flip (%) |",
              "|---|---|---:|---:|---:|---:|"]
    for proto, label, models, total in (("clevr", "CLEVR (oracle objects)", ("qwen", "internvl", "llava"), 180),
                                        ("ragig", "RAG-IGBench (full/crop)", ("qwen", "internvl"), 120),
                                        ("lvis", "LVIS (annotation instances)", ("qwen", "internvl"), 120)):
        for m in models:
            p = f"T2.{proto}.{m}"
            lines.append(f"| {label} | {names[m]} | {v(p + '.valid')}/{total} | {v(p + '.state'):.1f} | "
                         f"{v(p + '.answer'):.1f} | {v(p + '.silent'):.1f} |")
    lines += [f"", f"Replay disagreement, max over all seven cells: {v('T2.max_replay_disagreement_all_cells'):.1f}%",
              "", "## Table 3. Counterfactual future-query exposure (323-query bank)", "",
              "| Model | Silent | Exposed | Exposure % [95%] | Median diff. |", "|---|---:|---:|---:|---:|"]
    for m in ("qwen", "internvl", "llava", "pooled"):
        p = f"T3.{m}"
        lo, hi = ((v(p + ".wilson_lo"), v(p + ".wilson_hi")) if m == "llava"
                  else (v(p + ".lo"), v(p + ".hi")))
        label = names.get(m, "pooled")
        lines.append(f"| {label} | {v(p + '.silent')} | {v(p + '.exposed')} | "
                     f"{v(p + '.exposure'):.1f} [{lo:.1f},{hi:.1f}] | {v(p + '.median_diff'):.1f}% |")
    lines += ["", "## Linked in-text numbers", ""]
    lines += [f"- `{key}`: {value:.1f}" if isinstance(value, float) else f"- `{key}`: {value}"
              for key, value in sorted(values.items()) if key.startswith("S5")]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/frozen"),
                        help="directory of raw records, <protocol>/<model>.jsonl (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproduced"),
                        help="where main_results.{md,json} are written (default: %(default)s)")
    parser.add_argument("--no-check", action="store_true",
                        help="score without comparing against the paper values")
    args = parser.parse_args()
    config = load_config()
    values, details = compute(resolve(args.results), config)
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "main_results.json").write_text(json.dumps(
        {"values": values, "details": details}, indent=2, sort_keys=True) + "\n")
    report = markdown(values, config)
    (output / "main_results.md").write_text(report)
    print(report)
    if args.no_check:
        return 0
    mismatches = [(key, expected, values.get(key)) for key, expected in EXPECTED.items()
                  if values.get(key) is None or abs(round(values[key], 1) - expected) > 1e-6]
    print(f"Checked {len(EXPECTED)} paper values: {len(EXPECTED) - len(mismatches)} match, "
          f"{len(mismatches)} differ.")
    for key, expected, got in mismatches:
        print(f"  MISMATCH {key}: paper={expected} recomputed={got}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
