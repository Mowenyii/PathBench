"""Model-free scoring of raw PathBench records.

All rates condition on complete-valid cases (every condition parsed into a
complete table). States are compared after sorting rows by opaque observation ID.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from . import clevr
from .common import percentile, quantile

FIELDS = clevr.FIELDS


def clevr_state(entries: list[dict]) -> dict[str, dict]:
    return {entry["observation_id"]: {field: entry[field] for field in FIELDS}
            for entry in entries}


def category_state(entries: list[dict]) -> dict[str, str]:
    return {row["observation_id"]: row["category"] for row in entries}


def _valid(records: list[dict]) -> list[dict]:
    return [row for row in records if row["status"] == "complete_valid"]


# --- Generic order metrics --------------------------------------------------

def order_metrics(records: list[dict], state_fn, state_key: str = "state_entries",
                  seed: int | None = None, resamples: int = 10000) -> dict:
    """State-table flip, executable-answer flip, silent flip, replay disagreement."""
    valid = _valid(records)
    state_flags, answer_flags, replay_flags = [], [], []
    for row in valid:
        states = {c: state_fn(row[state_key][c]) for c in ("canonical", "reverse", "canonical_replay")}
        state_flags.append(states["canonical"] != states["reverse"])
        replay_flags.append(states["canonical"] != states["canonical_replay"])
        answer_flags.append(row["executable_predictions"]["canonical"] !=
                            row["executable_predictions"]["reverse"])
    n = len(valid)
    state, answer = sum(state_flags), sum(answer_flags)
    report = {
        "cases": len(records), "valid_cases": n,
        "state_flip_cases": state, "answer_flip_cases": answer,
        "silent_flip_cases": state - answer,
        "state_table_flip": state / n, "executable_answer_flip": answer / n,
        "silent_state_flip": (state - answer) / n,
        "replay_state_disagreement": sum(replay_flags) / n,
    }
    if seed is not None:
        # Case bootstrap over complete-valid cases, frozen seed per protocol/model.
        rng = random.Random(seed)
        state_rates, answer_rates = [], []
        for _ in range(resamples):
            picks = [rng.randrange(n) for _ in range(n)]
            state_rates.append(sum(state_flags[i] for i in picks) / n)
            answer_rates.append(sum(answer_flags[i] for i in picks) / n)
        report["state_table_flip_95"] = [quantile(state_rates, .025), quantile(state_rates, .975)]
        report["executable_answer_flip_95"] = [quantile(answer_rates, .025),
                                               quantile(answer_rates, .975)]
    return report


def clevr_change_rates(records: list[dict]) -> dict:
    """Fraction of attribute fields / object rows that differ between orders."""
    changed_objects = changed_fields = total_objects = 0
    for row in _valid(records):
        canonical = clevr_state(row["state_entries"]["canonical"])
        reverse = clevr_state(row["state_entries"]["reverse"])
        for token in canonical:
            differences = [canonical[token][f] != reverse[token][f] for f in FIELDS]
            changed_objects += any(differences)
            changed_fields += sum(differences)
            total_objects += 1
    return {"object_row_change_rate": changed_objects / total_objects,
            "attribute_field_change_rate": changed_fields / (total_objects * len(FIELDS))}


# --- CLEVR executable-state equivalence -------------------------------------

def ese_decomposition(records: list[dict]) -> dict:
    counts = Counter()
    for row in _valid(records):
        canonical = clevr_state(row["state_entries"]["canonical"])
        reverse = clevr_state(row["state_entries"]["reverse"])
        if canonical == reverse:
            continue
        if row["executable_predictions"]["canonical"] != row["executable_predictions"]["reverse"]:
            continue
        counts[clevr.classify_silent(row["question"], canonical, reverse)] += 1
    return {key: counts[key] for key in ("irrelevant_attribute_only",
                                         "predicate_membership_unchanged",
                                         "arithmetic_cancellation")}


# --- CLEVR counterfactual future-query exposure ------------------------------

def _count_vector(state: dict[str, dict], bank: list[dict]) -> list[int]:
    return [sum(all(attributes[field] == value for field, value in predicate.items())
                for attributes in state.values()) for predicate in bank]


def _bootstrap_rate(flags: list, seed: int, samples: int) -> list[float] | None:
    if not flags:
        return None
    rng = random.Random(seed)
    estimates = [sum(rng.choice(flags) for _ in flags) / len(flags) for _ in range(samples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def future_query_exposure(records: list[dict], manifest: dict[str, dict],
                          bank: list[dict]) -> list[dict]:
    """For every answer-silent case, compare both states on the fixed query bank."""
    cases = []
    for row in _valid(records):
        canonical = clevr_state(row["state_entries"]["canonical"])
        reverse = clevr_state(row["state_entries"]["reverse"])
        if canonical == reverse or (row["executable_predictions"]["canonical"] !=
                                    row["executable_predictions"]["reverse"]):
            continue
        truth = {item["observation_id"]: item["ground_truth_attributes"]
                 for item in manifest[row["case_id"]]["observations"]}
        cv, rv, tv = (_count_vector(s, bank) for s in (canonical, reverse, truth))
        differences = [a != b for a, b in zip(cv, rv)]
        cases.append({
            "case_id": row["case_id"],
            "mechanism": clevr.classify_silent(row["question"], canonical, reverse),
            "exposed": any(differences),
            "differing_query_fraction": sum(differences) / len(bank),
            "canonical_total_error": sum(abs(a - b) for a, b in zip(cv, tv)),
            "reverse_total_error": sum(abs(a - b) for a, b in zip(rv, tv)),
        })
    return cases


def summarize_exposure(cases: list[dict], seed: int, samples: int) -> dict:
    exposed = [case["exposed"] for case in cases]
    return {
        "silent_cases": len(cases), "exposed_cases": sum(exposed),
        "exposure_rate": sum(exposed) / len(cases),
        "exposure_95_bootstrap": _bootstrap_rate(exposed, seed, samples),
        "median_differing_query_fraction":
            percentile([c["differing_query_fraction"] for c in cases], 0.5),
        "canonical_lower_error": sum(c["canonical_total_error"] < c["reverse_total_error"]
                                     for c in cases),
        "reverse_lower_error": sum(c["reverse_total_error"] < c["canonical_total_error"]
                                   for c in cases),
    }


# --- Order-preserving nuisance control ----------------------------------------

def _indicators(record: dict, left: str, right: str) -> dict[str, int]:
    a = clevr_state(record["state_entries"][left])
    b = clevr_state(record["state_entries"][right])
    state_flip = int(a != b)
    answer_flip = int(record["executable_predictions"][left] !=
                      record["executable_predictions"][right])
    return {"state_flip": state_flip, "answer_flip": answer_flip,
            "silent_flip": int(state_flip == 1 and answer_flip == 0)}


def _exact_mcnemar(only_a: int, only_b: int) -> float:
    total = only_a + only_b
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(only_a, only_b) + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def _paired_difference(order: list[dict], nuisance: list[dict], metric: str,
                       seed: int, resamples: int) -> dict:
    size = len(order)
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        picks = [rng.randrange(size) for _ in range(size)]
        samples.append((sum(order[i][metric] for i in picks) -
                        sum(nuisance[i][metric] for i in picks)) / size)
    only_order = sum(1 for a, b in zip(order, nuisance) if a[metric] and not b[metric])
    only_nuisance = sum(1 for a, b in zip(order, nuisance) if b[metric] and not a[metric])
    return {"difference": (sum(r[metric] for r in order) - sum(r[metric] for r in nuisance)) / size,
            "difference_95": [quantile(samples, .025), quantile(samples, .975)],
            "exact_mcnemar_p": _exact_mcnemar(only_order, only_nuisance)}


def nuisance_control(order_records: list[dict], nuisance_records: list[dict],
                     seed: int, resamples: int = 10000,
                     metrics: tuple[str, ...] = ("state_flip", "answer_flip", "silent_flip")) -> dict:
    """Compare canonical-vs-reverse against canonical-vs-nuisance on common valid cases."""
    order_by = {row["case_id"]: row for row in order_records}
    nuis_by = {row["case_id"]: row for row in nuisance_records}
    common = sorted(c for c in order_by if order_by[c]["status"] == "complete_valid"
                    and nuis_by[c]["status"] == "complete_valid")
    anchor = sum(clevr_state(order_by[c]["state_entries"]["canonical"]) ==
                 clevr_state(nuis_by[c]["state_entries"]["canonical"]) for c in common)
    order_rows = [_indicators(order_by[c], "canonical", "reverse") for c in common]
    nuisance_rows = {cond: [_indicators(nuis_by[c], "canonical", cond) for c in common]
                     for cond in ("nuisance_pixel", "nuisance_relabel")}
    rate = lambda rows, m: sum(r[m] for r in rows) / len(rows)  # noqa: E731
    comparisons = {cond: {m: _paired_difference(order_rows, rows, m, seed + offset, resamples)
                          for offset, m in enumerate(metrics)}
                   for cond, rows in nuisance_rows.items()}
    worst = max(nuisance_rows, key=lambda cond: sum(r["state_flip"] for r in nuisance_rows[cond]))
    return {
        "common_valid": len(common), "canonical_anchor_identical": anchor,
        "order_state_flip": rate(order_rows, "state_flip"),
        "order_silent_flip": rate(order_rows, "silent_flip"),
        "pixel_state_flip": rate(nuisance_rows["nuisance_pixel"], "state_flip"),
        "relabel_state_flip": rate(nuisance_rows["nuisance_relabel"], "state_flip"),
        "pixel_silent_flip": rate(nuisance_rows["nuisance_pixel"], "silent_flip"),
        "relabel_silent_flip": rate(nuisance_rows["nuisance_relabel"], "silent_flip"),
        "worst_nuisance": worst,
        "order_minus_worst_state": comparisons[worst]["state_flip"]["difference"],
        "order_minus_worst_state_95": comparisons[worst]["state_flip"]["difference_95"],
        "order_minus_worst_state_mcnemar_p": comparisons[worst]["state_flip"]["exact_mcnemar_p"],
        "paired_comparisons": comparisons,
    }


# --- RAG-IGBench alternate-target reuse ---------------------------------------

def ragig_alternate_targets(records: list[dict], manifest: dict[str, dict]) -> list[dict]:
    """In answer-silent cases, evaluate the two query targets not used by the question."""
    cases = []
    for row in _valid(records):
        canonical, reverse = row["state_mappings"]["canonical"], row["state_mappings"]["reverse"]
        if canonical == reverse or (row["executable_predictions"]["canonical"] !=
                                    row["executable_predictions"]["reverse"]):
            continue
        source = manifest[row["case_id"]]
        truth = source["ground_truth_mapping"]
        alternatives = [q for q in source["query_ids"] if q != row["question_spec"]["query_id"]]
        answer_changes = [canonical[q] != reverse[q] for q in alternatives]
        correctness_changes = [(canonical[q] == truth[q]) != (reverse[q] == truth[q])
                               for q in alternatives]
        cases.append({"case_id": row["case_id"],
                      "answer_change_fraction": sum(answer_changes) / len(alternatives),
                      "correctness_change_fraction": sum(correctness_changes) / len(alternatives)})
    return cases


# --- LVIS changed-instance fraction and eight reuse programs -----------------

def _program_values(state: dict[str, str], truth: dict[str, str]) -> dict[str, object]:
    true_counts, predicted = Counter(truth.values()), Counter(state.values())
    ranked = sorted(true_counts, key=lambda key: (-true_counts[key], key))
    first = ranked[0]
    second = ranked[1] if len(ranked) > 1 else ranked[0]
    p1, p2 = predicted[first], predicted[second]
    return {"count_primary": p1, "count_secondary": p2, "sum": p1 + p2,
            "absolute_difference": abs(p1 - p2),
            "comparison": 1 if p1 > p2 else (-1 if p1 < p2 else 0),
            "primary_parity": p1 % 2, "primary_at_least_two": p1 >= 2,
            "category_diversity": len(predicted)}


def lvis_reuse(cases: list[dict], records: list[dict], seed: int = 20261005,
               resamples: int = 10000) -> dict:
    by_case = {row["case_id"]: row for row in records}
    valid = [c for c in cases if by_case[c["case_id"]]["status"] == "complete_valid"]
    scene_change, suite_silent = [], 0
    program_correct = defaultdict(int)
    for case in valid:
        record = by_case[case["case_id"]]
        truth = {row["observation_id"]: row["ground_truth_category"] for row in case["observations"]}
        canonical = category_state(record["state_entries"]["canonical"])
        reverse = category_state(record["state_entries"]["reverse"])
        changed = sum(canonical[key] != reverse[key] for key in truth)
        scene_change.append(changed / len(truth))
        out_c, out_r = _program_values(canonical, truth), _program_values(reverse, truth)
        expected = _program_values(truth, truth)
        for name in expected:
            program_correct[name] += out_c[name] == expected[name]
        suite_silent += changed > 0 and all(out_c[k] == out_r[k] for k in out_c)
    n = len(valid)
    rng = random.Random(seed)
    macro = sorted(sum(scene_change[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(resamples))
    return {
        "macro_scene_instance_change_rate": sum(scene_change) / n,
        "macro_scene_instance_change_95": [macro[int(.025 * resamples)],
                                           macro[int(.975 * resamples)]],
        "all_eight_programs_silent_state_flip_rate": suite_silent / n,
        "canonical_eight_program_accuracy": sum(program_correct.values()) / (8 * n),
    }
