"""Protocol definitions: how each case is serialized, parsed, and executed.

Each protocol yields one raw record per case with the same schema as the frozen
records under ``results/frozen`` so that ``scripts/reproduce_main.py`` can score
either the frozen or a regenerated run.
"""

from __future__ import annotations

import re
from collections import Counter

from . import clevr
from .common import read_jsonl

ORDER_CONDITIONS = ("canonical", "reverse", "canonical_replay")
NUISANCE_CONDITIONS = ("canonical", "nuisance_pixel", "nuisance_relabel")


# --- CLEVR typed state (order) ---------------------------------------------

class ClevrOrder:
    name = "clevr_confirmation"
    conditions = ORDER_CONDITIONS

    def __init__(self, protocol: dict, model: str):
        self.protocol = protocol

    def cases(self) -> list[dict]:
        return sorted(read_jsonl(self.protocol["manifest"]), key=lambda row: row["case_id"])

    def call(self, model, case: dict, condition: str) -> tuple[str, list[dict], str | None, int | None]:
        by_id = {item["observation_id"]: item for item in case["observations"]}
        turns = [(f"Tool observation {token}:", by_id[token]["crop_path"])
                 for token in case["condition_orders"][condition]]
        raw = model.history_chat(clevr.SYSTEM, turns, clevr.final_prompt(list(by_id)))
        entries, error = clevr.parse_state(raw, set(by_id))
        prediction = (clevr.execute_program(clevr.compile_question(case["question"]), entries)
                      if error is None else None)
        return raw, entries, error, prediction

    def record(self, case: dict) -> dict:
        return {"case_id": case["case_id"], "template": case["template"],
                "question": case["question"], "answer": case["answer"]}


class ClevrNuisance(ClevrOrder):
    """Order-preserving one-pixel and rank-aligned opaque-ID relabel conditions."""
    name = "clevr_nuisance"
    conditions = NUISANCE_CONDITIONS

    def call(self, model, case: dict, condition: str):
        view = case["conditions"][condition]
        by_id = {item["observation_id"]: item for item in view["observations"]}
        turns = [(f"Tool observation {token}:", by_id[token]["crop_path"])
                 for token in view["observation_order"]]
        raw = model.history_chat(clevr.SYSTEM, turns, clevr.final_prompt(list(by_id)))
        entries, error = clevr.parse_state(raw, set(by_id))
        if error is not None:
            return raw, [], error, None
        # Rewrite rows onto canonical observation IDs for cross-condition joins.
        back = {item["observation_id"]: item["canonical_observation_id"]
                for item in view["observations"]}
        entries = sorted(({**entry, "observation_id": back[entry["observation_id"]]}
                          for entry in entries), key=lambda entry: entry["observation_id"])
        prediction = clevr.execute_program(clevr.compile_question(case["question"]), entries)
        return raw, entries, None, prediction


# --- RAG-IGBench full/crop source binding ----------------------------------

def ragig_final_prompt(anchor_ids: list[str], query_ids: list[str]) -> str:
    return (
        "Match every local query crop to the full-image anchor from which it was cut. "
        f"Exact anchor IDs: {', '.join(anchor_ids)}. Exact query IDs: {', '.join(query_ids)}. "
        "Return one line per query and no other text. Use each query exactly once. "
        "Format: <QUERY-ID>: anchor=<ANCHOR-ID>. Use only the listed exact IDs."
    )


RAGIG_SYSTEM = ("You are a cross-view identity recorder. Preserve all opaque IDs and "
                "match each local crop to its source full-image anchor.")


def ragig_parse_mapping(raw: str, query_ids: set[str],
                        anchor_ids: set[str]) -> tuple[dict[str, str], str | None]:
    """Parse ``QUERY: anchor=ANCHOR`` rows; literal template brackets are tolerated."""
    queries_upper = {value.upper(): value for value in query_ids}
    anchors_upper = {value.upper(): value for value in anchor_ids}
    found = {}
    for line in raw.splitlines():
        match = re.search(
            r"<?(QUERY-[0-9a-f]{10})>?\s*:\s*"
            r"(?:anchor\s*=\s*)?<?(ANCHOR-[0-9a-f]{10})>?\s*\.?\s*$",
            line.strip(), re.IGNORECASE)
        if not match:
            continue
        query = queries_upper.get(match.group(1).upper())
        anchor = anchors_upper.get(match.group(2).upper())
        if query is None or anchor is None:
            continue
        if query in found:
            return {}, f"duplicate query ID: {query}"
        found[query] = anchor
    missing = query_ids - set(found)
    if missing:
        return {}, f"missing query IDs: {sorted(missing)}"
    return dict(sorted(found.items())), None


def ragig_execute(query_id: str, mapping: dict[str, str], anchor_ids: list[str]) -> int:
    """Program: 1-based rank of the anchor assigned to the selected query."""
    return sorted(anchor_ids).index(mapping[query_id]) + 1


class RagigIdentity:
    name = "ragig_confirmation"
    conditions = ORDER_CONDITIONS

    def __init__(self, protocol: dict, model: str):
        self.protocol = protocol

    def cases(self) -> list[dict]:
        return read_jsonl(self.protocol["manifest"])

    def call(self, model, case: dict, condition: str):
        by_id = {row["observation_id"]: row for row in case["observations"]}
        turns = [(f"{by_id[token]['role'].title()} observation {token}:",
                  by_id[token]["view_path"]) for token in case["condition_orders"][condition]]
        raw = model.history_chat(RAGIG_SYSTEM, turns,
                                 ragig_final_prompt(case["anchor_ids"], case["query_ids"]))
        mapping, error = ragig_parse_mapping(raw, set(case["query_ids"]), set(case["anchor_ids"]))
        prediction = (ragig_execute(case["question_spec"]["query_id"], mapping, case["anchor_ids"])
                      if error is None else None)
        return raw, mapping, error, prediction

    def record(self, case: dict) -> dict:
        return {"case_id": case["case_id"], "split": case["split"],
                "source_record_id": case["source_record_id"], "answer": case["answer"],
                "question_spec": case["question_spec"]}


# --- LVIS/COCO category state ----------------------------------------------

LVIS_SYSTEM = ("You are a visual category-state recorder. Store each opaque-ID crop, "
               "then produce the complete state table. Do not count or do arithmetic.")


def lvis_final_prompt(ids: list[str], vocabulary: list[str]) -> str:
    return (
        "Return one line for every provided observation and no other text. "
        f"Required exact IDs: {', '.join(sorted(ids))}. Use each ID exactly once. "
        "Format: <ID>: category=<label>. The label must be exactly one of: "
        + "; ".join(vocabulary) + ". Classify the designated central annotation "
        "instance in each crop; ignore incidental background objects."
    )


def lvis_parse_state(raw: str, expected_ids: set[str],
                     vocabulary: list[str]) -> tuple[list[dict], str | None]:
    allowed = {value.casefold(): value for value in vocabulary}
    found = {}
    for line in raw.splitlines():
        match = re.search(r"(OBS-[0-9a-f]{10})\s*:\s*(?:category\s*=\s*)?(.+?)\s*$",
                          line.strip(), re.IGNORECASE)
        if not match:
            continue
        visible = match.group(1).upper()
        token = next((value for value in expected_ids if value.upper() == visible), None)
        if token is None:
            continue
        if token in found:
            return [], f"duplicate observation ID: {token}"
        label = re.sub(r"\.$", "", match.group(2).strip()).strip().casefold()
        if label not in allowed:
            return [], f"out-of-vocabulary category: {match.group(2).strip()}"
        found[token] = {"observation_id": token, "category": allowed[label]}
    missing = expected_ids - set(found)
    if missing:
        return [], f"missing observation IDs: {sorted(missing)}"
    return [found[token] for token in sorted(found)], None


def lvis_execute(spec: dict, entries: list[dict]) -> int:
    counts = Counter(row["category"] for row in entries)
    values = [counts[name] for name in spec["categories"]]
    if spec["operator"] == "count":
        return values[0]
    if spec["operator"] == "sum":
        return sum(values)
    if spec["operator"] == "absolute_difference":
        return abs(values[0] - values[1])
    raise ValueError(f"unsupported operator: {spec['operator']}")


def lvis_codebook(vocabulary: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    width = max(2, len(str(len(vocabulary) - 1)))
    to_code = {category: f"C{index:0{width}d}" for index, category in enumerate(vocabulary)}
    return to_code, {code: category for category, code in to_code.items()}


def lvis_code_final_prompt(ids: list[str], vocabulary: list[str]) -> str:
    """Closed-set category-code interface (frozen for InternVL after its dry run)."""
    to_code, _ = lvis_codebook(vocabulary)
    mapping = "; ".join(f"{code}={category}" for category, code in to_code.items())
    return (
        f"There are exactly {len(ids)} stored observations. Return exactly "
        f"{len(ids)} lines and no other text. Required IDs in sorted order: "
        f"{', '.join(sorted(ids))}. Format each line exactly as <ID>: <CODE>. "
        f"Use each required ID once. Valid category codes are: {mapping}. "
        "Classify the designated central annotation instance in each crop and "
        "ignore incidental background objects."
    )


def lvis_code_parse_state(raw: str, expected_ids: set[str],
                          vocabulary: list[str]) -> tuple[list[dict], str | None]:
    """Accept ``Cxx`` or ``Cxx=<exact mapped category>``; anything else is invalid."""
    _, to_category = lvis_codebook(vocabulary)
    expected_upper = {value.upper(): value for value in expected_ids}
    found: dict[str, dict] = {}
    for line in raw.splitlines():
        match = re.fullmatch(r"\s*(OBS-[0-9a-f]{10})\s*:\s*(C\d{2})"
                             r"(?:\s*=\s*([^\n]+?))?\s*\.?\s*", line, re.IGNORECASE)
        if not match:
            continue
        token = expected_upper.get(match.group(1).upper())
        code = match.group(2).upper()
        if token is None:
            continue
        if token in found:
            return [], f"duplicate observation ID: {token}"
        if code not in to_category:
            return [], f"out-of-vocabulary category code: {code}"
        category = to_category[code]
        suffix = match.group(3)
        if suffix is not None and re.sub(r"\.$", "", suffix.strip()).casefold() != category.casefold():
            return [], f"category-code suffix mismatch: {code}={suffix.strip()}"
        found[token] = {"observation_id": token, "category": category, "category_code": code}
    missing = expected_ids - set(found)
    if missing:
        return [], f"missing observation IDs: {sorted(missing)}"
    return [found[token] for token in sorted(found)], None


LVIS_INTERFACES = {"label": (lvis_final_prompt, lvis_parse_state),
                   "code": (lvis_code_final_prompt, lvis_code_parse_state)}


class LvisCategory:
    name = "lvis_confirmation"
    conditions = ORDER_CONDITIONS

    def __init__(self, protocol: dict, model: str):
        self.protocol = protocol
        self.vocabulary = protocol["category_vocabulary"]
        self.prompt, self.parse = LVIS_INTERFACES[protocol["state_interface"][model]]

    def cases(self) -> list[dict]:
        return read_jsonl(self.protocol["manifest"])

    def call(self, model, case: dict, condition: str):
        by_id = {row["observation_id"]: row for row in case["observations"]}
        turns = [(f"Tool observation {token}:", by_id[token]["crop_path"])
                 for token in case["condition_orders"][condition]]
        raw = model.history_chat(LVIS_SYSTEM, turns, self.prompt(list(by_id), self.vocabulary))
        entries, error = self.parse(raw, set(by_id), self.vocabulary)
        prediction = lvis_execute(case["question_spec"], entries) if error is None else None
        return raw, entries, error, prediction

    def record(self, case: dict) -> dict:
        return {"case_id": case["case_id"], "split": case["split"],
                "image_id": case["image_id"], "answer": case["answer"],
                "question_spec": case["question_spec"]}


PROTOCOLS = {cls.name: cls for cls in (ClevrOrder, ClevrNuisance, RagigIdentity, LvisCategory)}
# RAG-IGBench records store parsed state under "state_mappings"; others under "state_entries".
STATE_KEY = {"ragig_confirmation": "state_mappings"}
