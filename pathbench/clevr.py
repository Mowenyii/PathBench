"""CLEVR typed-state protocol: prompt, strict parser, question compiler, executor.

The question compiler reads only the natural-language question; CLEVR's
ground-truth functional program and object attributes are never shown to a model.
"""

from __future__ import annotations

import itertools
import re

FIELDS = ("color", "shape", "size", "material")
COLORS = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")
SHAPES = ("cube", "sphere", "cylinder")
SIZES = ("small", "large")
MATERIALS = ("rubber", "metal")
VOCABULARY = {"color": COLORS, "shape": SHAPES, "size": SIZES, "material": MATERIALS}

SYSTEM = (
    "You are a visual state recorder. Tool observations are object-centered crops "
    "from one CLEVR scene, and each opaque observation ID appears exactly once. "
    "Preserve every ID and visible object attribute. Do not answer an arithmetic question."
)

TOKEN_MAP = {
    **{value: ("color", value) for value in COLORS},
    "cube": ("shape", "cube"), "cubes": ("shape", "cube"),
    "block": ("shape", "cube"), "blocks": ("shape", "cube"),
    "sphere": ("shape", "sphere"), "spheres": ("shape", "sphere"),
    "ball": ("shape", "sphere"), "balls": ("shape", "sphere"),
    "cylinder": ("shape", "cylinder"), "cylinders": ("shape", "cylinder"),
    "small": ("size", "small"), "tiny": ("size", "small"),
    "large": ("size", "large"), "big": ("size", "large"),
    "rubber": ("material", "rubber"), "matte": ("material", "rubber"),
    "metal": ("material", "metal"), "metallic": ("material", "metal"),
    "shiny": ("material", "metal"),
}
GENERIC_NOUNS = {"object", "objects", "thing", "things"}


def final_prompt(observation_ids: list[str]) -> str:
    required = ", ".join(sorted(observation_ids))
    return (
        "Return one line for every provided observation and no other text. "
        f"The required exact IDs are: {required}. Replace <ID> with one of those "
        "exact IDs and use each exactly once. Format: <ID>: "
        "color=gray/red/blue/green/brown/purple/cyan/yellow; "
        "shape=cube/sphere/cylinder; size=small/large; material=rubber/metal. "
        "Use only these literal words; never use synonyms such as gold."
    )


def parse_attributes(raw: str) -> dict:
    result = {}
    for key, values in VOCABULARY.items():
        pattern = rf"{key}\s*[:=]\s*({'|'.join(values)})\b"
        match = re.search(pattern, raw.lower())
        result[key] = match.group(1) if match else "unknown"
    return result


def parse_state(raw: str, expected_ids: set[str]) -> tuple[list[dict], str | None]:
    """Strictly parse one complete typed table; any gap invalidates the condition."""
    found = {}
    for line in raw.splitlines():
        match = re.search(r"(OBS-[0-9a-f]{10})\s*:", line, re.IGNORECASE)
        if not match:
            continue
        token = match.group(1).upper()
        canonical = next((value for value in expected_ids if value.upper() == token), None)
        if canonical is None:
            continue
        if canonical in found:
            return [], f"duplicate observation ID: {canonical}"
        # The only frozen alias: "gold" is mapped to the CLEVR color "yellow".
        normalized_line = re.sub(r"\bcolor\s*=\s*gold\b", "color=yellow",
                                 line, flags=re.IGNORECASE)
        attributes = parse_attributes(normalized_line)
        if any(attributes[field] == "unknown" for field in FIELDS):
            return [], f"incomplete attributes: {canonical}"
        found[canonical] = {"observation_id": canonical, **attributes}
    missing = expected_ids - set(found)
    if missing:
        return [], f"missing observation IDs: {sorted(missing)}"
    return [found[token] for token in sorted(found)], None


def parse_predicate(phrase: str) -> dict:
    """Map CLEVR-Math surface synonyms to a canonical attribute predicate."""
    predicate = {}
    tokens = re.findall(r"[a-z]+", phrase.lower())
    if not tokens:
        raise ValueError("empty predicate phrase")
    for token in tokens:
        if token in GENERIC_NOUNS:
            continue
        if token not in TOKEN_MAP:
            raise ValueError(f"unsupported predicate token: {token}")
        key, value = TOKEN_MAP[token]
        if key in predicate and predicate[key] != value:
            raise ValueError(f"conflicting {key} predicate")
        predicate[key] = value
    return predicate


def _count(predicate: dict) -> dict:
    return {"op": "count", "predicate": predicate}


def _predicate_contains(container: dict, subset: dict) -> bool:
    return all(container.get(key) == value for key, value in subset.items())


def compile_question(question: str) -> dict:
    """Compile the three CLEVR-Math arithmetic template families."""
    addition = re.fullmatch(
        r"Add (\d+) (.+)\. How many (.+) (?:exist|are left)\?", question)
    if addition:
        amount = int(addition.group(1))
        mutation = parse_predicate(addition.group(2))
        target = parse_predicate(addition.group(3))
        if target and mutation != target:
            raise ValueError("addition mutation/target predicate mismatch")
        return {"op": "add", "left": _count(target),
                "right": {"op": "constant", "value": amount},
                "mutation_predicate": mutation}

    multihop = re.fullmatch(
        r"Subtract all (.+)\. Subtract all (.+)\. How many objects are left\?",
        question)
    if multihop:
        first = parse_predicate(multihop.group(1))
        second = parse_predicate(multihop.group(2))
        return {"op": "subtract",
                "left": {"op": "subtract", "left": _count({}), "right": _count(first)},
                "right": _count(second)}

    subtraction = re.fullmatch(
        r"Subtract (all|\d+) (.+)\. How many (.+) are left\?", question)
    if subtraction:
        amount, removal_phrase, target_phrase = subtraction.groups()
        removal = parse_predicate(removal_phrase)
        target = parse_predicate(target_phrase)
        if target and not _predicate_contains(removal, target):
            raise ValueError("subtraction removal is not contained in target")
        right = (_count(removal) if amount == "all" else
                 {"op": "constant", "value": int(amount)})
        return {"op": "subtract", "left": _count(target), "right": right,
                "removal_predicate": removal}

    raise ValueError(f"unsupported question grammar: {question}")


def execute_program(program: dict, objects: list[dict]) -> int:
    op = program["op"]
    if op == "constant":
        return int(program["value"])
    if op == "count":
        predicate = program["predicate"]
        return sum(all(obj.get(key) == value for key, value in predicate.items())
                   for obj in objects)
    if op == "add":
        return execute_program(program["left"], objects) + execute_program(
            program["right"], objects)
    if op == "subtract":
        return execute_program(program["left"], objects) - execute_program(
            program["right"], objects)
    raise ValueError(f"unsupported executable op: {op}")


# --- Executable-state equivalence (ESE) ------------------------------------

def linear_terms(program: dict, sign: int = 1) -> tuple[int, list[tuple[int, dict]]]:
    """Express the arithmetic grammar as a constant plus signed predicate counts."""
    op = program["op"]
    if op == "constant":
        return sign * int(program["value"]), []
    if op == "count":
        return 0, [(sign, program["predicate"])]
    if op in ("add", "subtract"):
        left_constant, left_terms = linear_terms(program["left"], sign)
        right_sign = sign if op == "add" else -sign
        right_constant, right_terms = linear_terms(program["right"], right_sign)
        return left_constant + right_constant, left_terms + right_terms
    raise ValueError(f"unsupported executable op: {op}")


def contribution(attributes: dict, terms: list[tuple[int, dict]]) -> int:
    return sum(sign * int(all(attributes.get(key) == value
                              for key, value in predicate.items()))
               for sign, predicate in terms)


def candidate_attributes(left: dict, right: dict):
    values = [sorted({left[field], right[field]}) for field in FIELDS]
    for combination in itertools.product(*values):
        yield dict(zip(FIELDS, combination))


def answer_closure(program: dict, canonical: dict[str, dict],
                   reverse: dict[str, dict]) -> tuple[set[int], dict[str, set[int]]]:
    """All answers reachable when each field may take either order's value."""
    constant, terms = linear_terms(program)
    possible = {constant}
    per_object = {}
    for observation_id in sorted(canonical):
        values = {contribution(attributes, terms) for attributes in
                  candidate_attributes(canonical[observation_id],
                                       reverse[observation_id])}
        per_object[observation_id] = values
        possible = {left + right for left in possible for right in values}
    return possible, per_object


def classify_silent(question: str, canonical: dict[str, dict],
                    reverse: dict[str, dict]) -> str:
    """ESE mechanism of an answer-silent state flip."""
    program = compile_question(question)
    _, terms = linear_terms(program)
    _, per_object = answer_closure(program, canonical, reverse)
    used_fields = set().union(*(predicate for _, predicate in terms))
    changed_fields = [field for observation_id in canonical for field in FIELDS
                      if canonical[observation_id][field] != reverse[observation_id][field]]
    if all(field not in used_fields for field in changed_fields):
        return "irrelevant_attribute_only"
    if all(len(values) == 1 for values in per_object.values()):
        return "predicate_membership_unchanged"
    return "arithmetic_cancellation"


def build_query_bank() -> list[dict]:
    """All 323 nonempty attribute-conjunction count predicates."""
    bank = []
    for width in range(1, len(FIELDS) + 1):
        for selected in itertools.combinations(FIELDS, width):
            for values in itertools.product(*(VOCABULARY[field] for field in selected)):
                bank.append(dict(zip(selected, values)))
    return bank
