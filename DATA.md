# PathBench data format

All files are JSON Lines (one case per line). Paths inside files are relative to
the repository root.

## Directory overview

| Directory | Cases | Images included? | Contents |
|---|---:|---|---|
| `data/clevr_development/` | 72 | Yes | Development cases and CLEVR crops |
| `data/clevr_confirmation/` | 180 | Yes (957 crops) | CLEVR state-integrity cases |
| `data/clevr_nuisance/` | 180 | Yes (957 crops) | One-pixel and ID-relabeling controls |
| `data/ragig_confirmation/` | 120 | No; build locally | RAG-IGBench source-binding cases |
| `data/lvis_confirmation/` | 120 | No; build locally | LVIS category-state cases |
| `results/frozen/<protocol>/<model>.jsonl` | One row per case | — | Bundled raw model outputs |

## Case manifests: `data/<protocol>/manifest.jsonl`

Fields shared by all protocols:

| Field | Meaning |
|---|---|
| `case_id` | stable case identifier, e.g. `pathguard-001` |
| `answer` | ground-truth integer answer to the current question |
| `observations` | list of observations; each has an opaque `observation_id` and an image path |
| `condition_orders` | observation-ID order for each condition: `canonical`, `reverse`, `canonical_replay` (identical to `canonical`) |
| `selection_rank_sha256` | deterministic selection rank; selection never used model outputs |

Protocol-specific fields:

- **CLEVR** (`clevr_development`, `clevr_confirmation`): `question`, `template`
  (`addition.json`, `subtraction.json`, `subtraction-multihop.json`),
  `image_filename` (source CLEVR image). Each observation has `crop_path`,
  `object_index`, `pixel_center`, and `ground_truth_attributes`
  (`color`, `shape`, `size`, `material`; scoring only, never shown to models).
- **CLEVR nuisance** (`clevr_nuisance`): instead of `condition_orders`,
  `conditions` holds `canonical`, `nuisance_pixel` (every crop shifted one pixel
  diagonally with edge replication), and `nuisance_relabel` (opaque IDs replaced
  by a fresh rank-aligned bijection, given in `relabel_map`). All three keep the
  canonical order; each condition lists its own `observation_order` and
  `observations` (with `canonical_observation_id` and `crop_sha256`).
- **RAG-IGBench** (`ragig_confirmation`): `anchor_ids`, `query_ids`,
  `ground_truth_mapping` (query → anchor), `question_spec`
  (`{"operator": "selected_anchor_rank", "query_id": ...}`), `source_record_id`.
  Observations have `role` (`anchor` = full photo, `query` = 70% crop given by
  `crop_xywh`), `source_member` (path inside the upstream `images.tar.gz`),
  `source_sha256`, `view_path`, and `view_sha256`.
- **LVIS** (`lvis_confirmation`): `image_id` (COCO val2017), `question_spec`
  (`operator` ∈ `count`, `sum`, `absolute_difference`; `categories`).
  Observations have `box_xywh`, `instance_id`, `ground_truth_category`,
  `crop_path`, and `crop_sha256`.

The preparation scripts check every rebuilt image against its stored SHA-256.

## Raw model records: `results/frozen/<protocol>/<model>.jsonl`

| Field | Meaning |
|---|---|
| `case_id`, `answer`, `question` / `question_spec` | copied from the manifest |
| `raw_responses` | raw generated text per condition |
| `state_entries` | parsed typed state per condition: a list of rows keyed by `observation_id` (CLEVR: `color`/`shape`/`size`/`material`; LVIS: `category`, plus `category_code` for InternVL) |
| `state_mappings` | RAG-IGBench only: parsed query → anchor mapping per condition |
| `condition_errors` | parser error per condition, or `null` |
| `executable_predictions` | output of the fixed current program on each parsed state |
| `status` | `complete_valid` (every condition parsed into a complete table) or `invalid_single_attempt_state` |
| `model_repository`, `model_revision` | pinned checkpoint |
| `elapsed_seconds` | wall-clock time for the case |

Primary rates use `complete_valid` cases. Invalid cases remain in the files and
are included in all-case sensitivity bounds. Decoding uses one greedy attempt
per condition; failed outputs are never retried.

`scripts/run_inference.py` writes new outputs in the same schema. Analyze them
with `scripts/reproduce_main.py --results results/rerun --no-check`.
