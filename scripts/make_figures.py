#!/usr/bin/env python3
"""Regenerate Figure 1 (silent-corruption example) and Figure 2 (state-integrity audit).

    python scripts/make_figures.py --output-dir figures

Both figures are computed from the raw records in ``results/frozen`` (or
``--results``) and, for Figure 1, the shipped CLEVR crops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathbench import clevr, metrics  # noqa: E402
from pathbench.common import load_config, read_jsonl, resolve  # noqa: E402

DISPLAY = {"qwen": "Qwen", "internvl": "InternVL", "llava": "LLaVA", "idefics2": "IDEFICS2"}
COLORS = {"state": "#6a3d9a", "answer": "#ff7f00"}
DECOMP_COLORS = ("#4daf4a", "#377eb8", "#e41a1c")

# Figure 1 example: the minimum-SHA-256-rank silent flip in the InternVL confirmation cell.
EXAMPLE_CASE, EXAMPLE_MODEL, EXAMPLE_OBSERVATION = "pathguard-016", "internvl", "OBS-8456f4d667"


def paired_bars(ax, labels, reports, title):
    x = np.arange(len(labels))
    width = 0.34
    for offset, key, color, label in ((-width / 2, "state_table_flip", COLORS["state"], "state-table flip"),
                                      (width / 2, "executable_answer_flip", COLORS["answer"], "answer flip")):
        values = [r[key] * 100 for r in reports]
        err = [[v - r[key + "_95"][0] * 100 for v, r in zip(values, reports)],
               [r[key + "_95"][1] * 100 - v for v, r in zip(values, reports)]]
        bars = ax.bar(x + offset, values, width, color=color, yerr=err, capsize=2.5, label=label)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Complete-valid cases (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.2)


def figure2(results: Path, config: dict, stem: Path) -> None:
    protocols = config["protocols"]
    dev, conf = protocols["clevr_development"], protocols["clevr_confirmation"]
    dev_reports = [metrics.order_metrics(read_jsonl(results / "clevr_development" / f"{m}.jsonl"),
                                         metrics.clevr_state, seed=dev["bootstrap_seed"][m])
                   for m in dev["models"]]
    conf_records = {m: read_jsonl(results / "clevr_confirmation" / f"{m}.jsonl") for m in conf["models"]}
    conf_reports = [metrics.order_metrics(conf_records[m], metrics.clevr_state,
                                          seed=conf["bootstrap_seed"][m]) for m in conf["models"]]

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 8.5,
                         "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.2))
    paired_bars(axes[0], [DISPLAY[m] for m in dev["models"]], dev_reports, "A  Development (72 scenes)")
    paired_bars(axes[1], [DISPLAY[m] for m in conf["models"]], conf_reports,
                "B  Frozen confirmation (180 scenes)")
    labels = [DISPLAY[m] for m in conf["models"]]
    stacks = []
    for m, report in zip(conf["models"], conf_reports):
        ese = metrics.ese_decomposition(conf_records[m])
        stacks.append([100 * ese[k] / report["valid_cases"] for k in
                       ("irrelevant_attribute_only", "predicate_membership_unchanged",
                        "arithmetic_cancellation")])
    stacks = np.array(stacks)
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for column, (color, label) in enumerate(zip(DECOMP_COLORS, ("unused attributes", "predicate preserved",
                                                                "arithmetic cancellation"))):
        axes[2].bar(x, stacks[:, column], bottom=bottom, color=color, label=label)
        bottom += stacks[:, column]
    for index, value in enumerate(bottom):
        axes[2].text(index, value + 1.5, f"{value:.1f}", ha="center", fontsize=7)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Complete-valid cases (%)")
    axes[2].set_title("C  Silent state-flip mechanisms", loc="left", fontweight="bold")
    axes[2].grid(axis="y", alpha=.2)
    axes[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.15))
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2)
    fig.suptitle("Same evidence can preserve an answer while corrupting reusable state",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, .08, 1, .93))
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def figure1(results: Path, config: dict, stem: Path) -> None:
    manifest = {row["case_id"]: row for row in
                read_jsonl(config["protocols"]["clevr_confirmation"]["manifest"])}[EXAMPLE_CASE]
    record = {row["case_id"]: row for row in read_jsonl(
        results / "clevr_confirmation" / f"{EXAMPLE_MODEL}.jsonl")}[EXAMPLE_CASE]
    canonical = metrics.clevr_state(record["state_entries"]["canonical"])
    reverse = metrics.clevr_state(record["state_entries"]["reverse"])
    answers = record["executable_predictions"]
    assert answers["canonical"] == answers["reverse"] and canonical != reverse
    before, after = canonical[EXAMPLE_OBSERVATION], reverse[EXAMPLE_OBSERVATION]
    field = next(f for f in clevr.FIELDS if before[f] != after[f])
    query = {field: before[field], "shape": before["shape"]}
    future = {name: clevr.execute_program({"op": "count", "predicate": query}, list(state.values()))
              for name, state in (("canonical", canonical), ("reverse", reverse))}
    program = clevr.compile_question(manifest["question"])
    constant, terms = clevr.linear_terms(program)
    program_text = ""
    for sign, predicate in terms:
        term = f"count({' '.join(predicate.values()) or 'all'})"
        program_text += term if not program_text else f" {'+' if sign > 0 else '−'} {term}"
    if constant:
        program_text += f" {'+' if constant > 0 else '−'} {abs(constant)}"

    observations = {row["observation_id"]: row for row in manifest["observations"]}
    fig = plt.figure(figsize=(7.15, 2.85), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.45, 1.0), wspace=0.08, hspace=0.32)
    left = grid[:, 0].subgridspec(2, len(observations), wspace=0.05, hspace=0.40)
    for row_index, condition in enumerate(("canonical", "reverse")):
        for column, token in enumerate(manifest["condition_orders"][condition]):
            ax = fig.add_subplot(left[row_index, column])
            ax.imshow(Image.open(resolve(observations[token]["crop_path"])))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#2B6CB0" if condition == "canonical" else "#805AD5")
                spine.set_linewidth(1.4)
            ax.set_title(token[-4:], fontsize=5.7, pad=1.5)
            if column == 0:
                ax.set_ylabel(condition, fontsize=7.2, fontweight="bold", labelpad=4)
    fig.text(0.035, 0.965, "A  Same multiset, reversed order", ha="left", va="top",
             fontsize=8.5, fontweight="bold")
    right = fig.add_subplot(grid[:, 1])
    right.axis("off")
    right.text(0.0, 0.97, "B  Silent corruption changes future reuse", fontsize=8.5,
               fontweight="bold", va="top")
    boxes = (
        (0.67, 0.20, "Current program", "#E6FFFA", "#2C7A7B",
         f"{program_text}\ncanonical = {answers['canonical']}     reverse = {answers['reverse']}"),
        (0.37, 0.22, "Typed-state corruption", "#FFF5F5", "#C53030",
         f"{EXAMPLE_OBSERVATION}  (same ID)\n{field}: {before[field]}  →  {after[field]}"),
        (0.07, 0.20, "Frozen-bank future query", "#FAF5FF", "#6B46C1",
         f"count({' ∧ '.join(query.values())})\n"
         f"canonical = {future['canonical']}     reverse = {future['reverse']}"),
    )
    for y, h, heading, face, edge, body in boxes:
        right.add_patch(FancyBboxPatch((0.02, y), 0.96, h,
                                       boxstyle="round,pad=0.012,rounding_size=0.018",
                                       transform=right.transAxes, facecolor=face,
                                       edgecolor=edge, linewidth=1.2))
        right.text(0.05, y + h - 0.045, heading, transform=right.transAxes, fontsize=7.4,
                   fontweight="bold", color=edge, va="top")
        right.text(0.05, y + 0.035, body, transform=right.transAxes, fontsize=7.2,
                   color="#1A202C", va="bottom", family="monospace")
    for top in (0.67, 0.37):
        right.annotate("", xy=(0.50, top - 0.07), xytext=(0.50, top), xycoords="axes fraction",
                       arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#718096"})
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.04)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Figure 1: {EXAMPLE_CASE} ({EXAMPLE_MODEL}); {field} {before[field]}->{after[field]}; "
          f"current answer {answers['canonical']}; future {future}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/frozen"),
                        help="directory of raw records, <protocol>/<model>.jsonl (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"),
                        help="where fig1_*.{png,pdf} and fig2_*.{png,pdf} are written (default: %(default)s)")
    args = parser.parse_args()
    config = load_config()
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = resolve(args.results)
    figure1(results, config, output / "fig1_silent_corruption_example")
    figure2(results, config, output / "fig2_state_integrity_audit")
    print(f"wrote figures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
