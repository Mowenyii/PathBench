"""Fast checks: parsers/executors re-derive every frozen state, and Tables 1-3 match the paper."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathbench import clevr  # noqa: E402
from pathbench.common import load_config, read_jsonl  # noqa: E402
from pathbench.protocols import PROTOCOLS, STATE_KEY  # noqa: E402


class Replay:
    """Stands in for a VLM by returning a recorded raw response."""

    def __init__(self, raw):
        self.raw = raw

    def history_chat(self, *args):
        return self.raw


def test_question_compiler_and_bank():
    program = clevr.compile_question("Subtract 2 cylinders. How many cylinders are left?")
    objects = [{"shape": "cylinder"}] * 5 + [{"shape": "cube"}]
    assert clevr.execute_program(program, objects) == 3
    assert len(clevr.build_query_bank()) == 323


def test_frozen_states_are_rederived_from_raw_responses():
    config = load_config()
    for name, protocol_cls in PROTOCOLS.items():
        spec = config["protocols"][name]
        for model in spec["models"]:
            protocol = protocol_cls(spec, model)
            cases = {case["case_id"]: case for case in protocol.cases()}
            key = STATE_KEY.get(name, "state_entries")
            for record in read_jsonl(ROOT / "results/frozen" / name / f"{model}.jsonl"):
                for condition in protocol.conditions:
                    raw = record["raw_responses"][condition]
                    if not raw:
                        continue
                    _, state, error, prediction = protocol.call(
                        Replay(raw), cases[record["case_id"]], condition)
                    assert (error is None) == (record["condition_errors"][condition] is None)
                    assert prediction == record["executable_predictions"][condition]
                    if error is None:
                        assert state == record[key][condition]


def test_main_results_match_paper(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_main.py"),
                             "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "134 match, 0 differ" in result.stdout
