"""Ledger evaluation-row contract: emitted rows validate, checked-in ledgers validate and
their chains verify, and the (genome_key, calibration_hash) join key is present on every row."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("qiskit_aer")
jsonschema = pytest.importorskip("jsonschema")
import bridge  # noqa: E402,F401
from bridge.quantum_fitness import QuantumFitness  # noqa: E402
from dnalang.ledger import Ledger  # noqa: E402

SCHEMA = json.loads((ROOT / "schemas" / "ledger_eval_row.schema.json").read_text())


def test_emitted_ledger_rows_validate(tmp_path):
    f = QuantumFitness(tmp_path / "l.jsonl", seed=0)
    for g in f.space.baselines().values():
        f.score(g, 3)
    v = jsonschema.Draft7Validator(SCHEMA)
    rows = list(Ledger(tmp_path / "l.jsonl"))
    assert len(rows) == 4
    for r in rows:
        v.validate(r)


def test_checked_in_ledgers_validate_and_verify():
    files = sorted((ROOT / "results").glob("**/*.ledger.jsonl"))
    assert files, "no ledgers checked in"
    v = jsonschema.Draft7Validator(SCHEMA)
    for f in files:
        led = Ledger(f)
        assert led.verify() is None, f
        for r in led:
            v.validate(r)
            assert r["genome_key"] and r["calibration_hash"]
