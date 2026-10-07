"""Unit tests for bridge.verified_compare integration with osiris_governance."""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("qiskit_aer")

from bridge.compare import HARD
from bridge.verified_compare import verified_compare
from osiris_governance.ledger import DynamicEvidenceLedger


def test_verified_compare_runs_and_seals_ledger(tmp_path):
    ledger = DynamicEvidenceLedger(ledger_id="test-bridge-ledger")
    # Quick budget run
    out = verified_compare(
        workdir=tmp_path,
        seeds=[0],
        budget=10,
        cal=HARD,
        verify_shots=128,
        ledger=ledger,
    )
    assert "statistical_promotion_decision" in out
    assert "promoted" in out["statistical_promotion_decision"]
    assert out["verified_evidence_sealed"] is True
    assert (tmp_path / "verified_compare.json").exists()
    assert ledger.event_count >= 2
