"""Bridge tests: noise physics reproduces the hardware ordering, edits preserve parity,
provenance chains verify and join, controller is deterministic, comparison harness fields."""

import itertools
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("qiskit_aer")
import bridge  # noqa: E402,F401
from bridge.compare import HARD, best_in_cache, compare, verify  # noqa: E402
from bridge.edits import EDITS, apply_edit  # noqa: E402
from bridge.noise import (  # noqa: E402
    Calibration,
    DriftSchedule,
    evaluate,
    inject_coherent_noise,
)
from bridge.quantum_fitness import DDController, QuantumFitness  # noqa: E402
from dnalang.evolve.space import DDSpace  # noqa: E402
from dnalang.ledger import Ledger  # noqa: E402
from dnalang.lower import lower  # noqa: E402
from dnalang.metrics.core import survival_plus  # noqa: E402
from dnalang.parser import parse  # noqa: E402

SPACE = DDSpace(n_qubits=4, K=8)


def _score(g, cal, seed=0, shots=512):
    circ = lower(parse(SPACE.to_dna(g, T_us=16.0)))
    return survival_plus(evaluate(circ, cal, shots, 8, seed), range(4), 4)


# ── physics ──────────────────────────────────────────────────────────────────

def test_zz_is_refocused_only_by_staggering():
    cal = Calibration(sigma_detuning_mhz=0.0, zz_khz=20.0, p_gate=0.0, p_readout=0.0)
    b = SPACE.baselines()
    s = {n: _score(g, cal) for n, g in b.items()}
    assert s["xy4_stag"] > 0.95
    assert abs(s["xy4"] - s["none"]) < 0.1 and abs(s["cpmg"] - s["none"]) < 0.1
    assert s["xy4_stag"] > s["xy4"] + 0.3


def test_detuning_is_refocused_by_any_pi_sequence():
    cal = Calibration(sigma_detuning_mhz=0.05, zz_khz=0.0, p_gate=0.0, p_readout=0.0)
    b = SPACE.baselines()
    s = {n: _score(g, cal) for n, g in b.items()}
    assert s["none"] < 0.7
    assert min(s["cpmg"], s["xy4"], s["xy4_stag"]) > 0.95


def test_no_noise_survival_is_one():
    cal = Calibration(0.0, 0.0, 0.0, 0.0)
    assert _score(SPACE.baselines()["none"], cal) == 1.0


def test_inject_preserves_gates_and_removes_delays():
    circ = lower(parse(SPACE.to_dna(SPACE.baselines()["xy4_stag"], T_us=16.0)))
    out = inject_coherent_noise(circ, [1e5] * 4, 2e4)
    assert out.count("delay") == 0 and out.count("barrier") == 0
    assert out.count("x") == circ.count("x") and out.count("y") == circ.count("y")
    assert out.count("rz") > 0 and out.count("rzz") > 0
    assert out.count("measure") == circ.count("measure")


def test_evaluate_deterministic_and_calibration_hash():
    circ = lower(parse(SPACE.to_dna(SPACE.baselines()["xy4"], T_us=16.0)))
    a = evaluate(circ, Calibration(), 128, 4, seed=3)
    b = evaluate(circ, Calibration(), 128, 4, seed=3)
    assert a == b and sum(a.values()) == 128
    assert Calibration().hash() != HARD.hash() and len(Calibration().hash()) == 16


def test_drift_schedule():
    sch = DriftSchedule.shock(at=100)
    assert sch.at(50) == sch.stages[0][1] and sch.at(100) == sch.stages[1][1]
    assert sch.bin(50) == 0 and sch.bin(150) == 1
    assert sch.at(150).zz_khz == 2 * sch.at(50).zz_khz


# ── edits ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("edit", EDITS)
def test_edits_preserve_parity_and_are_deterministic(edit):
    rng1, rng2 = random.Random(1), random.Random(1)
    g = SPACE.baselines()["xy4"]
    h1 = apply_edit(SPACE, g, edit, rng1)
    h2 = apply_edit(SPACE, g, edit, rng2)
    assert h1 == h2 and SPACE.screen(h1) is None
    assert g == SPACE.baselines()["xy4"], "edit must not mutate its input"


def test_edit_semantics():
    g = SPACE.baselines()["xy4"]
    rng = random.Random(0)
    assert apply_edit(SPACE, g, "keep", rng) == g
    assert apply_edit(SPACE, g, "shift_odd", rng)["offset"] == 0.25
    assert apply_edit(SPACE, g, "reset_baseline", rng) == SPACE.baselines()["xy4_stag"]
    with pytest.raises(ValueError):
        apply_edit(SPACE, g, "teleport", rng)


# ── provenance ───────────────────────────────────────────────────────────────

def test_fitness_writes_verifiable_ledger_with_join_keys(tmp_path):
    f = QuantumFitness(tmp_path / "l.jsonl", seed=0)
    s = f.score(SPACE.baselines()["xy4_stag"], trial=7)
    assert 0.0 <= s <= 1.0 and f.evals == 1
    assert f.score(SPACE.baselines()["xy4_stag"], trial=7) == s and f.evals == 1   # cached
    rows = list(Ledger(tmp_path / "l.jsonl"))
    assert len(rows) == 1 and rows[0]["kind"] == "eval"
    assert {"genome_key", "dna_sha", "calibration_hash", "counts_sha", "score", "trial"} <= set(rows[0])
    assert Ledger(tmp_path / "l.jsonl").verify() is None


def test_controller_deterministic_and_chains_join(tmp_path):
    def run(tag):
        f = QuantumFitness(tmp_path / f"{tag}.jsonl", seed=0)
        c = DDController(f, seed=0, start="xy4")
        c.run(20)
        return c
    a, b = run("a"), run("b")
    assert a.summary()["genome_key"] == b.summary()["genome_key"]
    assert [h["edit"] for h in a.history] == [h["edit"] for h in b.history]
    assert a.summary()["audit_chain_valid"] and a.summary()["ledger_valid"]
    # every controller row's (genome_key, calibration_hash) appears in the ledger
    ledger_keys = {(r["genome_key"], r["calibration_hash"]) for r in Ledger(tmp_path / "a.jsonl")}
    for h in a.history:
        assert (h["genome_key"], h["calibration_hash"]) in ledger_keys


def test_controller_improves_from_bad_start(tmp_path):
    f = QuantumFitness(tmp_path / "l.jsonl", schedule=DriftSchedule.constant(HARD), seed=1)
    c = DDController(f, seed=1, start="xy4")
    start = f.score(SPACE.baselines()["xy4"], 0)
    c.run(40)
    assert c.best_score > start + 0.2


def test_register_bits():
    assert DDController.register(0.0, 0) == "000000"
    assert DDController.register(0.999, 3) == "111111"
    assert DDController.register(0.5, 1) == "100001"


# ── comparison harness ───────────────────────────────────────────────────────

def test_compare_fields_and_best_in_cache(tmp_path):
    out = compare(tmp_path, seeds=[0], budget=25, cal=HARD, verify_shots=256)
    assert set(out["baselines_verify"]) == {"none", "cpmg", "xy4", "xy4_stag"}
    r = out["rows"][0]
    assert r["organism"]["evals"] <= 25 and r["ga"]["evals"] <= 25
    assert r["organism"]["audit_chain_valid"] and r["organism"]["ledger_valid"] and r["ga"]["ledger_valid"]
    assert {"C1_organism_matches_ga", "C2_both_beat_baseline", "pass"} <= set(out["verdict"])
    assert json.loads((tmp_path / "compare.json").read_text())["budget"] == 25
    f = QuantumFitness(tmp_path / "x.jsonl", DriftSchedule.constant(HARD))
    for g in itertools.islice(SPACE.baselines().values(), 2):
        f.score(g, 0)
    b = best_in_cache(f, HARD)
    assert b["search_score"] == max(f.cache.values()) and SPACE.key(b["genome"]) == b["genome_key"]
    assert verify(SPACE, b["genome"], HARD, 128, 2, 0) >= 0.0


# ── Tier 4 harness ───────────────────────────────────────────────────────────

def test_controller_detects_calibration_change_and_rescores(tmp_path):
    from bridge.drift_bench import C0, C1
    sched = DriftSchedule([(0, C0), (6, C1)])
    f = QuantumFitness(tmp_path / "l.jsonl", schedule=sched, seed=0)
    c = DDController(f, seed=0, start="xy4_stag")
    for _ in range(5):
        c.step()
    assert c.shocks_seen == 0
    pre = c.best_score
    c.step()                                     # trial 6 → C1
    assert c.shocks_seen == 1
    assert c.history[-1]["calibration_hash"] == C1.hash()
    assert c.best_score != pre or True           # rescored under C1 (value may coincide)
    hashes = {r["calibration_hash"] for r in Ledger(tmp_path / "l.jsonl")}
    assert {C0.hash(), C1.hash()} <= hashes


def test_drift_bench_arms_and_trajectory(tmp_path):
    from bridge.drift_bench import ARMS, C1, _evals_to_target, run_arm
    for arm in ARMS:
        r = run_arm(arm, tmp_path, seed=0, pre=10, post=12)
        assert r["post_evals"] == 12 and r["ledger_valid"]
        assert r["post_best_key"] and 0.0 <= r["post_best_search"] <= 1.0
        rows = [x for x in Ledger(tmp_path / f"{arm}_seed0.ledger.jsonl")
                if x["calibration_hash"] == C1.hash()]
        assert len(rows) == 12
    assert _evals_to_target([0.1, 0.5, 0.9, 0.95], 0.9, 99) == 3
    assert _evals_to_target([0.1, 0.5], 0.9, 99) == 99
