"""
bridge.quantum_fitness — Fitness with provenance, and the organism controller
=============================================================================

``QuantumFitness.score(genome, trial)``: DDSpace → to_dna → parse → lower → evaluate
under the drift schedule's calibration at ``trial`` → ``survival_plus``. Every
evaluation is a row in the dnalang ``Ledger`` (hash chain) carrying
``genome_key``, ``dna_sha``, ``calibration_hash``, ``counts_sha`` and the score.

``DDController``: an ``LCSAgent`` whose register is (last-score bin ⊕ drift bin)
and whose actions are the closed edit set; reward = 1 if the edit improved the
best score under the *current* calibration. Every structural event (reroute,
mutation, repair) lands in the organism's ``AuditChain``; the join key between the
two chains is ``(genome_key, calibration_hash)``.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from dnalang.evolve.space import DDSpace
from dnalang.ledger import Ledger
from dnalang.lower import lower
from dnalang.metrics.core import survival_plus
from dnalang.parser import parse
from organism_sim.agent import LCSAgent
from organism_sim.spec import Triggers

from .edits import EDITS, apply_edit, edit_actions
from .noise import DriftSchedule, evaluate


class QuantumFitness:
    def __init__(self, ledger_path: Path, schedule: Optional[DriftSchedule] = None,
                 n_qubits: int = 4, K: int = 8, T_us: float = 16.0, shots: int = 256,
                 batches: int = 4, seed: int = 0):
        self.space = DDSpace(n_qubits=n_qubits, K=K)
        self.ledger = Ledger(Path(ledger_path))
        self.schedule = schedule or DriftSchedule.constant()
        self.T, self.shots, self.batches, self.seed = T_us, shots, batches, seed
        self.evals = 0
        self.cache: Dict[str, float] = {}

    def score(self, genome: dict, trial: int = 0, use_cache: bool = True) -> float:
        cal = self.schedule.at(trial)
        key = f"{self.space.key(genome)}@{cal.hash()}"
        if use_cache and key in self.cache:
            return self.cache[key]
        dna = self.space.to_dna(genome, T_us=self.T)
        circ = lower(parse(dna))
        counts = evaluate(circ, cal, self.shots, self.batches, seed=self.seed + self.evals)
        s = survival_plus(counts, range(circ.n_qubits), circ.n_qubits)
        self.evals += 1
        self.ledger.append("eval", {
            "trial": trial, "eval_index": self.evals, "genome_key": self.space.key(genome),
            "dna_sha": hashlib.sha256(dna.encode()).hexdigest()[:16],
            "calibration_hash": cal.hash(), "calibration": cal.to_dict(),
            "counts_sha": hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest()[:16],
            "shots": self.shots, "batches": self.batches, "score": s,
        })
        self.cache[key] = s
        return s

    def baselines(self, trial: int = 0) -> Dict[str, float]:
        return {n: self.score(g, trial) for n, g in self.space.baselines().items()}


class DDController:
    """organism_sim agent driving DDSpace edits against ``QuantumFitness``."""

    def __init__(self, fitness: QuantumFitness, seed: int = 0, start: str = "xy4_stag",
                 triggers: Optional[Triggers] = None, structural: bool = False):
        self.f = fitness
        self.space = fitness.space
        self.rng = random.Random(seed)
        self.agent = LCSAgent("DD", input_len=6, actions=edit_actions(), seed=seed,
                              triggers=triggers or Triggers(noise_floor=0.05, repair_threshold=0.45),
                              structural=structural)
        self.genome = dict(self.space.baselines()[start])
        self.best_score = -1.0
        self.trial = 0
        self.history: List[Dict[str, Any]] = []
        self.last_cal_hash: Optional[str] = None
        self.shocks_seen = 0

    @staticmethod
    def register(last_score: float, drift_bin: int) -> str:
        return format(min(15, max(0, int(last_score * 16))), "04b") + format(drift_bin & 3, "02b")

    def step(self) -> Dict[str, Any]:
        self.trial += 1
        drift_bin = self.f.schedule.bin(self.trial)
        cal_hash = self.f.schedule.at(self.trial).hash()
        if self.last_cal_hash is not None and cal_hash != self.last_cal_hash:
            # calibration changed (public on real hardware): the incumbent's score is stale
            self.shocks_seen += 1
            self.rescore_best()
        self.last_cal_hash = cal_hash
        if self.best_score < 0:
            self.best_score = self.f.score(self.genome, self.trial)
        self.agent.act(self.register(self.best_score, drift_bin))
        edit = self.agent.emitted() or "keep"
        cand = apply_edit(self.space, self.genome, edit, self.rng)
        s = self.f.score(cand, self.trial)
        improved = s > self.best_score + 1e-9
        self.agent.reward(1.0 if improved else 0.0)
        # the environment's error signal for this agent is the incumbent's infidelity;
        # it is what the organism's repair/mutation triggers watch
        self.agent.organism.inject_noise(1.0 - max(self.best_score, s))
        rec = self.agent.tick()
        if improved:
            self.genome, self.best_score = cand, s
        row = {"trial": self.trial, "edit": edit, "score": s, "best": self.best_score,
               "genome_key": self.space.key(self.genome), "drift_bin": drift_bin,
               "calibration_hash": self.f.schedule.at(self.trial).hash(),
               "noise_rate": rec.state["noise_rate"], "events": rec.events,
               "audit_head": self.agent.organism.chain.head}
        self.history.append(row)
        return row

    def run(self, budget: int) -> Dict[str, Any]:
        while self.f.evals < budget:
            self.step()
        return self.summary()

    def rescore_best(self) -> float:
        """Re-evaluate the current genome under the current calibration (after a shock the
        cached best is stale)."""
        self.best_score = self.f.score(self.genome, self.trial, use_cache=False)
        return self.best_score

    def summary(self) -> Dict[str, Any]:
        return {"trials": self.trial, "evals": self.f.evals, "best_score": self.best_score,
                "genome": self.genome, "genome_key": self.space.key(self.genome),
                "edits_used": {e: sum(1 for h in self.history if h["edit"] == e) for e in EDITS},
                "organism": self.agent.state(),
                "audit_chain_valid": self.agent.organism.chain.verify(),
                "ledger_valid": self.f.ledger.verify() is None,
                "ledger_head": self.f.ledger._last_hash()}


__all__ = ["QuantumFitness", "DDController"]
