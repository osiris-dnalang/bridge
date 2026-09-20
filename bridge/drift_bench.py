"""
bridge.drift_bench — Tier 4: calibration shock, four controllers
================================================================

Pre-shock phase under calibration C0 (``pre`` unique evaluations), then the
calibration hash changes to C1 (``DriftSchedule.shock``) and every arm gets
``post`` unique evaluations under C1. All arms can see the calibration hash
(it is public on real hardware); what differs is what they do with it:

    organism-structural  DDController, structural hooks ON: infidelity → noise →
                         repair (compaction) / mutation (GP variants + shock:
                         reset high-error rules, exploration boost)
    organism-plain       same controller, hooks OFF (isolates the layer)
    ga-continued         dnalang ``evolve`` warm-started from its pre-shock population
    ga-restarted         dnalang ``evolve`` from a fresh random population

Metric: unique post-shock evaluations until the arm's best post-shock search score
reaches ``target = oracle − margin``, where the oracle is a GA with 3× the post
budget under C1. Final: verified post-shock fidelity of each arm's best genome.

Pre-registered criterion (seeds 0–4; triggers tuned on 100–104 only):
    organism-structural reaches target in fewer evaluations than BOTH
    organism-plain and ga-continued on ≥ 4/5 seeds → PASS; else the layer is
    redundant for continuous parameter drift.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from dnalang.evolve.loop import GAConfig, evolve
from dnalang.ledger import Ledger
from organism_sim.spec import Triggers

from .compare import HARD, best_in_cache, verify
from .noise import Calibration, DriftSchedule
from .quantum_fitness import DDController, QuantumFitness

ARMS = ("organism-structural", "organism-plain", "ga-continued", "ga-restarted")
C0 = Calibration()
C1 = HARD
DEFAULT_TRIGGERS = Triggers(noise_floor=0.02, repair_threshold=0.08, unrepaired_integral=0.5,
                            window=10, entropy_floor_bits=0.0)


def _post_trajectory(ledger_path: Path, cal_hash: str) -> List[float]:
    """best-so-far search score per unique post-shock evaluation, in ledger order."""
    best, out = -1.0, []
    for row in Ledger(ledger_path):
        if row.get("calibration_hash") == cal_hash:
            best = max(best, row["score"])
            out.append(best)
    return out


def _evals_to_target(traj: List[float], target: float, cap: int) -> int:
    for i, b in enumerate(traj, 1):
        if b >= target:
            return i
    return cap


def _ga_budgeted(f: QuantumFitness, cal: Calibration, budget: int, seed: int,
                 seeds: Optional[List[dict]], pop: int = 12, gens: int = 60):
    start = f.evals

    def fitness(g: dict) -> float:
        key = f"{f.space.key(g)}@{cal.hash()}"
        if f.evals - start >= budget and key not in f.cache:
            return -1.0
        return f.score(g, 10 ** 6)          # trial index far past the shock → C1
    return evolve(f.space, fitness, GAConfig(pop_size=pop, generations=gens, seed=seed), seeds=seeds)


def run_arm(arm: str, workdir: Path, seed: int, pre: int, post: int,
            triggers: Triggers = DEFAULT_TRIGGERS) -> Dict[str, Any]:
    shock_at = 10 ** 5                      # controller trial index of the shock (set below)
    sched = DriftSchedule([(0, C0), (shock_at, C1)])
    f = QuantumFitness(workdir / f"{arm}_seed{seed}.ledger.jsonl", schedule=sched, seed=seed)
    info: Dict[str, Any] = {"arm": arm, "seed": seed}
    if arm.startswith("organism"):
        c = DDController(f, seed=seed, start="xy4", triggers=triggers,
                         structural=(arm == "organism-structural"))
        c.run(pre)                                              # phase 1 under C0
        sched.stages[1] = (c.trial + 1, C1)                     # shock now
        while f.evals < pre + post:
            c.step()
        info.update({"trials": c.trial, "shocks_seen": c.shocks_seen,
                     "organism": c.agent.state(), "audit_chain_valid": c.agent.organism.chain.verify()})
    else:
        # phase 1: GA under C0 (trial index 0 → C0)
        start = f.evals

        def fit0(g: dict) -> float:
            key = f"{f.space.key(g)}@{C0.hash()}"
            if f.evals - start >= pre and key not in f.cache:
                return -1.0
            return f.score(g, 0)
        res0 = evolve(f.space, fit0, GAConfig(pop_size=12, generations=60, seed=seed),
                      seeds=[dict(f.space.baselines()["xy4"])])
        sched.stages[1] = (1, C1)                               # every later trial is C1
        warm = res0.population if arm == "ga-continued" else None
        _ga_budgeted(f, C1, post, seed + (0 if warm else 1000), warm)
        info["pre_population"] = len(res0.population)
    traj = _post_trajectory(f.ledger.path, C1.hash())
    best = best_in_cache(f, C1)
    info.update({"evals_total": f.evals, "post_evals": len(traj), "post_trajectory": traj,
                 "post_best_search": best["search_score"], "post_best_key": best["genome_key"],
                 "post_best_genome": best["genome"], "ledger_valid": f.ledger.verify() is None})
    return info


def oracle(workdir: Path, seed: int, post: int, factor: int = 3) -> float:
    f = QuantumFitness(workdir / f"oracle_seed{seed}.ledger.jsonl",
                       schedule=DriftSchedule.constant(C1), seed=seed + 5000)
    _ga_budgeted(f, C1, post * factor, seed + 7, [dict(f.space.baselines()["xy4_stag"])],
                 pop=16, gens=200)
    return best_in_cache(f, C1)["search_score"]


def compare(workdir: Path, seeds: Sequence[int] = range(5), pre: int = 150, post: int = 250,
            margin: float = 0.005, triggers: Triggers = DEFAULT_TRIGGERS,
            verify_shots: int = 1024, arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    space = QuantumFitness(workdir / "probe.ledger.jsonl").space
    rows: List[Dict[str, Any]] = []
    for sd in seeds:
        orc = oracle(workdir, sd, post)
        target = orc - margin
        row: Dict[str, Any] = {"seed": sd, "oracle": orc, "target": target, "arms": {}}
        for arm in arms:
            r = run_arm(arm, workdir, sd, pre, post, triggers)
            r["evals_to_target"] = _evals_to_target(r["post_trajectory"], target, post)
            r["post_verify"] = verify(space, r["post_best_genome"], C1, verify_shots, 8, 90_000 + sd)
            r.pop("post_trajectory")
            r.pop("post_best_genome")
            row["arms"][arm] = r
        rows.append(row)
    wins = 0
    for row in rows:
        a = row["arms"]
        if "organism-structural" in a and "organism-plain" in a and "ga-continued" in a:
            s = a["organism-structural"]["evals_to_target"]
            if s < a["organism-plain"]["evals_to_target"] and s < a["ga-continued"]["evals_to_target"]:
                wins += 1
    med = {arm: float(np.median([r["arms"][arm]["evals_to_target"] for r in rows])) for arm in arms}
    fid = {arm: float(np.median([r["arms"][arm]["post_verify"] for r in rows])) for arm in arms}
    out = {"seeds": list(seeds), "pre": pre, "post": post, "margin": margin,
           "C0": C0.to_dict(), "C1": C1.to_dict(), "triggers": triggers.to_dict(),
           "median_evals_to_target": med, "median_post_verify": fid, "rows": rows,
           "verdict": {"structural_wins": wins, "n": len(rows),
                       "pass": wins >= 4 if len(rows) >= 5 else wins == len(rows)}}
    (workdir / "drift_compare.json").write_text(json.dumps(out, indent=2))
    return out


__all__ = ["ARMS", "C0", "C1", "DEFAULT_TRIGGERS", "run_arm", "oracle", "compare"]
