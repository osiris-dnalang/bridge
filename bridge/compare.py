"""
bridge.compare — Budget-matched comparison: organism controller vs dnalang GA vs baselines
==========================================================================================

Both searchers spend the same budget of *unique* circuit evaluations through the same
``QuantumFitness`` (shared cache, shared ledger). The GA is ``dnalang.evolve.loop.evolve``
unmodified; once the budget is exhausted its fitness returns −1 for unseen genomes so it
cannot exploit unevaluated candidates. Each side's best-within-budget genome is then
re-scored with fresh seeds and more shots (``verify_shots``) — the *verification score* —
which is what the criterion is judged on, so noisy in-search maxima cannot inflate a side.

Pre-registered criterion (5 seeds):
    C1  organism verification ≥ GA verification − tol on ≥ 4/5 seeds
    C2  both organism and GA beat the best baseline (verification) on ≥ 4/5 seeds
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from dnalang.evolve.loop import GAConfig, evolve
from dnalang.lower import lower
from dnalang.metrics.core import survival_plus
from dnalang.parser import parse

from .noise import Calibration, DriftSchedule, evaluate
from .quantum_fitness import DDController, QuantumFitness

HARD = Calibration(sigma_detuning_mhz=0.15, zz_khz=40.0, p_gate=0.004, p_readout=0.01)


def verify(space, genome: dict, cal: Calibration, shots: int = 1024, batches: int = 8,
           seed: int = 0, T_us: float = 16.0) -> float:
    circ = lower(parse(space.to_dna(genome, T_us=T_us)))
    counts = evaluate(circ, cal, shots, batches, seed=seed)
    return survival_plus(counts, range(circ.n_qubits), circ.n_qubits)


def best_in_cache(f: QuantumFitness, cal: Calibration) -> Dict[str, Any]:
    suffix = "@" + cal.hash()
    items = [(k[:-len(suffix)], v) for k, v in f.cache.items() if k.endswith(suffix)]
    key, score = max(items, key=lambda kv: kv[1])
    even, odd, off = key.split("|")
    return {"genome": {"even": list(even), "odd": list(odd), "offset": float(off)},
            "genome_key": key, "search_score": score}


def run_organism(workdir: Path, seed: int, budget: int, cal: Calibration,
                 structural: bool = False) -> Dict[str, Any]:
    f = QuantumFitness(workdir / f"organism_seed{seed}.ledger.jsonl",
                       schedule=DriftSchedule.constant(cal), seed=seed)
    c = DDController(f, seed=seed, start="xy4", structural=structural)
    s = c.run(budget)
    best = best_in_cache(f, cal)
    return {"method": "organism", "seed": seed, "evals": f.evals, "trials": c.trial,
            **best, "audit_chain_valid": s["audit_chain_valid"], "ledger_valid": s["ledger_valid"]}


def run_ga(workdir: Path, seed: int, budget: int, cal: Calibration, pop: int = 12,
           gens: int = 40) -> Dict[str, Any]:
    f = QuantumFitness(workdir / f"ga_seed{seed}.ledger.jsonl",
                       schedule=DriftSchedule.constant(cal), seed=seed)

    def fitness(g: dict) -> float:
        key = f"{f.space.key(g)}@{cal.hash()}"
        if f.evals >= budget and key not in f.cache:
            return -1.0
        return f.score(g, 0)

    res = evolve(f.space, fitness, GAConfig(pop_size=pop, generations=gens, seed=seed),
                 seeds=[dict(f.space.baselines()["xy4"])])
    best = best_in_cache(f, cal)
    return {"method": "ga", "seed": seed, "evals": f.evals, "generations": gens,
            "ga_reported_best": res.best_score, **best, "ledger_valid": f.ledger.verify() is None}


def compare(workdir: Path, seeds: Sequence[int] = range(5), budget: int = 300,
            cal: Calibration = HARD, tol: float = 0.005, verify_shots: int = 1024,
            structural: bool = False) -> Dict[str, Any]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    space = QuantumFitness(workdir / "probe.ledger.jsonl", DriftSchedule.constant(cal)).space
    vseed = 90_000
    baselines = {n: verify(space, g, cal, verify_shots, 8, vseed) for n, g in space.baselines().items()}
    best_base = max(baselines.values())
    rows: List[Dict[str, Any]] = []
    for sd in seeds:
        o = run_organism(workdir, sd, budget, cal, structural)
        g = run_ga(workdir, sd, budget, cal)
        o["verify"] = verify(space, o["genome"], cal, verify_shots, 8, vseed + sd + 1)
        g["verify"] = verify(space, g["genome"], cal, verify_shots, 8, vseed + sd + 1)
        rows.append({"seed": sd, "organism": o, "ga": g})
    c1 = sum(r["organism"]["verify"] >= r["ga"]["verify"] - tol for r in rows)
    c2 = sum(r["organism"]["verify"] > best_base and r["ga"]["verify"] > best_base for r in rows)
    n = len(rows)
    out = {"budget": budget, "calibration": cal.to_dict(), "calibration_hash": cal.hash(),
           "baselines_verify": baselines, "best_baseline": best_base, "tol": tol,
           "rows": rows,
           "organism_median_verify": float(np.median([r["organism"]["verify"] for r in rows])),
           "ga_median_verify": float(np.median([r["ga"]["verify"] for r in rows])),
           "verdict": {"C1_organism_matches_ga": c1 >= 4 if n >= 5 else c1 == n,
                       "C1_count": c1, "C2_both_beat_baseline": c2 >= 4 if n >= 5 else c2 == n,
                       "C2_count": c2, "n": n}}
    out["verdict"]["pass"] = bool(out["verdict"]["C1_organism_matches_ga"]
                                  and out["verdict"]["C2_both_beat_baseline"])
    (workdir / "compare.json").write_text(json.dumps(out, indent=2))
    return out


__all__ = ["HARD", "verify", "run_organism", "run_ga", "compare", "best_in_cache"]
