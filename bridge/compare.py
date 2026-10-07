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

``include_bo=True`` adds a third searcher, Bayesian optimisation (Optuna TPE, optional
``[bo]`` extra), under the same budget accounting. Its comparisons are reported as
``exploratory`` and never enter C1/C2: no BO criterion has been pre-registered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from dnalang.evolve.loop import GAConfig, evolve
from dnalang.lower import lower
from dnalang.metrics.core import survival_plus
from dnalang.parser import parse

from .noise import Calibration, DriftSchedule, evaluate
from .quantum_fitness import DDController, QuantumFitness

HARD = Calibration(sigma_detuning_mhz=0.15, zz_khz=40.0, p_gate=0.004, p_readout=0.01)


def verify_counts(space, genome: dict, cal: Calibration, shots: int = 1024, batches: int = 8,
                  seed: int = 0, T_us: float = 16.0) -> Tuple[float, Dict[str, int]]:
    circ = lower(parse(space.to_dna(genome, T_us=T_us)))
    counts = evaluate(circ, cal, shots, batches, seed=seed)
    return survival_plus(counts, range(circ.n_qubits), circ.n_qubits), counts


def verify(space, genome: dict, cal: Calibration, shots: int = 1024, batches: int = 8,
           seed: int = 0, T_us: float = 16.0) -> float:
    return verify_counts(space, genome, cal, shots, batches, seed, T_us)[0]


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


def run_bo(workdir: Path, seed: int, budget: int, cal: Calibration,
           max_trials_factor: int = 50) -> Dict[str, Any]:
    """Optuna TPE over the DDSpace genome: one categorical per slot plus the offset.

    Like the GA, parity-violating proposals are rejected before scoring and cost no
    budget; they are told to the study as FAIL so TPE models only screened genomes.
    Cached genomes are free. Budget therefore counts unique circuit evaluations exactly
    as for the GA, and xy4 is the first trial just as it seeds the GA population."""
    import optuna
    from optuna.trial import TrialState

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    f = QuantumFitness(workdir / f"bo_seed{seed}.ledger.jsonl",
                       schedule=DriftSchedule.constant(cal), seed=seed)
    space = f.space
    K = len(space.baselines()["xy4"]["even"])
    names = [f"{side}{k}" for side in ("e", "o") for k in range(K)]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    xy4 = space.baselines()["xy4"]
    study.enqueue_trial({**dict(zip(names, xy4["even"] + xy4["odd"])), "offset": xy4["offset"]})

    trials = rejected = 0
    while f.evals < budget and trials < max_trials_factor * budget:
        trial = study.ask()
        trials += 1
        pulses = [trial.suggest_categorical(n, list(space.pulses)) for n in names]
        g = {"even": pulses[:K], "odd": pulses[K:],
             "offset": trial.suggest_categorical("offset", list(space.offsets))}
        if space.screen(g) is not None:
            rejected += 1
            study.tell(trial, state=TrialState.FAIL)
            continue
        study.tell(trial, f.score(g, 0))
    best = best_in_cache(f, cal)
    return {"method": "bo", "seed": seed, "evals": f.evals, "trials": trials,
            "rejected_by_screen": rejected, "sampler": "optuna.TPESampler",
            "optuna_version": optuna.__version__, **best,
            "ledger_valid": f.ledger.verify() is None}


def compare(workdir: Path, seeds: Sequence[int] = range(5), budget: int = 300,
            cal: Calibration = HARD, tol: float = 0.005, verify_shots: int = 1024,
            structural: bool = False, include_bo: bool = False,
            keep_counts: bool = False) -> Dict[str, Any]:
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
        o["verify"], o_counts = verify_counts(space, o["genome"], cal, verify_shots, 8, vseed + sd + 1)
        g["verify"], g_counts = verify_counts(space, g["genome"], cal, verify_shots, 8, vseed + sd + 1)
        if keep_counts:
            o["verify_counts"], g["verify_counts"] = o_counts, g_counts
        row = {"seed": sd, "organism": o, "ga": g}
        if include_bo:
            b = run_bo(workdir, sd, budget, cal)
            b["verify"] = verify(space, b["genome"], cal, verify_shots, 8, vseed + sd + 1)
            row["bo"] = b
        rows.append(row)
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
    if include_bo:
        out["bo_median_verify"] = float(np.median([r["bo"]["verify"] for r in rows]))
        out["exploratory"] = {
            "note": "not pre-registered; does not affect verdict",
            "bo_ge_ga_minus_tol": sum(r["bo"]["verify"] >= r["ga"]["verify"] - tol for r in rows),
            "bo_ge_organism_minus_tol": sum(r["bo"]["verify"] >= r["organism"]["verify"] - tol
                                            for r in rows),
            "bo_beats_baseline": sum(r["bo"]["verify"] > best_base for r in rows),
            "n": n,
        }
    (workdir / "compare.json").write_text(json.dumps(out, indent=2))
    return out


__all__ = ["HARD", "verify", "run_organism", "run_ga", "run_bo", "compare", "best_in_cache"]
