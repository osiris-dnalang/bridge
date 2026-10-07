"""Apply the frozen BO pre-registration rule to results/bo_arm/eval/compare.json."""
import json
import sys
from pathlib import Path

from dnalang.ledger import Ledger

R = Path(__file__).resolve().parent
TOL = 0.005
out = json.loads((R / "eval/compare.json").read_text())
assert out["budget"] == 300 and out["calibration_hash"] == "2bb4f781c32940b8" and out["tol"] == TOL
rows = out["rows"]
assert [r["seed"] for r in rows] == [5, 6, 7, 8, 9]
best_base = out["best_baseline"]

table = []
for r in rows:
    g, b, o = r["ga"]["verify"], r["bo"]["verify"], r["organism"]["verify"]
    table.append({"seed": r["seed"], "ga": g, "bo": b, "organism": o, "delta": b - g,
                  "bo_minus_org": b - o, "bo_evals": r["bo"]["evals"], "ga_evals": r["ga"]["evals"],
                  "org_evals": r["organism"]["evals"],
                  "bo_rejected_frac": r["bo"]["rejected_by_screen"] / r["bo"]["trials"]})

n_better = sum(t["delta"] >= TOL for t in table)
n_not_worse = sum(t["delta"] >= -TOL for t in table)
n_p2 = sum(t["bo"] > best_base for t in table)
p1 = n_better >= 4
cls = "better (P1 PASS)" if p1 else ("tie" if n_not_worse >= 4 else "worse")


def first_reach(path):
    rs = list(Ledger(path))
    best = max(x["score"] for x in rs)
    return next(x["eval_index"] for x in rs if x["score"] == best), len(rs)


chains = {}
reach = {}
for p in sorted((R / "eval").glob("*.ledger.jsonl")):
    chains[p.name] = Ledger(p).verify() is None
    if p.stat().st_size:
        reach[p.name] = first_reach(p)

# determinism: aborted organism seed-5 prefix vs restarted run
old = [(x["genome_key"], x["score"]) for x in Ledger(R / "aborted_attempt_1/organism_seed5.ledger.jsonl")]
new = [(x["genome_key"], x["score"]) for x in Ledger(R / "eval/organism_seed5.ledger.jsonl")]
determinism = {"aborted_rows": len(old), "identical_prefix": new[:len(old)] == old}

report = {"baselines_verify": out["baselines_verify"], "best_baseline": best_base, "table": table,
          "P1_count": n_better, "P1_pass": p1, "not_worse_count": n_not_worse, "classification": cls,
          "P2_count": n_p2, "P2_pass": n_p2 >= 4, "verdict_C1C2_step3_style": out["verdict"],
          "medians": {"ga": out["ga_median_verify"], "bo": out["bo_median_verify"],
                      "organism": out["organism_median_verify"]},
          "exploratory_from_compare": out["exploratory"], "ledger_chains_valid": chains,
          "first_eval_reaching_final_best": reach, "determinism_check": determinism}
json.dump(report, sys.stdout, indent=2)
