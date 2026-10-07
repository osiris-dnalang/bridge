# BO arm — pre-registration

Status: **frozen 2026-10-07**, before any run on the seeds below. The record is the commit
that adds this file plus `PREREGISTRATION.md.ots` (OpenTimestamps proof of this exact
file). No BO comparison has been run on seeds 5–9; the only BO runs so far are unit-test
and smoke runs (budgets ≤ 25, seeds 0 and 3, scratch directories).

## Question

At an equal budget of unique circuit evaluations, does Bayesian optimisation (Optuna TPE)
find dynamical-decoupling sequences with higher verified survival than the unmodified
dnalang GA?

## Arms

| arm | implementation |
|---|---|
| GA | `bridge.compare.run_ga` — unmodified `dnalang.evolve.evolve`, pop 12, 40 generations, seeded with xy4 (as in Step 3) |
| BO | `bridge.compare.run_bo` — `optuna.samplers.TPESampler(seed=seed)`, 17 categoricals (16 pulse slots + offset), xy4 enqueued first, parity-violating proposals rejected before scoring (told as FAIL), trial cap 50 × budget |
| organism | `bridge.compare.run_organism` — reported for reference only; not part of the criterion |

Budget accounting is identical for all arms: one unit per unique `(genome, calibration)`
evaluation; cached genomes and screened-out genomes are free.

## Fixed settings

- Calibration: `HARD` (σ 0.15 MHz, J 40 kHz, p_gate 0.004, p_readout 0.01), hash `2bb4f781c32940b8`
- Budget: 300 unique evaluations per arm per seed
- Seeds: **5, 6, 7, 8, 9** (fresh: 0–4 were used by Step 3, 100–104 by Tier 4 tuning)
- Verification: each arm's best-in-search genome re-scored with 1024 shots × 8 batches at
  verification seed 90000 + seed + 1 (identical across arms), exactly as in `compare()`
- Tolerance: tol = 0.005 (≈ 3σ of the verification score, as in Step 3)

## Code and environment (pinned)

bridge `d30b675` · dnalang-core `227261e` · organism_sim `3935af1` · Python 3.13.5 ·
qiskit 2.5.2 · qiskit-aer 0.17.2 · optuna 5.0.0 · numpy 2.5.3

Command:

```bash
python -c "from pathlib import Path; from bridge.compare import compare, HARD; \
  print(compare(Path('results/bo_arm/eval'), seeds=range(5, 10), budget=300, cal=HARD, \
  include_bo=True)['exploratory'])"
```

## Criterion (stated before the run)

Let Δ = BO verification − GA verification, per seed.

- **P1 (primary): BO better than GA** — Δ ≥ +tol on ≥ 4/5 seeds.
- Otherwise, classified (descriptive, no separate hypothesis):
  - **tie** — Δ ≥ −tol on ≥ 4/5 seeds;
  - **worse** — anything else.
- **P2 (secondary): BO beats the best baseline** — BO verification > best baseline
  verification on ≥ 4/5 seeds.

Headroom is small: in Step 3 the GA median (0.9758) sat ≈ 0.005 above staggered XY4
(0.9705), close to the model's gate-error ceiling. A tie or a FAIL of P1 is a reportable
result.

## Analysis rules

- One run on seeds 5–9. No reruns, no seed substitution, no change of budget, tolerance,
  calibration or code after this document is frozen.
- `compare.json` and every ledger (`ga_`, `bo_`, `organism_seed*.ledger.jsonl`) are checked
  into `results/bo_arm/eval/` whatever the outcome; ledger chains must verify.
- The verdict is computed from `compare.json` by the rule above and written to this
  directory's README, with the per-seed Δ table.

## Exploratory (reported, not judged)

BO vs organism; fraction of BO proposals rejected by the parity screen; evaluations at which
each arm first reached its final best search score.
