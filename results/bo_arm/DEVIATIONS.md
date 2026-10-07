# Deviations from the pre-registration

## 1. Aborted first attempt (technical interruption), restarted unchanged

- **What happened.** The pre-registered run started 2026-10-07T05:53:40Z. At 05:54:40Z the
  process was killed together with the interactive session that launched it, a host-side
  termination unrelated to the experiment. It had completed 257 of 300 evaluations of the
  organism arm on seed 5, the first of 15 arm runs (3 arms × 5 seeds). No GA or BO run had
  started.
- **What was observed.** Nothing about the outcome. No `compare.json` was produced, no
  verification score existed, and the partial ledger's scores were not inspected before the
  restart decision.
- **What was done.** The partial outputs were moved unchanged to `aborted_attempt_1/`, and
  the identical command (same code, interpreter, seeds, budget, calibration) was restarted at
  2026-10-07T06:00:08Z, detached from the session (`setsid nohup`). The pre-registration's
  "one run, no reruns" rule is read as excluding outcome-dependent reruns; this restart
  precedes any outcome.
- **Check.** All arms are seeded and deterministic, so the restarted organism seed-5 run must
  reproduce the aborted run's 257 evaluations (genome keys and scores, in order).
  **Result: identical.** The restarted `organism_seed5` ledger's first 257 rows match the
  aborted run's genome keys and scores exactly (`verdict.json` → `determinism_check`).

## 2. Error in the pre-registration's description of the budget (not an execution change)

The pre-registration says "Budget: 300 unique evaluations per arm per seed". In the code it
pinned, 300 is a **cap**, not a quantity each arm spends. The GA (pop 12, 40 generations,
unchanged since Step 3) runs out of generations first and spent 70–99 unique evaluations
per seed; BO hit its trial cap (50 × budget, with 96–98% of proposals rejected by the parity
screen) on seeds 6–9 and spent 223–275; the organism spent 300 on every seed. The run itself
followed the pre-registered command exactly; the comparison it describes is therefore not
budget-matched, which `README.md` takes into account when reading the verdict.
