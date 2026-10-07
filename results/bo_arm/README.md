# BO arm vs GA — pre-registered result: P1 FAIL, classified as a tie; P2 PASS

Pre-registration: `PREREGISTRATION.md` (frozen in `eebe22b`, OpenTimestamps proof
`PREREGISTRATION.md.ots`). Run: 2026-10-07 06:00:08–06:12:20 UTC, seeds 5–9, HARD
calibration (hash `2bb4f781c32940b8`), code `d30b675`, Python 3.13.5, qiskit-aer 0.17.2,
optuna 5.0.0. Deviations: `DEVIATIONS.md` (one aborted, outcome-blind restart; one error in
how the pre-registration described the budget). Verdict computed by `verdict.py` →
`verdict.json`; all 16 ledger chains in `eval/` verify.

## Verdict (rule as pre-registered, tol = 0.005)

| | count | outcome |
|---|---|---|
| P1: BO ≥ GA + tol | 1 / 5 | **FAIL** |
| BO ≥ GA − tol (classification) | 5 / 5 | **tie** |
| P2: BO > best baseline (staggered XY4, 0.9705) | 5 / 5 | **PASS** |

Verified survival (1024 shots × 8 batches, identical verification seed per row):

| seed | GA | BO | organism | Δ = BO − GA | unique evals GA / BO / organism |
|---|---|---|---|---|---|
| 5 | 0.9663 | 0.9731 | 0.9763 | +0.0068 | 85 / 300 / 300 |
| 6 | 0.9751 | 0.9746 | 0.9771 | −0.0005 | 70 / 223 / 300 |
| 7 | 0.9685 | 0.9719 | 0.9741 | +0.0034 | 96 / 226 / 300 |
| 8 | 0.9719 | 0.9724 | 0.9692 | +0.0005 | 83 / 258 / 300 |
| 9 | 0.9695 | 0.9739 | 0.9739 | +0.0044 | 99 / 275 / 300 |
| median | 0.9695 | 0.9731 | 0.9741 | | |

## How to read it

- **The arms were not budget-matched.** 300 was a cap: the GA stopped after 70–99 unique
  evaluations (its 40 generations ran out), BO after 223–300 (trial cap, 96–98% of its
  proposals rejected by the parity screen), the organism always used 300. So the result is:
  *with roughly 3× the GA's evaluations, BO did not beat it by the pre-registered margin.*
  It does not show BO and the GA are equally efficient per evaluation.
- **Established:** BO beats staggered XY4 on 5/5 fresh seeds and is not worse than the GA on
  5/5. **Not established:** that BO is better than the GA, or better than the organism
  controller.

## Exploratory (not pre-registered as criteria)

- BO vs organism: BO ≥ organism − tol on 5/5 seeds; organism median 0.9741 vs BO 0.9731.
- The GA beat staggered XY4 on only 2/5 of these fresh seeds (6 and 8). In Step 3 (seeds 0–4)
  it did on 5/5. Step 3's comparison has the same budget property: its GA used 50–71 unique
  evaluations against the organism's 300 (`../step3_hard/compare.json`), so the bridge
  README's "300 unique evaluations per side … at equal hardware budget" overstates how
  matched that comparison was. Not corrected there by this run; flagged for review.
- Evaluation at which each arm first reached its final best search score: `verdict.json` →
  `first_eval_reaching_final_best` (BO 17–277, GA 7–80, organism 4–279).

## Implication for a follow-up (would need its own pre-registration)

A matched comparison needs a GA configured to spend the budget (more generations or a larger
population, stopping at exactly N unique evaluations) and a BO trial cap that cannot bind
first, e.g. a parity-respecting encoding instead of rejection.
