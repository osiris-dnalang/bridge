# bridge

`organism_sim` evolutionary controller × `dnalang-core` dynamical-decoupling search space
× Aer ground truth. Neither sibling imports the other; this is the only place both appear.

```bash
pytest                                        # 20 tests (needs qiskit-aer)
python -c "from bridge.compare import compare, HARD; from pathlib import Path; \
  print(compare(Path('results/run'), seeds=range(5), budget=300, cal=HARD)['verdict'])"
```

## Pieces

| module | role |
|---|---|
| `noise.py` | Environment DD actually fights, as exact unitaries inside the circuit: quasi-static detuning δ_q ~ N(0, σ) per batch (`rz` per idle interval) and static ZZ crosstalk J between chain neighbours (`rzz` per interval between pulses on *either* qubit, time-ordered — so simultaneous π pulses leave ZZ invariant and staggered ones refocus it). Plus Aer depolarising gate error and readout error. `Calibration.hash()` plays the role of `dnalang.backends.ibm.calibration_hash`; `DriftSchedule` supplies shocks. |
| `edits.py` | Closed, deterministic, parity-preserving edit set over `DDSpace` genomes — the controller's action alphabet. |
| `quantum_fitness.py` | `QuantumFitness`: `DDSpace → to_dna → parse → lower → evaluate → survival_plus`; one hash-chained `Ledger` row per unique evaluation (`genome_key, dna_sha, calibration_hash, counts_sha, score`). `DDController`: an `LCSAgent` whose register is (score bin ⊕ drift bin) and whose actions are edits; reward = improvement. Every controller row's `(genome_key, calibration_hash)` joins to a ledger row (tested). |
| `compare.py` | Budget-matched organism vs **unmodified** `dnalang.evolve.evolve` vs baselines, judged on fresh-seed verification scores (8,192 shots). |

The environment reproduces the `ibm_fez` finding: with ZZ present, CPMG and plain XY4 sit at
the no-DD level (≈ 0.44) while staggered XY4 survives (≈ 0.98); with detuning only, all
three refocus equally.

## Step 3 result — pre-registered, PASS as a tie

Criterion (stated before the run): C1 organism verification ≥ GA − 0.005 on ≥ 4/5 seeds;
C2 both beat the best baseline on ≥ 4/5. Hard calibration (σ 0.15 MHz, J 40 kHz,
p_gate 0.004), 300 unique evaluations per side, seeds 0–4 — `results/step3_hard/`.

| | verified survival |
|---|---|
| staggered XY4 (best baseline) | 0.9705 |
| dnalang GA (median of 5) | 0.9758 |
| organism controller (median of 5) | 0.9763 |

C1 5/5 (seed 3 by 0.0003), C2 5/5. This is a **tie with the GA**, ≈ 0.005 (≈ 3σ) above
staggered XY4, which is already near the model's gate-error ceiling. Established: the
environment, the provenance join, and that the organism controller is not worse than a
population GA at equal hardware budget. Not established: that it is better.

Next pre-registration (Tier 4): `DriftSchedule.shock` mid-run; criterion on re-convergence
time after the calibration hash changes, organism (structural on) vs GA restarted vs GA
continued.
