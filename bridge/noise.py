"""
bridge.noise — Environment for dynamical-decoupling search
==========================================================

Two noise mechanisms that DD actually refocuses, implemented as *explicit unitaries*
inside the circuit so the DD physics is exact for the model:

* quasi-static dephasing: per batch, each qubit gets a detuning δ_q ~ N(0, σ_δ);
  every delay of duration τ on q becomes rz(2π δ_q τ).
* static ZZ crosstalk J between chain neighbours: every delay τ on q contributes
  rzz(2π J τ / 2) with each neighbour (both sides sum to J·τ per pair). Simultaneous
  π pulses on neighbours leave ZZ invariant; *staggered* pulses refocus it.

Plus Aer incoherent error: depolarising p_gate on x/y/h, readout p_ro.

``Calibration`` is the environment state; ``drift(trial)`` returns the calibration at a
given trial index (the shock schedule) and ``calibration_hash`` fingerprints it, the same
role ``dnalang.backends.ibm.calibration_hash`` plays for real hardware.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from dnalang.qc_ir import Circuit, Op


@dataclass(frozen=True)
class Calibration:
    sigma_detuning_mhz: float = 0.05   # quasi-static detuning σ (MHz)
    zz_khz: float = 20.0               # ZZ coupling J (kHz)
    p_gate: float = 0.002              # depolarising error per pulse
    p_readout: float = 0.01

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class DriftSchedule:
    """Piecewise calibration: ``stages`` = [(start_trial, Calibration), ...]."""
    stages: List

    def at(self, trial: int) -> Calibration:
        cal = self.stages[0][1]
        for start, c in self.stages:
            if trial >= start:
                cal = c
        return cal

    def bin(self, trial: int) -> int:
        """Index of the active stage (used as the controller's drift register bits)."""
        idx = 0
        for i, (start, _) in enumerate(self.stages):
            if trial >= start:
                idx = i
        return min(idx, 3)

    @staticmethod
    def constant(cal: Optional[Calibration] = None) -> "DriftSchedule":
        return DriftSchedule([(0, cal or Calibration())])

    @staticmethod
    def shock(at: int, before: Optional[Calibration] = None,
              after: Optional[Calibration] = None) -> "DriftSchedule":
        before = before or Calibration()
        after = after or Calibration(sigma_detuning_mhz=before.sigma_detuning_mhz * 3,
                                     zz_khz=before.zz_khz * 2, p_gate=before.p_gate * 2,
                                     p_readout=before.p_readout)
        return DriftSchedule([(0, before), (at, after)])


def inject_coherent_noise(circ: Circuit, detuning_hz: Sequence[float], zz_hz: float) -> Circuit:
    """Time-order the per-qubit gene sequences and replace idle time with the coherent
    evolution it implies: rz(2π δ_q Δt) on each qubit per idle interval and, for every
    neighbour pair, rzz(2π J Δt) per interval between consecutive pulses on *either* qubit
    (so simultaneous π pulses leave ZZ invariant and staggered ones refocus it)."""
    nq = circ.n_qubits
    # per-qubit timeline: (t, op) for gates; delays only advance t
    events: List = []          # (t, order, kind, payload)
    t_q = [0.0] * nq
    order = 0
    for o in circ.ops:
        if o.name == "barrier":
            continue
        if o.name == "delay":
            q, tau = o.qubits[0], float(o.duration_s or 0.0)
            events.append((t_q[q], order, "idle", (q, tau)))
            t_q[q] += tau
        else:
            for q in o.qubits:
                events.append((t_q[q], order, "gate", o))
                break
        order += 1
    T = max(t_q) if t_q else 0.0
    # pulse times per qubit (for ZZ interval boundaries); measurements/prep count as boundaries
    pulse_t = [sorted({0.0, T} | {t for t, _, k, o in events if k == "gate" and q in o.qubits})
               for q in range(nq)]
    ops: List[Op] = []
    # single-qubit gates and detuning, in time order
    for t, _, kind, payload in sorted(events, key=lambda e: (e[0], e[1])):
        if kind == "idle":
            q, tau = payload
            phi = 2 * math.pi * detuning_hz[q] * tau
            if abs(phi) > 0:
                ops.append(Op("rz", (q,), params=(phi,)))
        else:
            ops.append(payload)
    # ZZ evolution: interleave by rebuilding in time order together with the above
    if zz_hz:
        zz_events = []
        for q in range(nq - 1):
            bounds = sorted(set(pulse_t[q]) | set(pulse_t[q + 1]))
            for a, b in zip(bounds, bounds[1:]):
                if b - a > 0:
                    zz_events.append((a, q, b - a))
        merged: List = []
        gi = 0
        gate_list = sorted(events, key=lambda e: (e[0], e[1]))
        zz_events.sort()
        zi = 0
        # emit gates at time t before ZZ intervals starting at t
        while gi < len(gate_list) or zi < len(zz_events):
            tg = gate_list[gi][0] if gi < len(gate_list) else math.inf
            tz = zz_events[zi][0] if zi < len(zz_events) else math.inf
            if tg <= tz:
                _, _, kind, payload = gate_list[gi]
                gi += 1
                if kind == "idle":
                    q, tau = payload
                    phi = 2 * math.pi * detuning_hz[q] * tau
                    if abs(phi) > 0:
                        merged.append(Op("rz", (q,), params=(phi,)))
                else:
                    merged.append(payload)
            else:
                a, q, dt = zz_events[zi]
                zi += 1
                merged.append(Op("rzz", (q, q + 1), params=(2 * math.pi * zz_hz * dt,)))
        ops = merged
    return Circuit(n_qubits=nq, n_cbits=circ.n_cbits, ops=ops, name=circ.name)


def aer_noise_model(cal: Calibration):
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel()
    if cal.p_gate > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(cal.p_gate, 1), ["x", "y", "h", "sx"])
    if cal.p_readout > 0:
        p = cal.p_readout
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


_SIM_CACHE: Dict[str, tuple] = {}


def _simulator(cal: Calibration):
    """AerSimulator per calibration. Unlike ``dnalang.backends.aer.run`` no transpile pass is
    needed: delays are already replaced by native rz/rzz unitaries, so there is nothing to
    schedule and Aer executes the gate set directly (0.05 s/eval instead of 0.7 s)."""
    key = cal.hash()
    if key not in _SIM_CACHE:
        from qiskit_aer import AerSimulator
        _SIM_CACHE[key] = AerSimulator(noise_model=aer_noise_model(cal))
    return _SIM_CACHE[key]


def evaluate(circ: Circuit, cal: Calibration, shots: int = 256, batches: int = 4,
             seed: int = 0) -> Dict[str, int]:
    """Run ``circ`` under calibration ``cal``: ``batches`` detuning draws, shots split evenly,
    counts merged. Deterministic for (circ, cal, seed)."""
    rng = np.random.default_rng(seed)
    per = max(1, shots // batches)
    total: Dict[str, int] = {}
    circs = []
    for _ in range(batches):
        det = rng.normal(0.0, cal.sigma_detuning_mhz * 1e6, size=circ.n_qubits)
        circs.append(inject_coherent_noise(circ, det, cal.zz_khz * 1e3))
    sim = _simulator(cal)
    res = sim.run([c.to_qiskit() for c in circs], shots=per, seed_simulator=seed).result()
    results = [res.get_counts(i) for i in range(len(circs))]
    for counts in results:
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    return total


__all__ = ["Calibration", "DriftSchedule", "inject_coherent_noise", "aer_noise_model", "evaluate"]
