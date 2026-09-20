"""
bridge.edits — Closed set of structural edits over a ``DDSpace`` genome
======================================================================

These are the controller's actions. Every edit is deterministic given the rng,
returns a *new* genome, and preserves the DD parity screen (even numbers of X and
of Y pulses per sublattice) so every edited genome compiles.
"""
from __future__ import annotations

import random
from typing import List

from dnalang.evolve.space import DDSpace

EDITS = ("keep", "shift_odd", "swap_xy", "add_pair", "drop_pair", "flip_sublattice",
         "random_mutation", "reset_baseline")


def _copy(g: dict) -> dict:
    return {"even": list(g["even"]), "odd": list(g["odd"]), "offset": g["offset"]}


def apply_edit(space: DDSpace, g: dict, edit: str, rng: random.Random) -> dict:
    h = _copy(g)
    if edit == "keep":
        return h
    if edit == "shift_odd":
        offs = list(space.offsets)
        h["offset"] = offs[(offs.index(h["offset"]) + 1) % len(offs)] if h["offset"] in offs else offs[0]
        return h
    sub = rng.choice(("even", "odd"))
    seq = h[sub]
    if edit == "swap_xy":
        h[sub] = ["Y" if p == "X" else "X" if p == "Y" else p for p in seq]
    elif edit == "add_pair":
        idle = [k for k, p in enumerate(seq) if p == "I"]
        if len(idle) >= 2:
            a, b = rng.sample(idle, 2)
            pulse = rng.choice([p for p in space.pulses if p != "I"])
            seq[a] = seq[b] = pulse
    elif edit == "drop_pair":
        for pulse in rng.sample([p for p in space.pulses if p != "I"], len(space.pulses) - 1):
            idx = [k for k, p in enumerate(seq) if p == pulse]
            if len(idx) >= 2:
                a, b = rng.sample(idx, 2)
                seq[a] = seq[b] = "I"
                break
    elif edit == "flip_sublattice":
        h["even"], h["odd"] = h["odd"], h["even"]
    elif edit == "random_mutation":
        for _ in range(20):
            cand = space.mutate(h, rng, 0.2)
            if space.screen(cand) is None:
                return cand
        return h
    elif edit == "reset_baseline":
        return _copy(space.baselines()["xy4_stag"])
    else:
        raise ValueError(f"unknown edit {edit!r}")
    if space.screen(h) is not None:      # parity broken (cannot happen for pairs; guard anyway)
        return _copy(g)
    return h


def edit_actions(edits: List[str] = list(EDITS)) -> List[str]:
    return [f"(emit {e})" for e in edits]


__all__ = ["EDITS", "apply_edit", "edit_actions"]
