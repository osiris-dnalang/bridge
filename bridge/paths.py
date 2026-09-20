"""Locate the two sibling projects. Prefer installed packages; fall back to sibling dirs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SIBLINGS = {"organism_sim": HERE.parent / "organism_sim", "dnalang": HERE.parent / "dnalang-core"}


def ensure() -> None:
    for mod, path in SIBLINGS.items():
        if importlib.util.find_spec(mod) is None and path.exists():
            sys.path.insert(0, str(path))


ensure()
