"""
bridge — organism_sim controller × dnalang-core DD search space × Aer ground truth
==================================================================================

Neither sibling imports the other; this package is the only place both names appear.
Importing ``bridge`` makes both siblings importable (installed, or sibling directories).
"""
from . import paths  # noqa: F401  (must run before any dnalang/organism_sim import)

__version__ = "0.1.0"
