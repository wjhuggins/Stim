"""lassynth package initialization."""
import sys

_pkg = __name__
sys.modules["lassynth"] = sys.modules[_pkg]

from .lattice_surgery_synthesis import LatticeSurgerySynthesizer, LatticeSurgerySolution

__all__ = ["LatticeSurgerySynthesizer", "LatticeSurgerySolution"]
