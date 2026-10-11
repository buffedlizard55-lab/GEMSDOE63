"""GEMSDOE57 -- fault-zone-anatomy lane for the DOE GEMS Prize Challenge.

Lane: predict where secondary strands sit around known faults, from shear-zone
mechanics, with every structural parameter fitted to a hide-and-recover holdout
rather than taken from a textbook.
"""

from .grid import EPSG, HEIGHT, WIDTH, TRANSFORM, Grid, load_grid, write_submission
from .metric import ALPHA, BETA, RADIUS_PX, dti_binary, dti_exact, dti_bruteforce

__all__ = [
    "EPSG", "HEIGHT", "WIDTH", "TRANSFORM", "Grid", "load_grid", "write_submission",
    "ALPHA", "BETA", "RADIUS_PX", "dti_binary", "dti_exact", "dti_bruteforce",
]
__version__ = "0.1.0"
