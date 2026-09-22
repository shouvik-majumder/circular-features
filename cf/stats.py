"""Small statistics helpers shared by the intervention scripts."""
from __future__ import annotations

import numpy as np


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k / n.

    Preferred over the normal approximation because the rates here are often near 0 and the
    trial counts are small (tens to low hundreds), where the normal interval misbehaves.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, float(centre - half)), min(1.0, float(centre + half)))
