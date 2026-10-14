"""Domain and shape markers for the signatures, from mathema when it is
installed.

Intent:
    Let the signatures state each parameter's domain and each array's
    shape in mathema's markers, while arbital itself needs only numpy.

With mathema installed these are its own markers, which it reads when
it checks arbital's claims. Without it they are plain stand-ins: every
module here uses `from __future__ import annotations`, so annotations
are never evaluated at run time, and if something does evaluate them a
range reads as `float` and a vector or matrix as `numpy.ndarray`.
"""

from __future__ import annotations

try:
    from mathema.types import InRange, Mat, UnitInterval, Vec
except ImportError:
    import numpy as np

    UnitInterval = float

    def InRange(lo, hi, *args, **kwargs):
        """A number between `lo` and `hi`: `float` without mathema.

        Intent:
            Stand in for mathema's range marker when it is absent.
        """
        return float

    def Vec(*dims, **kwargs):
        """An array of the given dimensions: `numpy.ndarray` without mathema.

        Intent:
            Stand in for mathema's shape marker when it is absent.
        """
        return np.ndarray

    Mat = Vec

__all__ = ["InRange", "Mat", "UnitInterval", "Vec"]
