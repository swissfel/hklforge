# hklforge

Diffractometer angle calculations for **any** geometry, with solvers that are
**generated exactly** from the geometry rather than hand-written for one
instrument (diffcalc) or found by numerical search (libhkl, ad_hoc_diffractometer).

Status: the geometry model, the exact sub-problems and the planner (geometry +
mode -> all solutions) exist and are tested against diffcalc, libhkl and the
Bernina two-reflection solver. See `CLAUDE.md` for the design, what was
verified, and the roadmap.

```python
from hklforge import Geometry

psic = Geometry.builtin("psic")              # You (1999) six-circle, as diffcalc
custom = Geometry.from_xrayutilities(        # xrayutilities' QConversion syntax
    sample=["x+", "z-", "y+", "z-"], detector=["x+", "z-"], r_i=(0, 1, 0))
psic.q_phi({"mu": 0, "eta": 10, "chi": 0, "phi": 0, "nu": 0, "delta": 20}, k=4.5)

from hklforge import Diffractometer
d = Diffractometer(psic, ub, energy=9000, surface_hkl=(0, 0, 1))   # ub: phi frame, q = UB h
d.solve((1, 1, 2), {"eta": 0, "chi": 0, "phi": 0})          # every branch, as dicts of degrees
d.solve((1, 1, 2), {"chi": 10, "phi": 20, "betain": 5})
d.solve((1, 1, 1), {"mu": 0, "nu": 0, "h2": (2, 2, 0)})     # double diffraction, detector on h1
d.solve((1, 1, 1), {"eta": 0, "chi": 0, "h2": (2, 2, 0)}, detector="between")
print(d.plan({"chi": 10, "phi": 20, "betain": 5}).describe())
```

Tests (`pytest`) compare against diffcalc-core and xrayutilities.
