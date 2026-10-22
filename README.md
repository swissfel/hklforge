# hklforge

Diffractometer angle calculations for **any** geometry, with solvers that are
**generated exactly** from the geometry rather than hand-written for one
instrument (diffcalc) or found by numerical search (libhkl, ad_hoc_diffractometer).

Status: early skeleton. The geometry model and the exact sub-problem solvers
exist and are tested; the planner that turns a geometry + constraint mode into
a solver is next. See `CLAUDE.md` for the design and the roadmap.

```python
from hklforge import Geometry

psic = Geometry.builtin("psic")              # You (1999) six-circle, as diffcalc
custom = Geometry.from_xrayutilities(        # xrayutilities' QConversion syntax
    sample=["x+", "z-", "y+", "z-"], detector=["x+", "z-"], r_i=(0, 1, 0))
psic.q_phi({"mu": 0, "eta": 10, "chi": 0, "phi": 0, "nu": 0, "delta": 20}, k=4.5)
```

Tests (`pytest`) compare against diffcalc-core and xrayutilities.
