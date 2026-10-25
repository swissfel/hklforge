# hklforge

Diffractometer angle calculations for **any** geometry, with solvers that are
**generated exactly** from the geometry rather than hand-written for one
instrument (diffcalc) or found by numerical search (libhkl, ad_hoc_diffractometer).

Status: the geometry model, the exact sub-problems, the planner (geometry +
mode -> all solutions) and all of diffcalc's virtual angles (psi, naz, qaz,
alpha/beta, betain/betaout, omega/bisect, ...) exist and are tested against
diffcalc (all 194 modes), libhkl (all 13 E6C modes), xrayutilities and the
Bernina two-reflection solver.

Documentation: [docs/](docs/README.md) - [virtual angles](docs/virtual_angles.md)
with 3D figures, [modes](docs/modes.md), [verification](docs/verification.md).
Example notebooks: [examples/](examples/). Design notes and roadmap: `CLAUDE.md`.

```python
from hklforge import Geometry

psic = Geometry.builtin("psic")              # You (1999) six-circle, as diffcalc
custom = Geometry.from_xrayutilities(        # xrayutilities' QConversion syntax
    sample=["x+", "z-", "y+", "z-"], detector=["x+", "z-"], r_i=(0, 1, 0))
psic.q_phi({"mu": 0, "eta": 10, "chi": 0, "phi": 0, "nu": 0, "delta": 20}, k=4.5)

from hklforge import Diffractometer
from hklforge.lattice import b_matrix
ub = b_matrix(5.431, 5.431, 5.431)                                   # phi frame, q = UB h (U = 1)
d = Diffractometer(psic, ub, energy=9000, surface_hkl=(0, 0, 1), reference_hkl=(0, 0, 1))
d.solve((1, 1, 2), {"eta": 0, "chi": 0, "phi": 0})          # every branch, as dicts of degrees
d.solve((1, 1, 2), {"chi": 10, "phi": 20, "betain": 5})
d.solve((1, 1, 1), {"mu": 0, "nu": 0, "h2": (2, 2, 0)})     # double diffraction, detector on h1
d.solve((1, 1, 1), {"eta": 0, "chi": 0, "h2": (2, 2, 0)}, detector="between")
d.solve((1, 1, 2), {"mu": 0, "nu": 0, "psi": 30})           # needs reference= / reference_hkl=
d.solve((1, 1, 2), {"nu": 0, "bisect": True, "omega": 0})   # bisecting
print(d.plan({"chi": 10, "phi": 20, "betain": 5}).describe())
angles = d.solve((1, 1, 2), {"mu": 0, "nu": 0, "betain": 2})[0]
d.pseudo(angles)                                            # all virtual angles of a position

from hklforge.draw import draw                              # 3D drawing (matplotlib)
draw(d, angles, show=("ttheta", "psi"))
```

Tests (`pytest`) compare against diffcalc-core and xrayutilities.

## License

BSD 3-Clause, see [LICENSE](LICENSE). The software is provided "as is",
without warranty of any kind. Check computed angles against your
instrument's limits before moving motors.
