# Crystallography and orientation

Tools for *planning* a measurement: which reflections exist, how strong they
are, which ones are equivalent, how the crystal is oriented, and where the
detector has to go. They need no gemmi or cctbx.

| module | needs | what |
|---|---|---|
| `hklforge.crystal.cell` | numpy | real/reciprocal basis, volume, d, crystal systems, cell checks |
| `hklforge.crystal.spacegroups` | numpy | all 564 settings of the 230 space groups: absences with the reason, equivalents, multiplicity |
| `hklforge.crystal.cif`, `.structure` | numpy, xraydb for F | a small CIF reader; structure factors with f0, f' + i f'' and Debye-Waller |
| `hklforge.crystal.reflections` | pandas (+ xraydb for F) | reflection lists, crystal columns, equivalence and rotation groups |
| `hklforge.orient` | numpy | U from one or two (hkl, direction) pairs |
| `hklforge.lattice.angle_between` | numpy | angle between two reflections |
| `Geometry.detector_towards` | numpy | detector circles that point the detector along a beam |

`pip install "hklforge[crystal]"` adds pandas and xraydb. Everything else
imports with numpy only.

Conventions are those of the rest of hklforge (and diffcalc): reciprocal
vectors carry 2π, so |B h| = 2π / d. B is Busing & Levy's matrix (a* along x,
b* in the x-y plane), and `ub = u @ b`.

## Space groups

```python
from hklforge.crystal.spacegroups import SpaceGroup

sg = SpaceGroup("P63/mmc")            # also "P 63/m m c", 194, "R-3m:R", "Fd-3m:2", ...
sg.why_forbidden((0, 0, 1))           # 'forbidden in P 63/m m c by the 6₃ screw axis along [0 0 1]
                                      #  and a c-glide plane ⊥ (1 -1 0)'
sg.is_allowed((0, 0, 2))              # True
SpaceGroup("Fm-3m").multiplicity((1, 1, 1))     # 8 (Friedel pairs included)
SpaceGroup("Fm-3m").representative((-1, 1, -1)) # (1, 1, 1)
SpaceGroup("Fm-3m").equivalents((1, 0, 0))      # the 6 reflections of {1 0 0}
```

The table (`spacegroups.json`) was generated from gemmi's by
`dev/make_spacegroup_table.py`. gemmi is needed only to regenerate it.
`why_forbidden` names the symmetry element that removes a reflection: the
centring, which screw axis, or which glide plane.

## Cells

```python
from hklforge.crystal import cell

c = cell.cell_for_system("hexagonal", a=4.143, c=28.636)   # fills in alpha = beta = 90, gamma = 120
cell.d_spacing(c, (0, 0, 3))
cell.volume(c)
cell.check_cell(c, "R-3m")            # [] if consistent, else warnings
```

## Structures and structure factors

```python
from hklforge.crystal.structure import read_cif

st = read_cif("Bi2Se3.cif")            # a path or the CIF text
st.formula(), st.density()             # composition from the sites, g/cm^3
st.structure_factor((0, 0, 3))         # |F| in electrons
st.structure_factor((0, 0, 3), energy=9000)   # with f' + i f'' at 9 keV
```

F is computed as

F(h) = Σ_j occ_j f_j(s) exp(−B_j s²) exp(2πi h·x_j), with s = 1/(2d),

over every atom of the cell. The form factors f0 are Waasmaier & Kirfel's,
and they use the ion's values when the CIF names an ion. With an energy, the
anomalous terms come from the Chantler tables (both via xraydb). Near an edge,
that makes F(h) ≠ F(−h) for non-centrosymmetric crystals. When a CIF leaves
the origin choice or the rhombohedral axes open, `read_cif` picks one and says
so in `st.notes`.

## Reflection lists and groups

```python
from hklforge.crystal import reflections

reflections.reflection_list(st.cell, 9000, structure=st, d_min=1.5)
# DataFrame, sorted by d: one row per set of equivalents, with mult, allowed / why not, |F|, I rel

reflections.target_check((0, 0, 1), st.cell, structure=st)   # warnings for an alignment target
reflections.rotation_groups(hkls, ub, axis)   # which reflections a turn about `axis` maps onto each other
```

There are two kinds of "the same reflection":

- **symmetry-equivalent** (`equivalence_groups`): related by the point group
  and Friedel's law. They have the same |Q| and, without anomalous
  scattering, the same |F|.
- **the same under a rotation** (`rotation_groups`): same |Q| and same angle
  to the axis. Turning the crystal about that axis (an azimuth scan) brings
  one onto the other. `collapse_groups` turns a table into one row per group.

## Orientation

```python
from hklforge.lattice import b_matrix, angle_between
from hklforge.orient import u_from_two, u_from_one

b = b_matrix(5.431, 5.431, 5.431)
u = u_from_two(b, (0, 0, 1), (0, 0, 1),     # (0 0 1) along z of the phi frame ...
                  (1, 0, 0), (1, 0, 0))     # ... and (1 0 0) towards x
ub = u @ b
angle_between(ub, (1, 1, 1), (1, 1, -1))    # 70.53 deg; any B or UB gives the same angle
```

An orientation is a pair (h, v): the reciprocal-lattice direction h points
along v, a direction in the phi frame (the lab frame with every circle at
zero).

- **`u_from_two`** uses Busing & Levy's construction, the same as diffcalc's
  `_calc_ub_from_two_references`. The first pair is matched exactly; the
  second only fixes the plane it spans with the first.
- **`u_from_one`** fixes U only up to a turn about v. It returns `u0`
  (default: identity) turned by the smallest rotation that takes h onto v.

## Pointing the detector

```python
import numpy as np
from hklforge import Geometry, wavevector

psic = Geometry.builtin("psic")
k = wavevector(9000)                                   # |k| in 1/angstrom
q2 = psic.sample_matrix(angles) @ ub @ np.array([2, 2, 0], float)   # a second reflection, lab frame
psic.detector_towards(k * np.asarray(psic.beam) + q2, near=angles)  # k_out = k_in + Q -> {'nu': ..., 'delta': ...}
```

`angles` is a sample position, for example one solution of
`Diffractometer.solve`. `k_out` can have any length; only its direction is used.

For a two-circle detector, there are at most two settings that point the
detector along a direction. `detector_towards` returns the one closest to
`near`'s detector circles.

A worked example of all of this, from a CIF to the detector position, is the
notebook [06_crystallography_planning](../examples/06_crystallography_planning.ipynb).

## What was verified

These checks are in `tests/test_crystal.py` and `tests/test_orient.py`, and
they run in CI:

- **space groups**: for every one of gemmi's 564 settings, the allowed /
  absent pattern of all reflections with indices −3..3 equals gemmi's.
  Equivalents are checked on a subset against gemmi's reciprocal asymmetric
  unit.
- **structure factors**: |F| for Bi2Se3, GaAs, NaCl and rutile (−3..3)
  equals gemmi within 0.5 % for every reflection above 5 % of the strongest.
  The remaining difference is the form-factor tables: Waasmaier-Kirfel here,
  IT92 in gemmi. Reflections that gemmi gives as zero are zero here too.
- **anomalous scattering**: for GaAs just above the Ga K edge (10.4 keV),
  |F(1 1 1)| ≠ |F(−1 −1 −1)|. For centrosymmetric NaCl they stay equal.
- **orientation**: `u_from_two` recovers 20 random U for a triclinic cell to
  1e-12, and refuses parallel pairs. `u_from_one` maps h onto v and returns
  an orthogonal U.
- **`detector_towards`**: for 50 random positions of psic, it gives back the
  detector circles that produced a given k_out, to 1e-7 deg.

Not verified: `u_from_two` was not compared numerically against diffcalc
(only the construction is the same), and the reflection-list intensities
were not compared against measured data.
