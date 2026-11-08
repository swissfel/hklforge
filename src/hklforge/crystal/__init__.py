"""Crystallography for planning an experiment, without gemmi or cctbx.

- :mod:`.cell` - unit cells: real/reciprocal basis, volume, d, crystal
  systems, cell checks against a space group;
- :mod:`.spacegroups` - all 564 settings of the 230 space groups (table from
  gemmi, ``dev/make_spacegroup_table.py``): systematic absences with the
  reason (centring, which screw axis, which glide plane), equivalent
  reflections, multiplicity;
- :mod:`.cif` / :mod:`.structure` - a small CIF reader and X-ray structure
  factors (f0 Waasmaier-Kirfel, f' + i f'' Chantler, Debye-Waller; needs
  ``xraydb``);
- :mod:`.reflections` - reflection lists with those columns, groups of
  symmetry-equivalent reflections and of reflections a rotation about an
  axis maps onto each other (needs ``pandas``).

Same conventions as the rest of hklforge: reciprocal vectors carry 2 pi,
B is Busing & Levy's (:func:`hklforge.lattice.b_matrix`). numpy only at
import; pure Python, so it also runs in the browser (Pyodide/xeus-python).
"""
from .spacegroups import SpaceGroup  # noqa: F401
