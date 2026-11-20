# hklforge documentation

- [Virtual angles](virtual_angles.md): every quantity (2θ, qaz, naz, alpha,
  beta, betain, betaout, tau, psi, omega, bisect), with its definition, a 3D
  figure and what holding it fixes.
- [Modes](modes.md): how to write a mode, how many things to hold, the closed-form
  chain used for each case, the energy as a free unknown within limits,
  double diffraction and the detector distance, other geometries.
- [Crystallography and orientation](crystallography.md): space groups
  (absences with the reason, equivalents), cells, CIF and structure factors,
  reflection lists and groups, U from reflections, `detector_towards`; what
  of it was checked against gemmi.
- [3D view, geometry explorer, stored geometries](viewer.md): the interactive
  three.js view of any geometry (virtual angles, Brillouin-zone points), the
  step-by-step geometry explorer, and the libhkl geometries with their check.
- [Verification](verification.md): what was checked against diffcalc, libhkl,
  xrayutilities, the Bernina reference and brute-force scans, with counts;
  the diffcalc bugs found on the way.

Example notebooks (executed, outputs included) are in [`../examples`](https://github.com/swissfel/hklforge/tree/main/examples):

| notebook | shows |
|---|---|
| [01_quickstart](https://github.com/swissfel/hklforge/blob/main/examples/01_quickstart.ipynb) | geometry, crystal, solving, plans, drawings, side by side with diffcalc |
| [02_virtual_angles](https://github.com/swissfel/hklforge/blob/main/examples/02_virtual_angles.ipynb) | each virtual angle drawn and checked against diffcalc, then held as a constraint |
| [03_diffcalc_comparison](https://github.com/swissfel/hklforge/blob/main/examples/03_diffcalc_comparison.ipynb) | all 194 diffcalc modes, and the diffcalc bugs demonstrated |
| [04_double_diffraction](https://github.com/swissfel/hklforge/blob/main/examples/04_double_diffraction.ipynb) | detector on h1 vs between, tdet, against the Bernina solver |
| [05_custom_geometries](https://github.com/swissfel/hklforge/blob/main/examples/05_custom_geometries.ipynb) | xrayutilities syntax (checked against QConversion), a kappa diffractometer |
| [06_crystallography_planning](https://github.com/swissfel/hklforge/blob/main/examples/06_crystallography_planning.ipynb) | Bi2Se3 from CIF: absences, reflection list (vs gemmi), GaAs Friedel pairs at the Ga K edge, azimuth groups, U from two reflections, the detector onto a second reflection |

Figures: `python docs/make_figures.py`. Drawing API: `hklforge.draw.draw` and
`draw_circles` (matplotlib).

Maintainers: [releasing](releasing.md) (PyPI via git tags, conda-forge).
