# hklforge documentation

- [Virtual angles](virtual_angles.md): every quantity (2θ, qaz, naz, alpha,
  beta, betain, betaout, tau, psi, omega, bisect), with its definition, a 3D
  figure and what holding it fixes.
- [Modes](modes.md): how to write a mode, how many things to hold, the closed-form
  chain used for each case, double diffraction and the detector distance,
  other geometries.
- [Verification](verification.md): what was checked against diffcalc, libhkl,
  xrayutilities, the Bernina reference and brute-force scans, with counts;
  the diffcalc bugs found on the way.

Example notebooks (executed, outputs included) are in [`../examples`](../examples):

| notebook | shows |
|---|---|
| [01_quickstart](../examples/01_quickstart.ipynb) | geometry, crystal, solving, plans, drawings, side by side with diffcalc |
| [02_virtual_angles](../examples/02_virtual_angles.ipynb) | each virtual angle drawn and checked against diffcalc, then held as a constraint |
| [03_diffcalc_comparison](../examples/03_diffcalc_comparison.ipynb) | all 194 diffcalc modes, and the diffcalc bugs demonstrated |
| [04_double_diffraction](../examples/04_double_diffraction.ipynb) | detector on h1 vs between, tdet, against the Bernina solver |
| [05_custom_geometries](../examples/05_custom_geometries.ipynb) | xrayutilities syntax (checked against QConversion), a kappa diffractometer |

Figures: `python docs/make_figures.py`. Drawing API: `hklforge.draw.draw` and
`draw_circles` (matplotlib).
