# What was verified, and how

Everything here can be rerun; the commands are given. Status as of 2026-10-22,
diffcalc-core 0.4.0, libhkl from conda-forge (E6C).

## Forward model and virtual angles

| check | how | result |
|---|---|---|
| psic Q (phi frame) | `tests/test_forward.py`, 100 random positions vs diffcalc `get_q_phi` | ≤ 1e-12 |
| xrayutilities axis syntax | 3 geometries incl. non-psic ones vs `QConversion` | ≤ 1e-10 |
| all 10 virtual angles | `tests/test_virtual.py`, 300 random positions and lattices vs diffcalc `get_virtual_angles` | ≤ 3e-13 deg |
| psi sign | same; libhkl `psi_constant_*` return psi = hklforge's psi at their solutions | identical |

## Solutions: diffcalc as the oracle (psic)

`tests/test_solve_diffcalc.py` covers **every constraint combination diffcalc
0.4 implements** (194, from `Constraints.is_current_mode_implemented`), with
random lattice, orientation, wavelength, reachable hkl and constraint values.
The *set* of positions must be equal, compared with angle wrapping; every
hklforge position is also re-checked through the forward model and its
virtual angles. Tally over 30 seeds (`python dev/diffcalc_stats.py 30`):

- 5,510 compared cases, 3,364 of them with solutions (2 to 8 positions):
  **identical sets in every case diffcalc answered correctly**;
- every case where diffcalc says the constraints "prohibit a solution",
  hklforge also returns none;
- 270 cases not compared: naz with two sample circles held (180; diffcalc
  lists the mode but raises "No code yet to handle this combination") and naz
  with bisect (90; neither diffcalc nor hklforge solves it).

### Where diffcalc is wrong

Found by this comparison and checked independently:

1. **eta, phi and one detector circle held, |eta| > 90**: diffcalc returns
   only wrong positions (hkl errors 0.5 to 9.5) and then its own self-check
   raises "The angles calculated for hkl … were …". hklforge's positions in those
   cases go back to the requested hkl through diffcalc's own `get_hkl` to
   ~1e-15. 15 such cases in the tally (eta/phi with delta, nu or qaz).
2. **bisect modes**: next to the correct positions diffcalc often returns
   others that satisfy neither bisect (chi axis 0.3 to 0.8 out of the
   scattering plane) nor the requested omega: an inconsistent pairing of its
   mu and eta branches. 984 such positions in 245 comparisons (in some modes
   it also lists correct positions twice). The correct ones agree with
   hklforge exactly.
3. **eta + bisect**: diffcalc sometimes crashes with `ValueError: math domain
   error` (25 cases); hklforge's solutions there satisfy hkl (diffcalc
   `get_hkl`) and bisect.
4. **naz with a surface constraint (betain, betaout, bin_eq_bout)**: diffcalc
   solves naz for the surface normal but then filters its results by naz of
   the reference vector, so it finds nothing unless the two vectors coincide.
   With them made equal, its results match hklforge's exactly.
5. `ubcalc.n_phi = v` is stored without normalising and alpha/beta are then
   computed with |n| ≠ 1; the tests give diffcalc unit vectors.

## Solutions: libhkl (E6C)

`dev/reference/compare_libhkl_planner.py` (needs libhkl; environment recipe in
the script). **All 13 E6C hkl modes**, 20 random lattices, orientations,
wavelengths and held values each; libhkl (a numerical solver) is run from 12
random starts per case and the union taken.

| modes | cases | libhkl positions | hklforge positions | libhkl ∉ hklforge |
|---|---|---|---|---|
| constant_omega/chi/phi_vertical, constant_mu_horizontal | 80 | 173 | 178 | 0 |
| lifting_detector_mu/omega/phi | 60 | 188 | 188 | 0 |
| double_diffraction_vertical/horizontal | 40 | 150 | 166 | 0 |
| bissector_vertical/horizontal | 40 | 446 | 480 | 0 |
| psi_constant_vertical/horizontal | 40 | 112 | 112 | 0 |

Every libhkl solution is one of hklforge's (within 1e-3 deg; libhkl's own hkl
error is ~1e-6, its constants differ slightly from diffcalc's), and hklforge
has more solutions in 27 cases. Notes: libhkl's bisector modes are
eta = delta/2 (or mu = gamma/2) modulo 180, i.e. hklforge omega = 0 or 180;
they match hklforge's bisect/omega only with the other held axes at 0, so they
were run that way. Its psi uses (h2, k2, l2) as the reference vector.

## Solutions: geometries without an oracle

`tests/test_solve_generic.py`: random geometries (random axis directions,
random beam and azimuth frame, 1-4 sample and 1-2 detector circles), 40
trials for each planner case (all x-group constraints, naz, qaz, omega, h2,
detector between). Random true angles set hkl and the constraint values; the
true position must be among the solutions, and every solution must satisfy
hkl and all constraints. Passes for every case.

naz with two sample circles (degree-8 polynomial, no diffcalc code):
`python dev/check_naz_scan.py 40` scans the two free circles on a 0.5° grid,
polishes every sign-change cell with scipy and compares: 33 cases, 48
sample settings found by both, none by only one.

## Double diffraction

`tests/test_double.py`, against the Bernina reference
(`dev/reference/bernina_two_reflections.py`, scan + root polish, itself checked
against diffcalc and libhkl): detector between, Si 111 + 220 for every held
pair, hexagonal (2,0,14) + (0,2,14), 12 random pairs: same sets of positions
and the same `tdet max`. The reference is ~30x slower and hangs for h1 = h2,
which hklforge reports as a continuum. Detector on h1: 8 positions for Si
111 + 220 at 9 keV in both libhkl double-diffraction modes, each in Bragg for
both reflections by diffcalc's forward model.

## Not verified

- psi for other reference conventions than diffcalc/libhkl's (there is one).
- Kappa geometries, axis limits: not implemented yet.
- The extra positions hklforge finds beyond libhkl were checked by hklforge's
  forward model only (which equals diffcalc's to 1e-12).
