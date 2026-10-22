# hklforge

Diffractometer angle solvers for **any declared geometry**, generated **exactly**
(closed-form sub-problems, all branches) instead of hand-written for one
instrument or found by numerical search. Started 2026-10-22 out of the
bernina-scripts work (`~/mypy/bernina-scripts`, SwissFEL Bernina endstation).

## Why this exists (what was measured, not assumed)

Tested side by side, same UB, in a Binder-like env (details and scripts in
`dev/reference/`, write-up in
`~/mypy/bernina-scripts/apps/dev/diffraction_two_reflections.ipynb`):

- **diffcalc-core 0.4.0**: You (1999) 6-circle only. Hand-derived closed forms,
  so every branch, in ~3 ms. Any valid 3 of sample/detector/reference constraints.
  No double diffraction. Last release 2022; upstream `main` has unreleased
  changes (#58 changes angle units in the API).
- **libhkl** (C/GObject, via PyGObject; behind hklpy and hklpy2's `hkl_soleil`
  solver): E6C is exactly the You geometry (eta = omega, nu = gamma; lab
  x = beam). Forward agrees with diffcalc to ~1e-6 (constants differ slightly).
  Numerical solver: returned **2 of diffcalc's 4** branches. Only 13 named E6C
  modes; no fixed incidence / alpha = beta / naz / qaz. Has `double_diffraction_*`
  (detector on h1): all 20 of its solutions were reproduced by the bernina solver.
  New geometries need C. GPL. Binder-compatible (no downgrades).
- **ad_hoc_diffractometer 1.0** (pure Python, YAML geometries, Argonne; hklpy2's
  `ad_hoc` solver): forward = diffcalc to 1e-14. Generic Newton-from-seeds
  solver: **misses branches** (1 of 4 for (1,0,10), eta=chi=phi=0; its own
  forward model confirms the missing ones), ~50-100x slower, fixed incidence
  20-30 s and half the solutions. Note: its `surface_normal` is Miller indices;
  stage limits default to +-180 (diffcalc happily returns mu = 183.4).
- **xrayutilities**: `QConversion` handles arbitrary geometries for angles -> Q
  (reciprocal-space maps); Q -> angles only via `Q2AngFit` (local fit, one solution).

So nobody combines *custom geometry* with *complete, exact* solving. That gap
is hklforge's purpose. diffcalc must come out as a special case.

## The owner's requirements (Henrik Lemke, SwissFEL Bernina)

- **Symbolic/exact solver generation first**; numeric scanning only as a
  fallback or cross-check.
- Geometry definable in **YAML and in xrayutilities' syntax** (`'x+'`, `'z-'`,
  outermost circle first, `r_i` = beam). Done, see below.
- **Two reflections at once** (double diffraction), with the detector either on
  reflection 1 (libhkl's choice) or between both outgoing beams, plus a
  suggested detector distance that keeps both spots on the face.
- **Good help with 3D drawings** explaining each quantity (k, k', Q, 2theta,
  surface normal, alpha/beta, betain/betaout, psi, naz, qaz, tau...), because
  these are "often difficult to track down". The bernina 3D view
  (`~/mypy/bernina-scripts/apps/dev/diffractometer_3d.html`, three.js,
  self-contained) is a reusable starting point.
- He prefers being shown evidence: say what was verified and how, and what was not.

## Model (done, tested)

`src/hklforge/geometry.py`: a geometry is two stacks of right-handed circles
about fixed lab axes (given at all angles zero), outermost first:
sample R = R_1...R_n (phi frame -> lab), detector D with k' = D k_in. Built-ins
in `src/hklforge/geometries/*.yml` (`psic`, `fourcv`). `Geometry.from_xrayutilities`
takes xu axis strings. Units: degrees in the API, |k| = 2 pi / lambda
(`wavevector()`), the 2 pi convention of diffcalc and libhkl, so
|UB h| = 2 pi / d.

`tests/test_forward.py` verifies: psic `q_phi` = diffcalc `get_q_phi` (1e-12,
random angles); xu-syntax geometries = xrayutilities `QConversion` output on
three different geometries including non-psic ones (so the xu sign convention
and "Q in sample frame" are confirmed, not assumed).

## Exact sub-problems (done, tested)

`src/hklforge/rotation.py`, all closed form, each returns **every** solution:

| function | solves | sols |
|---|---|---|
| `solve_projection(w, u, v, c)` | w . rot(u,t) v = c | <= 2 |
| `sp1(u, p, q)` | rot(u,t) p = q | 1 |
| `sp2(u1, u2, p, q)` | rot(u1,t1) rot(u2,t2) p = q (Paden-Kahan 2) | <= 2 |
| `sp3(u1, u2, u3, T)` | rot(u1)rot(u2)rot(u3) = T | <= 2 |
| `cone_points(u1,c1,u2,c2,L)` | z with u1.z=c1, u2.z=c2, abs(z)=L | <= 2 |
| `rotation_from_pairs` | rotation taking s1->t1, s2->t2 | 1 |
| `solve_projection_pair(eqs, u1, u2)` | two equations w_k . rot(u1,t1)rot(u2,t2) v_k = c_k | <= 8 |

The pair solver eliminates t2 (both equations linear in cos t2, sin t2) ->
degree-4 trig polynomial in t1 -> degree-8 polynomial in z = e^{i t1}; its
unit-circle roots are all t1. Coefficients come from a 16-point FFT (exact for
degree 4). `Degenerate` is raised for continua (parallel axes etc.).
`tests/test_rotation.py`: 200+ randomized checks that known solutions are
always found and every returned one satisfies the equation.

## Next: the planner (the core of "solver generation")

Fixed rotations between free ones are pushed to the right with
F rot(a,t) = rot(F a, t) F, so any chain becomes
rot(b1,t1) rot(b2,t2) rot(b3,t3) . F_all with b_i in the lab frame.
Notation: b = beam unit vector, k = |k|, q = UB h (phi frame), Q_lab = R q.
Bragg for a reflection is the projection equation **b . R q = -|q|^2 / (2k)**;
fixed incidence (betain) is **b . R n = -sin(betain)** (n = unit surface normal,
phi frame) - the same form. Exit angle betaout is **k'. R n = sin(betaout)**.

Mode = which circles are fixed (values), plus extra equations (incidence,
exit, second reflection), plus a detector policy. nS / nD = free sample /
detector circles. Cases, detector on h1 (default):

| case | equations -> steps | branches |
|---|---|---|
| nD=2, nS=1 (e.g. Bernina: eta, chi, phi fixed) | Bragg projection on the 1 sample circle -> k' = b + Rq/k -> `sp2` on detector | 2x2 = 4 |
| nD=1, nS=2 (1 detector + 2 sample fixed) | b . D b = cos 2theta on the 1 detector circle -> Q_lab -> `sp2` sample: (F q) -> Q_lab | 2x2 = 4 |
| nD=2, nS=2 + incidence (e.g. chi, phi, betain fixed) | Bragg + incidence projections -> `solve_projection_pair` -> k' -> `sp2` detector | <= 8x2 |
| nD=1, nS=3 + one extra (e.g. mu, nu, betain; or libhkl double_diffraction: mu, nu fixed + h2) | 2theta -> Q_lab; second vector's target from `cone_points` (angle to Q_lab preserved, projection on b fixed, length kept) -> `rotation_from_pairs` -> `sp3` | 2x2x2 = 8 |
| nD=1, exit-angle constraint | k' known after the detector step, so betaout is a plain projection on the sample | - |
| nD=0 (both detector fixed) | |Q| generically unreachable -> reject (diffcalc rejects it too) | - |

Detector **between** two reflections (bernina mode): both Bragg projections on
the sample -> `solve_projection_pair` (needs nS=2), then each k'_i, bisector,
`sp2` detector, plus tdet max. Reference implementation with tests:
`dev/reference/bernina_two_reflections.py` (1-D reduction + scan, verified
against diffcalc get_hkl and all 20 libhkl solutions) - the new exact version
should reproduce its branch sets.

Expected counts to check against: diffcalc gives 4 for Bernina eta/chi/phi
fixed, 8 for (mu, nu, betain); libhkl double_diffraction_vertical gives 8 for
Si 111+220 at 9 keV.

Later: psi, alpha/beta (reference vector instead of surface normal - same
equation form as betain/betaout), naz, qaz, bisect, alpha = beta, kappa
geometries, axis limits, UB from reflections (Busing-Levy), optional code
generation of a plan into readable Python, the 3D help.

## Verification standard (keep it)

- **diffcalc is the oracle for psic**: for every constraint combination it
  supports, the *set* of branches must match, not just one solution. Compare
  as sets with angle wrapping; diffcalc's angles can exceed +-180.
- Every solver result is re-checked through the forward model.
- Double diffraction: compare with `dev/reference/bernina_two_reflections.py`
  and, if libhkl is available, libhkl's modes (`dev/reference/compare_libhkl_e6c.py`).
- Report counts and max errors, not "looks right".

## Environment

- Dev: `/mnt/app/python/.pixi/envs/bpy312/bin/python` has numpy, pyyaml,
  pytest, scipy, sympy, diffcalc-core 0.4.0, xrayutilities. Run tests with
  `cd ~/mypy/hklforge && /mnt/app/python/.pixi/envs/bpy312/bin/python -m pytest`
  (pytest config adds `src` to the path). Do not pip-install into that shared
  env without asking the owner.
- The `dev/reference/compare_*.py` scripts ran in a throwaway env built like
  mybinder (repo2docker lock + `bernina-scripts/binder/environment.yml`) plus
  `micromamba install -c conda-forge hkl` and `pip install ad_hoc_diffractometer`.
  Rebuild a similar env to rerun them; they import bernina-scripts from
  `/home/lemke_h/mypy/bernina-scripts`.
- Naming at Bernina: the out-of-plane detector circle diffcalc calls **nu**
  is called **gamma** at the beamline; the SwissFEL lab frame is diffcalc's
  relabelled (SwissFEL x,y,z = diffcalc z,x,y). hklforge itself just uses the
  names a geometry declares.

## Git

Local repo only so far (`main`), no remote. The owner keeps related code under
github.com/swissfel; ask before creating a remote or pushing.
