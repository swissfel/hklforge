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

## The planner (done, tested)

`src/hklforge/solve.py`. `plan(geometry, mode, detector="h1"|"between")`
validates a mode once and picks the chain of sub-problems (`Plan.describe()`
prints it); `Plan.run(ub, k, hkl, surface, reference)` returns every position
as a dict of degrees (free circles wrapped to (-180, 180], held ones as
given). `Diffractometer(geometry, ub, wavelength=|energy=, surface=|surface_hkl=,
reference=|reference_hkl=)` wraps it: `.solve(hkl, mode)`, `.hkl(angles)`,
`.pseudo(angles)` (ttheta, betain/betaout, alpha/beta), `.spots(angles, hkls,
width, height)` (each k', separation, `tdet_max`, same face rule as the
bernina reference). A mode is a diffcalc-like dict: circle names -> degrees,
plus at most one of `betain`, `betaout`, `bin_eq_bout` (surface normal),
`alpha`, `beta`, `a_eq_b` (reference vector), `h2` (second reflection).

Fixed rotations are pushed right (F rot(a,t) = rot(F a, t) F), so a stack is
rot(b1,t1)...rot(bm,tm) F with lab axes b_i. Key reduction: **every
supported extra is a projection of x = R^T b** (beam seen from the sample):
Bragg b.Rq = -|q|^2/2k, betain b.Rn = -sin, betaout via k' = b + Rq/k gives
b.Rn = sin(betaout) - q.n/k, in=out gives b.Rn = -q.n/2k, alpha/beta the same
with the reference vector, h2 b.Rq2 = -|q2|^2/2k. x has 2 DOF and Bragg uses
one, so **at most one extra** (more is rejected with an explanation). Cases:

| case | steps | branches |
|---|---|---|
| nD=2, nS=1 (Bernina eta, chi, phi fixed) | Bragg `solve_projection` -> k' -> `sp2` detector | <= 4 |
| nD=2, nS=2 + extra (chi, phi, betain) | x from `cone_points` -> `sp2` for R x = b -> `sp2` detector | <= 8 |
| nD=1, nS=2 (delta, mu, chi) | b.k' = cos 2theta on the detector -> Q_lab -> `sp2` R q = Q_lab | <= 4 |
| nD=1, nS=3 + extra (mu, nu, betain; mu, nu, h2) | Q_lab; image of the extra's vector by `cone_points` -> `rotation_from_pairs` -> `sp3` | <= 8 |
| between (2 sample held + h2) | x from both Bragg conditions -> `sp2` sample -> `sp2` detector onto unit(k'1 + k'2) | <= 8 |

nD=0, nS>3, 3 detector circles: rejected (`ModeError`). Continua (parallel
free axes, h2 = h1, ...) raise `Degenerate` with the mode in the message;
parallel-but-contradictory equations return []. `solve_projection_pair` is
not needed by any of these (all equations share w = b); keep it for psi.

### What was verified (2026-10-22)

- `tests/test_solve_diffcalc.py`: all 100 diffcalc psic combinations the
  planner covers (3 sample; 1 det + 2 sample; 2 sample + 1 ref; 1 det + 1 ref +
  1 sample; refs betain, betaout, bin_eq_bout, alpha, beta, a_eq_b), random
  lattice/U/lambda/reachable hkl/values. `dev/diffcalc_stats.py 30`: 2,991
  cases, 1,597 with solutions (2-8 branches), **branch sets identical** in
  every case diffcalc answered; diffcalc's "prohibits a solution" cases all
  give [] here too.
- **diffcalc bug found**: with eta, phi and one detector circle fixed and
  |eta| > 90, diffcalc returns only wrong positions (hkl errors 0.5-9.5) and its
  own self-check raises ("The angles calculated for hkl ... were"). 9 of 60
  such cases. hklforge's solutions there verified with diffcalc's own
  `get_hkl` to ~1e-15. The test checks this explicitly.
- diffcalc quirk: `ubcalc.n_phi = v` is stored un-normalised and alpha/beta
  use it raw; give it unit vectors when comparing (betain/betaout normalise).
- `tests/test_solve_generic.py`: random geometries (random axes and beam),
  every case type incl. between: the true position is always among the
  solutions and all solutions satisfy the forward model; bad modes explained.
- `tests/test_double.py`: "between" reproduces the bernina reference's branch
  sets and `tdet max` (Si 111+220 all held pairs, hex (2,0,14)+(0,2,14),
  12 random); the reference is ~30x slower and **hangs for h1 = h2**.
  Detector on h1: 8 branches for Si 111+220 at 9 keV in both libhkl
  double-diffraction modes (the expected count), each in Bragg for both
  reflections by diffcalc's forward model.
- `dev/reference/compare_libhkl_planner.py` (throwaway env, see Environment):
  9 E6C modes (constant_omega/chi/phi_vertical, constant_mu_horizontal,
  lifting_detector_mu/omega/phi, double_diffraction_vertical/horizontal), 179
  random cases, libhkl from 12 random starts each: **every libhkl solution is
  in hklforge's set** (within 1e-3 deg; libhkl's own hkl error ~1e-6), and
  hklforge has more branches in 15 cases (22 positions). Not mapped yet:
  bissector_*, psi_constant_*.
- Speed: ~2 ms for 4 branches, ~5-7 ms for 8 (diffcalc 1.6-4.9 ms); not
  optimised (each solution is re-checked through the forward model).

## Next

psi (reference azimuth; needs `solve_projection_pair` with w != b), naz, qaz,
bisect / omega = delta/2 (libhkl bissector modes), kappa geometries, axis
limits and preferred-branch selection, UB from reflections (Busing-Levy),
optional code generation of a plan into readable Python, the 3D help.

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
- libhkl env used for `compare_libhkl_planner.py` (2026-10-22):
  `micromamba create -p <scratch>/hklenv -c conda-forge python=3.12 hkl pygobject numpy scipy pandas pyyaml pip`
  + `pip install diffcalc-core==0.4.0`. PyGObject pitfalls: keep the
  `Hkl.Sample`/`Detector` and the result of `pseudo_axis_values_set` alive in
  Python, or libhkl segfaults.
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
