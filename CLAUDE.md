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

## The planner and virtual angles (done, tested)

`src/hklforge/solve.py`: `plan(geometry, mode, detector="h1"|"between")`
checks a mode once and picks the chain (`Plan.case`, `Plan.describe()`);
`Plan.run` returns every position (free circles wrapped to (-180, 180], held
ones as given), each re-checked through the forward model and its constraints.
`Diffractometer(geometry, ub, wavelength=|energy=, surface=|surface_hkl=,
reference=|reference_hkl=)`: `.solve`, `.hkl`, `.pseudo` (all virtual angles),
`.spots` (k' per reflection, separation, `tdet_max`). A mode is a diffcalc-like
dict: circles -> degrees, plus `betain betaout bin_eq_bout alpha beta a_eq_b
psi h2 naz qaz bisect omega` (`vomega` if a circle is called omega).

`src/hklforge/virtual.py` defines every virtual angle from lab vectors (=
diffcalc to 3e-13; psi sign = diffcalc = libhkl). `omega` is
atan2(e.Q, e.p) (e = bisect-circle axis, p = beam across Q), *not* diffcalc's
asin form: with bisect it then fixes e uniquely, which is what diffcalc's
formulas mean. Geometry YAML keys `azimuth_zero` (qaz/naz frame; default the
lab axis after the beam, cyclic) and `bisect_circle` (psic/fourcv: chi).

Constraint groups (docs/modes.md has the full case table):
**x** (fix x = R^T b): betain/out, in=out, alpha/beta, a_eq_b, h2 (one
projection each), psi (x completely) - at most one, Bragg uses the other DOF;
**qaz** fixes k' (detector first); **naz**: m.R n = 0 + side, the one place
`solve_projection_pair` (degree 8) is used (naz + 2 sample circles);
**bisect/omega** act on the bisect-circle axis via the outer circles, then
`sp2` on the two inner ones. Cases: beam, beam_x, beam_naz, beam_x_naz, q,
q_x, q_naz, axis, between. Not supported: naz + bisect (nor in diffcalc),
nD=0, nD>2, nS>3 except axis.

Also: `src/hklforge/draw.py` (matplotlib 3D: `draw(d, angles, show=, view=,
hkls=)`, `draw_circles`), `src/hklforge/lattice.py` (`b_matrix`, = diffcalc's
B to 1e-12). Docs in `docs/` (figures from `docs/make_figures.py`), executed
example notebooks in `examples/`, generated by `dev/build_notebooks.py` and
then executed with nbconvert (edit the generator, not the notebooks).

### What was verified (2026-10-22) - details in docs/verification.md

- diffcalc, all 194 psic combinations it implements (`tests/test_solve_diffcalc.py`,
  `dev/diffcalc_stats.py 30`): 5,510 cases, 3,364 with solutions, identical
  sets wherever diffcalc answered correctly. Not comparable: naz + 2 sample
  (diffcalc "No code yet", 180 cases) -> checked by brute-force scan instead
  (`dev/check_naz_scan.py`, 33 cases, 48 = 48); naz + bisect (neither).
- diffcalc bugs found: (1) eta, phi + 1 detector held, |eta| > 90: only wrong
  positions, own self-check raises (15 cases); (2) bisect modes: extra
  positions violating bisect/omega (984 in 245 comparisons) and duplicates;
  (3) eta + bisect: ValueError math domain (25); (4) naz + surface
  constraint: solved for the surface normal, filtered by the reference vector
  -> answers only if they coincide; (5) `n_phi` not normalised.
- libhkl, all 13 E6C modes (`dev/reference/compare_libhkl_planner.py`, 260
  cases, 12 starts each): every libhkl solution is in hklforge's set, hklforge
  has more in 27 cases. libhkl bissector = omega 0 or 180 (held axes 0).
- random geometries, every planner case, truth recovered
  (`tests/test_solve_generic.py`); Bernina reference for "between" + tdet
  (`tests/test_double.py`; the reference hangs for h1 = h2).
- Speed ~2-7 ms per solve, about diffcalc's; not optimised.

## Next

Kappa <-> Euler conversion modes (a kappa *geometry* already works, see
examples/05), axis limits and preferred-branch selection, UB from reflections
(Busing-Levy) and refinement, an interactive (three.js) version of the 3D
help building on `~/mypy/bernina-scripts/apps/dev/diffractometer_3d.html`,
optional code generation of a plan into readable Python, naz + bisect.

## Verification standard (keep it)

- **diffcalc is the oracle for psic**: for every constraint combination it
  supports, the *set* of branches must match, not just one solution. Compare
  as sets with angle wrapping; diffcalc's angles can exceed +-180.
- Every solver result is re-checked through the forward model.
- Double diffraction: compare with `dev/reference/bernina_two_reflections.py`
  and, if libhkl is available, libhkl's modes (`dev/reference/compare_libhkl_planner.py`).
- Where no oracle exists: random-geometry truth recovery, and a brute-force
  scan for anything using the degree-8 polynomial.
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

Public at https://github.com/swissfel/hklforge (`main`), BSD-3-Clause,
copyright Henrik Lemke. On PyPI as `hklforge` since 0.0.1 (2026-10-25).
CI: `.github/workflows/ci.yml` (tests, Python 3.10/3.12/3.13; test_double
skipped there). Release: bump `version` in pyproject.toml, commit, then the
owner pushes a tag `vX.Y.Z` -> `publish.yml` tests, checks tag = version,
builds and uploads via PyPI trusted publishing (environment `pypi`). Tag
pushes need the owner's go-ahead (they publish). `.vscode/settings.json` holds the
owner's local editor changes - leave them out of commits.
