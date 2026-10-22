# Modes: what to hold, and how it is solved

```python
from hklforge import Diffractometer, Geometry
d = Diffractometer(Geometry.builtin("psic"), ub, energy=9000,
                   surface_hkl=(0, 0, 1), reference_hkl=(0, 0, 1))
d.solve((1, 1, 2), {"chi": 10, "phi": 20, "betain": 5})   # list of dicts, degrees
print(d.plan({"chi": 10, "phi": 20, "betain": 5}).describe())
```

A **mode** is a dict, as in diffcalc: circle names with the values they are held
at, plus virtual-angle constraints ([definitions](virtual_angles.md)):

| key | value |
|---|---|
| any circle name | degrees |
| `betain`, `betaout`, `alpha`, `beta`, `psi`, `naz`, `qaz`, `omega` | degrees |
| `bin_eq_bout`, `a_eq_b`, `bisect` | `True` |
| `h2` | (h, k, l) of a second reflection to hold in Bragg condition |
| `vomega` | `omega`, for a geometry that has a circle named omega |

`solve` returns **every** solution: free circles wrapped to (−180, 180], held
circles as given. Each solution is re-checked through the forward model
(hkl and every constraint to 1e-8) before it is returned. `detector="between"`
(with `h2`) puts the detector on the bisector of the two outgoing beams instead
of on reflection 1.

## How many things to hold

Count the free circles: nS on the sample, nD on the detector.

- The **detector is solved first** when nD = 1 (Bragg fixes it: b·k' = cos 2θ),
  or when nD = 2 and `qaz` is held (2θ and qaz give k'). Then Q in the lab is
  known and the sample must satisfy R q = Q_lab: **nS = 2**, plus one per
  sample constraint (x-group or naz, at most one of them).
- Otherwise (nD = 2, no qaz) the **sample is solved first**: Bragg is one
  equation on the sample, **nS = 1**, plus one per constraint (one x-group, and
  optionally naz). The detector then turns onto k' = b + R q/|k|.
- `bisect` / `omega` need the detector first and act on the circles outside
  the bisect circle: one free outer circle per axis constraint, and exactly two
  free circles from the bisect circle inwards.
- nD = 0 is rejected (|Q| would be fixed), and so is anything that leaves a
  continuum or over-determines; the `ModeError` says which and why.

The counts in the table below are for a near-cubic crystal (a = 4 Å, 9 keV,
(1,1,2)); a given hkl can have fewer (e.g. `{"qaz": 90, "chi": 0, "phi": 0}`
has none there, as the two held circles cannot reach that Q), or raise
`Degenerate` (mu = chi = 0 leaves eta and phi on one axis).

For psic that reproduces diffcalc's rule "one detector + one reference + one
sample constraint, or two of one kind…", but it is derived from the geometry,
so the same rules hold for any stack of circles.

## The cases

`Plan.case` names the chain of exact sub-problems; `Plan.describe()` prints it
for a mode. All are closed form (no search); the counts are the most
solutions each step can give.

| case | when | steps | max |
|---|---|---|---|
| `beam` | nD=2, nS=1 | Bragg: one projection equation in the sample circle → k' → `sp2` detector | 4 |
| `beam_x` | nD=2, nS=2, one x-group | x from Bragg + constraint (`cone_points`, or one point for psi) → `sp2` for R x = b → `sp2` detector | 8 |
| `beam_naz` | nD=2, nS=2, naz | Bragg and naz: two projection equations in two circles → degree-8 polynomial (`solve_projection_pair`) → `sp2` detector | 16 |
| `beam_x_naz` | nD=2, nS=3, x-group + naz | x → R = rot(b, t) R₀ with R₀ x = b; naz fixes t (projection) → `sp3` → `sp2` detector | 8 |
| `q` | detector first, nS=2 | `sp2` for R q = Q_lab | 4 |
| `q_x` | detector first, nS=3, one x-group | x → R from (q, x) ↦ (Q_lab, b) → `sp3` | 8 |
| `q_naz` | detector first, nS=3, naz | R = rot(Q̂, t) R₀ with R₀ q = Q_lab; naz fixes t → `sp3` | 8 |
| `axis` | detector first, bisect / omega | outer circles set e (projection, or e = cos ω p + sin ω Q̂ and `sp2`) → inner two: `sp2` | 8 |
| `between` | `h2`, detector between | x from both Bragg conditions → `sp2` sample → `sp2` detector onto unit(k'₁ + k'₂) | 8 |

Free and held circles can be anywhere in a stack: held rotations are moved to
the right with F rot(a, t) = rot(F a, t) F, so each stack becomes
rot(b₁, t₁)…rot(bₘ, tₘ)·F with lab axes bᵢ.

Where the solutions form a continuum (e.g. psic chi = 0 makes eta and phi the
same axis; h2 = h1) `Degenerate` is raised with the mode in the message;
contradictory equations on parallel vectors just give no solution.

## psic examples, with diffcalc's names

| what | mode | case | typical count |
|---|---|---|---|
| Bernina: eta, chi, phi held | `{"eta": 0, "chi": 0, "phi": 0}` | beam | 4 |
| fixed incidence, 2 sample held | `{"chi": 10, "phi": 20, "betain": 5}` | beam_x | 8 |
| psi, vertical (mu, nu held) | `{"mu": 0, "nu": 0, "psi": 30}` | q_x | 4 |
| vertical scattering plane, mu and phi held | `{"qaz": 90, "mu": 0, "phi": 0}` | q | 4 |
| reference azimuth + alpha | `{"naz": 30, "alpha": 5, "phi": 0}` | beam_x_naz | 8 |
| bisecting, vertical | `{"nu": 0, "bisect": True, "omega": 0}` | axis | 8 |
| bisecting with mu held | `{"delta": 40, "mu": 0, "bisect": True}` | axis | 8 |
| two reflections, detector on h1 (libhkl) | `{"mu": 0, "nu": 0, "h2": (2, 2, 0)}` | q_x | 8 |
| two reflections, detector between | `{"eta": 0, "chi": 0, "h2": (2, 2, 0)}`, `detector="between"` | between | 8 |

Not supported (and not in diffcalc either): naz together with bisect / omega;
more than one x-group constraint (it would over-determine x).

## Double diffraction and the detector distance

```python
sols = d.solve((1, 1, 1), {"eta": 0, "chi": 0, "h2": (2, 2, 0)}, detector="between")
d.spots(sols[0], [(1, 1, 1), (2, 2, 0)], width=77, height=77)
# {'kout': [...], 'off_axis': [a, a], 'separation': ..., 'tdet_max': ...}
```

`spots` gives each reflection's outgoing direction, its angle from the detector
axis, their separation and `tdet_max`: the largest sample-detector distance
(units of width / height) that keeps every spot within `margin` (0.9) of the
face's half sizes. The face axes default to the two lab axes after the beam in
cyclic order (psic: width along z, height along x), as in the Bernina script.

## Other geometries

```python
g = Geometry.from_xrayutilities(["x+", "z-", "y+", "z-"], ["x+", "z-"], r_i=(0, 1, 0),
                                sample_names=["mu", "eta", "chi", "phi"],
                                detector_names=["nu", "delta"])
g.bisect_circle = "chi"          # only needed for bisect / omega
```

or a YAML file like [`psic.yml`](../src/hklforge/geometries/psic.yml) with
`azimuth_zero` and `bisect_circle`. The planner needs nothing else: modes,
counting and solving follow from the axes.
