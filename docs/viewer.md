# The 3D view, the geometry explorer and the stored geometries

## Stored geometries

`Geometry.builtins()` lists them; `Geometry.builtin(name)` loads one.

| name | what | lab frame | checked against |
|---|---|---|---|
| `psic` | You (1999) 4S+2D six-circle (diffcalc; libhkl E6C) | x up, y beam | diffcalc, libhkl E6C |
| `fourcv` | Eulerian four-circle, vertical scattering (libhkl E4CV) | x up, y beam | diffcalc (psic with mu = nu = 0) |
| `e4ch` | Eulerian four-circle, horizontal scattering: libhkl E4CH | x beam, z up | libhkl, forward model |
| `k4cv` | kappa four-circle, vertical scattering: libhkl K4CV | x beam, z up | libhkl, forward model |
| `k6c` | kappa six-circle: libhkl K6C | x beam, z up | libhkl, forward model |

The libhkl ones keep libhkl's names, axes, order and frame. Their forward
models agree with libhkl to 3e-6 in hkl for 500 random positions each
(`dev/reference/compare_libhkl_builtins.py`; the remaining difference is
libhkl's constants, as for E6C). In K4CV/K6C the kappa axis is tilted by 50°
from the komega axis (−y) towards the vertical (−z).

Not stored: libhkl's ZAXIS and SIXS MED geometries turn the detector with the
sample's `mu` as well - one circle in both stacks - which hklforge's model
(two independent stacks) does not express yet.

`Geometry.to_dict()` / `Geometry.from_dict()` round-trip a geometry as plain
data; `hklforge.view.explorer.geometry_yaml(g)` writes it as YAML.

## The 3D view (`hklforge.view`)

One self-contained page (three.js inlined, no network): `VIEWER_HTML`. It
draws any geometry: a ring per circle about its axis, nested outermost first,
the sample on the innermost sample circle and the detector arm on the
innermost detector circle, with sliders for every circle. On top: k, k′, Q,
the Ewald sphere and the detector's cut of it, the reciprocal and real cell,
cells and Brillouin zones around Q and around target reflections, and:

- **Virtual angles**: each of 2θ, qaz, naz, α, β, βin, βout, τ, ψ, ω can be
  ticked on its own; it is drawn at the sample from the vectors that define it
  (the frame it is measured in, the plane it refers to, the arc with its
  value), and unfolds to its definition. The values follow the circles live.
- **Brillouin-zone special points and paths** (Γ, X, L, ... from
  `hklforge.crystal.bz.special_points`) around the zone Q is in.
- **Circle axes and names**, and a **focus** on one circle (the others fade):
  how the stack is built.
- **Structure factors as spheres**: one sphere per reflection at its
  reciprocal-lattice node, turning with the sample, its **volume proportional
  to |F|²** (radius ∝ |F|^(2/3)), so sizes compare intensities. A size slider
  (0.01× to 20×, logarithmic, or typed exactly) scales every sphere together;
  at 1× the strongest is 0.8 of the shortest of a*, b*, c* across. Forbidden
  reflections get none. The data come from the app:
  `structure_factor_nodes(structure, energy)` lists every allowed node with
  d ≥ λ/2 (all equivalents, nearest first, at most 4000) with its |F|² at that
  energy (f′ + i f″ included, so Friedel pairs differ near an edge); pass it as
  `message(..., structure_factors=...)`. Needs xraydb (`hklforge[crystal]`).

```python
from hklforge import Geometry
from hklforge.view import page, iframe, message, payload_html

html_text = page(Geometry.builtin("k6c"))          # a standalone page: save it and open it
# in a notebook (ipywidgets): the page in an iframe, plus a hidden element it reads
frame = widgets.HTML(iframe(page(geometry), height=700))
holder = widgets.HTML()
holder.value = payload_html(message(geometry, angles, ub=d.ub, k=d.k,
                                    surface=d.surface, reference=d.reference,
                                    special_points=sp, n=counter,
                                    structure_factors=structure_factor_nodes(st, energy)))
```

A message carries lab-frame vectors as hklforge has them (UB and the normals in
the phi frame); the page turns them into its own frame (scene X = up, Y =
beam). The page reads `<span id="{view_id}-payload">` from its parent
document every 150 ms - possible because an `<iframe srcdoc>` has its
parent's origin - so no custom widget and no server is needed, and standalone
the read simply fails. `view_id` lets two views share a notebook; `storage`
prefixes the browser-storage keys of the remembered state and snapshots.

**Verified** (`dev/check_viewer.py`, headless Chrome): for psic, fourcv, e4ch,
k4cv, k6c and psic in a beam-z/up-y frame, random UB, surface, reference and
positions, the page's own measured hkl equals `Diffractometer.hkl` to 3e-15
and its virtual angles equal `hklforge.virtual` to 3e-13 degrees.

Structure-factor spheres (2026-11-20, headless Chromium, GaAs at 9 keV, 560
reflections): the page draws one sphere per reflection with |F| > 0, every
radius equals 0.4·min(|a*|, |b*|, |c*|)·factor·(|F|²/max|F|²)^(1/3) from
Python's |F|² to 1e-4 at slider 1×, 2.5× and typed 0.5× (exact when typed,
the slider resolution otherwise), and none are drawn with the option off.
`tests/test_view.py` checks the data: the same |F|² as `reflection_list`,
all reachable allowed nodes present, nearest first, Friedel pairs differing at
the Ga K edge.

## The geometry explorer (`hklforge.view.explorer`)

`GeometryExplorer(geometries=None, default=None).widget` (ipywidgets): pick
a stored geometry (or pass your own, or type YAML), then step through how it
is built - the lab frame and the beam, each sample circle from the outermost
in, each detector circle - with the new circle highlighted in the 3D view and
a short text: what it turns about, what it sits on, what it carries, the
matrix product so far. The table lists every circle; the YAML can be edited
and shown.
