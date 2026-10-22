"""Write the example notebooks into examples/ (then execute them:
jupyter nbconvert --to notebook --execute --inplace examples/0*.ipynb, from examples/)."""
import os

import nbformat as nbf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
os.makedirs(OUT, exist_ok=True)

SETUP = '''import sys, os, logging, warnings
sys.path.insert(0, os.path.abspath("../src"))      # hklforge from this repository
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hklforge import Diffractometer, Geometry, ModeError, Degenerate
from hklforge.lattice import b_matrix
from hklforge.draw import draw, draw_circles
logging.getLogger("diffcalc").setLevel(logging.ERROR)
pd.set_option("display.precision", 3)
%matplotlib inline'''

DIFFCALC_HELPERS = '''from diffcalc.ub.calc import UBCalculation
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.util import DiffcalcException

import contextlib, io

def diffcalc_ubcalc(lattice, U=np.eye(3), surface=None, reference=None):
    ubc = UBCalculation("example")
    ubc.set_lattice("x", *lattice)
    with contextlib.redirect_stdout(io.StringIO()):     # diffcalc prints "Calculating UB matrix."
        ubc.set_u(U)
    if surface is not None:
        ubc.surf_nphi = tuple(np.asarray(surface) / np.linalg.norm(surface))
    if reference is not None:
        ubc.n_phi = tuple(np.asarray(reference) / np.linalg.norm(reference))
    return ubc

def diffcalc_solve(ubc, hkl, mode, wavelength):
    """diffcalc's positions as dicts, [] if it finds none, or the error text."""
    try:
        return [p.asdict for p, _ in HklCalculation(ubc, Constraints(mode)).get_position(*hkl, wavelength)]
    except DiffcalcException as e:
        msg = " ".join(str(e).replace("*", " ").split())
        if "No solutions" in msg or "prohibits a solution" in msg:
            return []
        return msg
    except (AssertionError, ValueError) as e:
        return f"{type(e).__name__}: {e}"

def same(a, b, tol=1e-5):
    return all(abs((a[n] - b[n] + 180) % 360 - 180) < tol for n in a)

def same_set(xs, ys):
    return all(any(same(x, y) for y in ys) for x in xs) and all(any(same(y, x) for x in xs) for y in ys)'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c)
               for c in cells]
    return n


# --------------------------------------------------------------------------- 01
quick = [
'''md:# hklforge quickstart

hklforge computes diffractometer angles for **any geometry you declare**, and
returns **every** solution of a mode in closed form - no numerical search,
so no missed branches. This notebook goes through the basics on the psic
six-circle (You 1999, as in diffcalc) and compares with diffcalc on the way.

Further notebooks: [02 virtual angles](02_virtual_angles.ipynb),
[03 diffcalc comparison](03_diffcalc_comparison.ipynb),
[04 double diffraction](04_double_diffraction.ipynb),
[05 custom geometries](05_custom_geometries.ipynb). Definitions are in
[docs/virtual_angles.md](../docs/virtual_angles.md), modes in
[docs/modes.md](../docs/modes.md).''',
SETUP,
'''md:## A geometry

Two stacks of circles, outermost first, each a right-handed rotation about a
fixed lab axis (given at all angles zero). psic: lab x vertical, y along the
beam, z transverse.''',
'''psic = Geometry.builtin("psic")
print(psic)
draw_circles(psic, {"mu": 10, "eta": 20, "chi": 30, "phi": 0, "nu": 15, "delta": 30});''',
'''md:## A crystal

UB is in the phi frame with q = UB h, |q| = 2π/d (the convention of diffcalc and libhkl).
Here: silicon, U = identity, 9 keV, surface normal along (0 0 1).''',
'''si = (5.431, 5.431, 5.431, 90, 90, 90)
ub = np.eye(3) @ b_matrix(*si)
d = Diffractometer(psic, ub, energy=9000, surface_hkl=(0, 0, 1), reference_hkl=(0, 0, 1))
print(f"wavelength {d.wavelength:.4f} A, |k| = {d.k:.4f} 1/A")''',
'''md:## Solving: a mode is what you hold

As in diffcalc, a dict of circle values and virtual-angle constraints. The
Bernina default - eta, chi, phi held - leaves mu and the two detector circles:''',
'''mode = {"eta": 0, "chi": 0, "phi": 0}
sols = d.solve((1, 1, 1), mode)
pd.DataFrame(sols)''',
'''md:Every solution, re-checked through the forward model. The plan says how it was found:''',
'''print(d.plan(mode).describe())''',
'''md:The virtual angles of each solution ([definitions](../docs/virtual_angles.md)):''',
'''pd.DataFrame([d.pseudo(s) for s in sols])''',
'''md:## More modes

Fixed incidence, psi, a scattering-plane azimuth, bisecting, a second reflection:''',
'''modes = [
    {"eta": 0, "chi": 0, "phi": 0},
    {"chi": 10, "phi": 20, "betain": 2},
    {"mu": 0, "nu": 0, "betain": 2},
    {"mu": 0, "nu": 0, "psi": 30},
    {"qaz": 90, "mu": 0, "phi": 0},
    {"naz": 30, "alpha": 5, "phi": 0},
    {"nu": 0, "bisect": True, "omega": 0},
    {"delta": 40, "mu": 0, "bisect": True},
    {"mu": 0, "nu": 0, "h2": (2, 2, 0)},
]
rows = []
for m in modes:
    p = d.plan(m)
    rows.append({"mode": m, "case": p.case, "solutions": len(d.solve((1, 1, 3), m))})
pd.DataFrame(rows)''',
'''md:## A drawing of one solution''',
'''s = d.solve((1, 1, 3), {"mu": 0, "nu": 0, "betain": 2})[0]
fig = plt.figure(figsize=(12, 5.5))
draw(d, s, show=("ttheta", "betain", "betaout"), ax=fig.add_subplot(1, 2, 1, projection="3d"),
     title="2theta, betain, betaout")
draw(d, s, show=("qaz", "naz"), view="beam", ax=fig.add_subplot(1, 2, 2, projection="3d"),
     title="qaz, naz (looking downstream)");''',
'''md:## Side by side with diffcalc

Same lattice, same U, same normals. diffcalc returns angles beyond ±180, so
the sets are compared with wrapping.''',
DIFFCALC_HELPERS,
'''ubc = diffcalc_ubcalc(si, surface=(0, 0, 1), reference=(0, 0, 1))
print("same UB as diffcalc:", np.allclose(ubc.UB, ub))
rows = []
for m in modes[:-1]:                                    # diffcalc has no second reflection
    mine = d.solve((1, 1, 3), m)
    theirs = diffcalc_solve(ubc, (1, 1, 3), m, d.wavelength)
    rows.append({"mode": m, "hklforge": len(mine),
                 "diffcalc": theirs if isinstance(theirs, str) else len(theirs),
                 "identical sets": None if isinstance(theirs, str) else same_set(mine, theirs)})
pd.DataFrame(rows)''',
'''md:In the last row diffcalc lists each of its 8 positions twice (the sets are equal). In other
bisect modes it also returns positions that violate the mode (see
[03 diffcalc comparison](03_diffcalc_comparison.ipynb)) - e.g. delta held, omega = 25:''',
'''from hklforge.virtual import bisect_axis
m = {"delta": 30, "bisect": True, "omega": 25}
theirs = diffcalc_solve(ubc, (1, 1, 3), m, d.wavelength)
mine = d.solve((1, 1, 3), m)
rows = []
for a in theirs:
    e, Q = bisect_axis(psic, a), psic.q_lab(a, 1.0)
    rows.append({"chi axis out of scattering plane": e @ np.cross(psic.beam, Q / np.linalg.norm(Q)),
                 "omega": d.pseudo(a)["omega"], "also found by hklforge": any(same(a, b) for b in mine)})
pd.DataFrame(rows)''',
'''md:## Modes that do not work are explained''',
'''for m in [{"eta": 0, "chi": 0}, {"eta": 0, "betain": 1, "betaout": 2}, {"nu": 0, "delta": 0, "eta": 0},
          {"eta": 0, "chi": 0, "omega": 0}]:
    try:
        d.plan(m)
    except ModeError as e:
        print(m, "->", e, "\\n")
try:
    d.solve((1, 1, 1), {"mu": 0, "chi": 0, "qaz": 90})
except Degenerate as e:
    print("Degenerate:", e)''',
]

# --------------------------------------------------------------------------- 02
virt = [
'''md:# Virtual angles, drawn and checked

Each quantity below is defined from lab vectors of one position
([docs/virtual_angles.md](../docs/virtual_angles.md)), drawn for a real
solution, and compared with diffcalc's `get_virtual_angles` for that
position. Then each is *held* as a constraint and the solutions checked.''',
SETUP,
DIFFCALC_HELPERS,
'''psic = Geometry.builtin("psic")
lattice = (4.0, 4.0, 4.0, 90, 90, 90)
ub = b_matrix(*lattice)
d = Diffractometer(psic, ub, energy=9000, surface_hkl=(0, 0, 1), reference_hkl=(0, -1, 1))
ubc = diffcalc_ubcalc(lattice, surface=ub @ (0, 0, 1), reference=ub @ (0, -1, 1))
pos = min(d.solve((1, 1, 2), {"mu": 10, "nu": 20, "betain": 5}), key=lambda a: abs(a["eta"]) + abs(a["chi"]))
print(pos)
mine = d.pseudo(pos)
theirs = HklCalculation(ubc, Constraints({})).get_virtual_angles(Position(**pos))
pd.DataFrame({"hklforge": mine, "diffcalc": theirs}).assign(
    difference=lambda t: (t.hklforge - t.diffcalc + 180) % 360 - 180)''',
'''md:`omega` is hklforge's (diffcalc has it only as a constraint). All others agree to ~1e-13 deg.

## 2θ: b, k', Q and the scattering plane
cos 2θ = b·k'. Q = |k|(k' − b) makes 90° + θ with the beam - that is the Bragg condition.''',
'''draw(d, pos, show=("ttheta",), view="plane");''',
'''md:## qaz, naz: azimuths about the beam
Looking downstream. r = z (azimuth 0), u = x (90°, vertical); qaz is the azimuth of k' (the
scattering plane), naz that of the reference vector n.''',
'''draw(d, pos, show=("qaz", "naz"), view="beam");''',
'''md:## alpha, beta / betain, betaout: beams to a plane
sin α = −b·n, sin β = k'·n (reference vector); betain / betaout the same with the surface
normal. Drawn here with n = surface normal = (0 0 1) so the plane is the surface.''',
'''d_up = Diffractometer(psic, ub, energy=9000, surface_hkl=(0, 0, 1), reference_hkl=(0, 0, 1))
p_up = min(d_up.solve((1, 1, 2), {"mu": 0, "nu": 0, "betain": 10}), key=lambda a: abs(a["eta"]) + abs(a["chi"]))
draw(d_up, p_up, show=("betain", "betaout"));
print({k: round(v, 3) for k, v in d_up.pseudo(p_up).items() if k in ("betain", "betaout", "alpha", "beta")})''',
'''md:Because k' = b + Q/|k|: sin β = −sin α + Q·n/|k|. The exit angle is fixed by the incidence
angle and hkl - which is why only one of them can be held:''',
'''Q = psic.q_lab(p_up, d_up.k); n = psic.sample_matrix(p_up) @ d_up.surface
va = d_up.pseudo(p_up)
print(np.sin(np.radians(va["betaout"])), -np.sin(np.radians(va["betain"])) + Q @ n / d_up.k)''',
'''md:## tau and psi: n about Q
cos τ = Q̂·n (fixed by the crystal). ψ is the azimuth of n about Q, from p - the direction
in the scattering plane normal to Q, along the beam's travel.''',
'''fig = plt.figure(figsize=(12, 5.5))
draw(d, pos, show=("tau",), view="Qn", ax=fig.add_subplot(1, 2, 1, projection="3d"), title="tau")
draw(d, pos, show=("psi",), view="Q", ax=fig.add_subplot(1, 2, 2, projection="3d"), title="psi (looking down Q)");''',
'''md:## omega and bisect: the chi axis
`bisect`: the chi axis lies in the scattering plane. `omega`: its direction there, from p towards Q.
omega = 0 with bisect is the classic bisecting position (chi axis normal to Q).''',
'''pb = min(d.solve((1, 1, 2), {"nu": 20, "bisect": True, "omega": 20}), key=lambda a: abs(a["eta"]) + abs(a["chi"]))
draw(d, pb, show=("omega",), view="plane");''',
'''md:## Holding each virtual angle

For each constraint, a mode holding it (plus circles), all solutions, and the value each
solution has - it must equal what was asked for.''',
'''tests = [
    ({"chi": 10, "phi": 20, "betain": 3}, "betain"),
    ({"chi": 10, "phi": 20, "betaout": 3}, "betaout"),
    ({"mu": 0, "nu": 0, "alpha": 4}, "alpha"),
    ({"mu": 0, "nu": 0, "beta": 4}, "beta"),
    ({"mu": 0, "nu": 0, "psi": 60}, "psi"),
    ({"qaz": 90, "mu": 0, "phi": 0}, "qaz"),
    ({"naz": 30, "alpha": 5, "phi": 0}, "naz"),
    ({"nu": 0, "bisect": True, "omega": 15}, "omega"),
]
rows = []
for m, name in tests:
    sols = d.solve((1, 1, 2), m)
    vals = [d.pseudo(s)[name] for s in sols]
    theirs = diffcalc_solve(ubc, (1, 1, 2), m, d.wavelength)
    rows.append({"mode": m, "held": f"{name} = {m[name]}", "solutions": len(sols),
                 "values found": sorted({round(v, 9) for v in vals}),
                 "diffcalc": theirs if isinstance(theirs, str) else len(theirs)})
pd.DataFrame(rows)''',
'''md:(diffcalc's count for omega includes positions that violate the mode -
[03](03_diffcalc_comparison.ipynb).)''',
]

# --------------------------------------------------------------------------- 03
cmp_ = [
'''md:# hklforge vs diffcalc, systematically

diffcalc-core 0.4 is the oracle for psic: for every constraint combination it
implements (194), random crystals, wavelengths, reflections and values, the
**set** of positions must be the same. This is a compact version of
`tests/test_solve_diffcalc.py` / `dev/diffcalc_stats.py`; the full tally is in
[docs/verification.md](../docs/verification.md).''',
SETUP,
DIFFCALC_HELPERS,
'''import itertools, collections, time
psic = Geometry.builtin("psic")
SAMPLE = ("mu", "eta", "chi", "phi")
DETECTOR = ("delta", "nu", "qaz", "naz")
REFS = ("betain", "betaout", "bin_eq_bout", "alpha", "beta", "a_eq_b", "psi")
BISECT = (("mu", "bisect"), ("eta", "bisect"), ("omega", "bisect"))
MODES = (list(itertools.combinations(SAMPLE, 3))
         + [(d_,) + s for d_ in DETECTOR for s in list(itertools.combinations(SAMPLE, 2)) + list(BISECT)]
         + [s + (r,) for s in itertools.combinations(SAMPLE, 2) for r in REFS]
         + [(d_, r, s) for d_ in DETECTOR for r in REFS for s in SAMPLE])
print(len(MODES), "combinations")

def value(name, rng):
    if name in ("bin_eq_bout", "a_eq_b", "bisect"): return True
    if name == "omega": return float(rng.uniform(-30, 30))
    if name in ("psi", "naz", "qaz"): return float(rng.uniform(-180, 180))
    if name in REFS: return float(rng.uniform(-20, 40))
    return float(rng.uniform(-170, 170))

def case(names, seed):
    rng = np.random.default_rng(seed)
    lattice = (*rng.uniform(3, 8, 3), *rng.uniform(70, 110, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3))); U = q * np.sign(np.linalg.det(q))
    surf, ref = (v / np.linalg.norm(v) for v in rng.normal(size=(2, 3)))
    lam = rng.uniform(0.7, 2.0)
    ub = U @ b_matrix(*lattice)
    while True:
        hkl = tuple(int(x) for x in rng.integers(-3, 4, 3))
        if 0 < np.linalg.norm(ub @ hkl) < 1.8 * 2 * np.pi / lam: break
    mode = {n: value(n, rng) for n in names}
    if "naz" in mode and set(mode) & {"betain", "betaout", "bin_eq_bout"}:
        ref = surf                         # see diffcalc bug 4 below
    return lattice, U, surf, ref, lam, hkl, mode''',
'''from hklforge.virtual import bisect_axis

def violates(d, a, mode):
    """How far a position is from satisfying hkl-independent parts of the mode."""
    err = 0.0
    if "bisect" in mode:
        e, Q = bisect_axis(psic, a), psic.q_lab(a, 1.0)
        err = abs(e @ np.cross(psic.beam, Q / np.linalg.norm(Q)))
        if "omega" in mode:
            err = max(err, np.radians(abs((d.pseudo(a)["omega"] - mode["omega"] + 180) % 360 - 180)))
    return err

rows = []
t_mine = t_theirs = 0.0
for names in MODES:
    for seed in range(3):
        lattice, U, surf, ref, lam, hkl, mode = case(names, seed)
        d = Diffractometer(psic, U @ b_matrix(*lattice), wavelength=lam, surface=surf, reference=ref)
        ubc = diffcalc_ubcalc(lattice, U, surf, ref)
        row = {"mode": "-".join(names), "seed": seed}
        try:
            t = time.perf_counter(); mine = d.solve(hkl, mode); t_mine += time.perf_counter() - t
        except ModeError as e:
            rows.append(row | {"result": "hklforge: mode not supported"}); continue
        t = time.perf_counter(); theirs = diffcalc_solve(ubc, hkl, mode, lam); t_theirs += time.perf_counter() - t
        if isinstance(theirs, str):
            fwd = HklCalculation(ubc, Constraints({}))
            ok = all(np.allclose(fwd.get_hkl(Position(**a), lam), hkl, atol=1e-9) for a in mine)
            kind = "no code" if "No code yet" in theirs else "diffcalc failed"
            rows.append(row | {"result": kind, "hklforge": len(mine), "hklforge ok by diffcalc get_hkl": ok})
            continue
        bad = [a for a in theirs if violates(d, a, mode) > 1e-6]
        good = [a for a in theirs if a not in bad]
        rows.append(row | {"result": "identical" if same_set(mine, good) else "DIFFERENT",
                           "hklforge": len(mine), "diffcalc": len(theirs), "diffcalc violating mode": len(bad)})
df = pd.DataFrame(rows)
print(f"hklforge {t_mine:.1f} s, diffcalc {t_theirs:.1f} s in total")
df.groupby("result").size()''',
'''md:Per mode family (with solutions only):''',
'''def family(m):
    n = m.split("-")
    if "bisect" in n: return f"{n[0]} + {n[1]} + bisect"
    kinds = ["det" if x in ("delta", "nu") else x if x in ("qaz", "naz", "psi") else "ref" if x in REFS else "sample" for x in n]
    return " + ".join(f"{kinds.count(k)} {k}" for k in dict.fromkeys(kinds))
ok = df[df.result == "identical"].assign(family=lambda t: t["mode"].map(family))
ok.groupby("family").agg(cases=("mode", "size"), positions=("hklforge", "sum"),
                         max_branches=("hklforge", "max"), diffcalc_violating=("diffcalc violating mode", "sum"))''',
'''md:## The diffcalc bugs this found

### 1. eta, phi and one detector circle held, |eta| > 90°
diffcalc returns only wrong positions, then its own self-check raises.''',
'''lattice, U = (5.0, 5.5, 6.0, 90, 95, 100), np.eye(3)
ub = U @ b_matrix(*lattice); lam = 1.2
d = Diffractometer(psic, ub, wavelength=lam); ubc = diffcalc_ubcalc(lattice, U)
for eta in (40, 140):
    m = {"nu": 10, "eta": eta, "phi": 30}
    calc = HklCalculation(ubc, Constraints(m))
    calc._HklCalculation__verify_pos_map_to_hkl = lambda *a, **k: None    # see what it would return
    theirs = [p.asdict for p, _ in calc.get_position(1, 2, 2, lam)]
    err = max(np.abs(np.array(HklCalculation(ubc, Constraints({})).get_hkl(Position(**a), lam)) - (1, 2, 2)).max() for a in theirs)
    mine = d.solve((1, 2, 2), m)
    print(f"eta = {eta}: diffcalc {len(theirs)} positions, worst hkl error {err:.2g}; hklforge {len(mine)}, "
          f"worst error {max(np.abs(d.hkl(a) - (1, 2, 2)).max() for a in mine):.1g}")''',
'''md:### 2. bisect: extra positions that violate the mode''',
'''bis = df[df["mode"].str.contains("bisect") & (df.result == "identical")]
print(f"{int(bis['diffcalc violating mode'].sum())} of {int(bis['diffcalc'].sum())} diffcalc positions in "
      f"{len(bis)} bisect cases violate bisect or omega; the rest equal hklforge's")''',
'''md:### 3. eta + bisect: diffcalc crashes''',
'''df[df.result == "diffcalc failed"]''',
'''md:### 4. naz with a surface constraint
diffcalc solves naz for the surface normal, then filters by naz of the reference vector:''',
'''lattice, U = (4.0, 4.0, 4.0, 90, 90, 90), np.eye(3)
ub = b_matrix(*lattice); lam = 1.0
m = {"naz": 40, "betain": 5, "mu": 0}
for ref in ((0, 1, 1), (0, 0, 1)):
    ubc = diffcalc_ubcalc(lattice, U, surface=(0, 0, 1), reference=ref)
    d = Diffractometer(psic, ub, wavelength=lam, surface=(0, 0, 1), reference=(0, 0, 1))
    theirs = diffcalc_solve(ubc, (1, 1, 2), m, lam)
    print(f"diffcalc reference vector {ref}: diffcalc {theirs if isinstance(theirs, str) else len(theirs)}, "
          f"hklforge (naz of the surface normal) {len(d.solve((1, 1, 2), m))}")''',
'''md:### 5. naz with two sample circles: "No code yet"
diffcalc lists the mode as implemented, but has no solver for it. hklforge's answer is
checked against a brute-force scan instead (`dev/check_naz_scan.py`):''',
'''sys.path.insert(0, os.path.abspath("../dev"))
import check_naz_scan
check_naz_scan.main(10);''',
]

# --------------------------------------------------------------------------- 04
dd = [
'''md:# Two reflections at once

A second reflection `h2` held in Bragg condition is one more constraint on the
sample. The detector can sit on reflection 1 (libhkl's `double_diffraction_*`
modes) or between the two outgoing beams (`detector="between"`, the Bernina
mode), with a suggested detector distance that keeps both spots on the face.''',
SETUP,
'''psic = Geometry.builtin("psic")
si = (5.431, 5.431, 5.431, 90, 90, 90)
# the orientation used at Bernina: (1,1,1) up-ish - here simply U = identity
ub = b_matrix(*si)
d = Diffractometer(psic, ub, energy=9000, surface_hkl=(1, 1, 1))
h1, h2 = (1, 1, 1), (2, 2, 0)''',
'''md:## Detector on h1 (libhkl double_diffraction_vertical: mu, nu held)''',
'''sols = d.solve(h1, {"mu": 0, "nu": 0, "h2": h2})
print(d.plan({"mu": 0, "nu": 0, "h2": h2}).describe())
pd.DataFrame([s | {"k' separation": d.spots(s, [h1, h2])["separation"]} for s in sols])''',
'''md:## Detector between the two reflections (eta, chi held)''',
'''m = {"eta": 0, "chi": 0, "h2": h2}
between = d.solve(h1, m, detector="between")
print(d.plan(m, "between").describe())
pd.DataFrame([s | {k: v for k, v in d.spots(s, [h1, h2], width=77, height=77).items() if k in ("separation", "tdet_max")}
              | {"off axis 1": d.spots(s, [h1, h2])["off_axis"][0], "off axis 2": d.spots(s, [h1, h2])["off_axis"][1]}
              for s in between])''',
'''md:`tdet_max` is the largest sample-detector distance (mm for a 77 × 77 mm face) that keeps both
spots within 90 % of the half sizes. Both reflections are the same angle off the detector axis.''',
'''s = between[0]
fig = plt.figure(figsize=(12, 5.5))
draw(d, s, show=("ttheta",), hkls=[h1, h2], ax=fig.add_subplot(1, 2, 1, projection="3d"),
     title="detector between k'1 and k'2")
draw(d, sols[0], show=("ttheta",), hkls=[h2], ax=fig.add_subplot(1, 2, 2, projection="3d"),
     title="detector on h1, h2 also in Bragg");''',
'''md:## Both reflections really are in Bragg condition
|k b + R UB h| = |k| for each, at every solution:''',
'''for name, ss in (("on h1", sols), ("between", between)):
    err = max(abs(np.linalg.norm(d.k * np.asarray(psic.beam) + psic.sample_matrix(s) @ ub @ np.asarray(h)) - d.k)
              for s in ss for h in (h1, h2))
    print(f"{name}: {len(ss)} solutions, worst Ewald-sphere error {err:.1e} 1/A")''',
'''md:## Against the Bernina reference solver
`dev/reference/bernina_two_reflections.py` (scan + root polish, checked against diffcalc and
libhkl) - if bernina-scripts is available:''',
'''try:
    sys.path[:0] = [os.path.expanduser("~/mypy/bernina-scripts"), os.path.abspath("../dev/reference")]
    import bernina_two_reflections as REF
    from bernina_prep import diffraction as BD
    ubc = BD.build_ubcalc("si", *si, ["(1,1,1) > (1,0,0)", "(1,1,-1) > (0,1,0)"], "(1,1,1)", frame="swissfel")
    import itertools, time
    dd = Diffractometer(psic, ubc.UB, energy=9000)
    rows = []
    for pair in itertools.combinations(["mu", "eta", "chi", "phi"], 2):
        fixed = {n: 0.0 for n in pair}
        t = time.perf_counter()
        try:
            df = REF.solve_two_reflections(ubc, h1, h2, 9000, fixed)
            ref = [{**{n: df[c][n] for n in ("mu", "eta", "chi", "phi")}, "nu": df[c]["gamma"], "delta": df[c]["delta"]} for c in df.columns]
        except ValueError:
            ref = []
        t_ref = time.perf_counter() - t
        t = time.perf_counter()
        try:
            mine = dd.solve(h1, fixed | {"h2": h2}, detector="between"); note = ""
        except Degenerate as e:
            mine, note = [], "continuum (chi = 0: eta and phi on one axis)"
        t_mine = time.perf_counter() - t
        same_ = lambda a, b: all(abs((a[n] - b[n] + 180) % 360 - 180) < 1e-4 for n in a)
        ident = all(any(same_(r, m) for m in mine) for r in ref) and all(any(same_(m, r) for r in ref) for m in mine)
        rows.append({"held": pair, "reference": len(ref), "hklforge": len(mine), "identical": ident,
                     "reference ms": 1e3 * t_ref, "hklforge ms": 1e3 * t_mine, "note": note})
    display(pd.DataFrame(rows))
except ImportError as e:
    print("bernina-scripts not available:", e)''',
]

# --------------------------------------------------------------------------- 05
custom = [
'''md:# Custom geometries

Nothing in the planner is specific to psic. A geometry is a list of axes; the
modes, the counting of what must be held and the closed-form steps all follow
from it. Two examples: one in xrayutilities' syntax (checked against
xrayutilities' own forward model), and a kappa diffractometer, whose kappa
axis is tilted - something no Euler-angle formula covers.''',
SETUP,
'''md:## xrayutilities syntax
A 2+2 surface diffractometer: sample omega (about z−, outermost) and a tilt (about y+);
detector nu (x+) and delta (z−), beam along y.''',
'''g = Geometry.from_xrayutilities(["z-", "y+"], ["x+", "z-"], r_i=(0, 1, 0),
                                 sample_names=["om", "tilt"], detector_names=["nu", "delta"], name="2+2")
print(g)
draw_circles(g, {"om": 20, "tilt": 10, "nu": 10, "delta": 30});''',
'''import xrayutilities as xu
qc = xu.experiment.QConversion(["z-", "y+"], ["x+", "z-"], (0, 1, 0))
lam = xu.en2lam(10000)
rng = np.random.default_rng(0)
worst = 0
for _ in range(200):
    a = dict(zip(g.names, rng.uniform(-150, 150, 4)))
    worst = max(worst, np.abs(g.q_phi(a, 2 * np.pi / lam) - np.ravel(qc(*[a[n] for n in g.names], wl=lam))).max())
print(f"forward model vs xrayutilities QConversion, 200 random positions: max |dQ| = {worst:.1e} 1/A")''',
'''md:Four circles: one thing to hold. With the incidence angle held (the usual surface-diffraction
mode), every solution:''',
'''ub = b_matrix(3.905, 3.905, 3.905)              # SrTiO3, (0 0 1) surface along the tilt-free z
d = Diffractometer(g, ub, energy=10000, surface_hkl=(0, 0, 1), reference_hkl=(0, 0, 1))
m = {"betain": 0.5}
print(d.plan(m).describe())
sols = d.solve((1, 0, 0.5), m)
pd.DataFrame([s | {k: d.pseudo(s)[k] for k in ("betain", "betaout", "ttheta")} for s in sols])''',
'''md:Each solution through xrayutilities' forward model:''',
'''for s in sols:
    q_xu = np.ravel(qc(*[s[n] for n in g.names], wl=lam))
    print(np.round(np.linalg.solve(ub, q_xu), 9))''',
'''md:## A kappa diffractometer
Kappa four-circle (as libhkl's K4CV): komega about z−, kappa about an axis tilted by
α = 50° from z towards the beam, kphi about z−; detector tth about z−. Written as YAML:''',
'''alpha = np.radians(50)
yaml_text = f"""
name: kappa4c
beam: y+
labels: {{x: vertical, y: beam, z: transverse}}
sample:
  - {{name: komega, axis: [0, 0, -1]}}
  - {{name: kappa,  axis: [0, {-np.sin(alpha):.12f}, {-np.cos(alpha):.12f}]}}
  - {{name: kphi,   axis: [0, 0, -1]}}
detector:
  - {{name: tth, axis: [0, 0, -1]}}
"""
k4 = Geometry.from_yaml(yaml_text)
print(k4)
draw_circles(k4, {"komega": 30, "kappa": 60, "kphi": 0, "tth": 40});''',
'''dk = Diffractometer(k4, b_matrix(5.431, 5.431, 5.431), energy=9000, surface_hkl=(0, 0, 1))
for m in ({"kphi": 0}, {"kappa": 30}, {"komega": 10}, {"betain": 5}):
    sols = dk.solve((2, 2, 0), m)
    print(dk.plan(m).describe())
    for s in sols:
        print("   ", {k: round(v, 4) for k, v in s.items()}, " hkl back:", np.round(dk.hkl(s), 10))''',
'''md:A kappa with α = 50° can tilt the sample by at most 2α = 100° (an Eulerian chi of ±100°),
so some reflections are out of reach with some circles held - then there are simply no solutions:''',
'''for h in [(1, 1, 1), (0, 0, 4), (1, 1, 3)]:
    print(h, {str(m): len(dk.solve(h, m)) for m in ({"kphi": 0}, {"kappa": 30}, {"komega": 10}, {"betain": 5})})''',
'''md:With kappa held, komega and kphi rotate about the same axis whenever kappa = 0 - then the
solutions form a continuum and that is said, rather than an arbitrary answer returned:''',
'''try:
    dk.solve((1, 1, 1), {"kappa": 0})
except Degenerate as e:
    print("Degenerate:", e)''',
]

for name, cells in (("01_quickstart", quick), ("02_virtual_angles", virt),
                    ("03_diffcalc_comparison", cmp_), ("04_double_diffraction", dd),
                    ("05_custom_geometries", custom)):
    nbf.write(nb(cells), os.path.join(OUT, f"{name}.ipynb"))
    print("wrote", name)
