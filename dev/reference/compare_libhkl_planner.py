"""The hklforge planner against libhkl's E6C modes.

Needs an env with libhkl (PyGObject) and diffcalc-core, e.g.
``micromamba create -p ./hklenv -c conda-forge python=3.12 hkl pygobject numpy scipy pip``
then ``pip install diffcalc-core==0.4.0``. Run from the repo root:
``hklenv/bin/python dev/reference/compare_libhkl_planner.py [n_cases]``.

For each E6C mode that is "these axes held" (plus h2 for double
diffraction), the held axes become an hklforge mode. libhkl solves
numerically from a start position, so it is run from several random starts
and the union taken. Reported: whether every libhkl solution is among
hklforge's, and the counts.
"""
import logging
import math
import sys

import numpy as np

sys.path.insert(0, "src")
import gi  # noqa: E402

gi.require_version("Hkl", "5.0")
from gi.repository import Hkl  # noqa: E402
from diffcalc.ub.calc import UBCalculation  # noqa: E402

from hklforge import Diffractometer, Geometry  # noqa: E402

logging.disable(logging.WARNING)
USER = Hkl.UnitEnum.USER
P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)     # diffcalc vector -> libhkl lab (x = beam)
E6C_AXES = ["mu", "omega", "chi", "phi", "gamma", "delta"]
TO_PSIC = {"mu": "mu", "omega": "eta", "chi": "chi", "phi": "phi", "gamma": "nu", "delta": "delta"}
MODES = ["constant_omega_vertical", "constant_chi_vertical", "constant_phi_vertical",
         "constant_mu_horizontal", "lifting_detector_mu", "lifting_detector_omega",
         "lifting_detector_phi", "double_diffraction_vertical", "double_diffraction_horizontal",
         "bissector_vertical", "bissector_horizontal", "psi_constant_vertical",
         "psi_constant_horizontal"]
# libhkl's bisector modes are omega = delta/2 (vertical) and mu = gamma/2 with
# omega = 0 (horizontal), each modulo 180: the chi axis normal to Q, either
# way round = hklforge's omega = 0 or 180 (with bisect). That holds only when
# the held axes are 0, so they are run with the held axes at 0.
HELD_AT_ZERO = ("bissector_vertical", "bissector_horizontal")
HKLFORGE_EXTRA = {"bissector_vertical": [{"omega": 0.0}, {"omega": 180.0}],
                  "bissector_horizontal": [{"bisect": True, "omega": 0.0},
                                           {"bisect": True, "omega": 180.0}]}
SKIPPED = {}


def libhkl_setup(lattice, ub, wavelength):
    f = Hkl.factories()["E6C"]
    g, el = f.create_new_geometry(), f.create_new_engine_list()
    s = Hkl.Sample.new("s")
    a, b, c, al, be, ga = lattice
    s.lattice_set(Hkl.Lattice.new(a, b, c, *map(math.radians, (al, be, ga))))
    m = s.UB_get()
    m.init(*(P @ ub).ravel())
    s.UB_set(m)
    g.wavelength_set(wavelength, USER)
    det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    el.init(g, det, s)
    el._keep = (s, det)        # the engine list only points at these: keep them alive
    return g, el


def libhkl_solutions(g, el, mode, hkl, held, h2, rng, n_starts=12, psi=None):
    eng = el.engine_get_by_name("hkl")
    eng.current_mode_set(mode)
    params = {} if h2 is None else dict(zip(("h2", "k2", "l2"), h2))
    if psi is not None:
        params["psi"] = psi
    for name, v in params.items():
        p = eng.parameter_get(name)
        p.value_set(float(v), USER)
        eng.parameter_set(name, p)
    out, errors = [], 0
    for _ in range(n_starts):
        start = [held.get(n, float(rng.uniform(-180, 180))) for n in E6C_AXES]
        g.axis_values_set(start, USER)
        el.get()
        try:
            sols = eng.pseudo_axis_values_set(list(map(float, hkl)), USER)
        except Exception:
            errors += 1
            continue
        for it in sols.items():             # keep `sols` alive: the items point into it
            a = dict(zip(E6C_AXES, it.geometry_get().axis_values_get(USER)))
            psic = {TO_PSIC[n]: v for n, v in a.items()}
            if not any(same(psic, o) for o in out):
                out.append(psic)
    return out, errors


PSI_SIGN = 1.0     # hklforge psi = PSI_SIGN * libhkl psi (checked by PSI_SEEN below)
PSI_SEEN = []


def same(a, b, tol=1e-3):           # libhkl agrees with diffcalc to ~1e-6 in hkl only
    return all(abs((a[n] - b[n] + 180) % 360 - 180) < tol for n in a)


def main(n_cases=20):
    rng = np.random.default_rng(11)
    psic = Geometry.builtin("psic")
    print("skipped modes:", SKIPPED)
    worst = {m: [] for m in MODES}
    for mode in MODES:
        tot = dict(cases=0, libhkl=0, hklforge=0, not_in_hklforge=0, libhkl_errors=0, hklforge_more=0)
        g0, el0 = None, None
        for _ in range(n_cases):
            lattice = (*rng.uniform(3, 7, 3), *rng.uniform(80, 100, 3))
            ubc = UBCalculation("t")
            ubc.set_lattice("x", *lattice)
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            ubc.set_u(q * np.sign(np.linalg.det(q)))
            ub = np.asarray(ubc.UB, float)
            lam = rng.uniform(0.8, 1.6)
            k = 2 * np.pi / lam

            def draw():
                while True:
                    h = tuple(int(x) for x in rng.integers(-3, 4, 3))
                    if 0 < np.linalg.norm(ub @ h) < 1.8 * k:
                        return h
            hkl = draw()
            h2 = draw() if mode.startswith(("double", "psi")) else None
            psi = float(rng.uniform(-180, 180)) if mode.startswith("psi") else None
            g0, el0 = libhkl_setup(lattice, ub, lam)
            eng = el0.engine_get_by_name("hkl")
            eng.current_mode_set(mode)
            write = eng.axis_names_get(Hkl.EngineAxisNamesGet.WRITE)
            held = {n: 0.0 if mode in HELD_AT_ZERO else float(rng.uniform(-60, 60))
                    for n in E6C_AXES if n not in write}
            hk_mode = {TO_PSIC[n]: v for n, v in held.items()}
            ref = None
            if psi is not None:                       # libhkl's psi is about Q from (h2, k2, l2)
                hk_mode["psi"] = PSI_SIGN * psi
                ref = ub @ np.asarray(h2, float)
            elif h2 is not None:
                hk_mode["h2"] = h2
            try:
                dm = Diffractometer(psic, ub, wavelength=lam, reference=ref)
                mine = [a for extra in HKLFORGE_EXTRA.get(mode, [{}]) for a in dm.solve(hkl, hk_mode | extra)]
            except Exception as e:
                print(f"  {mode}: hklforge {type(e).__name__}: {e}")
                continue
            theirs, errs = libhkl_solutions(g0, el0, mode, hkl, held, h2, rng, psi=psi)
            if psi is not None and theirs:           # report libhkl's psi in hklforge's convention
                dd = Diffractometer(psic, ub, wavelength=lam, reference=ref)
                PSI_SEEN.append((psi, [dd.pseudo(t)["psi"] for t in theirs]))
            tot["cases"] += 1
            tot["libhkl"] += len(theirs)
            tot["hklforge"] += len(mine)
            tot["libhkl_errors"] += errs
            d = Diffractometer(psic, ub, wavelength=lam)
            if psi is not None and h2 is not None:
                h2 = None                            # h2 was the psi reference, not a reflection
            for t in theirs:
                if any(same(t, m) for m in mine):
                    continue
                tot["not_in_hklforge"] += 1
                err = np.abs(d.hkl(t) - hkl).max()
                if h2 is not None:                   # h2 in Bragg: |k + Q2| = |k|
                    Q2 = psic.sample_matrix(t) @ ub @ np.asarray(h2, float)
                    err = max(err, abs(np.linalg.norm(k * np.asarray(psic.beam) + Q2) - k) / k)
                near = min((max(abs((t[n] - m[n] + 180) % 360 - 180) for n in t) for m in mine),
                           default=np.inf)
                worst[mode].append((err, near))
            tot["hklforge_more"] += len(mine) > len(theirs)
        print(f"{mode:30s} {tot}")
        for err, near in worst[mode]:
            print(f"    libhkl-only solution: its own hkl error {err:.1e}, nearest hklforge solution {near:.1e} deg")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
    for psi, seen in PSI_SEEN[:6]:
        print(f"libhkl psi {psi:9.3f} -> hklforge psi of libhkl's solutions {np.round(seen, 3)}")
