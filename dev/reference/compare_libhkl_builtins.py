"""The stored libhkl-based geometries against libhkl itself (forward model).

``e4ch``, ``k4cv`` and ``k6c`` are written in libhkl's frame
(x = beam, z = up) with libhkl's axis names and order. For random UB and
random positions this compares (h, k, l) from libhkl's "hkl" engine with
``Diffractometer.hkl``, and prints the largest difference per geometry.

Needs an env with libhkl, e.g.
``micromamba create -p ./hklenv -c conda-forge python=3.12 hkl pygobject numpy pyyaml``.
Run from the repo root: ``hklenv/bin/python dev/reference/compare_libhkl_builtins.py [n]``.
"""
import math
import sys

import numpy as np

sys.path.insert(0, "src")
import gi  # noqa: E402

gi.require_version("Hkl", "5.0")
from gi.repository import Hkl  # noqa: E402

from hklforge import Diffractometer, Geometry  # noqa: E402
from hklforge.lattice import b_matrix  # noqa: E402

USER = Hkl.UnitEnum.USER
PAIRS = {"e4ch": "E4CH", "k4cv": "K4CV", "k6c": "K6C"}
# libhkl ZAXIS (and SIXS MED2+x) turn the detector with the sample's mu as
# well (a circle in both stacks), which hklforge's model does not express yet.


def libhkl_hkl(factory, lattice, ub, wavelength, positions):
    f = Hkl.factories()[factory]
    g, el = f.create_new_geometry(), f.create_new_engine_list()
    s = Hkl.Sample.new("s")
    a, b, c, al, be, ga = lattice
    s.lattice_set(Hkl.Lattice.new(a, b, c, *map(math.radians, (al, be, ga))))
    m = s.UB_get()
    m.init(*np.asarray(ub, float).ravel())
    s.UB_set(m)
    g.wavelength_set(wavelength, USER)
    det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    el.init(g, det, s)
    names = list(g.axis_names_get())
    eng = el.engine_get_by_name("hkl")
    out = []
    for pos in positions:
        g.axis_values_set([pos[n] for n in names], USER)
        el.init(g, det, s)
        el.get()
        out.append(np.array(eng.pseudo_axis_values_get(USER)))
    return names, out


def main(n=200):
    rng = np.random.default_rng(3)
    lattice = (5.1, 6.3, 7.2, 90.0, 90.0, 90.0)
    wavelength = 1.1
    for ours, theirs in PAIRS.items():
        geom = Geometry.builtin(ours)
        # a random orientation of the lattice: UB = U B, U a random rotation
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        U = q * np.sign(np.linalg.det(q))
        ub = U @ b_matrix(*lattice)
        positions = [{c: float(rng.uniform(-170, 170)) for c in geom.names} for _ in range(n)]
        names, theirs_hkl = libhkl_hkl(theirs, lattice, ub, wavelength, positions)
        assert names == geom.names, (names, geom.names)
        d = Diffractometer(geom, ub, wavelength=wavelength)
        err = max(np.max(np.abs(d.hkl(p) - h)) for p, h in zip(positions, theirs_hkl))
        print(f"{ours:6s} vs libhkl {theirs:6s}: {n} positions, max |hkl difference| = {err:.2e}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
