"""naz + two held sample circles (no oracle: diffcalc has no code for it).

Independent completeness check on psic: scan the two free sample circles on
a 0.5 deg grid for cells where both equations (Bragg, naz) change sign,
polish with scipy, and compare the set of sample settings with hklforge's.
Run: python dev/check_naz_scan.py [n_cases]
"""
import itertools
import sys

import numpy as np
from scipy.optimize import fsolve

sys.path.insert(0, "src")
from hklforge import Diffractometer, Geometry  # noqa: E402
from hklforge.rotation import unit  # noqa: E402

P = Geometry.builtin("psic")
SAMPLE = ["mu", "eta", "chi", "phi"]


def residuals(a1, a2, free, held, q, n, k, naz):
    """Bragg and the naz projection, broadcast over angle arrays (degrees)."""
    b = np.asarray(P.beam)
    r, u = P.azimuth_frame()
    m = -np.sin(np.radians(naz)) * r + np.cos(np.radians(naz)) * u
    side = np.cos(np.radians(naz)) * r + np.sin(np.radians(naz)) * u
    shape = np.broadcast(a1, a2).shape
    R = np.broadcast_to(np.eye(3), shape + (3, 3)).copy()
    for c in P.sample:
        ang = a1 if c.name == free[0] else a2 if c.name == free[1] else held[c.name]
        t = np.radians(np.broadcast_to(ang, shape))
        ax = np.asarray(c.axis)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        M = (np.cos(t)[..., None, None] * np.eye(3) + np.sin(t)[..., None, None] * K
             + (1 - np.cos(t))[..., None, None] * np.outer(ax, ax))
        R = R @ M
    Rq, Rn = R @ q, R @ n
    return Rq @ b + (q @ q) / (2 * k), Rn @ m, Rn @ side


def main(n_cases=40):
    rng = np.random.default_rng(5)
    tot = dict(cases=0, scan=0, hklforge=0, scan_not_in_hklforge=0, hklforge_not_in_scan=0)
    g = np.arange(-180, 180, 0.5)
    A1, A2 = np.meshgrid(g, g, indexing="ij")
    for _ in range(n_cases):
        ub = rng.normal(size=(3, 3)) + 2 * np.eye(3)
        lam, n = rng.uniform(0.8, 1.6), unit(rng.normal(size=3))
        k = 2 * np.pi / lam
        hkl = rng.integers(-2, 3, 3)
        if not hkl.any() or np.linalg.norm(ub @ hkl) > 1.8 * k:
            continue
        free = list(rng.choice(SAMPLE, 2, replace=False))
        held = {c: float(rng.uniform(-90, 90)) for c in SAMPLE if c not in free}
        naz = float(rng.uniform(-180, 180))
        q = ub @ hkl
        d = Diffractometer(P, ub, wavelength=lam, reference=n)
        mine = {tuple(np.round([s[f] for f in free], 4)) for s in d.solve(hkl, held | {"naz": naz})}
        f1, f2, side = residuals(A1, A2, free, held, q, n, k, naz)
        cells = np.ones(A1[:-1, :-1].shape, bool)
        for f in (f1, f2):
            s = np.sign(f)
            corners = np.stack([s[:-1, :-1], s[1:, :-1], s[:-1, 1:], s[1:, 1:]])
            cells &= (corners.min(0) <= 0) & (corners.max(0) >= 0)
        roots = []
        for i, j in zip(*np.nonzero(cells)):
            x, info, ok, _ = fsolve(lambda v: residuals(v[0], v[1], free, held, q, n, k, naz)[:2],
                                    [g[i] + 0.25, g[j] + 0.25], full_output=True, xtol=1e-13)
            r = residuals(x[0], x[1], free, held, q, n, k, naz)
            if ok == 1 and max(abs(r[0]), abs(r[1])) < 1e-10 and r[2] > 0:
                x = (x + 180) % 360 - 180
                if not any(np.abs((x - y + 180) % 360 - 180).max() < 1e-4 for y in roots):
                    roots.append(x)
        close = lambda a, B: any(max(abs((a[0] - b[0] + 180) % 360 - 180), abs((a[1] - b[1] + 180) % 360 - 180)) < 1e-3 for b in B)  # noqa: E731
        tot["cases"] += 1
        tot["scan"] += len(roots)
        tot["hklforge"] += len(mine)
        tot["scan_not_in_hklforge"] += sum(not close(r, mine) for r in roots)
        tot["hklforge_not_in_scan"] += sum(not close(m, roots) for m in mine)
    print("naz + 2 held sample circles, psic, sample settings (detector follows):", tot)
    return tot


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
