"""Virtual angles against diffcalc's get_virtual_angles, and the drawings run."""
import logging

import numpy as np
import pytest

from hklforge import Diffractometer, Geometry
from hklforge.virtual import virtual_angles

PSIC = Geometry.builtin("psic")


def test_all_virtual_angles_match_diffcalc():
    pytest.importorskip("diffcalc")
    from diffcalc.hkl.calc import HklCalculation
    from diffcalc.hkl.constraints import Constraints
    from diffcalc.hkl.geometry import Position
    from diffcalc.ub.calc import UBCalculation
    logging.getLogger("diffcalc").setLevel(logging.ERROR)
    rng = np.random.default_rng(2)
    worst = {}
    for _ in range(300):
        ub = UBCalculation("t")
        ub.set_lattice("x", *rng.uniform(3, 8, 3), *rng.uniform(70, 110, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        ub.set_u(q * np.sign(np.linalg.det(q)))
        surf, ref = (v / np.linalg.norm(v) for v in rng.normal(size=(2, 3)))
        ub.surf_nphi, ub.n_phi = tuple(surf), tuple(ref)
        a = {n: rng.uniform(-170, 170) for n in PSIC.names}
        theirs = HklCalculation(ub, Constraints({})).get_virtual_angles(Position(**a))
        mine = virtual_angles(PSIC, a, surf, ref)
        for key, v in theirs.items():
            if not np.isnan(v):
                worst[key] = max(worst.get(key, 0.0), abs((mine[key] - v + 180) % 360 - 180))
    assert set(worst) == {"theta", "ttheta", "qaz", "naz", "alpha", "beta", "betain", "betaout",
                          "tau", "psi"}
    assert max(worst.values()) < 1e-9, worst


def test_omega_is_bisecting_angle():
    """omega = 0 with bisect: chi axis normal to Q, in the scattering plane."""
    d = Diffractometer(PSIC, 2 * np.pi / 4 * np.eye(3), energy=9000)
    for s in d.solve((1, 1, 2), {"nu": 0, "bisect": True, "omega": 0}):
        from hklforge.virtual import bisect_axis
        e, Q = bisect_axis(PSIC, s), PSIC.q_lab(s, 1.0)
        assert abs(e @ Q) < 1e-12 and abs(e @ np.cross(PSIC.beam, Q)) < 1e-12


def test_drawings_run():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hklforge.draw import draw, draw_circles
    d = Diffractometer(PSIC, 2 * np.pi / 4 * np.eye(3), energy=9000, surface_hkl=(0, 0, 1),
                       reference_hkl=(0, 1, 1))
    pos = d.solve((1, 1, 2), {"mu": 10, "nu": 20, "betain": 5})[0]
    for view in (None, "beam", "Q", "plane", "Qn"):
        draw(d, pos, show="all", view=view)
    draw_circles(PSIC, pos)
    plt.close("all")
