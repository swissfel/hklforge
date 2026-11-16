"""The stored geometries and Diffractometer.scan.

Every stored geometry: plain-data round trip, and truth recovery - random
true positions are among the solutions, for every choice of held circles
the planner accepts. Their forward models against libhkl are checked by
``dev/reference/compare_libhkl_builtins.py`` (needs libhkl).
"""
from itertools import combinations

import numpy as np
import pytest

from hklforge import Degenerate, Diffractometer, Geometry, ModeError
from hklforge.lattice import b_matrix

rng = np.random.default_rng(11)


def close(a, b, tol=1e-6):
    return all(abs((a[n] - b[n] + 180) % 360 - 180) < tol for n in a)


def test_builtins_listed():
    assert {"psic", "fourcv", "e4ch", "k4cv", "k6c"} <= set(Geometry.builtins())


@pytest.mark.parametrize("name", Geometry.builtins())
def test_round_trip(name):
    g = Geometry.builtin(name)
    g2 = Geometry.from_dict(g.to_dict())
    assert g2.to_dict() == g.to_dict()
    angles = {n: float(rng.uniform(-90, 90)) for n in g.names}
    assert np.allclose(g2.q_lab(angles, 3.0), g.q_lab(angles, 3.0))


@pytest.mark.parametrize("name", Geometry.builtins())
def test_truth_recovery(name):
    g = Geometry.builtin(name)
    ub = b_matrix(4.1, 5.3, 6.7, 90, 95, 90)
    d = Diffractometer(g, ub, wavelength=1.3)
    n_held = len(g.names) - 3
    solved = 0
    for held in combinations(g.names, n_held):
        for _ in range(4):
            true = {n: float(rng.uniform(-80, 80)) for n in g.names}
            hkl = d.hkl(true)
            mode = {n: true[n] for n in held}
            try:
                sols = d.solve(hkl, mode)
            except (Degenerate, ModeError):
                break                      # this choice of held circles is not a solvable mode
            assert any(close(s, true) for s in sols), (held, true)
            for s in sols:                 # every answer reproduces hkl
                assert np.allclose(d.hkl(s), hkl, atol=1e-7)
            solved += 1
    assert solved >= 4, f"{name}: no held-circle mode solved"


def test_scan_matches_solve():
    g = Geometry.builtin("psic")
    d = Diffractometer(g, b_matrix(5.431, 5.431, 5.431), energy=9000, reference_hkl=(0, 0, 1))
    mode = {"mu": 0.0, "nu": 0.0, "phi": 0.0}
    values = np.linspace(-180, 180, 37)
    res = d.scan((1, 1, 1), mode, "phi", values)
    assert [r["value"] for r in res] == list(values)
    for r in res:
        assert len(r["solutions"]) == len(d.solve((1, 1, 1), {**mode, "phi": r["value"]}))
        assert all(abs(s["phi"] - r["value"]) < 1e-9 or abs(abs(s["phi"] - r["value"]) - 360) < 1e-9
                   for s in r["solutions"])
    assert any(r["solutions"] for r in res)
    # a filter drops positions before they are counted
    kept = d.scan((1, 1, 1), mode, "phi", values, keep=lambda s: s["delta"] > 0)
    for r, k in zip(res, kept):
        assert len(k["solutions"]) == sum(s["delta"] > 0 for s in r["solutions"])


def test_scan_virtual_angle_and_errors():
    g = Geometry.builtin("psic")
    d = Diffractometer(g, b_matrix(5.431, 5.431, 5.431), energy=9000, surface_hkl=(0, 0, 1))
    res = d.scan((0, 0, 4), {"mu": 0.0, "nu": 0.0, "betain": 1.0}, "betain", [0.5, 1.0, 2.0])
    for r in res:
        for s in r["solutions"]:
            assert abs(d.pseudo(s)["betain"] - r["value"]) < 1e-7
    with pytest.raises(ValueError):
        d.scan((1, 1, 1), {"mu": 0.0}, "phi", [0.0])
    # (0 0 4) along phi's axis with phi free: a continuum, reported not raised
    res = d.scan((0, 0, 4), {"mu": 0.0, "nu": 0.0, "chi": 90.0}, "chi", [0.0])
    assert res[0]["solutions"] == [] and res[0]["error"]
