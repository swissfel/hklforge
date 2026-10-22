"""The planner on arbitrary geometries, where no oracle exists.

Each trial builds a random geometry (random axes, random beam), picks random
"true" angles, reads off the hkl and the constraint values those angles give,
and holds the mode's circles at their true values. The true position must be
among the solutions (completeness), and every solution must reproduce hkl
and the constraints through the forward model (correctness).
"""
import numpy as np
import pytest

from hklforge import Degenerate, Diffractometer, Geometry, ModeError
from hklforge.rotation import unit

rng = np.random.default_rng(7)

# (detector circles free, sample circles free, extra constraint or None)
CASES = [(2, 1, None), (2, 2, "betain"), (2, 2, "betaout"), (2, 2, "alpha"), (2, 2, "beta"),
         (2, 2, "h2"), (1, 2, None), (1, 3, "betain"), (1, 3, "betaout"), (1, 3, "beta"),
         (1, 3, "h2")]


def random_geometry(n_sample, n_detector):
    def circles(prefix, n):
        return [{"name": f"{prefix}{i}", "axis": list(unit(rng.normal(size=3)))} for i in range(n)]
    return Geometry.from_dict({"name": "random", "beam": list(unit(rng.normal(size=3))),
                               "sample": circles("s", n_sample), "detector": circles("d", n_detector)})


def close(a, b, tol=1e-6):
    return all(abs((a[n] - b[n] + 180) % 360 - 180) < tol for n in a)


def ewald_point(geom, k, R):
    """A random q (phi frame) in Bragg condition at sample rotation R."""
    b = np.asarray(geom.beam)
    return R.T @ (k * (unit(rng.normal(size=3)) - b))


def trial(nD, nS, extra, detector="h1"):
    geom = random_geometry(int(rng.integers(nS, 5)), int(rng.integers(nD, 3)))
    ub = rng.normal(size=(3, 3)) * 1.5
    k = rng.uniform(2, 6)
    d = Diffractometer(geom, ub, wavelength=2 * np.pi / k, surface=rng.normal(size=3),
                       reference=rng.normal(size=3))
    truth = {n: float(rng.uniform(-180, 180)) for n in geom.names}
    free = list(rng.choice(geom.sample_names, nS, replace=False)) + \
        list(rng.choice(geom.detector_names, nD, replace=False))
    mode = {n: truth[n] for n in geom.names if n not in free}
    R = geom.sample_matrix(truth)
    if detector == "between":
        # the true detector direction is the bisector: k'1 and k'2 mirror images about it
        b = np.asarray(geom.beam)
        bis = geom.k_out(truth)
        k1 = unit(bis + 0.6 * unit(np.cross(bis, rng.normal(size=3))))
        k2 = 2 * (k1 @ bis) * bis - k1
        q1, q2 = (R.T @ (k * (kk - b)) for kk in (k1, k2))
        hkl = np.linalg.solve(ub, q1)
        mode["h2"] = np.linalg.solve(ub, q2)
    else:
        hkl = d.hkl(truth)
        if extra == "h2":
            mode["h2"] = np.linalg.solve(ub, ewald_point(geom, k, R))
        elif extra:
            mode[extra] = d.pseudo(truth)[extra]
    try:
        sols = d.solve(hkl, mode, detector=detector)
    except Degenerate:
        return None
    assert any(close(s, truth) for s in sols), (mode, truth, sols)
    for s in sols:
        if detector == "between":
            sp = d.spots(s, [hkl, mode["h2"]])
            assert abs(sp["off_axis"][0] - sp["off_axis"][1]) < 1e-7
            for h in (hkl, mode["h2"]):              # both in Bragg: |k + Q| = |k|
                Q = geom.sample_matrix(s) @ ub @ h
                assert abs(np.linalg.norm(k * np.asarray(geom.beam) + Q) - k) < 1e-8 * k
        else:
            assert np.allclose(d.hkl(s), hkl, atol=1e-8)
            if extra == "h2":
                Q = geom.sample_matrix(s) @ ub @ mode["h2"]
                assert abs(np.linalg.norm(k * np.asarray(geom.beam) + Q) - k) < 1e-8 * k
            elif extra:
                assert abs(d.pseudo(s)[extra] - mode[extra]) < 1e-7
    return len(sols)


@pytest.mark.parametrize("nD,nS,extra", CASES)
def test_random_geometries_recover_truth(nD, nS, extra):
    counts = [trial(nD, nS, extra) for _ in range(40)]
    assert sum(c is not None for c in counts) >= 38       # Degenerate only by bad luck


def test_random_geometries_between():
    counts = [trial(2, 2, "h2", detector="between") for _ in range(40)]
    assert sum(c is not None for c in counts) >= 38


@pytest.mark.parametrize("mode,msg", [
    ({"eta": 0, "chi": 0}, "underdetermined"),
    ({"eta": 0, "chi": 0, "phi": 0, "mu": 0}, "overdetermined"),
    ({"eta": 0, "betain": 1, "betaout": 2}, "at most one"),
    ({"nu": 0, "delta": 0, "eta": 0}, "fixes |Q|"),
    ({"eta": 0, "chi": 0, "omega": 1}, "neither a circle"),
])
def test_bad_modes_are_explained(mode, msg):
    d = Diffractometer(Geometry.builtin("psic"), np.eye(3), wavelength=1.0)
    with pytest.raises(ModeError, match=msg.replace("|", r"\|")):
        d.plan(mode)


def test_fourcv_bisect_free_modes():
    """A 4-circle: one sample circle held, or incidence held."""
    g = Geometry.builtin("fourcv")
    d = Diffractometer(g, np.diag([1.2, 1.3, 1.1]), wavelength=1.5, surface=(0, 0, 1))
    assert len(d.solve((1, 1, 1), {"phi": 30})) == 4
    sols = d.solve((1, 1, 1), {"betain": 10})
    assert sols and all(abs(d.pseudo(s)["betain"] - 10) < 1e-9 for s in sols)


@pytest.mark.parametrize("detector,mode", [("h1", {"mu": 3, "nu": 4}), ("between", {"eta": 3, "chi": 40})])
def test_parallel_second_reflection(detector, mode):
    """h2 parallel to h1: contradictory Bragg conditions give no solution,
    identical ones a continuum."""
    d = Diffractometer(Geometry.builtin("psic"), np.diag([1.1, 1.2, 1.3]), wavelength=1.0)
    assert d.solve((1, 1, 0), mode | {"h2": (2, 2, 0)}, detector=detector) == []
    with pytest.raises(Degenerate):
        d.solve((1, 1, 0), mode | {"h2": (1, 1, 0)}, detector=detector)
