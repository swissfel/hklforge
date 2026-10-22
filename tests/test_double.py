"""Two reflections at once: against the bernina reference solver
(dev/reference/bernina_two_reflections.py, itself checked against diffcalc
and libhkl) and diffcalc's forward model."""
import itertools
import logging
import os
import sys

import numpy as np
import pytest

from hklforge import Degenerate, Diffractometer, Geometry

pytest.importorskip("diffcalc")
pytest.importorskip("pandas")
pytest.importorskip("scipy")
BERNINA = os.environ.get("BERNINA_SCRIPTS", os.path.expanduser("~/mypy/bernina-scripts"))
sys.path[:0] = [BERNINA, os.path.join(os.path.dirname(__file__), "..", "dev", "reference")]
D = pytest.importorskip("bernina_prep.diffraction")
REF = pytest.importorskip("bernina_two_reflections")
from diffcalc.hkl.calc import HklCalculation          # noqa: E402
from diffcalc.hkl.constraints import Constraints      # noqa: E402
from diffcalc.hkl.geometry import Position            # noqa: E402

logging.getLogger("diffcalc").setLevel(logging.ERROR)
PSIC = Geometry.builtin("psic")
NAMES = ["mu", "eta", "chi", "phi", "nu", "delta"]


@pytest.fixture(scope="module")
def si():
    return D.build_ubcalc("si", 5.43, 5.43, 5.43, 90, 90, 90,
                          ["(1,1,1) > (1,0,0)", "(1,1,-1) > (0,1,0)"], "(1,1,1)", frame="swissfel")


@pytest.fixture(scope="module")
def hexa():
    return D.build_ubcalc("x", 4.13, 4.13, 28.6, 90, 90, 120,
                          ["(0,0,1) > (-1,0,0)", "(1,0,0) > (0,0,1)"], "(0,0,1)", frame="swissfel")


def wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


def reference(ub, h1, h2, energy, fixed):
    try:
        df = REF.solve_two_reflections(ub, h1, h2, energy, fixed)
    except ValueError:
        return []
    return [{**{n: df[c][n] for n in NAMES[:4]}, "nu": df[c]["gamma"], "delta": df[c]["delta"],
             "tdet": df[c]["tdet max"]} for c in df.columns]


def assert_same_set(ref, mine, d, h1, h2):
    same = lambda a, b: all(abs(wrap(a[n] - b[n])) < 1e-4 for n in NAMES)  # noqa: E731
    assert all(any(same(r, m) for m in mine) for r in ref)
    assert all(any(same(m, r) for r in ref) for m in mine)
    for r in ref:                       # detector-distance suggestion, capped as the reference
        m = next(m for m in mine if same(r, m))
        assert abs(min(d.spots(m, [h1, h2])["tdet_max"], REF.TDET_LIMITS[1]) - r["tdet"]) < 1e-6


@pytest.mark.parametrize("pair", list(itertools.combinations(["mu", "eta", "chi", "phi"], 2)), ids="-".join)
def test_between_matches_reference_si(si, pair):
    d = Diffractometer(PSIC, si.UB, energy=9000)
    mode = {n: 0.0 for n in pair} | {"h2": (2, 2, 0)}
    ref = reference(si, (1, 1, 1), (2, 2, 0), 9000, {n: 0.0 for n in pair})
    try:
        mine = d.solve((1, 1, 1), mode, detector="between")
    except Degenerate:
        assert pair == ("mu", "chi") and not ref      # chi = 0: free eta and phi are one circle
        return
    assert_same_set(ref, mine, d, (1, 1, 1), (2, 2, 0))


def test_between_matches_reference_hex(hexa):
    d = Diffractometer(PSIC, hexa.UB, energy=5830)
    ref = reference(hexa, (2, 0, 14), (0, 2, 14), 5830, {"eta": 0, "chi": 0})
    mine = d.solve((2, 0, 14), {"eta": 0, "chi": 0, "h2": (0, 2, 14)}, detector="between")
    assert len(ref) == len(mine) > 0
    assert_same_set(ref, mine, d, (2, 0, 14), (0, 2, 14))


@pytest.mark.parametrize("seed", range(12))
def test_between_matches_reference_random(si, seed):
    rng = np.random.default_rng(seed)
    pair = list(itertools.combinations(["mu", "eta", "chi", "phi"], 2))[rng.integers(6)]
    fixed = {n: float(rng.uniform(-90, 90)) for n in pair}
    h1, h2 = (tuple(int(x) for x in rng.integers(-2, 3, 3)) for _ in range(2))
    if not any(h1) or not any(h2):
        return
    d = Diffractometer(PSIC, si.UB, energy=9000)
    if not np.cross(h1, h2).any():      # parallel: no solution or a continuum (the reference hangs)
        try:
            assert d.solve(h1, fixed | {"h2": h2}, detector="between") == []
        except Degenerate:
            pass
        return
    assert_same_set(reference(si, h1, h2, 9000, fixed),
                    d.solve(h1, fixed | {"h2": h2}, detector="between"), d, h1, h2)


@pytest.mark.parametrize("mode,hold", [({"mu": 0, "nu": 0}, ("mu", "chi")),
                                       ({"eta": 0, "delta": 0}, ("eta", "chi"))],
                         ids=["vertical", "horizontal"])
def test_detector_on_h1_libhkl_modes(si, mode, hold):
    """libhkl's double_diffraction_vertical / _horizontal: 8 branches for
    Si 111 + 220 at 9 keV; each is in Bragg for both reflections by diffcalc's
    forward model, and is found again by the 'between' solver with two of its
    sample circles held at the solution's values."""
    d = Diffractometer(PSIC, si.UB, energy=9000)
    sols = d.solve((1, 1, 1), mode | {"h2": (2, 2, 0)})
    assert len(sols) == 8
    calc, lam = HklCalculation(si, Constraints({})), d.wavelength
    for s in sols:
        assert np.allclose(calc.get_hkl(Position(**s), lam), (1, 1, 1), atol=1e-9)
        k2 = d.spots(s, [(2, 2, 0)])["kout"][0]          # put the detector on h2 instead
        delta2 = np.degrees(np.arcsin(np.clip(k2[0], -1, 1)))
        nu2 = np.degrees(np.arctan2(k2[2], k2[1]))
        assert np.allclose(calc.get_hkl(Position(**(s | {"delta": delta2, "nu": nu2})), lam),
                           (2, 2, 0), atol=1e-9)
        held = {n: s[n] for n in hold}
        between = d.solve((1, 1, 1), held | {"h2": (2, 2, 0)}, detector="between")
        assert any(abs(wrap(b["eta"] - s["eta"])) < 1e-6 and abs(wrap(b["phi"] - s["phi"])) < 1e-6
                   for b in between)
