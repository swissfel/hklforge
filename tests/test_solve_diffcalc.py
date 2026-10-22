"""The planner against diffcalc, the oracle for psic: for every constraint
combination both support, the *set* of positions must be the same."""
import itertools
import logging

import numpy as np
import pytest

from hklforge import Diffractometer, Geometry

diffcalc = pytest.importorskip("diffcalc")
from diffcalc.hkl.calc import HklCalculation          # noqa: E402
from diffcalc.hkl.constraints import Constraints      # noqa: E402
from diffcalc.hkl.geometry import Position            # noqa: E402
from diffcalc.ub.calc import UBCalculation            # noqa: E402
from diffcalc.util import DiffcalcException           # noqa: E402

logging.getLogger("diffcalc").setLevel(logging.ERROR)

PSIC = Geometry.builtin("psic")
SAMPLE = ("mu", "eta", "chi", "phi")
DETECTOR = ("delta", "nu")
REFS = ("betain", "betaout", "bin_eq_bout", "alpha", "beta", "a_eq_b")

MODES = (
    [s for s in itertools.combinations(SAMPLE, 3)]
    + [(d,) + s for d in DETECTOR for s in itertools.combinations(SAMPLE, 2)]
    + [s + (r,) for s in itertools.combinations(SAMPLE, 2) for r in REFS]
    + [(d, r, s) for d in DETECTOR for r in REFS for s in SAMPLE]
)


def random_setup(rng):
    ub = UBCalculation("test")
    ub.set_lattice("x", *rng.uniform(3, 8, 3), *rng.uniform(70, 110, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    ub.set_u(q * np.sign(np.linalg.det(q)))
    # unit vectors: diffcalc stores n_phi as given and its alpha/beta assume |n| = 1
    surf, ref = (v / np.linalg.norm(v) for v in rng.normal(size=(2, 3)))
    ub.surf_nphi = tuple(surf)
    ub.n_phi = tuple(ref)
    return ub, surf, ref


def random_value(name, rng):
    if name in ("bin_eq_bout", "a_eq_b"):
        return True
    if name in REFS:
        return float(rng.uniform(-20, 40))
    return float(rng.uniform(-170, 170))


def same(a, b, tol=1e-5):
    return all(abs((a[n] - b[n] + 180) % 360 - 180) < tol for n in a)


def compare(names, seed):
    """(n_hklforge, n_diffcalc) after asserting the two sets are equal."""
    rng = np.random.default_rng(seed)
    ub, surf, ref = random_setup(rng)
    lam = rng.uniform(0.7, 2.0)
    while True:                                       # reachable: |q| < 1.8 k
        hkl = tuple(int(x) for x in rng.integers(-3, 4, 3))
        if 0 < np.linalg.norm(ub.UB @ hkl) < 1.8 * 2 * np.pi / lam:
            break
    mode = {n: random_value(n, rng) for n in names}
    d = Diffractometer(PSIC, ub.UB, wavelength=lam, surface=surf, reference=ref)
    mine = d.solve(hkl, mode)
    try:
        theirs = [p.asdict for p, _ in HklCalculation(ub, Constraints(mode)).get_position(*hkl, lam)]
    except AssertionError as e:
        pytest.skip(f"diffcalc internal assertion: {str(e)[:60]}")
    except DiffcalcException as e:
        msg = " ".join(str(e).replace("*", " ").split())
        if "The angles calculated for hkl" in msg:
            # diffcalc's own self-check rejected its result (seen with eta, phi
            # and one detector circle fixed, |eta| > 90): judge ours by
            # diffcalc's forward model instead
            fwd = HklCalculation(ub, Constraints({}))
            for a in mine:
                assert np.allclose(fwd.get_hkl(Position(**a), lam), hkl, atol=1e-9)
            return len(mine), None
        if "No solutions were found" not in msg and "prohibits a solution" not in msg:
            pytest.skip(f"diffcalc refuses: {msg[:80]}")
        theirs = []
    missing = [x for x in theirs if not any(same(x, m) for m in mine)]
    extra = [m for m in mine if not any(same(m, x) for x in theirs)]
    assert not missing and not extra, (mode, hkl, lam, missing, extra)
    for a in mine:                                    # independent forward check
        assert np.allclose(d.hkl(a), hkl, atol=1e-9)
        pa = d.pseudo(a)
        for n, v in mode.items():
            if n in ("betain", "betaout", "alpha", "beta"):
                assert abs(pa[n] - v) < 1e-7
        if "bin_eq_bout" in mode:
            assert abs(pa["betain"] - pa["betaout"]) < 1e-7
        if "a_eq_b" in mode:
            assert abs(pa["alpha"] - pa["beta"]) < 1e-7
    return len(mine), len(theirs)


@pytest.mark.parametrize("names", MODES, ids="-".join)
@pytest.mark.parametrize("seed", range(6))
def test_same_branches_as_diffcalc(names, seed):
    compare(names, seed)


@pytest.mark.parametrize("names", [("delta", "eta", "phi"), ("nu", "eta", "phi")], ids="-".join)
@pytest.mark.parametrize("seed", range(6, 30))
def test_eta_phi_fixed_where_diffcalc_errs(names, seed):
    """More seeds for the modes in which diffcalc's own result can be wrong."""
    compare(names, seed)


def test_bernina_eta_chi_phi_gives_four():
    """CLAUDE.md's expected count: 4 branches with eta, chi, phi fixed."""
    n, m = compare(("eta", "chi", "phi"), seed=100)
    assert n == m == 4
