"""The forward model (angles -> Q) against diffcalc and xrayutilities."""
import numpy as np
import pytest

from hklforge.geometry import Geometry, wavevector

rng = np.random.default_rng(0)
PSIC = Geometry.builtin("psic")


def random_angles(geom, n=100):
    for _ in range(n):
        yield {name: rng.uniform(-170, 170) for name in geom.names}


def test_psic_matches_diffcalc():
    diffcalc = pytest.importorskip("diffcalc")
    from diffcalc.hkl.geometry import Position, get_q_phi

    for a in random_angles(PSIC):
        pos = Position(mu=a["mu"], delta=a["delta"], nu=a["nu"], eta=a["eta"], chi=a["chi"], phi=a["phi"])
        # diffcalc's get_q_phi is in units of |k|
        assert np.allclose(PSIC.q_phi(a, 1.0), np.ravel(get_q_phi(pos)), atol=1e-12)


def test_wavevector_matches_diffcalc_convention():
    # |k| = 2 pi / lambda with lambda = hc / E
    from scipy import constants as c
    lam = c.h * c.speed_of_light / (c.e * 12000) * 1e10
    assert np.isclose(wavevector(12000), 2 * np.pi / lam, rtol=1e-14)


@pytest.mark.parametrize("sample,detector,r_i", [
    (["x+", "z-", "y+", "z-"], ["x+", "z-"], (0, 1, 0)),     # psic in xu syntax
    (["z+", "y-", "x+"], ["z+"], (1, 0, 0)),                 # something else entirely
    (["y-", "x+", "z-"], ["x-", "y+"], (0, 0, 1)),
])
def test_xrayutilities_syntax_matches_qconversion(sample, detector, r_i):
    xu = pytest.importorskip("xrayutilities")
    geom = Geometry.from_xrayutilities(sample, detector, r_i)
    qc = xu.experiment.QConversion(sample, detector, r_i)
    energy = 9000.0
    lam = xu.en2lam(energy)
    for a in random_angles(geom, 40):
        args = [a[n] for n in geom.names]
        q_xu = np.ravel(qc(*args, wl=lam))
        # xrayutilities reports Q in the sample frame, |k| = 2 pi / lambda
        assert np.allclose(geom.q_phi(a, 2 * np.pi / lam), q_xu, atol=1e-10), (a, q_xu)


def test_xu_psic_equals_builtin_psic():
    geom = Geometry.from_xrayutilities(["x+", "z-", "y+", "z-"], ["x+", "z-"], (0, 1, 0),
                                       sample_names=["mu", "eta", "chi", "phi"],
                                       detector_names=["nu", "delta"])
    for a in random_angles(PSIC, 20):
        assert np.allclose(geom.q_phi(a, 3.0), PSIC.q_phi(a, 3.0))
