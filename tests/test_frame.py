"""No lab-frame bias: the same instrument written in another frame must give
the same angles, virtual angles and spots, with UB turned along."""
import numpy as np
import pytest

from hklforge import Diffractometer, Geometry
from hklforge.lattice import b_matrix
from hklforge.virtual import virtual_angles

PSIC = Geometry.builtin("psic")
# beam along z, up along y, x completing a right-handed frame
BEAM_Z = PSIC.reframed(beam="z+", up="y+", name="psic_beam_z")

MODES = [
    {"mu": 0, "nu": 0, "chi": 90},
    {"qaz": 90, "mu": 0, "eta": 10},
    {"delta": 20, "naz": 30, "phi": 15},
    {"nu": 5, "betain": 3, "mu": 0},
    {"mu": 0, "chi": 80, "psi": 40},
    {"mu": 0, "eta": 5, "h2": (1, 1, 1)},
]
VIRTUALS = ("ttheta", "qaz", "naz", "alpha", "beta", "betain", "betaout", "psi", "tau")


def test_axes_of_the_reframed_psic():
    axes = {c.name: c.axis for c in BEAM_Z.sample + BEAM_Z.detector}
    expect = {"mu": (0, 1, 0), "eta": (-1, 0, 0), "chi": (0, 0, 1), "phi": (-1, 0, 0),
              "nu": (0, 1, 0), "delta": (-1, 0, 0)}
    for name, v in expect.items():
        assert np.allclose(axes[name], v), name
    assert np.allclose(BEAM_Z.beam, (0, 0, 1)) and np.allclose(BEAM_Z.up, (0, 1, 0))
    assert BEAM_Z.labels == {"y": "vertical", "z": "beam", "x": "transverse"}
    assert np.isclose(np.linalg.det(PSIC.frame_rotation("z+", "y+")), 1)


def _setup(rng):
    B = b_matrix(*rng.uniform(3, 8, 3), *rng.uniform(80, 100, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q)) @ B


@pytest.mark.parametrize("mode", MODES, ids=lambda m: "+".join(m))
def test_same_solutions_in_another_frame(mode):
    rng = np.random.default_rng(3)
    M = PSIC.frame_rotation("z+", "y+")
    n = 0
    for _ in range(4):
        ub = _setup(rng)
        surf, ref = rng.normal(size=3), rng.normal(size=3)
        d0 = Diffractometer(PSIC, ub, energy=9000, surface=surf, reference=ref)
        d1 = Diffractometer(BEAM_Z, M @ ub, energy=9000, surface=M @ surf, reference=M @ ref)
        for hkl in rng.integers(-3, 4, size=(6, 3)):
            if not hkl.any():
                continue
            try:
                s0 = d0.solve(hkl, mode)
            except Exception as e:                    # the same failure on both
                with pytest.raises(type(e)):
                    d1.solve(hkl, mode)
                continue
            s1 = d1.solve(hkl, mode)
            key = lambda s: tuple(np.round([s[c] for c in sorted(s)], 6))
            assert sorted(map(key, s0)) == sorted(map(key, s1))
            for a in s0:
                v0 = virtual_angles(PSIC, a, d0.surface, d0.reference)
                v1 = virtual_angles(BEAM_Z, a, d1.surface, d1.reference)
                for name in VIRTUALS:
                    if name in v0 and np.isfinite(v0[name]):
                        assert np.isclose(v0[name], v1[name], atol=1e-7), name
                assert np.allclose(d0.hkl(a), d1.hkl(a))
                n += 1
            if s0:
                assert np.isclose(d0.spots(s0[0], [hkl])["tdet_max"], d1.spots(s0[0], [hkl])["tdet_max"])
    assert n


def test_up_alone_gives_the_vertical_azimuth():
    d = {"name": "g", "beam": "z+", "up": "y+",
         "sample": [{"name": "s", "axis": "y+"}], "detector": [{"name": "d", "axis": "x-"}]}
    r, u = Geometry.from_dict(d).azimuth_frame()
    assert np.allclose(u, (0, 1, 0)) and np.allclose(r, (1, 0, 0))
    psic_no_zero = Geometry.from_dict({"name": "p", "beam": "y+", "up": "x+",
                                       "sample": [{"name": "s", "axis": "x+"}],
                                       "detector": [{"name": "d", "axis": "z-"}]})
    assert np.allclose(psic_no_zero.azimuth_frame()[0], PSIC.azimuth_frame()[0])


def test_beam_is_required():
    with pytest.raises(ValueError, match="beam"):
        Geometry.from_dict({"name": "g", "sample": [], "detector": []})
    with pytest.raises(TypeError):
        Geometry.from_xrayutilities(["z-"], ["z-"])
