"""U from orientations, the angle between reflections and Geometry.detector_towards."""
import numpy as np
import pytest

from hklforge import Geometry
from hklforge.lattice import angle_between, b_matrix
from hklforge.orient import u_from_one, u_from_two

RNG = np.random.default_rng(3)


def _random_u():
    q, r = np.linalg.qr(RNG.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else -q


def test_u_from_two_recovers_u():
    b = b_matrix(4.1, 5.2, 6.3, 87, 95, 101)
    for _ in range(20):
        u = _random_u()
        h1, h2 = RNG.integers(-3, 4, 3), RNG.integers(-3, 4, 3)
        if np.linalg.norm(np.cross(b @ h1, b @ h2)) < 1e-3:
            continue
        np.testing.assert_allclose(u_from_two(b, h1, u @ b @ h1, h2, u @ b @ h2), u, atol=1e-12)
    with pytest.raises(ValueError):
        u_from_two(b, (1, 0, 0), (1, 0, 0), (2, 0, 0), (1, 0, 0))


def test_u_from_one():
    b = b_matrix(5, 5, 7, 90, 90, 120)
    for v in ([0, 0, 1.0], [1.0, 2, -0.5], -(b @ [1, 0, 2])):
        u = u_from_one(b, (1, 0, 2), v)
        a = u @ b @ [1, 0, 2]
        np.testing.assert_allclose(a / np.linalg.norm(a), np.asarray(v) / np.linalg.norm(v), atol=1e-12)
        np.testing.assert_allclose(u @ u.T, np.eye(3), atol=1e-12)
    u0 = _random_u()                       # from u0: only the smallest turn, so u0 b h on v stays u0
    np.testing.assert_allclose(u_from_one(b, (1, 1, 1), u0 @ b @ [1, 1, 1], u0), u0, atol=1e-12)


def test_angle_between():
    b = b_matrix(5.431, 5.431, 5.431)
    assert angle_between(b, (1, 0, 0), (0, 1, 0)) == pytest.approx(90)
    assert angle_between(b, (1, 1, 1), (1, 1, -1)) == pytest.approx(np.degrees(np.arccos(1 / 3)))
    assert angle_between(_random_u() @ b, (1, 1, 1), (1, 1, -1)) == pytest.approx(np.degrees(np.arccos(1 / 3)))


@pytest.mark.parametrize("name", ["psic", "fourcv"])
def test_detector_towards_recovers_the_detector(name):
    g = Geometry.builtin(name)
    if len(g.detector) != 2:
        with pytest.raises(ValueError):
            g.detector_towards(g.beam)
        return
    for _ in range(50):
        angles = {c.name: float(RNG.uniform(-170, 170)) for c in g.sample + g.detector}
        got = g.detector_towards(5 * g.k_out(angles), near=angles)
        np.testing.assert_allclose(g.k_out({**angles, **got}), g.k_out(angles), atol=1e-12)
        for c in g.detector:
            assert abs((got[c.name] - angles[c.name] + 180) % 360 - 180) < 1e-7
