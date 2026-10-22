"""The exact sub-problems: every returned solution satisfies the equation, and
known solutions are always among those returned."""
import numpy as np
import pytest

from hklforge.rotation import (Degenerate, rot, rotation_from_pairs, solve_projection,
                               solve_projection_pair, sp1, sp2, sp3, unit)

rng = np.random.default_rng(1)


def rand_unit():
    return unit(rng.normal(size=3))


def close_angle(a, b, tol=1e-7):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi) < tol


def test_rot_is_right_handed():
    assert np.allclose(rot([0, 0, 1], np.pi / 2) @ [1, 0, 0], [0, 1, 0])


@pytest.mark.parametrize("trial", range(50))
def test_sp1_sp2_sp3_recover_known_angles(trial):
    u1, u2, u3 = rand_unit(), rand_unit(), rand_unit()
    t = rng.uniform(-np.pi, np.pi, 3)
    p = rng.normal(size=3)
    assert close_angle(sp1(u1, p, rot(u1, t[0]) @ p), t[0])
    sols = sp2(u1, u2, p, rot(u1, t[0]) @ rot(u2, t[1]) @ p)
    assert 1 <= len(sols) <= 2
    assert any(close_angle(a, t[0]) and close_angle(b, t[1]) for a, b in sols)
    for a, b in sols:
        assert np.allclose(rot(u1, a) @ rot(u2, b) @ p, rot(u1, t[0]) @ rot(u2, t[1]) @ p)
    T = rot(u1, t[0]) @ rot(u2, t[1]) @ rot(u3, t[2])
    sols3 = sp3(u1, u2, u3, T)
    assert any(all(close_angle(x, y) for x, y in zip(s, t)) for s in sols3)
    for s in sols3:
        assert np.allclose(rot(u1, s[0]) @ rot(u2, s[1]) @ rot(u3, s[2]), T)


@pytest.mark.parametrize("trial", range(50))
def test_projection_one_circle(trial):
    w, u, v = rng.normal(size=3), rand_unit(), rng.normal(size=3)
    t0 = rng.uniform(-np.pi, np.pi)
    c = w @ rot(u, t0) @ v
    sols = solve_projection(w, u, v, c)
    assert any(close_angle(s, t0) for s in sols)
    for s in sols:
        assert np.isclose(w @ rot(u, s) @ v, c)


@pytest.mark.parametrize("trial", range(100))
def test_projection_pair_finds_known_solution(trial):
    u1, u2 = rand_unit(), rand_unit()
    t0 = rng.uniform(-np.pi, np.pi, 2)
    R = rot(u1, t0[0]) @ rot(u2, t0[1])
    eqs = []
    for _ in range(2):
        w, v = rng.normal(size=3), rng.normal(size=3)
        eqs.append((w, v, w @ R @ v))
    sols = solve_projection_pair(eqs, u1, u2)
    assert any(close_angle(a, t0[0], 1e-6) and close_angle(b, t0[1], 1e-6) for a, b in sols)
    for a, b in sols:
        Rs = rot(u1, a) @ rot(u2, b)
        for w, v, c in eqs:
            assert abs(w @ Rs @ v - c) < 1e-7 * (1 + abs(c))


def test_parallel_axes_are_reported_degenerate():
    with pytest.raises(Degenerate):
        sp2([0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0])


def test_rotation_from_pairs():
    R = rot(rand_unit(), 1.1)
    s1, s2 = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(rotation_from_pairs(s1, s2, R @ s1, R @ s2), R)
