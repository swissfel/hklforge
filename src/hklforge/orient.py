"""U from orientations: which crystal direction points where.

An orientation is a pair (h, v): the reciprocal-lattice direction ``h``
(Miller indices, through B) points along ``v`` (a direction in the phi frame,
i.e. the lab frame with every circle at zero). Two pairs fix U, one pair fixes
it up to a turn about ``v``. ``ub = u @ b`` with ``b`` from
:func:`hklforge.lattice.b_matrix`.
"""
from __future__ import annotations

import numpy as np

SMALL = 1e-4          # diffcalc's (Vlieg's) threshold for parallel references


def _rotation(axis, angle) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _any_perpendicular(v) -> np.ndarray:
    w = np.cross(v, [1.0, 0, 0] if abs(v[0]) < 0.9 else [0, 1.0, 0])
    return w / np.linalg.norm(w)


def u_from_two(b, h1, v1, h2, v2) -> np.ndarray:
    """U from two (h, v) pairs, Busing & Levy: the first pair is matched
    exactly, the second only in the plane it spans with the first (diffcalc's
    ``_calc_ub_from_two_references``)."""
    def triple(a, c):
        t3 = np.cross(a, c)
        cols = []
        for t in (a, np.cross(t3, a), t3):
            n = np.linalg.norm(t)
            if n < SMALL:
                raise ValueError("the two orientations are parallel; give two different directions")
            cols.append(t / n)
        return np.column_stack(cols)

    b = np.asarray(b, float)
    return triple(np.asarray(v1, float), np.asarray(v2, float)) @ np.linalg.inv(
        triple(b @ np.asarray(h1, float), b @ np.asarray(h2, float)))


def u_from_one(b, h, v, u0=None) -> np.ndarray:
    """U from one (h, v) pair: ``u0`` (default 1) turned by the smallest
    rotation that takes u0 b h onto v. The turn about v is kept from u0, so
    with u0 the crystal's current U the result does not depend on the lab
    frame; from u0 = 1 in the You frame it is diffcalc's primary-only U."""
    u0 = np.eye(3) if u0 is None or not len(u0) else np.asarray(u0, float)
    hc = u0 @ np.asarray(b, float) @ np.asarray(h, float)
    hc = hc / np.linalg.norm(hc)
    v = np.asarray(v, float) / np.linalg.norm(v)
    axis = np.cross(hc, v)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        r = np.eye(3) if hc @ v > 0 else _rotation(_any_perpendicular(hc), np.pi)
    else:
        r = _rotation(axis, np.arctan2(s, hc @ v))
    return r @ u0
