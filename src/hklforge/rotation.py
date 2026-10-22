"""Rotations about arbitrary axes, and the exact sub-problems every
diffractometer solve breaks down into.

All angles are radians here; the public API converts to degrees.

The sub-problems are the classic Paden-Kahan ones from robot kinematics
(rotations about axes through the origin), plus the variants a diffraction
geometry needs. Each returns *every* solution - at most two - in closed form,
which is what makes the generated solvers complete.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


class Degenerate(ValueError):
    """The sub-problem has a continuum of solutions (e.g. parallel axes)."""


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("zero vector has no direction")
    return v / n


def rot(axis, theta: float) -> np.ndarray:
    """Right-handed rotation by ``theta`` about unit ``axis`` (Rodrigues)."""
    u = np.asarray(axis, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return c * np.eye(3) + s * ux + (1 - c) * np.outer(u, u)


def _wrap(theta):
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


# --- sub-problem 0: one circle, one scalar equation -------------------------

def projection_coeffs(w, axis, v):
    """w . rot(axis, t) v  ==  a cos t + b sin t + g, returns (a, b, g)."""
    w, u, v = (np.asarray(x, dtype=float) for x in (w, axis, v))
    wu, uv = w @ u, u @ v
    return w @ v - wu * uv, w @ np.cross(u, v), wu * uv


def solve_trig_linear(a: float, b: float, c: float, scale: float = 1.0) -> list[float]:
    """All t with a cos t + b sin t = c (at most two)."""
    r = np.hypot(a, b)
    tol = 1e-12 * max(scale, r, abs(c), 1.0)
    if r < tol:
        if abs(c) < tol:
            raise Degenerate("equation holds for every angle")
        return []
    x = c / r
    if abs(x) > 1 + 1e-12:
        return []
    phi = np.arctan2(b, a)
    half = np.arccos(np.clip(x, -1.0, 1.0))
    if half < 1e-9:                         # tangent: one double root
        return [float(_wrap(phi))]
    return [float(_wrap(phi + half)), float(_wrap(phi - half))]


def solve_projection(w, axis, v, c: float) -> list[float]:
    """All t with w . rot(axis, t) v = c."""
    a, b, g = projection_coeffs(w, axis, v)
    return solve_trig_linear(a, b, c - g, scale=np.linalg.norm(w) * np.linalg.norm(v))


# --- sub-problem 1: rotate p onto q about one axis --------------------------

def sp1(axis, p, q) -> float:
    """The angle t with rot(axis, t) p = q (assumes it exists)."""
    u = np.asarray(axis, dtype=float)
    p_ = p - u * (u @ p)
    q_ = q - u * (u @ q)
    if np.linalg.norm(p_) < EPS or np.linalg.norm(q_) < EPS:
        raise Degenerate("vector lies on the rotation axis")
    return float(np.arctan2(u @ np.cross(p_, q_), p_ @ q_))


# --- sub-problem 2: rot(u1, t1) rot(u2, t2) p = q ---------------------------

def cone_points(u1, c1: float, u2, c2: float, length: float) -> list[np.ndarray]:
    """Vectors z with u1.z = c1, u2.z = c2 and |z| = length (u1, u2 unit).

    Geometrically the intersection of two cones (or a cone and a plane) - at
    most two points.
    """
    d = u1 @ u2
    if abs(abs(d) - 1) < 1e-12:
        raise Degenerate("the two directions are parallel")
    den = d * d - 1
    alpha = (d * c2 - c1) / den
    beta = (d * c1 - c2) / den
    n = np.cross(u1, u2)
    g2 = (length ** 2 - alpha ** 2 - beta ** 2 - 2 * alpha * beta * d) / (n @ n)
    if g2 < -1e-10 * max(length ** 2, 1.0):
        return []
    base = alpha * u1 + beta * u2
    if g2 <= 1e-14 * max(length ** 2, 1.0):
        return [base]
    g = np.sqrt(g2)
    return [base + g * n, base - g * n]


def sp2(u1, u2, p, q) -> list[tuple[float, float]]:
    """All (t1, t2) with rot(u1, t1) rot(u2, t2) p = q."""
    u1, u2, p, q = (np.asarray(x, float) for x in (u1, u2, p, q))
    if abs(np.linalg.norm(p) - np.linalg.norm(q)) > 1e-9 * max(np.linalg.norm(p), 1.0):
        return []
    out = []
    # z = rot(u2, t2) p = rot(u1, -t1) q
    for z in cone_points(u2, u2 @ p, u1, u1 @ q, np.linalg.norm(p)):
        t2 = sp1(u2, p, z)
        t1 = sp1(u1, z, q)
        out.append((t1, t2))
    return out


# --- sub-problem 3: three circles realising a full rotation -----------------

def sp3(u1, u2, u3, T) -> list[tuple[float, float, float]]:
    """All (t1, t2, t3) with rot(u1,t1) rot(u2,t2) rot(u3,t3) = T."""
    out = []
    for t1, t2 in sp2(u1, u2, u3, T @ u3):      # u3 is fixed by rot(u3, .)
        rest = rot(u2, -t2) @ rot(u1, -t1) @ T  # = rot(u3, t3)
        p = unit(np.cross(u3, [1.0, 0, 0]) if abs(u3[0]) < 0.9 else np.cross(u3, [0, 1.0, 0]))
        out.append((t1, t2, sp1(u3, p, rest @ p)))
    return out


def rotation_from_pairs(s1, s2, t1, t2) -> np.ndarray:
    """The rotation taking s1 -> t1 and s2 -> t2 (angles between them equal)."""
    def frame(a, b):
        e1 = unit(a)
        e2 = unit(b - e1 * (e1 @ b))
        return np.column_stack([e1, e2, np.cross(e1, e2)])
    return frame(t1, t2) @ frame(s1, s2).T


# --- sub-problem 4: two scalar equations in two circles ---------------------

def solve_projection_pair(eqs, u1, u2, scale: float = 1.0) -> list[tuple[float, float]]:
    """All (t1, t2) with  w_k . rot(u1, t1) rot(u2, t2) v_k = c_k,  k = 1, 2.

    For fixed t1 both equations are linear in (cos t2, sin t2); eliminating
    t2 with cos^2 + sin^2 = 1 leaves a trigonometric polynomial of degree 4 in
    t1. With z = exp(i t1) that is a degree-8 polynomial, whose roots on the
    unit circle are *all* the t1 - exact, no search.
    """
    def coeffs(t1):
        rows, rhs = [], []
        for w, v, c in eqs:
            a, b, g = projection_coeffs(rot(u1, -t1) @ w, u2, v)
            rows.append((a, b))
            rhs.append(c - g)
        return np.array(rows), np.array(rhs)

    def resultant(t1):
        M, r = coeffs(t1)
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        x = np.array([M[1, 1] * r[0] - M[0, 1] * r[1], -M[1, 0] * r[0] + M[0, 0] * r[1]])
        return x @ x - det * det

    # exact Fourier coefficients of a degree-4 trig polynomial from 16 samples
    n = 16
    ts = 2 * np.pi * np.arange(n) / n
    vals = np.array([resultant(t) for t in ts])
    F = np.fft.fft(vals) / n                     # F[k mod n] = coefficient of e^{ikt}
    ck = {k: F[k % n] for k in range(-4, 5)}     # E(t) = sum ck e^{ikt}
    big = max(abs(v) for v in ck.values())
    if big < 1e-14 * scale ** 4:
        raise Degenerate("the two equations do not fix the first circle")
    poly = np.array([ck[k] for k in range(4, -5, -1)])   # z^8 ... z^0 of z^4 E
    poly = np.trim_zeros(np.where(abs(poly) < 1e-13 * big, 0, poly), "f")
    roots = np.roots(poly) if poly.size > 1 else []
    cands = [float(np.angle(z)) for z in roots if abs(abs(z) - 1) < 1e-4]

    out = []
    for t1 in cands:
        M, r = coeffs(t1)
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if abs(det) > 1e-9 * max(np.abs(M).max(), 1e-300) ** 2:
            cs = np.linalg.solve(M, r)
            t2s = [float(np.arctan2(cs[1], cs[0]))]
        else:                                    # rows dependent: use one equation
            k = int(np.argmax(np.abs(M).sum(axis=1)))
            t2s = solve_trig_linear(M[k, 0], M[k, 1], r[k], scale)
        out += [(t1, t2) for t2 in t2s]
    return out
