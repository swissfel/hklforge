"""Two reflections in Bragg condition at once (experimental, apps/dev only).

diffcalc solves one reflection at a time; this module solves for two together
on top of diffcalc's UB and rotation conventions, and re-checks every result
with diffcalc's own forward model (HklCalculation.get_hkl).

A reflection h is in Bragg condition when
its scattering vector Q = R.UB.h (R = the sample rotation MU.ETA.CHI.PHI) lies
on the Ewald sphere |k_in + Q| = |k|. With the beam along diffcalc +y that
is one scalar equation per reflection,

    y . (R q) = -|q|^2 / (2 |k|),

so two reflections use up two of the sample's degrees of freedom: fix two of
mu/eta/chi/phi and the other two follow, as a finite set of branches. Each
reflection then leaves along its own k' = k_in + Q, and the detector is put
on the bisector of the two, at the largest distance that keeps both spots on
its face.

See apps/dev/diffraction_two_reflections.ipynb for what was learned,
including libhkl's built-in double_diffraction modes as an alternative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation

from bernina_prep.diffraction import en2lam, k_magnitude

SAMPLE_CIRCLES = ("mu", "eta", "chi", "phi")
TDET_LIMITS = (10.0, 5000.0)   # mm, the 3D view's tdet input range
DET_MARGIN = 0.9               # spots kept within 90% of the half width/height


def _rot(axis: str, a):
    """diffcalc's x/y/z_rotation, broadcast over an array of angles (rad)."""
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    rows = {
        "x": [[o, z, z], [z, c, -s], [z, s, c]],
        "y": [[c, z, s], [z, o, z], [-s, z, c]],
        "z": [[c, -s, z], [s, c, z], [z, z, o]],
    }[axis]
    return np.moveaxis(np.array(rows), (0, 1), (-2, -1))


def _sample_rotation(mu, eta, chi, phi):
    """MU.ETA.CHI.PHI as diffcalc builds it (angles in degrees, broadcast)."""
    r = np.radians
    return (_rot("x", r(mu)) @ _rot("z", -r(eta))
            @ _rot("y", r(chi)) @ _rot("z", -r(phi)))


def _detector_angles(u):
    """(delta, gamma) in degrees putting the detector along unit vector u.

    diffcalc's detector ray is NU.DELTA.y = (sin d, cos d cos g, cos d sin g),
    so d = asin(u_x) and g = atan2(u_z, u_y). The other arm configuration
    (180 - d, g + 180) points the same way.
    """
    return (np.degrees(np.arcsin(np.clip(u[0], -1.0, 1.0))),
            np.degrees(np.arctan2(u[2], u[1])))


def _wrap(a):
    """Degrees into (-180, 180]."""
    a = (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0
    return np.where(a == -180.0, 180.0, a)


def two_reflection_window(ubcalc: UBCalculation, hkl1, hkl2, energy: float) -> dict:
    """Whether two reflections can be in Bragg condition together at all.

    Independent of the circles: Q_i must make the angle 90 + theta_i with the
    beam, so two directions on those two cones can be separated by anything
    from |theta1 - theta2| to 180 - theta1 - theta2, and the angle between the
    two reflections is fixed by the crystal. Returns thetas, the angle between
    the reflections and that window (degrees), plus ``ok``.
    """
    k = k_magnitude(energy)
    q = [np.asarray(ubcalc.UB, dtype=float) @ np.asarray(h, dtype=float) for h in (hkl1, hkl2)]
    norms = [np.linalg.norm(v) for v in q]
    out = {"q": q, "k": k}
    for i, (n, h) in enumerate(zip(norms, (hkl1, hkl2)), start=1):
        if n < 1e-12:
            raise ValueError(f"reflection {i} {tuple(h)} is the origin")
        if n > 2 * k:
            raise ValueError(f"reflection {i} {tuple(h)} is out of reach at this "
                             f"energy (d = {2 * np.pi / n:.3f} Å < λ/2)")
    th = [np.degrees(np.arcsin(n / (2 * k))) for n in norms]
    sep = np.degrees(np.arccos(np.clip(q[0] @ q[1] / (norms[0] * norms[1]), -1.0, 1.0)))
    lo, hi = abs(th[0] - th[1]), 180.0 - th[0] - th[1]
    out.update(theta1=th[0], theta2=th[1], separation=sep, window=(lo, hi),
               ok=bool(lo - 1e-9 <= sep <= hi + 1e-9))
    return out


def solve_two_reflections(
    ubcalc: UBCalculation,
    hkl1, hkl2,
    energy: float,
    fixed: dict,
    det_width: float = 77.0,
    det_height: float = 77.0,
    scan_step: float = 0.05,
    grid_step: float = 2.0,
) -> pd.DataFrame:
    """Sample settings putting two reflections in Bragg condition at once.

    ``fixed`` maps exactly two of mu/eta/chi/phi to their (degree) values; the
    other two are solved for. Returns one column per solution branch, like
    :func:`solve_hkl`: ``gamma``/``delta`` place the detector on the bisector
    of the two outgoing beams (so :func:`solution_angles` drives the 3D view
    straight from this table), followed by each reflection's own detector
    angles, the angle between the two k', ``tdet max`` (mm, largest distance
    keeping both spots on a det_width x det_height face) and the incidence /
    exit angles relative to the surface.
    """
    from scipy.optimize import root

    sample_fixed = {n: float(v) for n, v in fixed.items() if n in SAMPLE_CIRCLES}
    free = [n for n in SAMPLE_CIRCLES if n not in sample_fixed]
    if len(free) != 2:
        raise ValueError(
            "two-reflection mode needs exactly two of mu, eta, chi, phi fixed "
            f"(now fixed: {', '.join(sample_fixed) or 'none'})")

    win = two_reflection_window(ubcalc, hkl1, hkl2, energy)
    if not win["ok"]:
        lo, hi = win["window"]
        raise ValueError(
            f"these two reflections are {win['separation']:.2f}° apart, but at this "
            f"energy two reflections can only be in Bragg condition together when "
            f"{lo:.2f}° ≤ angle ≤ {hi:.2f}°")
    q1, q2 = win["q"]
    k = win["k"]
    targets = np.array([-(q1 @ q1) / (2 * k), -(q2 @ q2) / (2 * k)])
    Q = np.stack([q1, q2], axis=1)                     # columns q1, q2

    def rotation(a, b):
        ang = dict(sample_fixed)
        ang[free[0]], ang[free[1]] = a, b
        return _sample_rotation(ang["mu"], ang["eta"], ang["chi"], ang["phi"])

    def beam_part(a, b):
        """Both Bragg residuals, y.(R q_i) - target_i, shape (..., 2)."""
        return (rotation(a, b) @ Q)[..., 1, :] - targets

    def residual(ab):
        return beam_part(ab[0], ab[1])

    scale = max(abs(targets).max(), 1e-12)
    seeds = []

    # (1) Main search, reduced to 1-D. Each circle enters R as one rotation
    # about a fixed axis, so with one free circle `a` held, either reflection's
    # equation is exactly al*cos(b) + be*sin(b) + ga = 0 in the other circle
    # `b`: closed form, two branches. Along each branch the other reflection is
    # a 1-D function of `a`, scanned finely for sign changes; where a branch
    # ends (the two meet at a fold) a sign difference between them seeds as
    # well. All four role assignments are scanned (which circle is `a`, which
    # reflection is solved in closed form), so a reflection that happens not to
    # depend on one circle - e.g. one lying along the phi axis - never hides a
    # solution.
    a_grid = np.arange(-180.0, 180.0, scan_step)
    nxt = np.roll(np.arange(a_grid.size), -1)          # periodic neighbours
    prv = np.roll(np.arange(a_grid.size), 1)
    for swap in (False, True):
        # `a` is the scanned circle, `b` the one solved in closed form
        def F(a, b, swap=swap):
            return beam_part(b, a) if swap else beam_part(a, b)

        f0, f90, f180 = (F(a_grid, np.full_like(a_grid, c)) for c in (0.0, 90.0, 180.0))
        for first in (0, 1):
            other = 1 - first
            ga = (f0[:, first] + f180[:, first]) / 2
            al = (f0[:, first] - f180[:, first]) / 2
            be = f90[:, first] - ga
            r = np.hypot(al, be)
            with np.errstate(divide="ignore", invalid="ignore"):
                c = -ga / r
            valid = (r >= 1e-9 * scale) & (np.abs(c) <= 1.0)
            if not valid.any():
                continue
            base = np.degrees(np.arctan2(be, al))
            half = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
            branches = [base + half, base - half]
            g2 = [F(a_grid, b)[:, other] for b in branches]
            found = []
            for b, g in zip(branches, g2):
                for i in np.nonzero(valid & valid[nxt] & (g * g[nxt] <= 0))[0]:
                    found.append((a_grid[i] + scan_step / 2, b[i]))
            edge = valid & (~valid[nxt] | ~valid[prv])
            for i in np.nonzero(edge & (g2[0] * g2[1] <= 0))[0]:
                found += [(a_grid[i], branches[0][i]), (a_grid[i], branches[1][i])]
            seeds += [(b, a) if swap else (a, b) for a, b in found]

    # (2) Independent second seed source: a coarse 2-D grid, keeping every
    # cell in which both residuals change sign.
    g = np.arange(-180.0, 180.0 + grid_step / 2, grid_step)
    A, B = np.meshgrid(g, g, indexing="ij")
    Fg = beam_part(A, B)
    crosses = np.ones(A[:-1, :-1].shape, dtype=bool)
    for comp in (0, 1):
        f = np.sign(Fg[..., comp])
        corners = np.stack([f[:-1, :-1], f[1:, :-1], f[:-1, 1:], f[1:, 1:]])
        crosses &= (corners.min(axis=0) <= 0) & (corners.max(axis=0) >= 0)
    seeds += [(g[i] + grid_step / 2, g[j] + grid_step / 2) for i, j in zip(*np.nonzero(crosses))]

    roots = []
    for seed in seeds:
        sol = root(residual, seed, method="hybr", options={"xtol": 1e-14})
        if np.abs(residual(sol.x)).max() > 1e-10 * scale:
            continue
        ab = _wrap(sol.x)
        if not any(np.abs(_wrap(ab - prev)).max() < 1e-6 for prev in roots):
            roots.append(ab)
    if not roots:
        fixed_txt = " and ".join(f"{n} = {v:g}" for n, v in sample_fixed.items())
        # a free circle whose axis lies along the beam changes no Bragg
        # condition at all, leaving one circle for two equations
        probe = np.arange(-180.0, 180.0, 15.0)
        inert = []
        for pos, name in enumerate(free):
            vals = [beam_part(*((probe, np.full_like(probe, v)) if pos == 1
                                 else (np.full_like(probe, v), probe)))
                    for v in (0.0, 90.0, 180.0)]
            if max(np.abs(v - vals[0]).max() for v in vals[1:]) < 1e-9 * scale:
                inert.append(name)
        why = (f" ({' and '.join(inert)} then rotates about the beam, which "
               "changes no Bragg condition)" if inert else "")
        raise ValueError(
            "the two reflections fit on the Ewald sphere together, but not with "
            f"{fixed_txt} fixed{why} - try fixing a different pair of sample circles")
    roots.sort(key=lambda ab: (abs(ab[0]) + abs(ab[1]), ab[0], ab[1]))

    # Every solution is re-checked with diffcalc's own forward model before it
    # is returned: at the solved sample setting and each reflection's own
    # detector angles, get_hkl must give that reflection back.
    check = HklCalculation(ubcalc, Constraints({}))
    lam = en2lam(energy)

    beam = np.array([0.0, 1.0, 0.0])
    surf_phi = np.asarray(ubcalc.surf_nphi, dtype=float).ravel()
    surf_phi = surf_phi / np.linalg.norm(surf_phi)
    columns = []
    for a, b in roots:
        R = rotation(a, b)
        kout = [k * beam + R @ q for q in (q1, q2)]
        u = [v / np.linalg.norm(v) for v in kout]
        centre = u[0] + u[1]
        centre = centre / np.linalg.norm(centre)
        surf = R @ surf_phi
        ang = dict(sample_fixed)
        ang[free[0]], ang[free[1]] = float(a), float(b)
        d1, g1 = _detector_angles(u[0])
        d2, g2 = _detector_angles(u[1])
        for (d, gm), h in (((d1, g1), hkl1), ((d2, g2), hkl2)):
            pos = Position(mu=ang["mu"], delta=d, nu=gm,
                           eta=ang["eta"], chi=ang["chi"], phi=ang["phi"])
            got = np.array(check.get_hkl(pos, lam))
            if np.abs(got - np.asarray(h, dtype=float)).max() > 1e-6:
                raise RuntimeError(f"internal check failed: diffcalc puts {tuple(h)} "
                                   f"at {tuple(np.round(got, 6))} for this solution")
        common = {
            "2θ 1": 2 * win["theta1"], "delta 1": d1, "gamma 1": g1,
            "2θ 2": 2 * win["theta2"], "delta 2": d2, "gamma 2": g2,
            "k′ sep.": float(np.degrees(np.arccos(np.clip(u[0] @ u[1], -1, 1)))),
            "betain": float(np.degrees(np.arcsin(np.clip(-beam @ surf, -1, 1)))),
            "betaout 1": float(np.degrees(np.arcsin(np.clip(u[0] @ surf, -1, 1)))),
            "betaout 2": float(np.degrees(np.arcsin(np.clip(u[1] @ surf, -1, 1)))),
        }
        dc, gc = _detector_angles(centre)
        # both arm configurations reach the bisector; filters pick between them
        for delta, gamma in ((dc, gc), (180.0 - dc, gc + 180.0)):
            delta, gamma = float(_wrap(delta)), float(_wrap(gamma))
            D = _rot("x", np.radians(gamma)) @ _rot("z", -np.radians(delta))
            # face axes: width along diffcalc z, height along diffcalc x
            e_w, e_h = D @ np.array([0.0, 0.0, 1.0]), D @ np.array([1.0, 0.0, 0.0])
            tdet = TDET_LIMITS[1]
            for v in u:
                cos_c = v @ centre
                for e, size in ((e_w, det_width), (e_h, det_height)):
                    per_mm = abs(v @ e) / cos_c       # spot offset per mm of tdet
                    if per_mm > 1e-12:
                        tdet = min(tdet, DET_MARGIN * size / 2 / per_mm)
            columns.append({"gamma": gamma, "delta": delta,
                            "mu": ang["mu"], "eta": ang["eta"],
                            "chi": ang["chi"], "phi": ang["phi"],
                            **common, "tdet max": tdet})
    return pd.DataFrame({f"sol. {n}": col for n, col in enumerate(columns)})