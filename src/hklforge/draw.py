"""3D drawings of a position and its virtual angles (matplotlib, optional).

``draw(d, angles, show=...)`` puts the lab vectors of one position into a 3D
axes - beam b, outgoing beam k', Q, surface normal, reference vector, the
bisect circle's axis - and draws each requested virtual angle as a labelled
arc, using the definitions of :mod:`hklforge.virtual`. ``draw_circles``
draws the geometry itself: every circle as a ring about its current axis.

The plot is oriented so that the geometry's "vertical" lab axis (from its
``labels``, e.g. psic x) points up and the beam runs left to right.
"""
from __future__ import annotations

import numpy as np

from .rotation import rot, unit
from .virtual import bisect_axis, scattering_frame, virtual_angles

COLORS = {"beam": "#1f77b4", "kout": "#d62728", "Q": "#2ca02c", "surface": "#8c564b",
          "reference": "#9467bd", "bisect": "#ff7f0e", "frame": "#7f7f7f", "p": "#17becf"}
ALL = ("ttheta", "qaz", "naz", "alpha", "beta", "betain", "betaout", "tau", "psi", "omega")


def _perp(v):
    v = unit(v)
    return unit(np.cross(v, [1.0, 0, 0] if abs(v[0]) < 0.9 else [0, 1.0, 0]))


def view_basis(geometry) -> np.ndarray:
    """Rows: plot x (beam), plot y (across), plot z (up), as lab vectors."""
    b = np.asarray(geometry.beam)
    up = None
    for axis, label in geometry.labels.items():
        if str(label).lower() == "vertical":
            up = np.eye(3)["xyz".index(axis)]
    if up is None:
        up = geometry.azimuth_frame()[1]
    up = unit(up - b * (b @ up))
    return np.array([b, np.cross(up, b), up])


class _Scene:
    def __init__(self, ax, basis):
        self.ax, self.B = ax, basis

    def p(self, v):
        return self.B @ np.asarray(v, float)

    def arrow(self, vec, color, label=None, origin=(0, 0, 0), lw=2.0, ls="-"):
        o, v = self.p(origin), self.p(vec)
        self.ax.quiver(*o, *v, color=color, lw=lw, arrow_length_ratio=0.08, linestyle=ls)
        if label:
            t = o + 1.08 * v
            self.ax.text(*t, label, color=color, fontsize=11, ha="center", va="center")

    def line(self, pts, color, lw=1.0, ls="-"):
        P = np.array([self.p(x) for x in pts])
        self.ax.plot(P[:, 0], P[:, 1], P[:, 2], color=color, lw=lw, ls=ls)

    def arc(self, v1, v2, radius, color, label, axis=None):
        """Arc from direction v1 to v2 (about their common normal, or about ``axis``)."""
        v1 = unit(v1)
        if axis is None:
            axis = np.cross(v1, unit(v2))
            if np.linalg.norm(axis) < 1e-9:
                return
            axis = unit(axis)
        v1p = unit(v1 - axis * (axis @ v1))
        v2p = unit(v2 - axis * (axis @ v2))
        ang = np.arctan2(axis @ np.cross(v1p, v2p), v1p @ v2p)
        ts = np.linspace(0, ang, 40)
        pts = [radius * rot(axis, t) @ v1p for t in ts]
        self.line(pts, color, lw=1.6)
        mid = (radius + 0.12) * rot(axis, ang / 2) @ v1p
        self.ax.text(*self.p(mid), label, color=color, fontsize=11, ha="center", va="center")

    def plane(self, e1, e2, color, size=1.0, alpha=0.08):
        e1, e2 = unit(e1), unit(e2)
        corners = [size * (s1 * e1 + s2 * e2) for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self.ax.add_collection3d(Poly3DCollection([[self.p(c) for c in corners]],
                                                  facecolor=color, alpha=alpha, edgecolor="none"))

    def finish(self, lim=1.25, title=None):
        ax = self.ax
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        if title:
            ax.set_title(title)


def _ax(ax):
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure(figsize=(6.5, 6))
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=18, azim=-62)
    return ax


def _lab_axes(sc, geometry):
    """A small lab x, y, z triad in a corner of the plot."""
    origin = sc.B.T @ np.array([-1.0, -1.05, -1.0])
    for i, name in enumerate("xyz"):
        e = np.eye(3)[i]
        label = geometry.labels.get(name, "")
        sc.arrow(0.28 * e, COLORS["frame"], None, origin=origin, lw=0.8)
        sc.ax.text(*sc.p(origin + 0.42 * e), f"{name}" + (f" ({label})" if label else ""),
                   color=COLORS["frame"], fontsize=8, ha="center", va="center")


def _look_from(sc, direction):
    """Point the camera from lab ``direction`` towards the origin, from the
    side that keeps the beam running left to right."""
    v = sc.p(unit(direction))
    if v[1] > 0:
        v = v * np.array([1, -1, 1]) if abs(v[0]) > 0.9 else -v * np.array([1, 1, -1])
    sc.ax.view_init(elev=float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))),
                    azim=float(np.degrees(np.arctan2(v[1], v[0]))))


def draw(d, angles: dict, show=("ttheta",), ax=None, title=None, lab_axes=True, view=None,
         hkls=()):
    """Draw one position of Diffractometer ``d`` with the virtual angles in ``show``.

    ``show``: any of "ttheta", "qaz", "naz", "alpha", "beta", "betain",
    "betaout", "tau", "psi", "omega", or "all". ``view``: None (default
    oblique view), "beam" (look downstream along the beam - azimuths in
    true size), "Q" (look down Q - psi in true size), "plane" (normal to the
    scattering plane - 2theta, omega), "Qn" (normal to Q and n - tau) or a
    lab vector to look from. ``hkls``: further reflections whose outgoing
    beams b + R UB h / |k| are drawn too (double diffraction). Vectors are
    unit length; returns the matplotlib 3D axes.
    """
    g = d.geometry
    ax = _ax(ax)
    sc = _Scene(ax, view_basis(g))
    show = ALL if show == "all" else tuple(show)
    b = np.asarray(g.beam)
    kout = g.k_out(angles)
    R = g.sample_matrix(angles)
    Q = kout - b
    Qh, p, s = scattering_frame(b, Q)
    va = virtual_angles(g, angles, d.surface, d.reference)
    r, u = g.azimuth_frame()

    if isinstance(view, str):
        n_lab = None if d.reference is None else R @ d.reference
        view = {"beam": -b + 0.15 * sc.B[2], "Q": Qh + 0.25 * s,
                "plane": np.cross(b, Qh) + 0.3 * sc.B[2],
                "Qn": None if n_lab is None else np.cross(Qh, n_lab) + 0.2 * sc.B[2]}[view]
    if view is not None:
        _look_from(sc, view)
    if lab_axes:
        _lab_axes(sc, g)
    sc.arrow(b, COLORS["beam"], None, origin=-b, lw=2.2)          # incoming, ends at the sample
    sc.ax.text(*sc.p(-1.1 * b), "k (beam b)", color=COLORS["beam"], fontsize=11)
    sc.arrow(kout, COLORS["kout"], "k' (detector)" if hkls else "k'")
    sc.arrow(Qh, COLORS["Q"], "Q")
    for i, h in enumerate(hkls):
        ki = unit(b + R @ d.ub @ np.asarray(h, float) / d.k)
        sc.arrow(ki, "#e377c2", f"k'{i + 1} {tuple(int(x) if float(x).is_integer() else x for x in h)}",
                 lw=1.5, ls="--")
    sc.plane(b, Qh, COLORS["Q"], size=1.0, alpha=0.06)            # scattering plane

    n = None if d.reference is None else R @ d.reference
    sv = None if d.surface is None else R @ d.surface
    if n is not None and set(show) & {"naz", "alpha", "beta", "tau", "psi"}:
        sc.arrow(n, COLORS["reference"], "n (reference)")
        if set(show) & {"alpha", "beta"}:                       # the plane alpha/beta refer to
            sc.plane(_perp(n), np.cross(n, _perp(n)), COLORS["reference"], size=0.8, alpha=0.10)
    if sv is not None and set(show) & {"betain", "betaout"}:
        sc.arrow(sv, COLORS["surface"], "surface normal")
        sc.plane(_perp(sv), np.cross(sv, _perp(sv)), COLORS["surface"], size=0.8, alpha=0.10)

    if "ttheta" in show:
        sc.arc(b, kout, 0.5, COLORS["kout"], f"2θ={va['ttheta']:.1f}°")
        sc.line([np.zeros(3), 0.9 * b], COLORS["beam"], ls=":")      # beam continued
    if {"qaz", "naz"} & set(show):
        sc.arrow(0.8 * r, COLORS["frame"], "r (az. 0)", lw=1.0, ls="--")
        sc.arrow(0.8 * u, COLORS["frame"], "u (az. 90)", lw=1.0, ls="--")
        sc.plane(r, u, COLORS["frame"], size=0.9, alpha=0.05)
    if "qaz" in show:
        kp = kout - b * (b @ kout)
        sc.line([np.zeros(3), kp], COLORS["kout"], ls=":")
        sc.arc(r, kp, 0.4, COLORS["kout"], f"qaz={va['qaz']:.1f}°", axis=b)
    if "naz" in show and n is not None:
        np_ = n - b * (b @ n)
        sc.line([np.zeros(3), np_], COLORS["reference"], ls=":")
        sc.arc(r, np_, 0.75, COLORS["reference"], f"naz={va['naz']:.1f}°", axis=b)
    for name, vec, beam_vec, sign in (("alpha", n, -b, 1), ("beta", n, kout, 1),
                                      ("betain", sv, -b, 1), ("betaout", sv, kout, 1)):
        if name in show and vec is not None:
            proj = beam_vec - vec * (vec @ beam_vec)        # the beam's shadow on the plane
            col = COLORS["reference"] if name in ("alpha", "beta") else COLORS["surface"]
            sc.line([np.zeros(3), unit(proj) * 0.9], col, ls=":")
            radius = 0.75 if name in ("alpha", "betain") else 0.55
            sc.arc(proj, beam_vec, radius, col, f"{name}={va[name]:.1f}°")
    if "tau" in show and n is not None:
        sc.arc(Qh, n, 0.35, COLORS["reference"], f"τ={va['tau']:.1f}°")
    if "psi" in show and n is not None:
        sc.arrow(0.7 * p, COLORS["p"], "p", lw=1.2)
        npp = n - Qh * (Qh @ n)
        sc.line([np.zeros(3), unit(npp) * 0.7], COLORS["reference"], ls=":")
        sc.arc(p, npp, 0.5, COLORS["reference"], f"ψ={va['psi']:.1f}°", axis=Qh)
    if "omega" in show and g.bisect_circle is not None:
        e = bisect_axis(g, angles)
        sc.arrow(e, COLORS["bisect"], f"{g.bisect_circle} axis")
        sc.arrow(0.7 * p, COLORS["p"], "p", lw=1.2)
        sc.arc(p, e, 0.6, COLORS["bisect"], f"ω={va['omega']:.1f}°")
    sc.finish(title=title)
    return ax


def draw_circles(geometry, angles: dict | None = None, ax=None, title=None):
    """Every circle of ``geometry`` as a ring about its current lab axis.

    Sample circles are nested (outermost = largest ring), detector circles
    drawn on the right; the rotation sense is marked by an arrow head.
    """
    angles = angles or {}
    ax = _ax(ax)
    sc = _Scene(ax, view_basis(geometry))
    _lab_axes(sc, geometry)
    b = np.asarray(geometry.beam)
    sc.arrow(b, COLORS["beam"], None, origin=-b, lw=2.2)
    sc.ax.text(*sc.p(-1.15 * b), "beam", color=COLORS["beam"], fontsize=11)

    def ring(axis, radius, color, label, centre=np.zeros(3)):
        a = unit(axis)
        v = unit(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
        pts = [centre + radius * rot(a, t) @ v for t in np.linspace(0, 2 * np.pi * 0.92, 60)]
        sc.line(pts, color, lw=2.0)
        tip, tail = pts[-1], pts[-4]
        sc.arrow(tip - tail, color, None, origin=tail, lw=2.0)    # rotation sense
        sc.line([centre - 0.3 * radius * a, centre + 1.25 * radius * a], color, lw=0.8, ls="--")
        sc.ax.text(*sc.p(centre + 1.35 * radius * a), label, color=color, fontsize=11)

    M = np.eye(3)
    n = len(geometry.sample)
    for i, c in enumerate(geometry.sample):
        ring(M @ np.asarray(c.axis), 1.0 - 0.6 * i / max(n, 1), "#444444", c.name)
        M = M @ c.matrix(angles.get(c.name, 0.0))
    D = np.eye(3)
    for i, c in enumerate(geometry.detector):
        ring(D @ np.asarray(c.axis), 0.45 - 0.15 * i, COLORS["kout"], c.name)
        D = D @ c.matrix(angles.get(c.name, 0.0))
    sc.arrow(geometry.k_out(angles), COLORS["kout"], "k' (detector)")
    sc.finish(title=title or geometry.name)
    return ax
