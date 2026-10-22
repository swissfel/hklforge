"""Virtual (pseudo) angles of a position: what they are and how to compute them.

All are defined from lab-frame vectors, so they apply to any geometry:

- b: incoming beam (unit), k' = D b: outgoing beam (unit), |k| = k
- Q = k (k' - b): scattering vector; Q-hat its direction
- n_ref = R n (reference vector), n_surf = R s (surface normal), both unit
- (r, u): the azimuth frame about the beam (``Geometry.azimuth_frame``);
  for psic r = z, u = x, so an azimuth of 90 deg means "vertical"
- e: lab direction of the bisect circle's axis (psic: chi)
- p = (b - (b.Q-hat) Q-hat) / cos(theta): the beam's direction across Q,
  i.e. the unit vector in the scattering plane perpendicular to Q

Definitions (degrees), the same as diffcalc's (You 1999) for psic:

=========  ==========================================================
theta      half the scattering angle; cos(2 theta) = b . k'
ttheta     2 theta
qaz        azimuth of the scattering plane: atan2(k'.u, k'.r)
naz        azimuth of n_ref: atan2(n_ref.u, n_ref.r)
alpha      incidence on the reference plane: sin(alpha) = -b . n_ref
beta       exit from the reference plane:   sin(beta)  = k' . n_ref
betain     as alpha with the surface normal: sin = -b . n_surf
betaout    as beta with the surface normal:  sin = k' . n_surf
tau        angle between Q and n_ref: cos(tau) = Q-hat . n_ref
psi        azimuth of n_ref about Q, from the scattering plane:
           atan2(n_ref . (Q-hat x p), n_ref . p)
omega      direction of e within the scattering plane, from p towards Q:
           atan2(e . Q-hat, e . p); with bisect, e = cos(omega) p + sin(omega)
           Q-hat, so omega = 0 is the classic bisecting position (e normal
           to Q, tilted towards the beam). |omega| < 90 also means
           sin(omega) = e . Q-hat, the angle between Q and the circle's plane
bisect     (a condition, not an angle) e lies in the scattering plane,
           e . (b x Q-hat) = 0
=========  ==========================================================
"""
from __future__ import annotations

import numpy as np

from .rotation import unit

VIRTUAL_ANGLES = ("theta", "ttheta", "qaz", "naz", "alpha", "beta", "betain", "betaout",
                  "tau", "psi", "omega")


def _asin(x):
    return float(np.degrees(np.arcsin(np.clip(x, -1.0, 1.0))))


def scattering_frame(b, Q):
    """(Q-hat, p, s): Q direction, the beam's direction across Q (p) and
    s = Q-hat x p, completing the frame that psi is measured in."""
    Qh = unit(Q)
    p = unit(b - Qh * (b @ Qh))
    return Qh, p, np.cross(Qh, p)


def psi_of(n, b, Q) -> float:
    Qh, p, s = scattering_frame(b, Q)
    return float(np.degrees(np.arctan2(n @ s, n @ p)))


def omega_of(e, b, Q) -> float:
    Qh, p, _ = scattering_frame(b, Q)
    return float(np.degrees(np.arctan2(e @ Qh, e @ p)))


def bisect_axis(geometry, angles: dict) -> np.ndarray:
    """Lab direction of the bisect circle's axis (outer circles applied)."""
    if geometry.bisect_circle is None:
        raise ValueError(f"geometry {geometry.name} declares no bisect_circle")
    M = np.eye(3)
    for c in geometry.sample:
        if c.name == geometry.bisect_circle:
            return M @ np.asarray(c.axis)
        M = M @ c.matrix(angles.get(c.name, 0.0))


def virtual_angles(geometry, angles: dict, surface=None, reference=None) -> dict:
    """Every virtual angle defined for this position (degrees).

    ``surface`` / ``reference``: phi-frame vectors (any length) or None;
    angles that need a missing vector, or are undefined at this position
    (e.g. psi with Q along the beam), are left out or NaN.
    """
    b = np.asarray(geometry.beam)
    kout = geometry.k_out(angles)
    R = geometry.sample_matrix(angles)
    r, u = geometry.azimuth_frame()
    Q = kout - b                                   # in units of k
    out = {"theta": float(np.degrees(np.arccos(np.clip(b @ kout, -1, 1)))) / 2}
    out["ttheta"] = 2 * out["theta"]
    small = np.linalg.norm(Q) < 1e-12
    out["qaz"] = float("nan") if small else float(np.degrees(np.arctan2(kout @ u, kout @ r)))
    if surface is not None:
        s = R @ unit(surface)
        out["betain"], out["betaout"] = _asin(-b @ s), _asin(kout @ s)
    if reference is not None:
        n = R @ unit(reference)
        out["alpha"], out["beta"] = _asin(-b @ n), _asin(kout @ n)
        perp = n - b * (b @ n)
        out["naz"] = (float("nan") if np.linalg.norm(perp) < 1e-12
                      else float(np.degrees(np.arctan2(n @ u, n @ r))))
        if small:
            out["tau"] = out["psi"] = float("nan")
        else:
            Qh = unit(Q)
            out["tau"] = float(np.degrees(np.arccos(np.clip(Qh @ n, -1, 1))))
            across = np.linalg.norm(np.cross(Qh, b)) < 1e-12 or np.linalg.norm(np.cross(Qh, n)) < 1e-12
            out["psi"] = float("nan") if across else psi_of(n, b, Q)
    if geometry.bisect_circle is not None and not small:
        out["omega"] = omega_of(bisect_axis(geometry, angles), b, Q)
    return out
