"""The planner: from a geometry and a mode to every exact solution.

A *mode* is a dict naming what is held: circles with their values (degrees)
and extra constraints, as diffcalc does it::

    {"eta": 0, "chi": 0, "phi": 0}           # 3 sample circles
    {"chi": 10, "phi": 20, "betain": 5}       # 2 sample circles + incidence
    {"mu": 0, "nu": 0, "h2": (2, 2, 0)}       # second reflection in Bragg too

``plan(geometry, mode)`` checks the mode once and picks the chain of exact
sub-problems (``Plan.describe()`` says which); ``Plan.run`` executes it for
one hkl. ``Diffractometer`` ties a geometry to a UB, wavelength and surface
normal and is the convenient entry point.

The reduction that makes every case closed form (b = beam, k = |k|,
q = UB h in the phi frame, R = sample rotation, x = R^T b = the beam seen
from the sample):

- Bragg:     b . R q = -|q|^2 / (2k)
- betain:    b . R n = -sin(betain)            (n = unit surface normal)
- betaout:   k'. R n =  sin(betaout),  and k' = b + R q / k, so
             b . R n =  sin(betaout) - q.n / k
- bin = bout: b . R n = -q.n / (2k)
- alpha, beta, a_eq_b: the same three with the reference vector
- h2 in Bragg: b . R q2 = -|q2|^2 / (2k)

Each is a projection of x onto a known phi-frame vector. x has two degrees
of freedom and Bragg takes one, so **at most one** of these extras can be
added, and with it x is one of at most two points (``cone_points``).

Cases (nS / nD = free sample / detector circles):

- nD = 2, nS = 1: Bragg alone fixes the sample circle (<= 2), then the
  detector is turned onto k' = b + R q / k (``sp2``, <= 2).
- nD = 2, nS = 2, one extra: x from Bragg + extra (<= 2), the two sample
  circles from R x = b (``sp2``, <= 2), detector (``sp2``, <= 2).
- nD = 1, nS = 2: Bragg on the detector circle, b . k' = cos 2theta (<= 2),
  which gives Q_lab; then R q = Q_lab (``sp2``, <= 2).
- nD = 1, nS = 3, one extra: as above to Q_lab; the image R v of the
  extra's vector keeps its angle to q and has the required projection on b
  (``cone_points``, <= 2); R follows from the two vector pairs and the three
  circles from R (``sp3``, <= 2).
- detector "between" (with h2): nD = 2, nS = 2; the sample from both Bragg
  conditions as above, the detector on the bisector of the two outgoing
  beams.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Geometry, wavevector, HC_EV_ANGSTROM
from .rotation import (Degenerate, cone_points, rot, rotation_from_pairs, solve_projection,
                       sp2, sp3, unit)

# extra constraints and the vector they act on
SURFACE_CONSTRAINTS = ("betain", "betaout", "bin_eq_bout")
REFERENCE_CONSTRAINTS = ("alpha", "beta", "a_eq_b")
EXTRA_CONSTRAINTS = SURFACE_CONSTRAINTS + REFERENCE_CONSTRAINTS + ("h2",)
DETECTOR_POLICIES = ("h1", "between")

ANGLE_TOL = 1e-6        # degrees, for merging duplicate branches
RESIDUAL_TOL = 1e-8     # relative, forward-model re-check of every solution


class ModeError(ValueError):
    """The mode does not fix a finite set of positions, or is not supported."""


@dataclass
class Plan:
    geometry: Geometry
    fixed: dict                     # circle name -> degrees
    extras: dict                    # constraint name -> value
    detector: str = "h1"
    free_sample: list = field(init=False)
    free_detector: list = field(init=False)

    def __post_init__(self):
        g = self.geometry
        self.free_sample = [n for n in g.sample_names if n not in self.fixed]
        self.free_detector = [n for n in g.detector_names if n not in self.fixed]
        self._validate()

    # --- checking the mode --------------------------------------------------

    def _validate(self):
        nS, nD, nX = len(self.free_sample), len(self.free_detector), len(self.extras)
        if self.detector not in DETECTOR_POLICIES:
            raise ModeError(f"detector policy {self.detector!r}: use one of {DETECTOR_POLICIES}")
        if self.detector == "between":
            if set(self.extras) != {"h2"}:
                raise ModeError("detector 'between' needs h2 and no other extra constraint")
            if (nS, nD) != (2, 2):
                raise ModeError("detector 'between' needs 2 free sample and 2 free detector circles, "
                                f"got {nS} and {nD}")
            return
        if nX > 1:
            raise ModeError(f"{sorted(self.extras)}: each of these fixes the beam direction seen from "
                            "the sample, which Bragg already constrains; at most one can be added")
        if nD > 2:
            raise ModeError(f"{nD} free detector circles; at most 2 are supported")
        if nD == 0:
            raise ModeError("all detector circles fixed: that fixes |Q|, so a general hkl is unreachable")
        need = (1 if nD == 2 else 2) + nX
        if nS != need:
            what = "underdetermined (a continuum of positions)" if nS > need else "overdetermined"
            raise ModeError(f"{nS} free sample circles {self.free_sample} with {nD} free detector "
                            f"circle(s) and {nX} extra constraint(s) is {what}; need {need} free "
                            "sample circles")
        if nS > 3:
            raise ModeError("more than 3 free sample circles is not supported")

    @property
    def case(self) -> str:
        nS, nD = len(self.free_sample), len(self.free_detector)
        if self.detector == "between":
            return "between"
        return f"nD{nD}_nS{nS}"

    def describe(self) -> str:
        S, Dn = ", ".join(self.free_sample), ", ".join(self.free_detector)
        x = next(iter(self.extras), None)
        steps = {
            "nD2_nS1": [f"Bragg condition b.R q = -|q|^2/2k: one projection equation in {S} (<= 2)",
                        f"k' = b + R q / k; turn the detector ({Dn}) onto it, sp2 (<= 2)"],
            "nD2_nS2": [f"beam in the phi frame x = R^T b from Bragg + {x}: cone_points (<= 2)",
                        f"sample ({S}) from R x = b: sp2 (<= 2)",
                        f"k' = b + R q / k; detector ({Dn}): sp2 (<= 2)"],
            "nD1_nS2": [f"Bragg on the detector ({Dn}): b.k' = cos 2theta, projection (<= 2)",
                        f"Q_lab = k (k' - b); sample ({S}) from R q = Q_lab: sp2 (<= 2)"],
            "nD1_nS3": [f"Bragg on the detector ({Dn}): b.k' = cos 2theta, projection (<= 2)",
                        f"Q_lab = k (k' - b); image of the {x} vector: cone_points (<= 2)",
                        "R from the two vector pairs; sample "
                        f"({S}) from R: sp3 (<= 2)"],
            "between": [f"x = R^T b from Bragg for h1 and h2: cone_points (<= 2)",
                        f"sample ({S}) from R x = b: sp2 (<= 2)",
                        f"detector ({Dn}) on the bisector of k'1 and k'2: sp2 (<= 2)"],
        }[self.case]
        held = ", ".join([f"{n}={v}" for n, v in self.fixed.items()]
                         + [f"{n}={v}" for n, v in self.extras.items()])
        lines = [f"{self.geometry.name}: {held}, detector on {self.detector}"]
        lines += [f"  {i + 1}. {s}" for i, s in enumerate(steps)]
        return "\n".join(lines)

    # --- running it ---------------------------------------------------------

    def run(self, ub, k: float, hkl, surface=None, reference=None) -> list[dict]:
        """Every position (dict of degrees) putting ``hkl`` in Bragg condition.

        ``ub``: phi frame, 2 pi convention (q = UB h, |q| = 2 pi / d);
        ``k`` = 2 pi / lambda; ``surface`` / ``reference``: phi-frame vectors.
        Raises ``Degenerate`` where the solutions form a continuum (e.g. two
        free circles about the same axis at these fixed values).
        """
        try:
            return self._run(ub, k, hkl, surface, reference)
        except Degenerate as e:
            held = ", ".join(f"{n}={v:g}" for n, v in self.fixed.items())
            raise Degenerate(f"{e} (free: {', '.join(self.free_sample + self.free_detector)}; "
                             f"held: {held})") from None

    def _run(self, ub, k, hkl, surface, reference):
        g = self.geometry
        ub = np.asarray(ub, float)
        b = np.asarray(g.beam)
        q = ub @ np.asarray(hkl, float)
        if np.linalg.norm(q) < 1e-12 or np.linalg.norm(q) > 2 * k:
            return []
        eqs = self._sample_equations(ub, k, q, surface, reference)   # [(v, c)] for b . R v = c

        S_axes, F = _chain(g.sample, self.fixed)
        D_axes, G = _chain(g.detector, self.fixed)
        Gb = G @ b
        cos2th = 1 - (q @ q) / (2 * k * k)

        sols = []                                  # (sample radians, detector radians)
        if len(D_axes) == 2:
            for ts, R in _sample_from_beam(S_axes, F, b, eqs):
                if self.detector == "between":
                    q2 = ub @ np.asarray(self.extras["h2"], float)
                    kout = unit(2 * b + R @ (q + q2) / k)
                else:
                    kout = b + R @ q / k
                sols += [(ts, td) for td in sp2(*D_axes, Gb, kout)]
        else:
            (d1,) = D_axes
            for s in solve_projection(b, d1, Gb, cos2th):
                Q_lab = k * (rot(d1, s) @ Gb - b)
                for ts in _sample_from_q(S_axes, F, b, q, Q_lab, eqs[1:]):
                    sols.append((ts, (s,)))

        out = []
        for ts, td in sols:
            a = dict(self.fixed)
            a.update({n: _wrap_deg(np.degrees(t)) for n, t in zip(self.free_sample, ts)})
            a.update({n: _wrap_deg(np.degrees(t)) for n, t in zip(self.free_detector, td)})
            a = {n: float(a[n]) for n in g.names}
            if self._residual(a, ub, k, q, eqs) < RESIDUAL_TOL and not any(_same(a, o) for o in out):
                out.append(a)
        return out

    def _sample_equations(self, ub, k, q, surface, reference):
        """Bragg first, then the extra (if any), each as (v, c): b . R v = c."""
        eqs = [(q, -(q @ q) / (2 * k))]
        for name, val in self.extras.items():
            if name == "h2":
                q2 = ub @ np.asarray(val, float)
                eqs.append((q2, -(q2 @ q2) / (2 * k)))
                continue
            vec = surface if name in SURFACE_CONSTRAINTS else reference
            if vec is None:
                which = "surface normal" if name in SURFACE_CONSTRAINTS else "reference vector"
                raise ModeError(f"{name} needs a {which}")
            n = unit(vec)
            if name in ("betain", "alpha"):
                eqs.append((n, -np.sin(np.radians(val))))
            elif name in ("betaout", "beta"):           # k'. R n = sin(val), k' = b + R q / k
                eqs.append((n, np.sin(np.radians(val)) - (q @ n) / k))
            else:                                       # in = out
                eqs.append((n, -(q @ n) / (2 * k)))
        return eqs

    def _residual(self, a, ub, k, q, eqs) -> float:
        """Largest relative error of all equations, through the forward model."""
        g = self.geometry
        R = g.sample_matrix(a)
        b = np.asarray(g.beam)
        if self.detector == "between":
            res = max(abs(b @ R @ v - c) / max(np.linalg.norm(v), 1e-300) for v, c in eqs)
            q2 = ub @ np.asarray(self.extras["h2"], float)
            kout = unit(2 * b + R @ (q + q2) / k)
            return max(res, np.linalg.norm(g.k_out(a) - kout))
        res = np.linalg.norm(g.q_phi(a, k) - q) / np.linalg.norm(q)
        for v, c in eqs[1:]:
            res = max(res, abs(b @ R @ v - c) / np.linalg.norm(v))
        return res


def plan(geometry: Geometry, mode: dict, detector: str = "h1") -> Plan:
    """Check a mode against a geometry and pick its chain of sub-problems."""
    fixed, extras = {}, {}
    for name, val in mode.items():
        if name in geometry.names:
            fixed[name] = float(val)
        elif name in EXTRA_CONSTRAINTS:
            extras[name] = val
        else:
            raise ModeError(f"{name!r} is neither a circle of {geometry.name} {geometry.names} "
                            f"nor a constraint {EXTRA_CONSTRAINTS}")
    return Plan(geometry, fixed, extras, detector)


class Diffractometer:
    """A geometry with a UB matrix, a wavelength and optional normal vectors.

    ``ub`` is in the phi frame with the 2 pi convention (q = UB h). The
    surface normal (for betain/betaout) and reference vector (alpha/beta)
    are phi-frame vectors; ``*_hkl`` sets them from reciprocal-lattice
    indices instead, n = UB h.
    """

    def __init__(self, geometry: Geometry, ub, wavelength: float | None = None,
                 energy: float | None = None, surface=None, surface_hkl=None,
                 reference=None, reference_hkl=None):
        self.geometry = geometry
        self.ub = np.asarray(ub, float)
        if (wavelength is None) == (energy is None):
            raise ValueError("give exactly one of wavelength (angstrom) or energy (eV)")
        self.k = wavevector(energy) if energy is not None else 2 * np.pi / wavelength
        self.surface = _normal(self.ub, surface, surface_hkl)
        self.reference = _normal(self.ub, reference, reference_hkl)
        self._plans = {}

    @property
    def wavelength(self) -> float:
        return 2 * np.pi / self.k

    @property
    def energy(self) -> float:
        return HC_EV_ANGSTROM / self.wavelength

    def plan(self, mode: dict, detector: str = "h1") -> Plan:
        key = (tuple(sorted((n, repr(v)) for n, v in mode.items())), detector)
        if key not in self._plans:
            self._plans[key] = plan(self.geometry, mode, detector)
        return self._plans[key]

    def solve(self, hkl, mode: dict, detector: str = "h1") -> list[dict]:
        return self.plan(mode, detector).run(self.ub, self.k, hkl, self.surface, self.reference)

    def hkl(self, angles: dict) -> np.ndarray:
        return np.linalg.solve(self.ub, self.geometry.q_phi(angles, self.k))

    def pseudo(self, angles: dict) -> dict:
        """2theta and, where the vectors are set, betain/betaout, alpha/beta (degrees)."""
        g = self.geometry
        b, kout, R = np.asarray(g.beam), g.k_out(angles), g.sample_matrix(angles)
        out = {"ttheta": np.degrees(np.arccos(np.clip(b @ kout, -1, 1)))}
        for vec, names in ((self.surface, ("betain", "betaout")), (self.reference, ("alpha", "beta"))):
            if vec is not None:
                n = R @ unit(vec)
                out[names[0]] = np.degrees(np.arcsin(np.clip(-b @ n, -1, 1)))
                out[names[1]] = np.degrees(np.arcsin(np.clip(kout @ n, -1, 1)))
        return out

    def spots(self, angles: dict, hkls, width: float = 77.0, height: float = 77.0,
              margin: float = 0.9, face=None) -> dict:
        """Where reflections ``hkls`` leave the sample, seen from the detector.

        Returns each reflection's unit k' and the angle between it and the
        detector direction, the largest pairwise separation, and ``tdet_max``:
        the largest sample-detector distance (in the units of width/height)
        that keeps every spot within ``margin`` of the face's half sizes.
        ``face`` = (width axis, height axis) in the lab at all angles zero;
        by default the two lab axes after the beam in cyclic order (for psic,
        beam y: width along z, height along x).
        """
        g = self.geometry
        b = np.asarray(g.beam)
        if face is None:
            i = int(np.argmax(np.abs(b)))
            face = (np.eye(3)[(i + 1) % 3], np.eye(3)[(i + 2) % 3])
        Dm = g.detector_matrix(angles)
        centre = Dm @ b
        axes = [Dm @ unit(a) for a in face]
        R = g.sample_matrix(angles)
        kouts = [unit(b + R @ self.ub @ np.asarray(h, float) / self.k) for h in hkls]
        tdet = np.inf
        for v in kouts:
            c = v @ centre
            if c <= 0:
                tdet = 0.0
                continue
            for e, size in zip(axes, (width, height)):
                per = abs(v @ e) / c                     # offset on the face per unit distance
                if per > 1e-12:
                    tdet = min(tdet, margin * size / 2 / per)
        sep = max((np.degrees(np.arccos(np.clip(u @ w, -1, 1)))
                   for i, u in enumerate(kouts) for w in kouts[i + 1:]), default=0.0)
        return {"kout": kouts,
                "off_axis": [float(np.degrees(np.arccos(np.clip(v @ centre, -1, 1)))) for v in kouts],
                "separation": float(sep), "tdet_max": float(tdet)}


# --- helpers ----------------------------------------------------------------

def _chain(circles, fixed):
    """Lab axes of the free circles and the product F of the fixed ones, with
    the stack's matrix = rot(b_1, t_1) ... rot(b_m, t_m) F
    (fixed rotations pushed right: F rot(a, t) = rot(F a, t) F)."""
    L = np.eye(3)
    free = []
    for c in circles:
        if c.name in fixed:
            L = L @ c.matrix(fixed[c.name])
        else:
            free.append(L @ np.asarray(c.axis))
    return free, L


def _sample_from_beam(axes, F, b, eqs):
    """Sample rotations R with b . R v = c for all (v, c), one or two equations."""
    if len(axes) == 1:
        ((v, c),) = eqs
        for t in solve_projection(b, axes[0], F @ v, c):
            yield (t,), rot(axes[0], t) @ F
        return
    (v1, c1), (v2, c2) = eqs
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if np.linalg.norm(np.cross(v1, v2)) < 1e-12 * n1 * n2:
        s = np.sign(v1 @ v2)                # same line: contradictory or the same equation
        if abs(c1 / n1 - s * c2 / n2) > 1e-12 * (1 + abs(c1 / n1)):
            return
        raise Degenerate("both equations act on parallel vectors, leaving a continuum")
    for x in cone_points(v1 / n1, c1 / n1, v2 / n2, c2 / n2, 1.0):   # x = R^T b
        for t1, t2 in sp2(axes[0], axes[1], F @ x, b):
            yield (t1, t2), rot(axes[0], t1) @ rot(axes[1], t2) @ F


def _sample_from_q(axes, F, b, q, Q_lab, extra):
    """Sample circle angles with R q = Q_lab, plus b . R v = c if one extra."""
    if not extra:
        return sp2(axes[0], axes[1], F @ q, Q_lab)
    ((v, c),) = extra
    out = []
    nq, nv = np.linalg.norm(q), np.linalg.norm(v)
    if np.linalg.norm(np.cross(q, v)) < 1e-10 * nq * nv:
        # R v = (q.v / |q|^2) Q_lab is already known: the extra holds or never does
        if abs(b @ Q_lab * (q @ v) / (nq * nq) - c) > 1e-10 * nv:
            return []
        raise Degenerate("the constraint's vector is parallel to Q, so it does not fix the azimuth")
    for z in cone_points(Q_lab / nq, (q @ v) / nq, b, c, nv):        # z = R v
        R = rotation_from_pairs(q, v, Q_lab, z)
        out += sp3(*axes, R @ F.T)
    return out


def _normal(ub, vec, hkl):
    if vec is not None and hkl is not None:
        raise ValueError("give the vector or its hkl, not both")
    if hkl is not None:
        return unit(ub @ np.asarray(hkl, float))
    return None if vec is None else unit(vec)


def _wrap_deg(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _same(a: dict, b: dict) -> bool:
    return all(abs(_wrap_deg(a[n] - b[n])) < ANGLE_TOL for n in a)
