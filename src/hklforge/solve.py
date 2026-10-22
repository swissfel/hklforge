"""The planner: from a geometry and a mode to every exact solution.

A *mode* is a dict naming what is held: circles with their values (degrees)
and virtual-angle constraints, as diffcalc does it::

    {"eta": 0, "chi": 0, "phi": 0}           # 3 sample circles
    {"chi": 10, "phi": 20, "betain": 5}       # 2 sample circles + incidence
    {"mu": 0, "nu": 0, "psi": 30}             # azimuth about Q
    {"nu": 0, "bisect": True, "omega": 0}     # bisecting
    {"mu": 0, "nu": 0, "h2": (2, 2, 0)}       # second reflection in Bragg too

``plan(geometry, mode)`` checks the mode once and picks the chain of exact
sub-problems (``Plan.describe()`` says which); ``Plan.run`` executes it for
one hkl. ``Diffractometer`` ties a geometry to a UB, wavelength and the
surface / reference vectors and is the convenient entry point. The virtual
angles themselves are defined in :mod:`hklforge.virtual`.

Notation: b = beam, k = |k|, q = UB h (phi frame), R = sample rotation,
k' = outgoing beam, Q_lab = R q = k (k' - b), x = R^T b (the beam seen from
the sample), (r, u) = azimuth frame about the beam.

Every constraint falls in one of four groups, by what it fixes:

**x constraints** fix the beam direction seen from the sample. Bragg is
``b . R q = -|q|^2 / (2k)``, i.e. x . q-hat = -sin(theta); the others are

- betain:  b . R s = -sin(betain)    (s = unit surface normal)
- betaout: k'. R s = sin(betaout), and k' = b + R q / k, so
           b . R s = sin(betaout) - q.s / k
- bin_eq_bout: b . R s = -q.s / (2k)
- alpha, beta, a_eq_b: the same with the reference vector n
- h2 in Bragg: b . R q2 = -|q2|^2 / (2k)
- psi: x lies in the plane through q at azimuth psi from n about q, so
  x = -sin(theta) q-hat + cos(theta) rot(q-hat, -psi) n_perp-hat  (one point)

x has two degrees of freedom and Bragg takes one, so at most one of these;
with it x is one or two points (``cone_points``).

**qaz** fixes k' itself (its azimuth about the beam, 2theta being given).

**naz** fixes the azimuth of R n about the beam: m . R n = 0 with
m = -sin(naz) r + cos(naz) u, plus R n on the naz side of the beam.

**bisect / omega** constrain the lab axis e of the bisect circle (psic: chi),
which only the circles outside it move: bisect is e . (b x Q-hat) = 0 (e in
the scattering plane); omega is the direction of e in that plane, from p
(the beam's direction across Q) towards Q: e . (cos(omega) Q-hat -
sin(omega) p) = 0 on the side e . (cos(omega) p + sin(omega) Q-hat) > 0.
Both together fix e = cos(omega) p + sin(omega) Q-hat.

Cases (nS / nD = free sample / detector circles; "detector first" means the
detector is solved before the sample: nD = 1, or nD = 2 with qaz):

=================  ======================================================  =====
case               steps                                                   sols
=================  ======================================================  =====
beam               nD=2, nS=1: Bragg projection -> k' -> sp2 detector       <= 4
beam_x             nD=2, nS=2 + x: x points -> sp2 R x = b -> sp2 det       <= 8
beam_naz           nD=2, nS=2 + naz: Bragg & naz projection pair (degree 8)  <= 16
                   -> sp2 detector
beam_x_naz         nD=2, nS=3 + x + naz: x -> R = rot(b, t) R0, naz fixes   <= 8
                   t -> sp3 -> sp2 detector
q                  detector first, nS=2: sp2 R q = Q_lab                    <= 4
q_x                detector first, nS=3 + x: R from (q, x) -> (Q_lab, b)    <= 8
                   -> sp3
q_naz              detector first, nS=3 + naz: R = rot(Q-hat, t) R0, naz    <= 8
                   fixes t -> sp3
axis               detector first, bisect and/or omega: outer circles set   <= 16
                   e (projection or cone_points + sp2), inner two: sp2
between            h2, detector on the bisector of k'1 and k'2: as beam_x   <= 8
=================  ======================================================  =====
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Geometry, wavevector, HC_EV_ANGSTROM
from .rotation import (Degenerate, cone_points, rot, rotation_from_pairs, solve_projection,
                       solve_projection_pair, sp2, sp3, unit)
from .virtual import bisect_axis, omega_of, psi_of, virtual_angles

SURFACE_CONSTRAINTS = ("betain", "betaout", "bin_eq_bout")
REFERENCE_CONSTRAINTS = ("alpha", "beta", "a_eq_b", "psi")
X_CONSTRAINTS = SURFACE_CONSTRAINTS + REFERENCE_CONSTRAINTS + ("h2",)
AXIS_CONSTRAINTS = ("bisect", "omega")
CONSTRAINTS = X_CONSTRAINTS + ("naz", "qaz") + AXIS_CONSTRAINTS
ALIASES = {"vomega": "omega"}   # use 'vomega' when the geometry has a circle named omega
DETECTOR_POLICIES = ("h1", "between")

ANGLE_TOL = 1e-6        # degrees, for merging duplicate branches
RESIDUAL_TOL = 1e-8     # relative / radians, forward-model re-check of every solution


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
    case: str = field(init=False)

    def __post_init__(self):
        g = self.geometry
        self.free_sample = [n for n in g.sample_names if n not in self.fixed]
        self.free_detector = [n for n in g.detector_names if n not in self.fixed]
        self.case = self._validate()

    # --- checking the mode --------------------------------------------------

    @property
    def x_constraint(self):
        return next((n for n in self.extras if n in X_CONSTRAINTS), None)

    @property
    def axis_constraints(self):
        return [n for n in AXIS_CONSTRAINTS if n in self.extras]

    def _validate(self) -> str:
        g = self.geometry
        nS, nD = len(self.free_sample), len(self.free_detector)
        xs = [n for n in self.extras if n in X_CONSTRAINTS]
        naz, qaz = "naz" in self.extras, "qaz" in self.extras
        axis = self.axis_constraints
        if self.detector not in DETECTOR_POLICIES:
            raise ModeError(f"detector policy {self.detector!r}: use one of {DETECTOR_POLICIES}")
        if self.detector == "between":
            if set(self.extras) != {"h2"}:
                raise ModeError("detector 'between' needs h2 and no other constraint")
            if (nS, nD) != (2, 2):
                raise ModeError("detector 'between' needs 2 free sample and 2 free detector circles, "
                                f"got {nS} and {nD}")
            return "between"
        if len(xs) > 1:
            raise ModeError(f"{xs}: each of these fixes the beam direction seen from the sample, "
                            "which Bragg already constrains; at most one can be used")
        if nD > 2:
            raise ModeError(f"{nD} free detector circles; at most 2 are supported")
        if nD == 0:
            raise ModeError("all detector circles held: that fixes |Q|, so a general hkl is unreachable")
        if qaz and nD == 1:
            raise ModeError("qaz fixes the outgoing beam; with a detector circle held as well "
                            "that is overdetermined")
        det_first = nD == 1 or qaz
        if axis:
            if not det_first:
                raise ModeError(f"{axis} need the scattering plane first: hold one detector circle "
                                "or give qaz")
            if xs or naz:
                raise ModeError(f"{axis} together with {xs + ['naz'] * naz} is not supported")
            if g.bisect_circle is None:
                raise ModeError(f"geometry {g.name} declares no bisect_circle for {axis}")
            j = g.sample_names.index(g.bisect_circle)
            outer = [n for n in g.sample_names[:j] if n in self.free_sample]
            inner = [n for n in g.sample_names[j:] if n in self.free_sample]
            if len(outer) != len(axis) or len(inner) != 2:
                raise ModeError(
                    f"{axis} act on the {g.bisect_circle} axis, which only the circles outside it "
                    f"({', '.join(g.sample_names[:j]) or 'none'}) move: need {len(axis)} of those "
                    f"free (got {outer}) and 2 free from {g.sample_names[j:]} (got {inner})")
            return "axis"
        extra = len(xs) + naz
        need = (2 if det_first else 1) + extra
        if nS != need:
            what = "underdetermined (a continuum of positions)" if nS > need else "overdetermined"
            raise ModeError(f"{nS} free sample circles {self.free_sample} with "
                            f"{'the detector solved first' if det_first else f'{nD} free detector circles'}"
                            f" and {extra} constraint(s) on the sample is {what}; need {need}")
        if det_first:
            if extra > 1:
                raise ModeError(f"{xs + ['naz']}: with the detector solved first, {xs[0]} already "
                                "fixes the sample rotation; naz would overdetermine it")
            return "q_x" if xs else "q_naz" if naz else "q"
        return {(1, 0, 0): "beam", (2, 1, 0): "beam_x", (2, 0, 1): "beam_naz",
                (3, 1, 1): "beam_x_naz"}[(nS, len(xs), int(naz))]

    def describe(self) -> str:
        S, Dn = ", ".join(self.free_sample), ", ".join(self.free_detector)
        x = self.x_constraint
        g = self.geometry
        det = (f"qaz and 2theta fix k'; detector ({Dn}): sp2 (<= 2)" if "qaz" in self.extras else
               f"Bragg on the detector ({Dn}): b.k' = cos 2theta, projection (<= 2)")
        xstep = (f"beam in the phi frame x = R^T b from Bragg + {x}: "
                 + ("one point" if x == "psi" else "cone_points (<= 2)"))
        detector_last = f"k' = b + R q / k; detector ({Dn}): sp2 (<= 2)"
        steps = {
            "beam": [f"Bragg b.R q = -|q|^2/2k: one projection equation in {S} (<= 2)", detector_last],
            "beam_x": [xstep, f"sample ({S}) from R x = b: sp2 (<= 2)", detector_last],
            "beam_naz": [f"Bragg and naz: two projection equations in ({S}), degree-8 "
                         "polynomial (<= 8)", detector_last],
            "beam_x_naz": [xstep, "R = rot(b, t) R0 with R0 x = b; naz fixes t: projection (<= 2)",
                           f"sample ({S}) from R: sp3 (<= 2)", detector_last],
            "q": [det, f"Q_lab = k (k' - b); sample ({S}) from R q = Q_lab: sp2 (<= 2)"],
            "q_x": [det, xstep, f"R from (q, x) -> (Q_lab, b); sample ({S}): sp3 (<= 2)"],
            "q_naz": [det, "R = rot(Q-hat, t) R0 with R0 q = Q_lab; naz fixes t: projection (<= 2)",
                      f"sample ({S}) from R: sp3 (<= 2)"],
            "axis": [det, f"{g.bisect_circle} axis e from {' + '.join(self.axis_constraints)}: "
                     + ("cone_points (<= 2) then sp2 (<= 2)" if len(self.axis_constraints) == 2
                        else "projection (<= 2)"),
                     f"inner circles from R_inner q = R_outer^T Q_lab: sp2 (<= 2)"],
            "between": ["x = R^T b from Bragg for h1 and h2: cone_points (<= 2)",
                        f"sample ({S}) from R x = b: sp2 (<= 2)",
                        f"detector ({Dn}) on the bisector of k'1 and k'2: sp2 (<= 2)"],
        }[self.case]
        held = ", ".join([f"{n}={v}" for n, v in self.fixed.items()]
                         + [f"{n}={v}" for n, v in self.extras.items()])
        lines = [f"{g.name}: {held}, detector on {self.detector}  [{self.case}]"]
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
        for name in self.extras:
            if name in SURFACE_CONSTRAINTS and surface is None:
                raise ModeError(f"{name} needs a surface normal")
            if name in REFERENCE_CONSTRAINTS + ("naz",) and reference is None:
                raise ModeError(f"{name} needs a reference vector")
        try:
            return self._run(np.asarray(ub, float), k, np.asarray(hkl, float),
                             None if surface is None else unit(surface),
                             None if reference is None else unit(reference))
        except Degenerate as e:
            held = ", ".join(f"{n}={v:g}" for n, v in self.fixed.items())
            raise Degenerate(f"{e} (free: {', '.join(self.free_sample + self.free_detector)}; "
                             f"held: {held})") from None

    def _run(self, ub, k, hkl, s_vec, n_vec):
        g = self.geometry
        b = np.asarray(g.beam)
        q = ub @ hkl
        nq = np.linalg.norm(q)
        if nq < 1e-12 or nq > 2 * k:
            return []
        ctx = _Context(g, self, ub, k, q, s_vec, n_vec)
        S_axes, F = _chain(g.sample, self.fixed)
        D_axes, G = _chain(g.detector, self.fixed)
        Gb = G @ b

        sols = []                                    # (sample radians, detector radians)
        if self.case.startswith("q") or self.case == "axis":
            for td, kout in ctx.detector_first(D_axes, Gb):
                Q_lab = k * (kout - b)
                sols += [(ts, td) for ts in ctx.sample_from_q(S_axes, F, Q_lab)]
        else:
            for ts, R in ctx.sample_from_beam(S_axes, F):
                if self.case == "between":
                    kout = unit(2 * b + R @ (q + ctx.q2) / k)
                else:
                    kout = b + R @ q / k
                sols += [(ts, td) for td in sp2(*D_axes, Gb, kout)]

        out = []
        for ts, td in sols:
            a = dict(self.fixed)
            a.update({n: _wrap_deg(np.degrees(t)) for n, t in zip(self.free_sample, ts)})
            a.update({n: _wrap_deg(np.degrees(t)) for n, t in zip(self.free_detector, td)})
            a = {n: float(a[n]) for n in g.names}
            if ctx.residual(a) < RESIDUAL_TOL and not any(_same(a, o) for o in out):
                out.append(a)
        return out


class _Context:
    """One run of a plan: the vectors and the step implementations."""

    def __init__(self, g, plan, ub, k, q, s_vec, n_vec):
        self.g, self.plan, self.ub, self.k, self.q = g, plan, ub, k, q
        self.s, self.n = s_vec, n_vec
        self.b = np.asarray(g.beam)
        self.nq = np.linalg.norm(q)
        self.sin_th = self.nq / (2 * k)
        self.cos_th = np.sqrt(max(0.0, 1 - self.sin_th ** 2))
        self.cos2th = 1 - 2 * self.sin_th ** 2
        ex = plan.extras
        self.q2 = ub @ np.asarray(ex["h2"], float) if "h2" in ex else None
        if "naz" in ex:
            r, u = g.azimuth_frame()
            a = np.radians(ex["naz"])
            self.naz_dir = np.cos(a) * r + np.sin(a) * u
            self.naz_normal = -np.sin(a) * r + np.cos(a) * u

    # --- x = R^T b ------------------------------------------------------------

    def x_points(self) -> list[np.ndarray]:
        """The beam directions in the phi frame allowed by Bragg + the x constraint."""
        name = self.plan.x_constraint
        val = self.plan.extras[name]
        qh, k, q = self.q / self.nq, self.k, self.q
        if name == "psi":
            n_perp = self.n - qh * (qh @ self.n)
            if np.linalg.norm(n_perp) < 1e-12:
                raise Degenerate("the reference vector is parallel to Q, so psi is undefined")
            p = rot(qh, -np.radians(val)) @ unit(n_perp)
            return [-self.sin_th * qh + self.cos_th * p]
        if name == "h2":
            v, c = self.q2, -(self.q2 @ self.q2) / (2 * k)
        else:
            v = self.s if name in SURFACE_CONSTRAINTS else self.n
            if name in ("betain", "alpha"):
                c = -np.sin(np.radians(val))
            elif name in ("betaout", "beta"):
                c = np.sin(np.radians(val)) - (q @ v) / k
            else:                                       # in = out
                c = -(q @ v) / (2 * k)
        nv = np.linalg.norm(v)
        if np.linalg.norm(np.cross(qh, v)) < 1e-12 * nv:
            s = np.sign(qh @ v)                         # same line as q: contradictory or redundant
            if abs(-self.sin_th - s * c / nv) > 1e-12:
                return []
            raise Degenerate(f"{name} acts on a vector parallel to Q, leaving a continuum")
        return cone_points(qh, -self.sin_th, v / nv, c / nv, 1.0)

    def naz_ok(self, R) -> bool:
        return (R @ self.n) @ self.naz_dir > 0

    # --- sample first (detector after) --------------------------------------

    def sample_from_beam(self, axes, F):
        b, qh = self.b, self.q / self.nq
        case = self.plan.case
        if case == "beam":
            for t in solve_projection(b, axes[0], F @ qh, -self.sin_th):
                yield (t,), rot(axes[0], t) @ F
        elif case in ("beam_x", "between"):
            for x in self.x_points():
                for t1, t2 in sp2(axes[0], axes[1], F @ x, b):
                    yield (t1, t2), rot(axes[0], t1) @ rot(axes[1], t2) @ F
        elif case == "beam_naz":
            eqs = [(b, F @ qh, -self.sin_th), (self.naz_normal, F @ self.n, 0.0)]
            for t1, t2 in solve_projection_pair(eqs, axes[0], axes[1]):
                R = rot(axes[0], t1) @ rot(axes[1], t2) @ F
                if self.naz_ok(R):
                    yield (t1, t2), R
        elif case == "beam_x_naz":
            for x in self.x_points():
                R0 = _rotation_onto(x, b)
                for t in solve_projection(self.naz_normal, b, R0 @ self.n, 0.0):
                    R = rot(b, t) @ R0
                    if self.naz_ok(R):
                        for ts in sp3(*axes, R @ F.T):
                            yield ts, R

    # --- detector first ------------------------------------------------------

    def detector_first(self, axes, Gb):
        """(detector radians, unit k') for every detector setting."""
        b = self.b
        if "qaz" in self.plan.extras:
            r, u = self.g.azimuth_frame()
            a = np.radians(self.plan.extras["qaz"])
            sin2th = 2 * self.sin_th * self.cos_th
            kout = self.cos2th * b + sin2th * (np.cos(a) * r + np.sin(a) * u)
            return [(td, kout) for td in sp2(*axes, Gb, kout)]
        (d1,) = axes
        return [((s,), rot(d1, s) @ Gb) for s in solve_projection(b, d1, Gb, self.cos2th)]

    def sample_from_q(self, axes, F, Q_lab):
        q, b, case = self.q, self.b, self.plan.case
        if case == "q":
            return sp2(axes[0], axes[1], F @ q, Q_lab)
        out = []
        if case == "q_x":
            for x in self.x_points():
                if np.linalg.norm(np.cross(q, x)) < 1e-12 * self.nq:
                    raise Degenerate("Q is along the beam (2theta = 180): its azimuth is free")
                out += sp3(*axes, rotation_from_pairs(q, x, Q_lab, b) @ F.T)
        elif case == "q_naz":
            Qh = unit(Q_lab)
            R0 = _rotation_onto(q, Q_lab)
            for t in solve_projection(self.naz_normal, Qh, R0 @ self.n, 0.0):
                R = rot(Qh, t) @ R0
                if self.naz_ok(R):
                    out += sp3(*axes, R @ F.T)
        elif case == "axis":
            out = self._axis(Q_lab)
        return out

    def _axis(self, Q_lab):
        g, ex, b = self.g, self.plan.extras, self.b
        j = g.sample_names.index(g.bisect_circle)
        outer_axes, F_out = _chain(g.sample[:j], self.plan.fixed)
        inner_axes, F_in = _chain(g.sample[j:], self.plan.fixed)
        a_c = F_out @ np.asarray(g.sample[j].axis)      # e = rot(outer free) a_c
        Qh = unit(Q_lab)
        if np.linalg.norm(np.cross(b, Qh)) < 1e-12:
            raise Degenerate("Q is along the beam: no scattering plane")
        p = unit(b - Qh * (b @ Qh))
        outers = []                                       # (outer radians, R_outer)
        if len(outer_axes) == 2:                          # bisect and omega: e is one vector
            om = np.radians(ex["omega"])
            e = np.cos(om) * p + np.sin(om) * Qh
            for t1, t2 in sp2(outer_axes[0], outer_axes[1], a_c, e):
                outers.append(((t1, t2), rot(outer_axes[0], t1) @ rot(outer_axes[1], t2) @ F_out))
        else:
            if "bisect" in ex:
                w, side = unit(np.cross(b, Qh)), None
            else:
                om = np.radians(ex["omega"])
                w, side = np.cos(om) * Qh - np.sin(om) * p, np.cos(om) * p + np.sin(om) * Qh
            for t in solve_projection(w, outer_axes[0], a_c, 0.0):
                R_out = rot(outer_axes[0], t) @ F_out
                if side is None or (R_out @ np.asarray(g.sample[j].axis)) @ side > 0:
                    outers.append(((t,), R_out))
        out = []
        for to, R_out in outers:
            for ti in sp2(inner_axes[0], inner_axes[1], F_in @ self.q, R_out.T @ Q_lab):
                out.append(tuple(to) + tuple(ti))
        return out

    # --- the re-check ----------------------------------------------------------

    def residual(self, a) -> float:
        """Largest error over all equations, through the forward model."""
        g, b, k, q = self.g, self.b, self.k, self.q
        R = g.sample_matrix(a)
        kout = g.k_out(a)
        if self.plan.case == "between":
            res = max(abs(np.linalg.norm(k * b + R @ v) - k) / k for v in (q, self.q2))
            return max(res, np.linalg.norm(kout - unit(2 * b + R @ (q + self.q2) / k)))
        res = np.linalg.norm(g.q_phi(a, k) - q) / self.nq
        Q = R @ q
        for name, val in self.plan.extras.items():
            if name in SURFACE_CONSTRAINTS or name in ("alpha", "beta", "a_eq_b"):
                v = R @ (self.s if name in SURFACE_CONSTRAINTS else self.n)
                sin_in, sin_out = -b @ v, kout @ v
                if name in ("betain", "alpha"):
                    err = sin_in - np.sin(np.radians(val))
                elif name in ("betaout", "beta"):
                    err = sin_out - np.sin(np.radians(val))
                else:
                    err = sin_in - sin_out
            elif name == "h2":
                err = (np.linalg.norm(k * b + R @ self.q2) - k) / k
            elif name == "psi":
                err = np.radians(_wrap_deg(psi_of(R @ self.n, b, Q) - val))
            elif name == "naz":
                va = virtual_angles(g, a, reference=self.n)["naz"]
                err = np.radians(_wrap_deg(va - val))
            elif name == "qaz":
                err = np.radians(_wrap_deg(virtual_angles(g, a)["qaz"] - val))
            elif name == "bisect":
                err = bisect_axis(g, a) @ unit(np.cross(b, unit(Q)))
            elif name == "omega":
                err = np.radians(_wrap_deg(omega_of(bisect_axis(g, a), b, Q) - val))
            res = max(res, abs(err))
        return res


def plan(geometry: Geometry, mode: dict, detector: str = "h1") -> Plan:
    """Check a mode against a geometry and pick its chain of sub-problems."""
    fixed, extras = {}, {}
    for name, val in mode.items():
        if name in geometry.names:
            fixed[name] = float(val)
        elif ALIASES.get(name, name) in CONSTRAINTS:
            name = ALIASES.get(name, name)
            if name in ("bin_eq_bout", "a_eq_b", "bisect"):
                if not val:
                    continue
                val = True
            elif name != "h2":
                val = float(val)
            extras[name] = val
        else:
            raise ModeError(f"{name!r} is neither a circle of {geometry.name} {geometry.names} "
                            f"nor a constraint {CONSTRAINTS}")
    return Plan(geometry, fixed, extras, detector)


class Diffractometer:
    """A geometry with a UB matrix, a wavelength and optional normal vectors.

    ``ub`` is in the phi frame with the 2 pi convention (q = UB h). The
    surface normal (for betain/betaout) and reference vector (alpha, beta,
    psi, naz, tau) are phi-frame vectors; ``*_hkl`` sets them from
    reciprocal-lattice indices instead, n = UB h.
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
        """All virtual angles of a position (degrees), see :mod:`hklforge.virtual`."""
        return virtual_angles(self.geometry, angles, self.surface, self.reference)

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


def _perp(v):
    v = unit(v)
    return unit(np.cross(v, [1.0, 0, 0] if abs(v[0]) < 0.9 else [0, 1.0, 0]))


def _rotation_onto(src, dst) -> np.ndarray:
    """Some rotation taking direction src onto direction dst."""
    return rotation_from_pairs(src, _perp(src), dst, _perp(dst))


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
