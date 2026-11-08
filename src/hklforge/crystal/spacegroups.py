"""Space groups from their symmetry operations: absences, equivalents, multiplicity.

All 564 settings of the 230 groups come from ``spacegroups.json``, written
from gemmi's table by ``dev/make_spacegroup_table.py`` (in the hklforge repository) (gemmi is only
needed to regenerate it). A group is found by the names gemmi accepts:
"R-3m", "R -3 m", "R-3m:R", "P21/c", "P 1 21/c 1", "Fd-3m:2", or a number.

Reflections are rows h; an operation x' = R x + t maps the structure factor
F(h R) = exp(-2 pi i h.t) F(h), so

- h is **systematically absent** when some operation has h R = h and h.t not
  an integer (a centring vector, or the intrinsic translation of a screw
  axis or glide plane along h);
- the **equivalent reflections** are h R over the point group's rotations,
  with Friedel's -h (the Laue class): same |F| without anomalous scattering.
"""
from __future__ import annotations

import json
import re
from fractions import Fraction
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np

TABLE = Path(__file__).with_name("spacegroups.json")

CENTRING_RULES = {
    "A": "A centring: k + l even",
    "B": "B centring: h + l even",
    "C": "C centring: h + k even",
    "I": "I centring: h + k + l even",
    "F": "F centring: h, k, l all even or all odd",
    "R": "R centring (hexagonal axes, obverse): -h + k + l = 3n",
}
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@lru_cache(maxsize=None)
def _table():
    return json.loads(TABLE.read_text())


def _normal(name) -> str:
    return "".join(str(name).split()).lower()


def parse_triplet(text: str):
    """'-y+1/4,x+1/4,z+1/4' (or a CIF symop like '1/2-x, y, -z') -> (R, t):
    integer 3x3 R and fractional t with x' = R x + t."""
    parts = text.replace(" ", "").lower().split(",")
    if len(parts) != 3:
        raise ValueError(f"not a symmetry operation: {text!r}")
    R, t = np.zeros((3, 3), int), np.zeros(3)
    for i, part in enumerate(parts):
        for sign, coef, var in re.findall(r"([+-]?)(\d+(?:\.\d+)?(?:/\d+)?)?\*?([xyz]?)", part):
            if not coef and not var:
                continue
            s = -1 if sign == "-" else 1
            if var:
                R[i, "xyz".index(var)] += s * (int(coef) if coef else 1)
            else:
                t[i] += s * float(Fraction(coef))
    return R, t


def _reduced(v) -> tuple:
    v = [int(round(x)) for x in v]
    g = 0
    for x in v:
        g = gcd(g, abs(x))
    v = [x // g for x in v] if g else v
    # first nonzero component positive
    for x in v:
        if x:
            return tuple(v) if x > 0 else tuple(-y for y in v)
    return tuple(v)


def _fmt_dir(v, brackets="[]") -> str:
    return brackets[0] + " ".join(str(x) for x in _reduced(v)) + brackets[1]


class SpaceGroup:
    """A space group setting by name or number, e.g. ``SpaceGroup("R-3m")``.

    ``name`` (extended Hermann-Mauguin, e.g. "R -3 m:H"), ``number``,
    ``centring`` (the lattice letter), ``system``, ``point_group``,
    ``laue``; ``ops`` the coset representatives (R, t), ``centring_vectors``.
    """

    full = True                      # all absences are known (not only centring)

    def __init__(self, symbol):
        if isinstance(symbol, SpaceGroup):
            symbol = symbol.name
        table = _table()
        key = _normal(symbol)
        i = table["aliases"].get(key)
        if i is None and key.endswith(":h"):
            i = table["aliases"].get(key[:-2])
        if i is None:
            raise ValueError(f"unknown space group {str(symbol).strip()!r}")
        g = table["groups"][i]
        self.name, self.number, self.hall = g["name"], g["number"], g["hall"]
        self.system, self.point_group, self.laue = g["system"], g["point_group"], g["laue"]
        self.centring = self.name.lstrip()[0].upper()
        self.ops = [parse_triplet(t) for t in g["ops"]]
        self.centring_vectors = [np.array([float(Fraction(x)) for x in v.split(",")])
                                 for v in g["centring_vectors"]]
        rots = {tuple(R.ravel()) for R, _ in self.ops}
        self._rot = np.array(sorted(rots)).reshape(-1, 3, 3)
        laue = rots | {tuple((-np.array(r)).ravel()) for r in rots}
        self._laue = np.array(sorted(laue)).reshape(-1, 3, 3)

    def __repr__(self):
        return f"SpaceGroup({self.name!r})"

    def __eq__(self, other):
        return isinstance(other, SpaceGroup) and other.name == self.name

    def __hash__(self):
        return hash(self.name)

    @property
    def rhombohedral_axes(self) -> bool:
        return self.name.endswith(":R")

    @property
    def base_symbol(self) -> str:
        return self.name.split(":")[0]

    def all_ops(self):
        """Every operation (R, t) of one cell: coset representatives times centring."""
        return [(R, t + c) for R, t in self.ops for c in self.centring_vectors]

    # --- absences ---

    def why_forbidden(self, hkl) -> str:
        """'' if (h k l) is allowed, else which symmetry removes it: the
        centring, or the screw axis or glide plane (with its direction)."""
        h = np.array([int(round(x)) for x in hkl])
        if not h.any():
            return ""
        for c in self.centring_vectors:
            if _not_integer(h @ c):
                rule = CENTRING_RULES.get(self.centring) if not self.rhombohedral_axes else None
                return f"forbidden in {self.name} by {rule or 'the lattice centring'}"
        reasons = []
        for R, t in self.all_ops():
            if np.array_equal(h @ R, h) and _not_integer(h @ t):
                text = _describe(R, t)
                if text not in reasons:
                    reasons.append(text)
        if not reasons:
            return ""
        # a screw axis and a glide plane often remove the same reflection: name both
        kinds = [next((r for r in reasons if "screw" in r), None), next((r for r in reasons if "glide" in r), None)]
        return f"forbidden in {self.name} by " + " and ".join(r for r in kinds if r)

    def is_allowed(self, hkl) -> bool:
        return not self.why_forbidden(hkl)

    def allowed_mask(self, H) -> np.ndarray:
        """``is_allowed`` for an array of reflections, shape (n, 3)."""
        H = np.asarray(H, float).round().astype(int).reshape(-1, 3)
        ok = np.ones(len(H), bool)
        for R, t in self.all_ops():
            fixed = np.all(H @ R == H, axis=1)
            ok &= ~(fixed & _not_integer(H @ t))
        return ok

    def condition_summary(self) -> str:
        rule = CENTRING_RULES.get(self.centring) if not self.rhombohedral_axes else None
        text = rule or "no centring condition"
        screws = sum(1 for R, t in self.ops if _intrinsic(R, t).any())
        return text + (f"; plus glide planes / screw axes ({screws} operations)" if screws else "")

    # --- equivalent reflections ---

    def equivalents(self, hkl, friedel: bool = True) -> list[tuple]:
        """The distinct reflections equivalent to (h k l): h R over the point
        group, and with ``friedel`` also -h (the Laue class)."""
        h = np.array([int(round(x)) for x in hkl])
        rots = self._laue if friedel else self._rot
        return sorted({tuple(int(x) for x in h @ R) for R in rots}, key=_preference, reverse=True)

    def multiplicity(self, hkl, friedel: bool = True) -> int:
        return len(self.equivalents(hkl, friedel))

    def representative(self, hkl, friedel: bool = True) -> tuple:
        """The equivalent shown for a whole set: most indices non-negative,
        then the largest h, k, l - e.g. (1 1 1) for {1 1 1}."""
        return self.equivalents(hkl, friedel)[0]

    def is_unique(self, hkl) -> bool:
        """True for the representative of its set of equivalents."""
        return tuple(int(round(x)) for x in hkl) == self.representative(hkl)


def _preference(h):
    return (sum(1 for x in h if x >= 0), h[0], h[1], h[2])


def _not_integer(x, tol=1e-6):
    x = np.asarray(x, float)
    return np.abs(x - np.round(x)) > tol


def _order(R) -> int:
    M = np.eye(3, dtype=int)
    for n in range(1, 7):
        M = M @ R
        if np.array_equal(M, np.eye(3, dtype=int)):
            return n
    return 1


def _intrinsic(R, t) -> np.ndarray:
    """The translation of (R, t)^n / n along the axis or in the plane (mod 1
    removed only if it is a lattice vector): nonzero for screws and glides."""
    n = _order(R)
    w, M = np.zeros(3), np.eye(3)
    for _ in range(n):
        w += M @ t
        M = R @ M
    w = w / n
    return np.where(np.abs(w - np.round(w)) < 1e-9, 0.0, w)


def _describe(R, t) -> str:
    """'the 2₁ screw axis along [0 1 0]', 'a c-glide plane ⊥ (0 1 0)'."""
    det = round(np.linalg.det(R))
    w = _intrinsic(R, t)
    if det > 0:
        n = _order(R)
        vals, vecs = np.linalg.eig(R.astype(float))
        axis = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
        axis = axis / np.max(np.abs(axis))
        u = np.array(_reduced(np.round(axis * 6)))           # lattice vector along the axis
        j = int(np.argmax(np.abs(u)))
        m = int(round(n * w[j] / u[j])) % n if u[j] else 0
        return f"the {n}{str(m).translate(_SUB)} screw axis along {_fmt_dir(u)}"
    # improper: a glide plane; its normal is the reciprocal direction with h R = -h
    vals, vecs = np.linalg.eig(R.T.astype(float))
    normal = np.real(vecs[:, np.argmin(np.abs(vals + 1))])
    normal = np.round(normal / np.max(np.abs(normal)) * 6)
    halves = [abs(x) for x in w if abs(x) > 1e-9]
    if any(abs(x - 0.25) < 1e-9 or abs(x - 0.75) < 1e-9 for x in halves):
        letter = "d"
    elif len(halves) == 1:
        letter = "abc"[int(np.argmax(np.abs(w)))]
    elif len(halves) >= 2:
        letter = "n"
    else:
        letter = "e"
    article = "an" if letter in "aen" else "a"
    return f"{article} {letter}-glide plane ⊥ {_fmt_dir(normal, '()')}"
