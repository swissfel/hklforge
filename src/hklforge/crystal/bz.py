"""High-symmetry points of the first Brillouin zone and the standard paths.

Points and paths are Setyawan & Curtarolo's (Comput. Mater. Sci. 49, 299
(2010)), but expressed as (h, k, l) in the reciprocal basis of the user's
**conventional** cell, the basis UB uses (q = UB h, 2 pi convention), so a
viewer can draw them at node + hkl.

S&C give each lattice a primitive cell P = M C (rows: primitive and
conventional real vectors) and the points as fractions f of the primitive
reciprocal vectors. With reciprocal vectors as rows, B_p = M^-T B_c, so the
conventional indices are ``hkl = f M^-T``.

Supported: cP (CUB), cF (FCC), cI (BCC), tP (TET), tI (BCT1/BCT2), oP (ORC),
hP (HEX), hR (RHL1/RHL2, in hexagonal or rhombohedral axes). For oP, S&C
sort a < b < c; here X, Y, Z stay along a*, b*, c* of the cell as given. The
centred orthorhombic, monoclinic and triclinic lattices, whose points depend
on the cell's ratios in ways that have no single unambiguous table, return
Γ only with a note.

Every point is checked numerically before it is returned: it must lie on the
boundary of the first Brillouin zone of the lattice (as far from the origin
as from its nearest other node, and no node closer), and be equidistant from
at least as many nodes as the same point of a generic cell of its variant
(a corner stays a corner). A point that fails (a wrong table, or an
unexpected setting such as hexagonal axes with gamma = 60) is dropped, the
path split there, and the note says so.

S&C's Table 6 (BCT1) has M = (-1/2, 1/2, 1/2) and no H; a version with
H = (-1/2, 1/2, 1/2), M = (1/2, 1/2, -1/2) circulates, and that M is outside
the zone for c < a.

numpy only.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..lattice import b_matrix

LATTICES = ("cP", "cF", "cI", "tP", "tI", "oP", "oS", "oF", "oI", "hP", "hR", "mP", "mS", "aP")
UNSUPPORTED = ("oS", "oF", "oI", "mP", "mS", "aP")
_TOL = 1e-4          # metric inference: equal lengths / angles
_BOUNDARY_TOL = 1e-6

_h, _t = 0.5, 1.0 / 3.0
_M = {
    "P": np.eye(3),
    "F": np.array([[0, _h, _h], [_h, 0, _h], [_h, _h, 0]]),
    "I": np.array([[-_h, _h, _h], [_h, -_h, _h], [_h, _h, -_h]]),
    "R": np.array([[2 * _t, _t, _t], [-_t, _t, _t], [-_t, -2 * _t, _t]]),   # obverse, hexagonal axes
}


def _cell(cell: dict) -> tuple:
    return tuple(float(cell[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma"))


def _eq(x, y, tol=_TOL):
    return abs(x - y) <= tol * max(1.0, abs(y))


def _sg(space_group):
    if space_group is None:
        return None
    from .spacegroups import SpaceGroup
    if isinstance(space_group, SpaceGroup):
        return space_group
    try:
        return SpaceGroup(space_group)
    except ValueError:
        if isinstance(space_group, str) and "_" in space_group:    # "P6_3/mmc" -> "P63/mmc"
            return SpaceGroup(space_group.replace("_", ""))
        raise


def _rhombohedral_metric(cell) -> bool:
    a, b, c, al, be, ga = _cell(cell)
    return _eq(a, b) and _eq(a, c) and _eq(al, be) and _eq(al, ga) and not _eq(al, 90)


def bravais_lattice(cell: dict, space_group=None) -> str:
    """The Bravais lattice symbol ('cP', 'cF', ..., 'hR', 'mS', 'aP').

    With ``space_group`` (a name, number or ``SpaceGroup``): its crystal
    system and centring letter (A/B/C -> S, R -> hR also in rhombohedral
    axes; trigonal groups without R are hP). Without one, only from the cell
    metric (lengths/angles equal within 1e-4), and the centring is unknown,
    so P (or hR for a rhombohedral metric a = b = c, alpha = beta = gamma != 90).
    """
    sg = _sg(space_group)
    if sg is not None:
        letter = sg.centring
        system = sg.system
        if system == "trigonal":
            return "hR" if letter == "R" else "hP"
        if system == "hexagonal":
            return "hP"
        if letter in "ABC":
            letter = "S"
        if system == "monoclinic" and letter != "P":
            letter = "S"         # C, A, I settings of the same lattice
        return {"cubic": "c", "tetragonal": "t", "orthorhombic": "o", "monoclinic": "m",
                "triclinic": "a"}[system] + letter
    a, b, c, al, be, ga = _cell(cell)
    right = [_eq(x, 90) for x in (al, be, ga)]
    if all(right):
        if _eq(a, b) and _eq(a, c):
            return "cP"
        if _eq(a, b):
            return "tP"
        return "oP"
    if right[0] and right[1] and _eq(a, b) and (_eq(ga, 120) or _eq(ga, 60)):
        return "hP"
    if _rhombohedral_metric(cell):
        return "hR"
    if sum(right) == 2:
        return "mP"
    return "aP"


# --- the tables (fractions of the primitive reciprocal vectors) ----------------------

def _table(variant: str, **p):
    """(points, path) of S&C for one variant; ``p`` holds eta, nu, zeta."""
    h, q = 0.5, 0.25
    if variant == "CUB":
        pts = {"X": (0, h, 0), "M": (h, h, 0), "R": (h, h, h)}
        path = [["Γ", "X", "M", "Γ", "R", "X"], ["M", "R"]]
    elif variant == "FCC":
        pts = {"K": (3 / 8, 3 / 8, 3 / 4), "L": (h, h, h), "U": (5 / 8, q, 5 / 8),
               "W": (h, q, 3 / 4), "X": (h, 0, h)}
        path = [["Γ", "X", "W", "K", "Γ", "L", "U", "W", "L", "K"], ["U", "X"]]
    elif variant == "BCC":
        pts = {"H": (h, -h, h), "N": (0, 0, h), "P": (q, q, q)}
        path = [["Γ", "H", "N", "Γ", "P", "H"], ["P", "N"]]
    elif variant == "TET":
        pts = {"A": (h, h, h), "M": (h, h, 0), "R": (0, h, h), "X": (0, h, 0), "Z": (0, 0, h)}
        path = [["Γ", "X", "M", "Γ", "Z", "R", "A", "Z"], ["X", "R"], ["M", "A"]]
    elif variant == "BCT1":
        e = p["eta"]
        # S&C Table 6 (arXiv:1004.2974): M = (-1/2, 1/2, 1/2), no H point
        pts = {"M": (-h, h, h), "N": (0, h, 0), "P": (q, q, q),
               "X": (0, 0, h), "Z": (e, e, -e), "Z1": (-e, 1 - e, e)}
        path = [["Γ", "X", "M", "Γ", "Z", "P", "N", "Z1", "M"], ["X", "P"]]
    elif variant == "BCT2":
        e, z = p["eta"], p["zeta"]
        pts = {"N": (0, h, 0), "P": (q, q, q), "Σ": (-e, e, e), "Σ1": (e, 1 - e, -e),
               "X": (0, 0, h), "Y": (-z, z, h), "Y1": (h, h, -z), "Z": (h, h, -h)}
        path = [["Γ", "X", "Y", "Σ", "Γ", "Z", "Σ1", "N", "P", "Y1", "Z"], ["X", "P"]]
    elif variant == "ORC":
        pts = {"R": (h, h, h), "S": (h, h, 0), "T": (0, h, h), "U": (h, 0, h),
               "X": (h, 0, 0), "Y": (0, h, 0), "Z": (0, 0, h)}
        path = [["Γ", "X", "S", "Y", "Γ", "Z", "U", "R", "T", "Z"], ["Y", "T"], ["U", "X"], ["S", "R"]]
    elif variant == "HEX":
        pts = {"A": (0, 0, h), "H": (1 / 3, 1 / 3, h), "K": (1 / 3, 1 / 3, 0), "L": (h, 0, h),
               "M": (h, 0, 0)}
        path = [["Γ", "M", "K", "Γ", "A", "L", "H", "A"], ["L", "M"], ["K", "H"]]
    elif variant == "RHL1":
        e, n = p["eta"], p["nu"]
        pts = {"B": (e, h, 1 - e), "B1": (h, 1 - e, e - 1), "F": (h, h, 0), "L": (h, 0, 0),
               "L1": (0, 0, -h), "P": (e, n, n), "P1": (1 - n, 1 - n, 1 - e), "P2": (n, n, e - 1),
               "Q": (1 - n, n, 0), "X": (n, 0, -n), "Z": (h, h, h)}
        path = [["Γ", "L", "B1"], ["B", "Z", "Γ", "X"], ["Q", "F", "P1", "Z"], ["L", "P"]]
    elif variant == "RHL2":
        e, n = p["eta"], p["nu"]
        pts = {"F": (h, -h, 0), "L": (h, 0, 0), "P": (1 - n, -n, 1 - n), "P1": (n, n - 1, n - 1),
               "Q": (e, e, e), "Q1": (1 - e, -e, -e), "Z": (h, -h, h)}
        path = [["Γ", "P", "Z", "Q", "Γ", "F", "P1", "Q1", "L", "Z"]]
    else:  # pragma: no cover
        raise ValueError(variant)
    return pts, path


def _variant(lattice: str, cell: dict, rh_axes: bool):
    """(variant, M, node centring, extra note) for a supported lattice."""
    a, b, c, al, be, ga = _cell(cell)
    if lattice == "cP":
        return "CUB", _M["P"], "P", {}, ""
    if lattice == "cF":
        return "FCC", _M["F"], "F", {}, ""
    if lattice == "cI":
        return "BCC", _M["I"], "I", {}, ""
    if lattice == "tP":
        return "TET", _M["P"], "P", {}, ""
    if lattice == "tI":
        if abs(c - a) <= 1e-6 * a:
            return "BCC", _M["I"], "I", {}, "tI with c = a is body-centred cubic: BCC points used."
        if c < a:
            return "BCT1", _M["I"], "I", {"eta": (1 + c * c / (a * a)) / 4}, ""
        return "BCT2", _M["I"], "I", {"eta": (1 + a * a / (c * c)) / 4, "zeta": a * a / (2 * c * c)}, ""
    if lattice == "oP":
        return "ORC", _M["P"], "P", {}, ""
    if lattice == "hP":
        return "HEX", _M["P"], "P", {}, ""
    if lattice == "hR":
        M, cen = (_M["P"], "P") if rh_axes else (_M["R"], "R")
        prim = M @ _real_rows(cell)
        n = np.linalg.norm(prim, axis=1)
        cos_al = float(prim[0] @ prim[1] / (n[0] * n[1]))
        alpha = np.degrees(np.arccos(np.clip(cos_al, -1, 1)))
        if alpha < 90:
            eta = (1 + 4 * cos_al) / (2 + 4 * cos_al)
            return "RHL1", M, cen, {"eta": eta, "nu": 0.75 - eta / 2}, ""
        eta = 1 / (2 * np.tan(np.radians(alpha) / 2) ** 2)
        return "RHL2", M, cen, {"eta": eta, "nu": 0.75 - eta / 2}, ""
    raise ValueError(lattice)  # pragma: no cover


def _real_rows(cell) -> np.ndarray:
    from .cell import real_basis
    return real_basis(*_cell(cell)).T


# --- the boundary check ------------------------------------------------------------------

_CENTRING = {
    "P": lambda h, k, l: np.ones_like(h, bool),
    "I": lambda h, k, l: (h + k + l) % 2 == 0,
    "F": lambda h, k, l: ((h % 2 == k % 2) & (k % 2 == l % 2)),
    "A": lambda h, k, l: (k + l) % 2 == 0,
    "B": lambda h, k, l: (h + l) % 2 == 0,
    "C": lambda h, k, l: (h + k) % 2 == 0,
    "R": lambda h, k, l: (-h + k + l) % 3 == 0,
}


def _nodes_near(cell: dict, centring: str, radius: float):
    """Cartesian reciprocal nodes G (2 pi) with |G| <= radius, the origin excluded,
    and their hkl."""
    B = b_matrix(*_cell(cell))
    lengths = np.linalg.norm(_real_rows(cell), axis=1)
    # h = G . a / 2 pi, so |h| <= |G| |a| / 2 pi
    lim = [int(np.ceil(radius * L / (2 * np.pi))) + 1 for L in lengths]
    H = np.array(list(itertools.product(*(range(-m, m + 1) for m in lim))))
    H = H[_CENTRING[centring](H[:, 0], H[:, 1], H[:, 2]) & np.any(H != 0, axis=1)]
    G = H @ B.T
    keep = np.linalg.norm(G, axis=1) <= radius * (1 + 1e-9)
    return G[keep], H[keep]


def boundary_nodes(cell: dict, hkl, centring: str = "P", tol: float = _BOUNDARY_TOL):
    """Check that ``hkl`` (conventional indices) lies on the first-BZ boundary of
    the lattice with that node centring (P, I, F, A, B, C or R obverse).

    Returns ``(on_boundary, n_equidistant)``: whether no node is closer than
    the origin and at least one other is as close (relative ``tol``), and how
    many nodes (the origin included) are at that distance.
    """
    q = b_matrix(*_cell(cell)) @ np.asarray(hkl, float)
    r = float(np.linalg.norm(q))
    if r == 0:
        return False, 1
    G, _ = _nodes_near(cell, centring, 2 * r * (1 + tol))
    d = np.linalg.norm(q - G, axis=1)
    closer = np.any(d < r * (1 - tol))
    same = int(np.sum(np.abs(d - r) <= tol * r))
    return (not closer) and same >= 1, same + 1


# --- main entry ---------------------------------------------------------------------------

def _clean(v) -> tuple:
    out = []
    for x in v:
        x = round(float(x), 12)
        out.append(0.0 if x == 0 else x)
    return tuple(out)


def _split_path(path, drop):
    out = []
    for seg in path:
        cur = []
        for lab in seg:
            if lab in drop:
                if len(cur) > 1:
                    out.append(cur)
                cur = []
            else:
                cur.append(lab)
        if len(cur) > 1:
            out.append(cur)
    return out


def special_points(cell: dict, space_group=None) -> dict:
    """High-symmetry points and the standard path of the first Brillouin zone.

    Returns ``{"lattice": "cF", "variant": "FCC", "points": {"Γ": (0, 0, 0),
    "X": (0., 1., 0.), ...}, "path": [["Γ", "X", ...], ["U", "X"]], "note": ""}``
    with each point as (h, k, l) in the reciprocal basis of the cell as given
    (Q = UB hkl). ``path`` is a list of segments, split where S&C write "|".
    Unsupported lattices give Γ only and say so in ``note``; points that fail
    the boundary check are dropped (and the note names them).
    """
    sg = _sg(space_group)
    lattice = bravais_lattice(cell, sg)
    gamma_only = {"Γ": (0.0, 0.0, 0.0)}
    if lattice in UNSUPPORTED:
        return {"lattice": lattice, "variant": "", "points": gamma_only, "path": [],
                "note": f"no unambiguous standard path implemented for {lattice}"}
    rh_axes = sg.rhombohedral_axes if sg is not None else _rhombohedral_metric(cell)
    variant, M, centring, par, note = _variant(lattice, cell, rh_axes)
    found, path = _evaluate(cell, variant, M, centring, par)
    expected = _reference_counts(variant)
    points = dict(gamma_only)
    dropped = []
    for lab, (hkl, ok, n) in found.items():
        if ok and n >= expected.get(lab, 2):
            points[lab] = _clean(hkl)
        else:
            dropped.append(lab)
    if dropped:
        path = _split_path(path, set(dropped))
        msg = (f"dropped {', '.join(dropped)}: not on the Brillouin-zone boundary of this "
               f"{lattice} cell, or not the corner/edge it should be (unexpected setting?)")
        note = f"{note} {msg}".strip()
    return {"lattice": lattice, "variant": variant, "points": points, "path": path, "note": note}


def _evaluate(cell, variant, M, centring, par):
    """{label: (hkl, on_boundary, n_equidistant)} and the path, unchecked."""
    frac, path = _table(variant, **par)
    Minv_T = np.linalg.inv(M).T
    out = {}
    for lab, f in frac.items():
        hkl = np.asarray(f, float) @ Minv_T
        ok, n = boundary_nodes(cell, hkl, centring)
        out[lab] = (hkl, ok, n)
    return out, path


# A generic cell of each variant (no accidental degeneracy). How many nodes a
# point is equidistant from (face centre 2, edge 3, corner >= 4) is the same
# for every cell of a variant - S&C split the variants exactly where the zone
# changes shape - and can only grow at special ratios. So a point that is on
# the boundary but meets fewer nodes than here is not the point it claims to
# be (e.g. H of a hexagonal cell given with gamma = 60: on the top face, but
# not at its corner).
def _rh(alpha):
    return dict(a=5.0, b=5.0, c=5.0, alpha=alpha, beta=alpha, gamma=alpha)


_REFERENCE = {
    "CUB": ("cP", dict(a=4.0, b=4.0, c=4.0, alpha=90, beta=90, gamma=90)),
    "FCC": ("cF", dict(a=4.0, b=4.0, c=4.0, alpha=90, beta=90, gamma=90)),
    "BCC": ("cI", dict(a=4.0, b=4.0, c=4.0, alpha=90, beta=90, gamma=90)),
    "TET": ("tP", dict(a=4.0, b=4.0, c=6.3, alpha=90, beta=90, gamma=90)),
    "BCT1": ("tI", dict(a=4.0, b=4.0, c=3.1, alpha=90, beta=90, gamma=90)),
    "BCT2": ("tI", dict(a=4.0, b=4.0, c=6.7, alpha=90, beta=90, gamma=90)),
    "ORC": ("oP", dict(a=3.1, b=4.3, c=5.7, alpha=90, beta=90, gamma=90)),
    "HEX": ("hP", dict(a=3.7, b=3.7, c=5.9, alpha=90, beta=90, gamma=120)),
    "RHL1": ("hR", _rh(71.3)),
    "RHL2": ("hR", _rh(107.9)),
}
_COUNTS: dict = {}


def _reference_counts(variant: str) -> dict:
    if variant not in _COUNTS:
        lattice, cell = _REFERENCE[variant]
        v, M, centring, par, _ = _variant(lattice, cell, rh_axes=True)
        assert v == variant
        found, _ = _evaluate(cell, v, M, centring, par)
        _COUNTS[variant] = {lab: n for lab, (_, ok, n) in found.items() if ok}
    return _COUNTS[variant]
