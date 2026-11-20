"""An interactive 3D view of any geometry: the circles, k, k', Q, the Ewald
sphere, the reciprocal lattice, Brillouin zones and the virtual angles.

The view is one self-contained page (three.js inlined, works offline):
``VIEWER_HTML``. It is built from a :class:`~hklforge.Geometry` - one ring
per circle, nested outermost first, about the circle's axis - so it shows
every geometry hklforge can solve. Without a geometry it shows psic (You's
six-circle).

How it is fed (no custom widget, no server):

- :func:`page` returns the page's HTML with a geometry (and optionally an
  id, a browser-storage prefix and detector sizes) built in. Standalone it
  is fully usable: sliders for every circle.
- Embedded in a notebook with ``<iframe srcdoc=...>`` (:func:`iframe`), the
  page has the notebook's origin and reads a hidden element the notebook
  writes into, ``<span id="{id}-payload">`` holding the JSON of
  :func:`message` (polled every 150 ms; a changing ``n`` marks a new one).
  :func:`payload_html` writes that element.

Vectors in a message are in the geometry's lab frame (UB and the surface
normal in the phi frame, as everywhere in hklforge); the page turns them
into its own frame. numpy only.
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path

import numpy as np

from ..geometry import Geometry

VIEWER_HTML = Path(__file__).with_name("viewer.html")
CONFIG_MARK = "<!--HKLFORGE_VIEW_CONFIG-->"


def _vec(v):
    return None if v is None else [float(x) + 0.0 for x in np.asarray(v, float)]


def geometry_message(geometry: Geometry) -> dict:
    """The geometry as the page needs it: ``Geometry.to_dict()`` with ``up``
    resolved (``up_direction``) and the azimuth frame (r, u) of qaz/naz."""
    d = geometry.to_dict()
    r, u = geometry.azimuth_frame()
    d["up"] = _vec(geometry.up_direction())
    d["azimuth"] = {"r": _vec(r), "u": _vec(u)}
    return d


def message(geometry: Geometry | None = None, angles: dict | None = None, *, ub=None,
            k: float | None = None, energy: float | None = None, surface=None, reference=None,
            label: str = "", targets=None, special_points: dict | None = None, n: int = 0,
            structure_factors: dict | None = None, structure: dict | None = None, **extra) -> dict:
    """The JSON the page reads (see the module docstring).

    ``geometry``: only needed when the page should change geometry (it is
    sent each time, the page rebuilds only when it differs). ``ub``: phi
    frame, 2 pi convention; ``k`` = abs(k) = 2 pi / lambda (or ``energy``
    in eV); ``surface`` / ``reference``: phi-frame vectors for betain/betaout
    and alpha/beta/psi/naz/tau; ``targets``: reflections drawn as nodes;
    ``special_points``: :func:`hklforge.crystal.bz.special_points`;
    ``structure_factors``: :func:`structure_factor_nodes`, drawn as spheres
    (volume proportional to abs(F)^2) when the page's option is on;
    ``structure``: :func:`structure_atoms`, the atoms drawn in real space
    (experimental option; without it the page draws the lattice points). Any
    further key (``display``, ``ghosts``, ``detectors``, ``memory`` ...)
    is passed on as is.
    """
    msg = {"label": label, "n": n}
    if geometry is not None:
        msg["geometry"] = geometry_message(geometry)
    if angles is not None:
        msg["angles"] = {name: float(v) for name, v in angles.items()}
    if energy is not None:
        from ..geometry import wavevector
        k = wavevector(energy)
    if k is not None:
        msg["kMag"] = float(k)
    if ub is not None:
        msg["ub"] = np.asarray(ub, float).tolist()
    if surface is not None:
        msg["nPhi"] = _vec(surface)
    if reference is not None:
        msg["refPhi"] = _vec(reference)
    if targets:
        msg["targets"] = [_vec(t) for t in targets]
    if special_points is not None:
        msg["bz"] = {"lattice": special_points.get("lattice"), "variant": special_points.get("variant"),
                     "points": {name: _vec(h) for name, h in special_points.get("points", {}).items()},
                     "path": [list(seg) for seg in special_points.get("path", [])],
                     "note": special_points.get("note", "")}
    if structure_factors is not None:
        msg["structureFactors"] = structure_factors
    if structure is not None:
        msg["structure"] = structure
    msg.update(extra)
    return msg


def structure_factor_nodes(structure, energy: float | None = None, d_min: float | None = None,
                           max_nodes: int = 4000, thermal: bool = True) -> dict:
    """Every reflection of ``structure`` with d >= ``d_min`` and its abs(F)^2,
    as the page's ``structureFactors`` (``message(structure_factors=...)``).

    ``energy`` (eV) includes f' + i f'' and sets the default ``d_min`` to
    lambda / 2 (all reflections reachable at that energy); without it,
    ``d_min`` is needed. Each node of the lattice is listed (all equivalents,
    not one per set), allowed by the space group, nearest to the origin first,
    at most ``max_nodes``. Needs xraydb (``hklforge[crystal]``).

    Returns ``{"hkl": [[h, k, l], ...], "f2": [...], "energy", "dMin",
    "total", "name"}``; ``total`` counts the reflections before the cap.
    """
    from ..crystal.cell import b_matrix, cell_tuple, wavelength
    if d_min is None:
        if energy is None:
            raise ValueError("give the energy or d_min")
        d_min = wavelength(energy) / 2
    B = b_matrix(*cell_tuple(structure.cell))
    qmax = 2 * np.pi / d_min
    lim = [int(np.floor(qmax / np.linalg.norm(B @ e))) for e in np.eye(3)]
    if np.prod([2 * m + 1 for m in lim]) > 2e6:
        raise ValueError("too many reflections: raise d_min")
    H = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in lim], indexing="ij"), -1).reshape(-1, 3)
    q = np.linalg.norm(H @ B.T, axis=1)
    keep = (q > 0) & (q <= qmax * (1 + 1e-9))
    if structure.space_group is not None:
        keep &= structure.space_group.allowed_mask(H)
    H, q = H[keep], q[keep]
    order = np.argsort(q, kind="stable")
    total = len(order)
    H = H[order[:max_nodes]]
    f2 = np.abs(structure.structure_factors(H, energy, thermal=thermal)) ** 2
    return {"hkl": H.astype(int).tolist(), "f2": [float(x) for x in f2],
            "energy": None if energy is None else float(energy), "dMin": float(d_min),
            "total": int(total), "name": structure.name}


def page(geometry: Geometry | None = None, view_id: str = "diff3d", storage: str | None = None,
         detectors=None, title: str | None = None) -> str:
    """The page's HTML with its configuration built in.

    ``view_id`` names the elements it talks to in the parent document
    (``{view_id}-payload`` ...), so two views can share a notebook.
    ``storage``: prefix of the browser-storage keys of the remembered state
    and snapshots (default ``hklforge-viewer``). ``detectors``: the preset
    list ``[[name, width_mm, height_mm], ...]``.
    """
    cfg = {"id": view_id}
    if geometry is not None:
        cfg["geometry"] = geometry_message(geometry)
    if storage:
        cfg["storage"] = storage
    if detectors is not None:
        cfg["detectors"] = [[str(d[0]), float(d[1]), float(d[2])] for d in detectors]
    if title:
        cfg["title"] = title
    text = VIEWER_HTML.read_text(encoding="utf-8")
    script = "<script>window.HKLFORGE_VIEW = " + json.dumps(cfg).replace("</", "<\\/") + ";</script>"
    if CONFIG_MARK not in text:
        raise RuntimeError("viewer.html lost its configuration marker")
    return text.replace(CONFIG_MARK, script, 1)


def iframe(page_html: str, height: int = 760, view_id: str = "diff3d") -> str:
    """An ``<iframe srcdoc>`` holding ``page_html`` (e.g. for an ipywidgets ``HTML``)."""
    return (f'<iframe class="{view_id}-frame" srcdoc="{_html.escape(page_html, quote=True)}" '
            f'style="width:100%;height:{int(height)}px;border:0" '
            'allow="fullscreen"></iframe>')


def payload_html(msg: dict, view_id: str = "diff3d") -> str:
    """The hidden element the page polls, holding ``msg``."""
    return (f'<span id="{view_id}-payload" style="display:none">'
            + _html.escape(json.dumps(msg)) + "</span>")


ELEMENTS_JSON = Path(__file__).resolve().parent.parent / "crystal" / "elements.json"
_ELEMENTS = None


def element_style(symbol: str) -> tuple:
    """(covalent radius in angstrom, colour "#rrggbb") of an element, for
    drawing (Cordero 2008 radii, Jmol colours; ``elements.json``). Unknown
    symbols get 1.5 angstrom and grey."""
    global _ELEMENTS
    if _ELEMENTS is None:
        _ELEMENTS = json.loads(ELEMENTS_JSON.read_text())["elements"]
    e = _ELEMENTS.get(symbol) or _ELEMENTS.get(symbol[:1].upper() + symbol[1:].lower())
    return (1.5, "#909090") if e is None else (float(e[1]), str(e[2]))


def structure_atoms(structure) -> dict:
    """The atoms of one unit cell for the page's real-space drawing
    (``message(structure=...)``): ``{"name", "atoms": [[label, element, x,
    y, z, b_iso, occupancy], ...] (fractional, every atom of the cell),
    "elements": {element: [covalent radius A, colour]}}``. numpy only."""
    sites, X = structure.atoms()
    atoms = [[s.label, s.element, *(round(float(v), 6) for v in x), round(float(s.b_iso), 4),
              round(float(s.occupancy), 4)] for s, x in zip(sites, X)]
    elements = {el: list(element_style(el)) for el in dict.fromkeys(s.element for s in sites)}
    return {"name": structure.name, "atoms": atoms, "elements": elements}
