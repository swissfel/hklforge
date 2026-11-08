"""Unit cells: the reciprocal lattice, crystal systems and consistency checks.

Conventions (the same as hklforge and diffcalc): reciprocal vectors carry
2 pi, |a*| = 2 pi / a for an orthogonal cell, so |B h| = 2 pi / d(hkl). The
B matrix is Busing & Levy's: a* along x, b* in the x-y plane.
"""
from __future__ import annotations

import numpy as np

from ..geometry import HC_EV_ANGSTROM  # noqa: F401
from ..lattice import b_matrix  # noqa: F401  (one Busing-Levy implementation)

AVOGADRO = 6.02214076e23

CELL_KEYS = ("a", "b", "c", "alpha", "beta", "gamma")


def wavelength(energy: float) -> float:
    """Wavelength in angstrom of photons of ``energy`` eV."""
    return HC_EV_ANGSTROM / float(energy)


def real_basis(a: float, b: float, c: float, alpha: float = 90.0, beta: float = 90.0,
               gamma: float = 90.0) -> np.ndarray:
    """Columns a, b, c in an orthonormal frame (a along x, b in the x-y plane)."""
    al, be, ga = np.radians([alpha, beta, gamma])
    A = np.array([[a, b * np.cos(ga), c * np.cos(be)],
                  [0, b * np.sin(ga), c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)],
                  [0, 0, 0.0]])
    A[2, 2] = np.sqrt(c * c - A[0, 2] ** 2 - A[1, 2] ** 2)
    return A


def cell_tuple(cell: dict) -> tuple:
    return tuple(float(cell[k]) for k in CELL_KEYS)


def volume(cell: dict) -> float:
    """Cell volume in cubic angstrom."""
    return float(abs(np.linalg.det(real_basis(*cell_tuple(cell)))))


def d_spacing(cell: dict, hkl) -> np.ndarray:
    """d of (h, k, l) (or an array of them, shape (..., 3)) in angstrom."""
    q = np.asarray(hkl, float) @ b_matrix(*cell_tuple(cell)).T
    with np.errstate(divide="ignore"):
        return 2 * np.pi / np.linalg.norm(q, axis=-1)


def real_direction_in_reciprocal_frame(ub: np.ndarray, uvw) -> np.ndarray:
    """The real-space direction [u v w] in the frame of ``ub`` (a UB or B
    matrix with 2 pi): real basis = 2 pi (UB)^-T."""
    return 2 * np.pi * np.linalg.inv(np.asarray(ub, float)).T @ np.asarray(uvw, float)


# --- crystal systems ---------------------------------------------------------------

SYSTEMS = {
    # name: (which lengths are free, fixed angles or None if free, help)
    "cubic": ("a", {"alpha": 90, "beta": 90, "gamma": 90}, "a = b = c, all angles 90"),
    "tetragonal": ("ac", {"alpha": 90, "beta": 90, "gamma": 90}, "a = b, c; all angles 90"),
    "orthorhombic": ("abc", {"alpha": 90, "beta": 90, "gamma": 90}, "a, b, c; all angles 90"),
    "hexagonal": ("ac", {"alpha": 90, "beta": 90, "gamma": 120}, "a = b, c; alpha = beta = 90, gamma = 120"),
    "trigonal (hexagonal axes)": ("ac", {"alpha": 90, "beta": 90, "gamma": 120},
                                  "a = b, c; gamma = 120 (how R-3m, R-3c crystals are usually given)"),
    "trigonal (rhombohedral axes)": ("a", None, "a = b = c, alpha = beta = gamma (not 90)"),
    "monoclinic": ("abc", {"alpha": 90, "gamma": 90}, "a, b, c, beta; alpha = gamma = 90 (unique axis b)"),
    "triclinic": ("abc", None, "a, b, c, alpha, beta, gamma"),
}


def cell_for_system(system: str, a: float, b: float | None = None, c: float | None = None,
                    alpha: float | None = None, beta: float | None = None, gamma: float | None = None) -> dict:
    """The full cell from the parameters a crystal system leaves free, e.g.
    ``cell_for_system("hexagonal", a=4.143, c=28.636)`` -> gamma = 120."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown crystal system {system!r}; one of {', '.join(SYSTEMS)}")
    free, angles, _ = SYSTEMS[system]
    cell = {"a": float(a), "b": float(b if "b" in free and b is not None else a),
            "c": float(c if "c" in free and c is not None else a)}
    if system == "trigonal (rhombohedral axes)":
        al = float(alpha if alpha is not None else 90.0)
        cell.update(alpha=al, beta=al, gamma=al)
    elif angles is None:
        cell.update(alpha=float(alpha or 90), beta=float(beta or 90), gamma=float(gamma or 90))
    else:
        cell.update({k: float(v) for k, v in angles.items()})
        if "beta" not in angles:
            cell["beta"] = float(beta if beta is not None else 90.0)
    return cell


def system_of_number(number: int) -> str:
    """Crystal system of space group ``number`` (1-230)."""
    for top, name in ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
                      (167, "trigonal"), (194, "hexagonal"), (230, "cubic")):
        if number <= top:
            return name
    raise ValueError(f"space group number {number} is not 1-230")


def _close(x, y, tol=1e-3):
    return abs(float(x) - float(y)) <= tol * max(1.0, abs(float(y)))


def check_cell(cell: dict, space_group=None) -> list[str]:
    """Warnings about a cell, alone and against a space group (a
    ``SpaceGroup``, a symbol or a number). Empty if all is consistent."""
    from .spacegroups import SpaceGroup
    a, b, c = (float(cell[k]) for k in "abc")
    al, be, ga = (float(cell[k]) for k in ("alpha", "beta", "gamma"))
    out = []
    if _close(a, b) and _close(al, 90) and _close(be, 90) and _close(ga, 90) and not _close(a, c):
        if space_group is None:
            out.append("a = b with all angles 90 is a tetragonal cell. A hexagonal or trigonal crystal "
                       "given by a and c needs gamma = 120.")
    if space_group is None:
        return out
    sg = space_group if isinstance(space_group, SpaceGroup) else SpaceGroup(space_group)
    system = sg.system
    sq = _close(al, 90) and _close(be, 90) and _close(ga, 90)
    if system == "cubic" and not (_close(a, b) and _close(a, c) and sq):
        out.append(f"{sg.name} is cubic: a = b = c and all angles 90.")
    elif system == "tetragonal" and not (_close(a, b) and sq):
        out.append(f"{sg.name} is tetragonal: a = b and all angles 90.")
    elif system == "orthorhombic" and not sq:
        out.append(f"{sg.name} is orthorhombic: all angles 90.")
    elif system == "hexagonal" and not (_close(a, b) and _close(al, 90) and _close(be, 90) and _close(ga, 120)):
        out.append(f"{sg.name} is hexagonal: a = b, alpha = beta = 90, gamma = 120.")
    elif system == "trigonal":
        hex_axes = _close(a, b) and _close(al, 90) and _close(be, 90) and _close(ga, 120)
        rh_axes = _close(a, b) and _close(a, c) and _close(al, be) and _close(al, ga) and not _close(al, 90)
        if sg.centring == "R" and rh_axes and not sg.rhombohedral_axes:
            out.append(f"{sg.name} is given here in hexagonal axes, the cell is rhombohedral: "
                       f"use the space group {sg.base_symbol}:R, or the hexagonal cell (gamma = 120).")
        elif sg.centring == "R" and hex_axes and sg.rhombohedral_axes:
            out.append(f"{sg.name} is in rhombohedral axes, the cell in hexagonal ones: "
                       f"use {sg.base_symbol}:H.")
        elif not (hex_axes or rh_axes):
            out.append(f"{sg.name} is trigonal: hexagonal axes (a = b, gamma = 120) or, for R groups, "
                       "rhombohedral axes (a = b = c, alpha = beta = gamma).")
    elif system == "monoclinic" and not (_close(al, 90) and _close(ga, 90)):
        out.append(f"{sg.name} is monoclinic (unique axis b): alpha = gamma = 90.")
    return out
