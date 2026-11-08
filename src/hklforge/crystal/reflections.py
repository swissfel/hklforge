"""Reflection lists, structure-factor columns and groups of equivalent reflections.

Two kinds of "the same reflection":

- **symmetry-equivalent** (``equivalence_groups``): related by the crystal's
  point group and Friedel's law, so the same |Q| and, without anomalous
  scattering, the same |F|;
- **the same under a rotation about an axis** (``rotation_groups``): same
  |Q| and same angle to the axis, i.e. the same component Q_par along it and
  Q_perp across it. Rotating the crystal about that axis brings one onto
  the other, at the same 2theta and the same exit direction relative to the
  axis. With the axis along a sample circle (phi, a surface normal, a fibre
  or film texture axis) a scan of that circle passes all of them; the
  ``rotation`` column gives the turn about the axis that takes each one to
  where the group's first member is.

Symmetry-equivalent reflections are always in the same rotation group only
when the axis is a symmetry axis of the crystal; a rotation group can also
hold reflections that are not equivalent at all (other |F|).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cell import b_matrix, cell_tuple, wavelength
from .spacegroups import SpaceGroup, _preference


def as_space_group(space_group) -> SpaceGroup | None:
    if space_group is None or isinstance(space_group, SpaceGroup):
        return space_group
    return SpaceGroup(space_group)


def fmt_hkl(h) -> str:
    return "(" + " ".join(f"{int(round(x))}" if abs(x - round(x)) < 1e-9 else f"{x:g}" for x in h) + ")"


def reflection_list(cell: dict, energy: float, space_group=None, structure=None,
                    d_min: float | None = None, max_rows: int = 2000, unique: bool = True,
                    allowed_only: bool = False) -> pd.DataFrame:
    """Reflections reachable at ``energy`` (d >= lambda / 2, or ``d_min``),
    sorted by d (largest first); with ``unique`` and a space group one per set
    of equivalents (with its multiplicity).

    Columns: h, k, l, d [A], 2theta, and with a space group allowed, why not,
    mult; with a structure |F|, |F|^2 (energy-dependent: f' and f'') and
    I rel (|F|^2 over the strongest listed).
    """
    sg = as_space_group(space_group)
    if structure is not None and sg is None:
        sg = structure.space_group
    lam = wavelength(energy)
    d_min = max(d_min or 0.0, lam / 2)
    B = b_matrix(*cell_tuple(cell))
    qmax = 2 * np.pi / d_min
    lim = [int(np.floor(qmax / np.linalg.norm(B @ e))) for e in np.eye(3)]
    if np.prod([2 * m + 1 for m in lim]) > 5e6:
        raise ValueError("too many reflections: raise d_min or lower the energy")
    rng = [np.arange(-m, m + 1) for m in lim]
    H = np.stack(np.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    H = H[np.any(H != 0, axis=1)]
    d = 2 * np.pi / np.linalg.norm(H @ B.T, axis=1)
    keep = d >= d_min * (1 - 1e-9)
    if allowed_only and sg is not None:
        keep &= sg.allowed_mask(H)
    H, d = H[keep], d[keep]
    # by d; at equal d the nicer indices first (mostly non-negative, then larger)
    order = sorted(range(len(H)), key=lambda i: (-round(d[i], 9), tuple(-x for x in _preference(tuple(H[i])))))
    rows = []
    for i in order:
        hkl = tuple(int(x) for x in H[i])
        if unique and sg is not None and not sg.is_unique(hkl):
            continue
        if unique and sg is None and tuple(-x for x in hkl) > hkl:
            continue                                             # Friedel pairs only
        rows.append({"h": hkl[0], "k": hkl[1], "l": hkl[2], "d [A]": float(d[i]),
                     "2theta": float(np.degrees(2 * np.arcsin(min(1.0, lam / (2 * d[i])))))})
        if len(rows) >= max_rows:
            break
    df = pd.DataFrame(rows)
    if len(df):
        df = pd.concat([df, crystal_columns(df[["h", "k", "l"]].to_numpy(), sg, structure, energy,
                                            multiplicity=unique)], axis=1)
    return df


def crystal_columns(hkls, space_group=None, structure=None, energy: float | None = None,
                    multiplicity: bool = False) -> pd.DataFrame:
    """Per reflection: allowed and why not (space group), mult, |F| and |F|^2
    (structure, at ``energy``), I rel; an "allowed by symmetry, F = 0" note
    where the atom positions cancel it."""
    H = np.asarray(hkls, float).reshape(-1, 3)
    sg = as_space_group(space_group)
    if structure is not None and sg is None:
        sg = structure.space_group
    cols = {}
    if sg is not None:
        cols["allowed"] = [sg.is_allowed(h) for h in H]
        cols["why not"] = [sg.why_forbidden(h) for h in H]
        if multiplicity:
            cols["mult"] = [sg.multiplicity(h) for h in H]
    if structure is not None and len(H):
        F = np.abs(structure.structure_factors(H, energy))
        cols["|F|"] = F
        cols["|F|²"] = F ** 2
        fmax = F.max()
        cols["I rel"] = (F / fmax) ** 2 if fmax > 0 else np.zeros(len(F))
        if sg is not None:
            weak = np.array(cols["allowed"]) & (F < 1e-6 * max(fmax, 1))
            cols["why not"] = ["allowed by symmetry, F = 0 (atom positions)" if w else t
                               for w, t in zip(weak, cols["why not"])]
    return pd.DataFrame(cols)


def target_check(hkl, cell: dict, space_group=None, structure=None, energy: float | None = None) -> list[str]:
    """Warnings for an alignment target: forbidden, or zero/weak intensity."""
    out = []
    if any(abs(x - round(x)) > 1e-6 for x in hkl):
        return out
    name = fmt_hkl(hkl)
    sg = structure.space_group if structure is not None and space_group is None else as_space_group(space_group)
    if sg is not None:
        why = sg.why_forbidden(hkl)
        if why:
            return [f"{name} is {why}"]
    if structure is not None:
        f = structure.structure_factor(hkl, energy)
        f000 = structure.z_electrons
        if f < 1e-6 * f000:
            out.append(f"{name} has F = 0 from the atom positions")
        elif f < 0.02 * f000:
            out.append(f"{name} is weak: |F| = {f:.3g} of F(000) = {f000:.0f}")
    return out


# --- groups ---------------------------------------------------------------------------


def equivalence_groups(hkls, space_group) -> pd.DataFrame:
    """For each reflection: ``group`` (numbered in order of appearance),
    ``family`` (the set's representative, e.g. {1 1 1}) and ``mult``."""
    sg = as_space_group(space_group)
    reps, groups, fams, mults = {}, [], [], []
    for h in np.asarray(hkls, float).reshape(-1, 3):
        rep = sg.representative(h)
        groups.append(reps.setdefault(rep, len(reps)))
        fams.append("{" + fmt_hkl(rep)[1:-1] + "}")
        mults.append(sg.multiplicity(h))
    return pd.DataFrame({"group": groups, "family": fams, "mult": mults})


def rotation_groups(hkls, ub, axis, tol: float = 2e-3) -> pd.DataFrame:
    """Group reflections that a rotation about ``axis`` maps onto each other.

    ``ub``: a UB (or B) matrix with 2 pi, ``axis`` a vector in the same frame
    (e.g. a circle's axis, ``ub @ hkl`` or a real-space direction). Two
    reflections share a group when their Q_par and Q_perp agree within
    ``tol`` (1/A). Columns: group, |Q|, Q_par, Q_perp, rotation: the angle
    (deg, right-handed about ``axis``) that takes the reflection to where the
    group's first member is (0 for that one; NaN on the axis, Q_perp = 0).
    """
    H = np.asarray(hkls, float).reshape(-1, 3)
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    Q = H @ np.asarray(ub, float).T
    qpar = Q @ a
    perp = Q - np.outer(qpar, a)
    qperp = np.linalg.norm(perp, axis=1)
    groups, first = [], []                  # first: index of each group's first member
    for i in range(len(H)):
        for g, j in enumerate(first):
            if abs(qpar[i] - qpar[j]) < tol and abs(qperp[i] - qperp[j]) < tol:
                groups.append(g)
                break
        else:
            groups.append(len(first))
            first.append(i)
    rot = np.full(len(H), np.nan)
    for i, g in enumerate(groups):
        j = first[g]
        if qperp[i] > tol and qperp[j] > tol:
            rot[i] = np.degrees(np.arctan2(np.cross(perp[i], perp[j]) @ a, perp[i] @ perp[j]))
    return pd.DataFrame({"group": groups, "|Q|": np.linalg.norm(Q, axis=1), "Q_par": qpar,
                         "Q_perp": qperp, "rotation": rot})


def collapse_groups(df: pd.DataFrame, label=fmt_hkl, angle: str | None = None) -> pd.DataFrame:
    """One row per ``group`` (its first member) with ``n`` and the other
    ``members`` listed, each with its ``angle`` column if given ("(0 1 5) 60°").
    ``df`` has h, k, l as columns or as its index."""
    flat = df.reset_index() if "h" not in df.columns else df
    rows = []
    for _, sub in flat.groupby("group", sort=False):
        row = sub.iloc[0].copy()
        row["n"] = len(sub)
        names = []
        for _, m in sub.iloc[1:].iterrows():
            text = label(m[["h", "k", "l"]].to_numpy(float))
            if angle is not None and np.isfinite(m[angle]):
                text += f" {m[angle]:.4g}°"
            names.append(text)
        row["members"] = ", ".join(names)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.set_index(["h", "k", "l"]) if "h" not in df.columns else out.reset_index(drop=True)
