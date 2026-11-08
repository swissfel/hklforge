"""Crystal structures: atoms in the unit cell and X-ray structure factors.

    F(h) = sum_j occ_j f_j(s) exp(-B_j s^2) exp(2 pi i h . x_j),   s = 1 / (2 d)

over every atom of the unit cell (the CIF's sites under all symmetry
operations), with

- f0(s) from Waasmaier & Kirfel (xraydb; the ion's when the CIF names one),
- with an ``energy``, the anomalous terms f' + i f'' (Chantler tables,
  xraydb) - these change |F| near an absorption edge and make F(h) and
  F(-h) differ for non-centrosymmetric crystals,
- the isotropic Debye-Waller factor B = 8 pi^2 U from the CIF (an
  anisotropic U is reduced to the mean of U11, U22, U33).

|F|^2 is in electrons^2 per unit cell; F(0 0 0) is the number of electrons
(plus f' with an energy).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import cif
from .cell import AVOGADRO, b_matrix, cell_tuple, volume
from .spacegroups import SpaceGroup, parse_triplet

try:
    import xraydb
except ImportError:          # needed only for structure factors, formula and density
    xraydb = None


@dataclass
class Site:
    label: str
    element: str             # "Fe"
    scatterer: str           # what f0 is taken for: the ion if known ("Fe3+"), else the element
    xyz: np.ndarray          # fractional
    occupancy: float = 1.0
    b_iso: float = 0.0       # angstrom^2


@dataclass
class Structure:
    """Cell, symmetry and atom sites, e.g. from ``read_cif``."""
    name: str
    cell: dict
    space_group: SpaceGroup | None
    sites: list
    ops: list = field(default=None, repr=False)        # (R, t); default: the space group's
    notes: list = field(default_factory=list)          # what reading had to decide
    _atoms: tuple = field(default=None, repr=False, init=False)

    def __post_init__(self):
        if self.ops is None:
            if self.space_group is None:
                raise ValueError("a structure needs a space group or its symmetry operations")
            self.ops = self.space_group.all_ops()

    # --- the unit cell ---

    def atoms(self):
        """(sites, fractional positions in [0, 1)): every atom of one cell."""
        if self._atoms is None:
            sites, pos = [], []
            for s in self.sites:
                found = []
                for R, t in self.ops:
                    x = (R @ s.xyz + t) % 1.0
                    if not any(np.all(np.abs((x - y + 0.5) % 1.0 - 0.5) < 1e-3) for y in found):
                        found.append(x)
                sites += [s] * len(found)
                pos += found
            self._atoms = (sites, np.array(pos).reshape(-1, 3))
        return self._atoms

    def composition(self) -> dict:
        """{element: atoms per unit cell} (occupancies as given)."""
        out: dict = {}
        for s in self.atoms()[0]:
            out[s.element] = out.get(s.element, 0.0) + s.occupancy
        return out

    @property
    def formula(self) -> str:
        counts = self.composition()
        ints = [int(round(v)) for v in counts.values()]
        g = 1.0
        if all(abs(v - i) < 1e-6 for v, i in zip(counts.values(), ints)) and all(ints):
            g = float(np.gcd.reduce(ints))
        return "".join(f"{el}{_num(n / g)}" for el, n in counts.items())

    @property
    def density(self) -> float:
        """g / cm^3."""
        _need_xraydb("the density")
        mass = sum(n * xraydb.atomic_mass(el) for el, n in self.composition().items())
        return mass / (volume(self.cell) * 1e-24 * AVOGADRO)

    @property
    def z_electrons(self) -> float:
        """Electrons per unit cell, F(0 0 0) without anomalous scattering."""
        _need_xraydb("electron counts")
        return float(sum(n * xraydb.atomic_number(el) for el, n in self.composition().items()))

    # --- structure factors ---

    def structure_factors(self, hkls, energy: float | None = None, thermal: bool = True) -> np.ndarray:
        """Complex F for reflections ``hkls`` (shape (n, 3)); with ``energy``
        (eV) including f' + i f''."""
        _need_xraydb("structure factors")
        H = np.asarray(hkls, float).reshape(-1, 3)
        sites, X = self.atoms()
        q = np.linalg.norm(H @ b_matrix(*cell_tuple(self.cell)).T, axis=1)
        s = q / (4 * np.pi)                                   # sin(theta) / lambda = 1 / (2 d)
        phase = np.exp(2j * np.pi * H @ X.T)                 # (n, atoms)
        F = np.zeros(len(H), complex)
        for scat in {si.scatterer for si in sites}:
            idx = [i for i, si in enumerate(sites) if si.scatterer == scat]
            f = np.asarray(xraydb.f0(scat, s), float).reshape(-1)
            if energy is not None:
                el = sites[idx[0]].element
                f = f + float(xraydb.f1_chantler(el, energy)) + 1j * float(xraydb.f2_chantler(el, energy))
            w = np.array([sites[i].occupancy for i in idx])
            if thermal:
                b = np.array([sites[i].b_iso for i in idx])
                w = w[None, :] * np.exp(-np.outer(s ** 2, b))
            else:
                w = np.broadcast_to(w, (len(H), len(idx)))
            F += f * np.sum(w * phase[:, idx], axis=1)
        return F

    def structure_factor(self, hkl, energy: float | None = None) -> float:
        """|F(h k l)| in electrons."""
        return float(abs(self.structure_factors([hkl], energy)[0]))


def _num(x):
    return "" if abs(x - 1) < 1e-6 else (f"{x:.0f}" if abs(x - round(x)) < 1e-6 else f"{x:.3g}")


def _need_xraydb(what):
    if xraydb is None:
        raise ImportError(f"{what} need xraydb (pip install xraydb)")


# --- reading CIF -------------------------------------------------------------------

_ELEMENT = re.compile(r"[A-Z][a-z]?")


def _element_and_ion(type_symbol: str | None, label: str) -> tuple[str, str]:
    text = (type_symbol or label or "").strip()
    m = _ELEMENT.match(text) or _ELEMENT.match(label.strip().capitalize())
    if m is None:
        raise ValueError(f"no element in atom site {label!r}")
    element = m.group(0)
    ion = re.match(r"([A-Z][a-z]?)(\d*)([+-])", text)
    scatterer = element
    if ion and xraydb is not None:
        name = f"{ion.group(1)}{ion.group(2) or ''}{ion.group(3)}"
        if name in xraydb.f0_ions(element):
            scatterer = name
    return element, scatterer


def read_cif(text_or_path, block: int | str = 0) -> Structure:
    """A CIF (path or its text) -> ``Structure``. Uses the CIF's own symmetry
    operations when it lists them, else those of its space group."""
    text = str(text_or_path)
    if not ("\n" in text or text.lstrip().startswith("data_")):
        with open(text_or_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    blocks = [b for b in cif.parse(text) if b.get("_cell_length_a", "_cell.length_a") is not None]
    if not blocks:
        raise ValueError("no data block with a unit cell in this CIF")
    b = blocks[block] if isinstance(block, int) else next(x for x in blocks if x.name == block)
    cell = {k: cif.number(b.get(f"_cell_length_{k}", f"_cell.length_{k}")) for k in "abc"}
    cell.update({k: cif.number(b.get(f"_cell_angle_{k}", f"_cell.angle_{k}"), 90.0)
                 for k in ("alpha", "beta", "gamma")})
    name = b.get("_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m", "_space_group.name_h-m_alt")
    num = b.get("_space_group_it_number", "_symmetry_int_tables_number", "_space_group.it_number")
    sg = None
    for key in (name, num):
        if key and key not in ("?", "."):
            try:
                sg = SpaceGroup(key)
                break
            except ValueError:
                pass
    ops = None
    lp = b.loop("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz", "_space_group_symop.operation_xyz")
    if lp:
        col = next(v for k, v in lp.items() if k.endswith(("operation_xyz", "pos_as_xyz")))
        ops = [parse_triplet(t) for t in col]
    if sg is None and ops is None:
        raise ValueError("the CIF has neither a known space group nor symmetry operations")
    notes = []
    if ops is None:
        sg, note = _choose_setting(sg, str(name or num), cell, b)
        if note:
            notes.append(note)
    atoms = b.loop("_atom_site_fract_x", "_atom_site.fract_x")
    if not atoms:
        raise ValueError("the CIF has no atom sites (_atom_site_fract_x)")
    col = lambda k: atoms.get(f"_atom_site_{k}", atoms.get(f"_atom_site.{k}"))      # noqa: E731
    labels = col("label") or [f"atom{i}" for i in range(len(col("fract_x")))]
    aniso = b.loop("_atom_site_aniso_u_11", "_atom_site_aniso.u_11")
    u_aniso = {}
    if aniso:
        alab = aniso.get("_atom_site_aniso_label", aniso.get("_atom_site_aniso.label", []))
        for i, lab in enumerate(alab):
            us = [cif.number(aniso.get(f"_atom_site_aniso_u_{k}", aniso.get(f"_atom_site_aniso.u_{k}"))[i], 0.0)
                  for k in ("11", "22", "33")]
            u_aniso[lab] = sum(us) / 3
    sites = []
    for i, label in enumerate(labels):
        get = lambda k: (col(k) or [None] * len(labels))[i]                          # noqa: E731
        element, scat = _element_and_ion(get("type_symbol"), label)
        u = cif.number(get("u_iso_or_equiv"))
        bval = cif.number(get("b_iso_or_equiv"))
        if bval is None:
            u = u if u is not None else u_aniso.get(label, 0.0)
            bval = 8 * np.pi ** 2 * u
        sites.append(Site(label, element, scat,
                          np.array([cif.number(get(f"fract_{k}"), 0.0) for k in "xyz"]),
                          cif.number(get("occupancy"), 1.0), float(bval)))
    st = Structure(b.name, cell, sg, sites, ops, notes)
    if notes and notes[-1] == "origin":
        notes.pop()
        st = _fewest_atoms(st, notes)
    return st


def _choose_setting(sg, given: str, cell: dict, block) -> tuple:
    """The setting the CIF means when it only names the group: hexagonal or
    rhombohedral axes from the cell; the origin choice decided later."""
    if ":" in given:
        return sg, ""
    if sg.centring == "R":
        rh = (abs(cell["a"] - cell["b"]) < 1e-3 and abs(cell["a"] - cell["c"]) < 1e-3
              and abs(cell["alpha"] - cell["beta"]) < 1e-3 and abs(cell["alpha"] - 90) > 1e-3)
        want = f"{sg.base_symbol}:{'R' if rh else 'H'}"
        if want != sg.name:
            return SpaceGroup(want), f"rhombohedral axes ({want}): the cell has a = b = c, alpha = beta = gamma"
        return sg, ""
    if sg.name.endswith((":1", ":2")):
        return sg, "origin"
    return sg, ""


def _fewest_atoms(st: Structure, notes: list) -> Structure:
    """Two origin choices and the CIF says neither: take the one that puts
    the sites on special positions (fewer atoms per cell)."""
    options = []
    for choice in ("1", "2"):
        alt = SpaceGroup(f"{st.space_group.base_symbol}:{choice}")
        cand = Structure(st.name, st.cell, alt, st.sites, None, list(notes))
        options.append((len(cand.atoms()[1]), choice, cand))
    n1, n2 = options[0][0], options[1][0]
    best = min(options, key=lambda o: (o[0], o[1] != "2"))       # a tie: origin choice 2 (ICSD, COD)
    if n1 != n2:
        best[2].notes.append(f"origin choice {best[1]} ({best[2].space_group.name}): the CIF does not say which, "
                             f"and that one puts the atoms on special positions ({best[0]} atoms per cell)")
    else:
        best[2].notes.append(f"origin choice {best[1]} ({best[2].space_group.name}) assumed: the CIF does not say")
    return best[2]
