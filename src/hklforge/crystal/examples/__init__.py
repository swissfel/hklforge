"""Example structures to try things with: small CIFs (textbook values)
shipped with hklforge.

>>> from hklforge.crystal import examples
>>> examples.names()
['Bi2Se3', 'GaAs', 'Si', 'SrTiO3', 'TiO2_rutile']
>>> st = examples.load("SrTiO3")          # a Structure (read_cif)
>>> text = examples.cif_text("Si")        # the CIF itself, e.g. for an upload field
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent

TITLES = {"Si": "silicon (diamond, F d -3 m)",
          "GaAs": "GaAs (zinc blende, F -4 3 m)",
          "SrTiO3": "SrTiO3 (cubic perovskite, P m -3 m)",
          "TiO2_rutile": "TiO2 rutile (P 42/m n m)",
          "Bi2Se3": "Bi2Se3 (R -3 m, hexagonal axes)"}


def names() -> list[str]:
    """The example names, sorted."""
    return sorted(p.stem for p in _DIR.glob("*.cif"))


def title(name: str) -> str:
    """A one-line description for a menu."""
    return TITLES.get(name, name)


def cif_text(name: str) -> str:
    """The CIF of example ``name``."""
    path = _DIR / f"{name}.cif"
    if not path.is_file():
        raise ValueError(f"no example {name!r}; there are {names()}")
    return path.read_text(encoding="utf-8")


def load(name: str):
    """Example ``name`` as a :class:`~hklforge.crystal.structure.Structure`."""
    from ..structure import read_cif
    return read_cif(cif_text(name))
