"""Write src/hklforge/crystal/elements.json: per element its atomic number,
covalent radius and display colour, for drawing atoms (hklforge.view).

Radii: Cordero et al., Dalton Trans. (2008) 2832 (covalent radii, angstrom);
colours: Jmol's CPK scheme. Both taken from ASE's tables (ase.data), so run
once when the table should change (needs ase; nothing at run time does):

    python dev/make_element_table.py
"""
import json
from pathlib import Path

from ase.data import chemical_symbols, covalent_radii
from ase.data.colors import jmol_colors

OUT = Path(__file__).resolve().parent.parent / "src" / "hklforge" / "crystal" / "elements.json"


def main():
    table = {}
    for z, sym in enumerate(chemical_symbols):
        if z == 0 or z >= len(jmol_colors):
            continue
        r, g, b = (int(round(255 * float(c))) for c in jmol_colors[z])
        table[sym] = [z, round(float(covalent_radii[z]), 3), f"#{r:02x}{g:02x}{b:02x}"]
    OUT.write_text(json.dumps({"source": "covalent radii: Cordero et al. 2008; colours: Jmol (via ase.data)",
                               "fields": ["Z", "covalent_radius", "color"], "elements": table},
                              separators=(",", ":")) + "\n")
    print(f"{len(table)} elements -> {OUT}")


if __name__ == "__main__":
    main()
