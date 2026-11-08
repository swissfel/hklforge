"""Write src/hklforge/crystal/spacegroups.json from gemmi's space-group table.

Run once when the table should change (needs gemmi; nothing at run time
does): every setting gemmi knows (564), each with its symmetry operations
as x,y,z triplets and its centring vectors, plus the names gemmi resolves to
it, so that hklforge.crystal.spacegroups finds a group by the same names
gemmi accepts ("R-3m", "R -3 m", "P21/c", "P 1 21/c 1", "R-3m:R", 166, ...).

    python dev/make_spacegroup_table.py
"""
import json
from fractions import Fraction
from pathlib import Path

import gemmi

OUT = Path(__file__).resolve().parent.parent / "src" / "hklforge" / "crystal" / "spacegroups.json"


def frac(n):
    f = Fraction(n, gemmi.Op.DEN)
    return str(f) if f.denominator > 1 else str(f.numerator)


def normal(name: str) -> str:
    return "".join(str(name).split()).lower()


def main():
    groups, aliases = [], {}
    for sg in gemmi.spacegroup_table():
        ops = sg.operations()
        groups.append({
            "name": sg.xhm(), "number": sg.number, "hall": sg.hall.strip(),
            "centring": sg.centring_type(), "system": sg.crystal_system_str(),
            "point_group": sg.point_group_hm(), "laue": sg.laue_str(),
            "ops": [op.triplet() for op in ops.sym_ops],
            "centring_vectors": [",".join(frac(x) for x in v) for v in ops.cen_ops],
        })
    index = {g["name"]: i for i, g in enumerate(groups)}
    for sg in gemmi.spacegroup_table():
        for variant in {sg.xhm(), sg.hm, sg.short_name(), sg.hall, f"{sg.hm}:{sg.ext}" if sg.ext else sg.hm}:
            if not variant.strip():
                continue
            found = gemmi.find_spacegroup_by_name(variant)
            if found is not None:
                aliases[normal(variant)] = index[found.xhm()]
    for number in range(1, 231):
        aliases[str(number)] = index[gemmi.find_spacegroup_by_number(number).xhm()]
    OUT.write_text(json.dumps({"source": f"gemmi {gemmi.__version__} space-group table",
                               "groups": groups, "aliases": aliases}, separators=(",", ":")))
    print(f"{OUT}: {len(groups)} settings, {len(aliases)} names, {OUT.stat().st_size // 1024} kB")


if __name__ == "__main__":
    main()
