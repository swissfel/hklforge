"""hklforge.crystal: space groups, CIF, structure factors and reflection groups,
against gemmi where it is installed (only the tests need it)."""
import itertools
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
from hklforge import lattice
from hklforge.crystal import cell, reflections
from hklforge.crystal.spacegroups import SpaceGroup, parse_triplet
from hklforge.crystal.structure import read_cif

DATA = Path(__file__).parent / "data"
try:
    import xraydb  # noqa: F401
except ImportError:
    xraydb = None
needs_xraydb = pytest.mark.skipif(xraydb is None, reason="structure factors need xraydb")
H = np.array([h for h in itertools.product(range(-3, 4), repeat=3) if any(h)])

NACL = """data_NaCl
_cell_length_a 5.6402
_cell_length_b 5.6402
_cell_length_c 5.6402
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0 0 0
Cl1 Cl 0.5 0.5 0.5
"""

SI_NUMBER_ONLY = """data_Si
_cell_length_a 5.431
_cell_length_b 5.431
_cell_length_c 5.431
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 227
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.125 0.125 0.125
"""


def test_cell_basics():
    cell_ = dict(zip(cell.CELL_KEYS, (4.1, 5.2, 6.3, 87, 95, 101)))
    B = lattice.b_matrix(*cell.cell_tuple(cell_))
    A = cell.real_basis(*cell.cell_tuple(cell_))
    in_b_frame = 2 * np.pi * np.linalg.inv(B).T          # a, b, c in B's frame: a_i . b*_j = 2 pi delta
    np.testing.assert_allclose(in_b_frame.T @ in_b_frame, A.T @ A, atol=1e-10)   # same metric
    assert cell.volume(cell_) == pytest.approx(abs(np.linalg.det(A)))
    assert cell.d_spacing(cell_, (1, 2, 3)) == pytest.approx(lattice.d_spacing(B, (1, 2, 3)))
    assert cell.wavelength(12398.419843320026) == pytest.approx(1.0)


def test_triplets():
    R, t = parse_triplet("1/2-y, x+1/2 ,-z+3/4")
    np.testing.assert_array_equal(R, [[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    np.testing.assert_allclose(t, [0.5, 0.5, 0.75])
    R, t = parse_triplet("x-y,x,z+1/6")
    np.testing.assert_array_equal(R[0], [1, -1, 0])


def test_every_setting_matches_gemmi():
    gemmi = pytest.importorskip("gemmi")
    for g in gemmi.spacegroup_table():
        sg = SpaceGroup(g.xhm())
        assert sg.name == g.xhm() and sg.number == g.number
        ops = g.operations()
        ref = np.array([not ops.is_systematically_absent(list(map(int, h))) for h in H])
        np.testing.assert_array_equal(sg.allowed_mask(H), ref, err_msg=g.xhm())
        asu = gemmi.ReciprocalAsu(g)
        for h in H[::41]:
            rep = asu.to_asu(list(map(int, h)), ops)[0]
            same = {tuple(x) for x in H.tolist() if asu.to_asu(list(x), ops)[0] == rep}
            assert set(sg.equivalents(h)) & set(map(tuple, H.tolist())) == same, (g.xhm(), h)


def test_names_and_reasons():
    assert SpaceGroup("R -3 m").name == SpaceGroup(166).name == "R -3 m:H"
    assert SpaceGroup("P 1 21/c 1") == SpaceGroup("P21/c")
    with pytest.raises(ValueError):
        SpaceGroup("Q99")
    sg = SpaceGroup("P63/mmc")
    assert "6₃ screw axis along [0 0 1]" in sg.why_forbidden((0, 0, 1)) and sg.is_allowed((0, 0, 2))
    assert SpaceGroup("Fm-3m").multiplicity((1, 1, 1)) == 8 and SpaceGroup("Fm-3m").multiplicity((1, 0, 0)) == 6
    assert SpaceGroup("Fm-3m").representative((-1, 1, -1)) == (1, 1, 1)


def _gemmi_sf(text, hkls):
    gemmi = pytest.importorskip("gemmi")
    small = gemmi.make_small_structure_from_block(gemmi.cif.read_string(text).sole_block())
    small.change_occupancies_to_crystallographic()
    calc = gemmi.StructureFactorCalculatorX(small.cell)
    return np.array([abs(calc.calculate_sf_from_small_structure(small, list(map(int, h)))) for h in hkls])


@pytest.mark.parametrize("source", ["Bi2Se3.cif", "GaAs.cif", "NaCl", "rutile neutral, no aniso"])
@needs_xraydb
def test_structure_factors_match_gemmi(source):
    if source == "NaCl":
        text = NACL
    elif source.startswith("rutile"):        # gemmi uses neutral atoms and ignores this aniso loop
        text = (DATA / "TiO2_rutile.cif").read_text().replace("Ti4+", "Ti").replace("O2-", "O")
        text = text[:text.index("loop_\n_atom_site_aniso_label")]
    else:
        text = (DATA / source).read_text()
    hkls = [h for h in H if h[2] >= 0]
    ref = _gemmi_sf(text, hkls)
    mine = np.abs(read_cif(text).structure_factors(hkls))
    strong = ref > 0.05 * ref.max()
    assert np.max(np.abs(mine[strong] / ref[strong] - 1)) < 5e-3     # Waasmaier-Kirfel vs IT92 form factors
    assert np.all(mine[ref < 1e-6] < 1e-6 * ref.max())


@needs_xraydb
def test_cif_reader_details():
    r = read_cif(DATA / "TiO2_rutile.cif")
    assert r.formula == "TiO2" and len(r.atoms()[1]) == 6 and r.density == pytest.approx(4.25, abs=0.01)
    ti, o = r.sites
    assert ti.scatterer == "Ti4+" and ti.b_iso == pytest.approx(0.42)
    assert o.scatterer == "O2-" and o.b_iso == pytest.approx(8 * np.pi ** 2 * 0.0063333, rel=1e-3)
    assert o.xyz[0] == pytest.approx(0.3048)
    # only the number of a group with two origins: the one with the atoms on special positions
    si = read_cif(SI_NUMBER_ONLY)
    assert si.space_group.name == "F d -3 m:2" and len(si.atoms()[1]) == 8 and "origin choice 2" in si.notes[0]
    assert si.structure_factor((2, 2, 2)) < 1e-6 and si.structure_factor((1, 1, 1)) == pytest.approx(59.6, abs=0.5)


@needs_xraydb
def test_anomalous_breaks_friedel_for_acentric():
    g = read_cif(DATA / "GaAs.cif")
    plain = np.abs(g.structure_factors([(1, 1, 1), (-1, -1, -1)]))
    assert plain[0] == pytest.approx(plain[1])
    edge = np.abs(g.structure_factors([(1, 1, 1), (-1, -1, -1)], energy=10400))    # just above Ga K
    assert abs(edge[0] - edge[1]) > 5 and edge.max() < plain[0]
    nacl = read_cif(NACL)                                   # centrosymmetric: stays equal
    e = np.abs(nacl.structure_factors([(1, 1, 1), (-1, -1, -1)], energy=10400))
    assert e[0] == pytest.approx(e[1])


def test_rotation_groups_cubic_about_c():
    B = lattice.b_matrix(5.431, 5.431, 5.431)
    hkls = [(2, 2, 0), (2, -2, 0), (-2, -2, 0), (-2, 2, 0), (0, 0, 4), (4, 0, 0), (1, 1, 1), (1, -1, 1), (1, 1, -1)]
    g = reflections.rotation_groups(hkls, B, B @ [0, 0, 1])
    groups = dict(zip(hkls, g["group"]))
    assert len({groups[h] for h in hkls[:4]}) == 1                     # {2 2 0} turn into each other
    assert groups[(0, 0, 4)] != groups[(4, 0, 0)]                      # same |Q|, other angle to c
    assert groups[(1, 1, 1)] == groups[(1, -1, 1)] != groups[(1, 1, -1)]
    rot = dict(zip(hkls, g["rotation"]))
    assert rot[(2, 2, 0)] == 0 and abs(rot[(2, -2, 0)]) == pytest.approx(90) and abs(rot[(-2, -2, 0)]) == pytest.approx(180)
    assert np.isnan(rot[(0, 0, 4)])                                     # on the axis
    # the rotation really maps it: rot(axis, angle) Q = Q of the first member
    a = np.array([0, 0, 1.0])
    t = np.radians(rot[(2, -2, 0)])
    Rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    np.testing.assert_allclose(Rz @ (B @ [2, -2, 0]), B @ [2, 2, 0], atol=1e-12)


def test_rotation_groups_hexagonal_and_collapse():
    B = lattice.b_matrix(4.143, 4.143, 28.636, 90, 90, 120)
    hkls = [(1, 0, 5), (0, 1, 5), (-1, 1, 5), (-1, 0, 5), (0, -1, 5), (1, -1, 5), (1, 0, -5)]
    g = reflections.rotation_groups(hkls, B, B @ [0, 0, 1])
    assert list(g["group"]) == [0] * 6 + [1]
    steps = sorted(np.round(np.abs(g["rotation"][:6])) % 360)
    assert steps == [0, 60, 60, 120, 120, 180]
    import pandas as pd
    df = pd.DataFrame({"h": [h[0] for h in hkls], "k": [h[1] for h in hkls], "l": [h[2] for h in hkls]})
    one = reflections.collapse_groups(pd.concat([df, g], axis=1))
    assert list(one["n"]) == [6, 1] and one.iloc[0]["members"].startswith("(0 1 5)")
    with_angle = reflections.collapse_groups(pd.concat([df, g], axis=1), angle="rotation")
    assert with_angle.iloc[0]["members"].startswith("(0 1 5) ") and "°" in with_angle.iloc[0]["members"]


@needs_xraydb
def test_equivalence_groups_and_columns():
    sg = SpaceGroup("Fd-3m")
    e = reflections.equivalence_groups([(1, 1, 1), (-1, 1, 1), (2, 2, 0), (0, 2, -2)], sg)
    assert list(e["group"]) == [0, 0, 1, 1] and list(e["family"]) == ["{1 1 1}", "{1 1 1}", "{2 2 0}", "{2 2 0}"]
    st = read_cif(SI_NUMBER_ONLY)
    cols = reflections.crystal_columns([(1, 1, 1), (2, 2, 2), (2, 0, 0)], structure=st, energy=9000)
    assert list(cols["allowed"]) == [True, True, False]
    assert cols["why not"][1] == "allowed by symmetry, F = 0 (atom positions)"
    assert cols["|F|²"][0] == pytest.approx(cols["|F|"][0] ** 2) and cols["I rel"][0] == pytest.approx(1)
