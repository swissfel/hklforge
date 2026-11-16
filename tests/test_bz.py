"""hklforge.crystal.bz: Brillouin-zone points (Setyawan & Curtarolo) in the
conventional reciprocal basis, each checked to lie on the zone boundary."""
import numpy as np
import pytest

from hklforge.lattice import b_matrix
from hklforge.crystal.bz import boundary_nodes, bravais_lattice, special_points

RNG = np.random.default_rng(7)


def cubic(a):
    return dict(a=a, b=a, c=a, alpha=90, beta=90, gamma=90)


def tetragonal(a, c):
    return dict(a=a, b=a, c=c, alpha=90, beta=90, gamma=90)


def hexagonal(a, c, gamma=120):
    return dict(a=a, b=a, c=c, alpha=90, beta=90, gamma=gamma)


def length(cell, hkl):
    """|q| in units of 2 pi / a."""
    return np.linalg.norm(b_matrix(*(cell[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")))
                          @ np.asarray(hkl, float)) * cell["a"] / (2 * np.pi)


CENTRING = {"cP": "P", "cF": "F", "cI": "I", "tP": "P", "tI": "I", "oP": "P", "hP": "P"}


def node_centring(res, sg):
    if res["lattice"] == "hR":
        return "P" if sg.endswith(":R") else "R"
    return CENTRING[res["lattice"]]


def check_all(cell, sg, variant):
    res = special_points(cell, sg)
    assert res["variant"] == variant
    assert res["note"] == "", res["note"]
    assert res["points"]["Γ"] == (0.0, 0.0, 0.0)
    cen = node_centring(res, sg)
    for lab, hkl in res["points"].items():
        if lab != "Γ":
            ok, _ = boundary_nodes(cell, hkl, cen)
            assert ok, (lab, hkl)
    labels = {lab for seg in res["path"] for lab in seg}
    assert labels <= set(res["points"])
    assert all(len(seg) >= 2 for seg in res["path"])
    return res


# --- 1. known positions -------------------------------------------------------------------

def test_simple_cubic():
    res = special_points(cubic(4.0), "Pm-3m")
    assert res["variant"] == "CUB"
    assert res["points"]["X"] == (0.0, 0.5, 0.0)
    assert res["points"]["M"] == (0.5, 0.5, 0.0)
    assert res["points"]["R"] == (0.5, 0.5, 0.5)


def test_fcc():
    c = cubic(5.43)
    res = special_points(c, "Fd-3m")
    p = res["points"]
    assert p["X"] == (0.0, 1.0, 0.0)
    assert p["L"] == (0.5, 0.5, 0.5)
    assert p["W"] == (0.5, 1.0, 0.0)
    assert length(c, p["X"]) == pytest.approx(1)
    assert length(c, p["L"]) == pytest.approx(np.sqrt(3) / 2)
    assert length(c, p["W"]) == pytest.approx(np.sqrt(5) / 2)
    assert length(c, p["K"]) == pytest.approx(3 * np.sqrt(2) / 4)
    assert length(c, p["U"]) == pytest.approx(3 * np.sqrt(2) / 4)   # U is equivalent to K
    assert res["path"] == [["Γ", "X", "W", "K", "Γ", "L", "U", "W", "L", "K"], ["U", "X"]]
    # vertices: W meets 4 nodes (origin included), K 3; L, X are face centres (2)
    assert boundary_nodes(c, p["W"], "F") == (True, 4)
    assert boundary_nodes(c, p["K"], "F") == (True, 3)
    assert boundary_nodes(c, p["L"], "F") == (True, 2)
    assert boundary_nodes(c, p["X"], "F") == (True, 2)


def test_bcc():
    c = cubic(3.3)
    p = special_points(c, "Im-3m")["points"]
    assert length(c, p["H"]) == pytest.approx(1)
    assert length(c, p["N"]) == pytest.approx(np.sqrt(2) / 2)
    assert length(c, p["P"]) == pytest.approx(np.sqrt(3) / 2)
    ok_p, n_p = boundary_nodes(c, p["P"], "I")
    ok_h, n_h = boundary_nodes(c, p["H"], "I")
    assert ok_p and n_p >= 4      # P: corner of 3 faces
    assert ok_h and n_h >= 3      # H: corner of 4 faces
    assert boundary_nodes(c, p["N"], "I") == (True, 2)


# --- 2. every supported variant, random cells ------------------------------------------------

def test_cubic_random():
    for a in RNG.uniform(2, 10, 5):
        check_all(cubic(a), "Pm-3m", "CUB")
        check_all(cubic(a), "Fm-3m", "FCC")
        check_all(cubic(a), "Im-3m", "BCC")
        check_all(cubic(a), None, "CUB")


def test_tetragonal_random():
    for a in RNG.uniform(2, 10, 5):
        for ratio in (RNG.uniform(0.3, 0.95), RNG.uniform(1.05, 4)):
            c = tetragonal(a, a * ratio)
            check_all(c, "P4/mmm", "TET")
            check_all(c, "I4/mmm", "BCT1" if ratio < 1 else "BCT2")


def test_bct_with_c_equal_a_is_bcc():
    res = special_points(tetragonal(4, 4), "I4/mmm")
    assert res["variant"] == "BCC" and "body-centred cubic" in res["note"]
    assert set(res["points"]) == {"Γ", "H", "N", "P"}


def test_orthorhombic_random():
    for _ in range(8):
        a, b, c = RNG.uniform(2, 12, 3)     # any order: X, Y, Z stay along a*, b*, c*
        cell = dict(a=a, b=b, c=c, alpha=90, beta=90, gamma=90)
        res = check_all(cell, "Pnma", "ORC")
        assert res["points"]["X"] == (0.5, 0.0, 0.0)
        assert res["points"]["Z"] == (0.0, 0.0, 0.5)
        check_all(cell, None, "ORC")


def test_hexagonal_random():
    for a, c in RNG.uniform(2, 12, (6, 2)):
        res = check_all(hexagonal(a, c), "P6_3/mmc", "HEX")
        assert res["points"]["K"] == pytest.approx((1 / 3, 1 / 3, 0))
        assert boundary_nodes(hexagonal(a, c), res["points"]["K"], "P")[1] == 3
        check_all(hexagonal(a, c), "P-3m1", "HEX")


def rh_alpha(a, c):
    """Rhombohedral angle of the primitive cell of a hexagonal R cell."""
    ar2 = a * a / 3 + c * c / 9
    return np.degrees(np.arccos((c * c / 9 - a * a / 6) / ar2))


def test_rhombohedral_hexagonal_axes():
    # Bi2Se3: alpha ~ 24 deg
    assert rh_alpha(4.14, 28.6) < 90
    check_all(hexagonal(4.14, 28.6), "R-3m", "RHL1")
    for a in RNG.uniform(3, 6, 4):
        for c in (a * RNG.uniform(1.4, 8), a * RNG.uniform(0.3, 1.1)):
            v = "RHL1" if rh_alpha(a, c) < 90 else "RHL2"
            check_all(hexagonal(a, c), "R-3m", v)
    assert rh_alpha(4, 3) > 90
    check_all(hexagonal(4, 3), "R-3c", "RHL2")


def test_rhombohedral_axes():
    for al in list(RNG.uniform(20, 85, 4)) + list(RNG.uniform(95, 115, 4)):
        cell = dict(a=5, b=5, c=5, alpha=al, beta=al, gamma=al)
        v = "RHL1" if al < 90 else "RHL2"
        check_all(cell, "R-3m:R", v)
        assert special_points(cell)["variant"] == v      # metric alone -> hR, primitive


def test_hexagonal_and_rhombohedral_settings_agree():
    """Same lattice in hexagonal and rhombohedral axes: the same Cartesian
    point lengths."""
    a, c = 4.14, 28.6
    hx = hexagonal(a, c)
    ar = np.sqrt(a * a / 3 + c * c / 9)
    al = rh_alpha(a, c)
    rh = dict(a=ar, b=ar, c=ar, alpha=al, beta=al, gamma=al)
    p1 = special_points(hx, "R-3m")["points"]
    p2 = special_points(rh, "R-3m:R")["points"]
    assert set(p1) == set(p2)
    for lab in p1:
        assert length(hx, p1[lab]) / a == pytest.approx(length(rh, p2[lab]) / ar, rel=1e-9)


# --- 3. Bravais lattices -----------------------------------------------------------------------

@pytest.mark.parametrize("sg, lat", [
    ("Fd-3m", "cF"), ("Im-3m", "cI"), ("Pm-3m", "cP"), ("I4/mmm", "tI"), ("P4/mmm", "tP"),
    ("P6_3/mmc", "hP"), ("R-3m", "hR"), ("R-3m:R", "hR"), ("Cmcm", "oS"), ("Fddd", "oF"),
    ("Immm", "oI"), ("Pnma", "oP"), ("P2_1/c", "mP"), ("C2/c", "mS"), ("P-1", "aP"), ("P-3m1", "hP"),
    (166, "hR"), (225, "cF")])
def test_bravais_from_space_group(sg, lat):
    assert bravais_lattice(cubic(4), sg) == lat


def test_bravais_from_metric():
    assert bravais_lattice(cubic(4)) == "cP"
    assert bravais_lattice(tetragonal(4, 6)) == "tP"
    assert bravais_lattice(dict(a=3, b=4, c=5, alpha=90, beta=90, gamma=90)) == "oP"
    assert bravais_lattice(hexagonal(3, 5)) == "hP"
    assert bravais_lattice(dict(a=5, b=5, c=5, alpha=70, beta=70, gamma=70)) == "hR"
    assert bravais_lattice(dict(a=3, b=4, c=5, alpha=90, beta=100, gamma=90)) == "mP"
    assert bravais_lattice(dict(a=3, b=4, c=5, alpha=80, beta=100, gamma=95)) == "aP"


# --- 4. an unexpected setting is not returned wrong ---------------------------------------------

def test_hexagonal_gamma_60_is_guarded():
    cell = hexagonal(4, 6, gamma=60)
    res = special_points(cell, "P6/mmm")
    for lab, hkl in res["points"].items():
        if lab != "Γ":
            assert boundary_nodes(cell, hkl, "P")[0], lab
    # (1/3, 1/3, 0) is not the K corner with gamma = 60: dropped, and said so
    # K = (1/3, 1/3, 0) is not on the boundary with gamma = 60; H = (1/3, 1/3, 1/2)
    # is (on the top face) but not at its corner: both dropped, and said so
    assert "K" not in res["points"] and "H" not in res["points"]
    assert "K" in res["note"] and "H" in res["note"]
    assert set(res["points"]) == {"Γ", "A", "L", "M"}       # these are equivalent
    assert all("K" not in seg and "H" not in seg for seg in res["path"])
    assert all(len(seg) >= 2 for seg in res["path"])


def test_a_wrong_point_is_dropped():
    """The guard itself: a point inside the zone and one outside both fail."""
    c = cubic(4)
    assert not boundary_nodes(c, (0.2, 0, 0), "P")[0]
    assert not boundary_nodes(c, (0.7, 0, 0), "P")[0]
    assert boundary_nodes(c, (0, 0, 1), "I")[0]            # H of bcc
    assert not boundary_nodes(c, (0, 0, 1), "P")[0]        # a node of the P lattice
    assert not boundary_nodes(c, (0, 0, 0.9), "I")[0]
    # the BCT1 table as sometimes quoted (M = (1/2, 1/2, -1/2) -> (0, 0, 1)) is outside for c < a
    assert not boundary_nodes(tetragonal(4, 3), (0, 0, 1), "I")[0]


# --- 5. unsupported ------------------------------------------------------------------------------

@pytest.mark.parametrize("cell, sg", [
    (dict(a=3, b=5, c=7, alpha=90, beta=90, gamma=90), "Cmcm"),
    (dict(a=3, b=5, c=7, alpha=90, beta=90, gamma=90), "Fddd"),
    (dict(a=3, b=5, c=7, alpha=90, beta=90, gamma=90), "Immm"),
    (dict(a=3, b=5, c=7, alpha=90, beta=105, gamma=90), "P2_1/c"),
    (dict(a=3, b=5, c=7, alpha=90, beta=105, gamma=90), "C2/c"),
    (dict(a=3, b=5, c=7, alpha=80, beta=105, gamma=95), "P-1"),
    (dict(a=3, b=5, c=7, alpha=80, beta=105, gamma=95), None),
])
def test_unsupported(cell, sg):
    res = special_points(cell, sg)
    assert res["points"] == {"Γ": (0.0, 0.0, 0.0)}
    assert res["path"] == []
    assert "no unambiguous standard path" in res["note"]


def test_corner_counts_are_the_same_across_a_variant():
    """The minimum node counts the guard uses come from one generic cell per
    variant; random cells of the same variant give exactly those counts."""
    from hklforge.crystal import bz
    cells = []
    for _ in range(6):
        a = RNG.uniform(2, 8)
        cells += [(cubic(a), "Pm-3m"), (cubic(a), "Fm-3m"), (cubic(a), "Im-3m"),
                  (tetragonal(a, a * RNG.uniform(1.05, 3)), "P4/mmm"),
                  (tetragonal(a, a * RNG.uniform(0.3, 0.95)), "I4/mmm"),
                  (tetragonal(a, a * RNG.uniform(1.05, 3)), "I4/mmm"),
                  (dict(a=a, b=a * RNG.uniform(1.05, 2), c=a * RNG.uniform(2.1, 4),
                        alpha=90, beta=90, gamma=90), "Pmmm"),
                  (hexagonal(a, RNG.uniform(2, 12)), "P6/mmm")]
        for al in (RNG.uniform(25, 85), RNG.uniform(95, 115)):
            cells.append((dict(a=a, b=a, c=a, alpha=al, beta=al, gamma=al), "R-3m:R"))
    n = 0
    for cell, sg in cells:
        lat = bravais_lattice(cell, sg)
        v, M, cen, par, _ = bz._variant(lat, cell, sg.endswith(":R"))
        found, _ = bz._evaluate(cell, v, M, cen, par)
        assert {lab: k for lab, (_, ok, k) in found.items()} == bz._reference_counts(v), (v, cell)
        n += 1
    assert n == len(cells) == 60
