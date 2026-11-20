"""hklforge.view: the page and its messages (no browser).

That the page computes the same hkl and virtual angles as hklforge was
checked in Chrome (dev/check_viewer.py); here only what Python controls.
"""
import html
import json
import re

import numpy as np

from hklforge import Diffractometer, Geometry
from hklforge.crystal.bz import special_points
from hklforge.lattice import b_matrix
from hklforge.view import (CONFIG_MARK, VIEWER_HTML, geometry_message, iframe, message, page,
                           payload_html)


def test_default_geometry_in_page_is_psic():
    """The page's built-in default must stay what Python would send for psic."""
    text = VIEWER_HTML.read_text(encoding="utf-8")
    m = re.search(r"window\.HKLFORGE_VIEW = Object\.assign\((\{.*?\}),\s*\n", text, re.S)
    assert m, "default configuration not found"
    default = json.loads(m.group(1))
    assert default["geometry"] == json.loads(json.dumps(geometry_message(Geometry.builtin("psic"))))
    assert default["id"] == "diff3d"


def test_page_offline_and_configured():
    text = VIEWER_HTML.read_text(encoding="utf-8")
    assert text.count(CONFIG_MARK) == 1
    assert not re.search(r"<script[^>]+src=", text), "the page must not load scripts from anywhere"
    assert not re.search(r"<link[^>]+href=\"http", text)
    g = Geometry.builtin("k6c")
    p = page(g, view_id="geo3d", storage="test-prefix", detectors=[["a", 10, 20]], title="T")
    assert CONFIG_MARK not in p
    cfg = json.loads(re.search(r"<script>window\.HKLFORGE_VIEW = (\{.*?\});</script>", p).group(1))
    assert cfg["id"] == "geo3d" and cfg["storage"] == "test-prefix" and cfg["title"] == "T"
    assert cfg["detectors"] == [["a", 10.0, 20.0]]
    assert [c["name"] for c in cfg["geometry"]["sample"]] == g.sample_names


def test_geometry_message():
    for name in Geometry.builtins():
        g = Geometry.builtin(name)
        m = geometry_message(g)
        assert np.allclose(m["up"], g.up_direction())
        r, u = g.azimuth_frame()
        assert np.allclose(m["azimuth"]["r"], r) and np.allclose(m["azimuth"]["u"], u)
        assert abs(np.dot(m["up"], m["beam"])) < 1e-12
        json.dumps(m)                                    # plain data


def test_message_and_payload():
    g = Geometry.builtin("psic")
    d = Diffractometer(g, b_matrix(5.431, 5.431, 5.431), energy=9000, surface_hkl=(0, 0, 1))
    sp = special_points({"a": 5.431, "b": 5.431, "c": 5.431, "alpha": 90, "beta": 90, "gamma": 90}, "Fd-3m")
    msg = message(g, {"mu": 1, "eta": 2}, ub=d.ub, energy=9000, surface=d.surface, label="x <y>",
                  targets=[(1, 1, 1)], special_points=sp, n=3, ghosts=[])
    assert abs(msg["kMag"] - d.k) < 1e-12
    assert msg["angles"] == {"mu": 1.0, "eta": 2.0} and msg["ghosts"] == []
    assert msg["bz"]["variant"] == "FCC" and "X" in msg["bz"]["points"]
    assert "refPhi" not in msg                           # only sent when given
    text = payload_html(msg, "geo3d")
    assert text.startswith('<span id="geo3d-payload"')
    inner = re.search(r">(.*)</span>", text).group(1)
    assert json.loads(html.unescape(inner)) == json.loads(json.dumps(msg))
    fr = iframe(page(), 500)
    assert "srcdoc=" in fr and "<script>" not in fr      # escaped into the attribute


def test_geometry_yaml_round_trip():
    from hklforge.view.explorer import geometry_yaml, steps
    for name in Geometry.builtins():
        g = Geometry.builtin(name)
        g2 = Geometry.from_yaml(geometry_yaml(g))
        assert g2.names == g.names and g2.bisect_circle == g.bisect_circle
        assert np.allclose([c.axis for c in g2.sample + g2.detector], [c.axis for c in g.sample + g.detector])
        assert np.allclose(g2.up_direction(), g.up_direction()) and np.allclose(g2.beam, g.beam)
        st = steps(g)
        assert len(st) == len(g.names) + 2                 # frame, each circle, whole
        assert [s["focus"] for s in st[1:-1]] == g.names


def test_explorer_widget():
    import pytest
    pytest.importorskip("ipywidgets")
    from hklforge.view.explorer import GeometryExplorer
    e = GeometryExplorer()
    assert e.pick.value == "psic" and e.step.value == e.step.max
    e.pick.value = "k6c"
    e.step.value = 3
    assert "kappa" in e.title.value
    msg = json.loads(html.unescape(re.search(r">(.*)</span>", e.holder.value).group(1)))
    assert [c["name"] for c in msg["geometry"]["sample"]] == ["mu", "komega", "kappa"]
    assert msg["focus"] == "kappa" and msg["geometry"]["detector"] == []
    e.yaml.value = "name: broken"
    e._on_yaml()
    assert "beam" in e.yaml_msg.value                       # an error shown, not raised


def test_structure_factor_nodes():
    """The spheres' data: every allowed node within d >= lambda/2 (all
    equivalents), |F|^2 as the reflection list computes it, nearest first."""
    import pytest
    pytest.importorskip("xraydb")
    pytest.importorskip("pandas")
    from pathlib import Path
    from hklforge.crystal import reflections
    from hklforge.crystal.cell import d_spacing, wavelength
    from hklforge.crystal.structure import read_cif
    from hklforge.view import structure_factor_nodes
    st = read_cif((Path(__file__).parent / "data" / "GaAs.cif").read_text())
    energy = 10367.0                                   # at the Ga K edge: f', f'' matter
    sf = structure_factor_nodes(st, energy)
    H = np.array(sf["hkl"])
    f2 = np.array(sf["f2"])
    assert sf["total"] == len(H) and sf["dMin"] == pytest.approx(wavelength(energy) / 2)
    d = d_spacing(st.cell, H)
    assert np.all(d >= sf["dMin"] * (1 - 1e-9)) and np.all(np.diff(d) <= 1e-9)   # nearest first
    assert np.all(st.space_group.allowed_mask(H))
    assert {(1, 1, 1), (-1, -1, -1), (1, -1, 1), (2, 0, 0), (0, 0, -2)} <= set(map(tuple, H))
    assert (1, 0, 0) not in set(map(tuple, H))         # F-centred: forbidden, no sphere
    # anomalous terms: Friedel pairs differ near the edge
    i, j = (sf["hkl"].index(h) for h in ([1, 1, 1], [-1, -1, -1]))
    assert abs(f2[i] - f2[j]) > 1e-3 * f2[i]
    # the same |F|^2 as the reflection list (one per set of equivalents)
    lst = reflections.reflection_list(st.cell, energy, structure=st, unique=False)
    ref = {(int(r.h), int(r.k), int(r.l)): r["|F|²"] for _, r in lst.iterrows()}
    common = [k for k in ref if k in set(map(tuple, H))]
    assert len(common) > 50
    for k in common:
        assert f2[sf["hkl"].index(list(k))] == pytest.approx(ref[k], rel=1e-9)
    # every reachable node is there: none with d >= lambda/2 missing
    assert set(ref) <= set(map(tuple, H)) | {k for k in ref if ref[k] < 1e-9}
    # the cap keeps the nearest, counts the rest
    small = structure_factor_nodes(st, energy, max_nodes=10)
    assert len(small["hkl"]) == 10 and small["total"] == sf["total"]
    assert small["hkl"] == sf["hkl"][:10]
    msg = message(structure_factors=small)
    assert json.loads(json.dumps(msg))["structureFactors"]["hkl"] == small["hkl"]
    with pytest.raises(ValueError):
        structure_factor_nodes(st)


def test_examples_and_structure_atoms():
    """The example CIFs load, and the atoms the page draws are every atom of
    one cell with its element's radius and colour."""
    from hklforge.crystal import examples
    from hklforge.view import element_style, structure_atoms
    assert examples.names() == ["Bi2Se3", "GaAs", "Si", "SrTiO3", "TiO2_rutile"]
    per_cell = {"Bi2Se3": 15, "GaAs": 8, "Si": 8, "SrTiO3": 5, "TiO2_rutile": 6}
    for name in examples.names():
        st = examples.load(name)
        assert examples.title(name) and "data_" in examples.cif_text(name)
        sa = structure_atoms(st)
        assert len(sa["atoms"]) == per_cell[name]
        assert json.loads(json.dumps(sa)) == sa                       # JSON-ready
        for label, el, x, y, z, b, occ in sa["atoms"]:
            assert 0 <= x < 1 and 0 <= y < 1 and 0 <= z < 1 and b >= 0 and 0 < occ <= 1
            assert sa["elements"][el] == list(element_style(el))
    import pytest
    with pytest.raises(ValueError):
        examples.cif_text("nope")
    # Cordero covalent radii and Jmol colours (the published values)
    assert element_style("Si") == (1.11, "#f0c8a0") and element_style("O") == (0.66, "#ff0d0d")
    assert element_style("Ti")[0] == 1.6 and element_style("Xx") == (1.5, "#909090")
    assert message(structure=structure_atoms(examples.load("Si")))["structure"]["name"] == "Si"
