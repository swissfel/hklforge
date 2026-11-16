"""The 3D viewer in Chrome against hklforge in Python (needs google-chrome and websocket-client).

Run: ``python dev/check_viewer.py <output dir>``. For several geometries the
page is hosted as a notebook hosts it (iframe srcdoc + hidden payload
element); messages from hklforge.view.message set geometry, UB, surface,
reference and angles; the page's own measured hkl and virtual angles are
compared with Diffractometer.hkl / virtual_angles. Writes screenshots."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cdp import Chrome  # noqa: E402

from hklforge import Diffractometer, Geometry  # noqa: E402
from hklforge.crystal.bz import special_points  # noqa: E402
from hklforge.lattice import b_matrix  # noqa: E402
from hklforge.view import iframe, message, page  # noqa: E402

OUT = Path(sys.argv[1])
rng = np.random.default_rng(5)


def bernina():
    g = Geometry.builtin("psic").reframed(beam="z+", up="y+", name="bernina_you",
                                          labels={"x": "transverse", "y": "vertical", "z": "beam"})
    return g


GEOMS = [Geometry.builtin(n) for n in ("psic", "e4ch", "k6c", "k4cv", "fourcv")] + [bernina()]


def wrap(a, b):
    return abs((a - b + 180) % 360 - 180)


def main():
    parent = OUT / "parent.html"
    parent.write_text("<!doctype html><html><meta charset='utf-8'><body style='margin:0'>" + iframe(page(), 900)
                      + "<span id='diff3d-payload' style='display:none'></span></body></html>")
    c = Chrome()
    report = {}
    try:
        c.goto(parent.as_uri(), 4.0)
        F = "document.querySelector('iframe').contentWindow"
        names0 = c.js(f"{F}.__diffApp.debugCounts().circleNames")
        report["standalone circles"] = names0
        n = 0
        cell = (4.1, 5.3, 6.7, 90, 95, 90)
        ub0 = b_matrix(*cell)
        for g in GEOMS:
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            U = q * np.sign(np.linalg.det(q))
            ub = U @ ub0
            surface, reference = rng.normal(size=3), rng.normal(size=3)
            d = Diffractometer(g, ub, energy=9000, surface=surface, reference=reference)
            worst_h = worst_v = 0.0
            for trial in range(6):
                angles = {nm: float(rng.uniform(-70, 70)) for nm in g.names}
                n += 1
                msg = message(g, angles, ub=ub, k=d.k, surface=surface, reference=reference, n=n,
                              label=f"{g.name} {trial}")
                c.js(f"document.getElementById('diff3d-payload').textContent = {json.dumps(json.dumps(msg))}")
                c.sleep(0.5)
                dbg = c.js(f"(() => {{ {F}.document.getElementById('show-hkl-cell').click(); "
                           f"const d = {F}.__diffApp.debugCounts(); {F}.document.getElementById('show-hkl-cell').click(); return d; }})()")
                if dbg["measuredHkl"] is None:     # needs the hkl label switched on: computed anyway
                    pass
                assert dbg["geometry"] == g.name, dbg["geometry"]
                assert dbg["circleNames"] == g.names, (dbg["circleNames"], g.names)
                h_py = d.hkl(angles)
                worst_h = max(worst_h, float(np.max(np.abs(np.array(dbg["measuredHkl"]) - h_py))))
                va = d.pseudo(angles)
                for k, v in dbg["virtual"].items():
                    if k in va and np.isfinite(va[k]) and v is not None:
                        worst_v = max(worst_v, wrap(v, va[k]))
                missing = sorted(set(k for k in va if np.isfinite(va[k]) and k != "theta") - set(dbg["virtual"]))
                assert not missing, (g.name, missing)
            report[g.name] = {"max |hkl page - python|": worst_h, "max virtual angle diff [deg]": worst_v}

        # drawing: virtual angles all on, BZ points on
        sp = special_points({"a": 5.431, "b": 5.431, "c": 5.431, "alpha": 90, "beta": 90, "gamma": 90}, "Fd-3m")
        g = Geometry.builtin("psic")
        d = Diffractometer(g, b_matrix(5.431, 5.431, 5.431), energy=9000, surface_hkl=(0, 0, 1), reference_hkl=(1, 0, 0))
        sol = d.solve((1, 1, 1), {"mu": 0, "nu": 0, "phi": 0})[0]
        n += 1
        msg = message(g, sol, ub=d.ub, k=d.k, surface=d.surface, reference=d.reference,
                      special_points=sp, targets=[(1, 1, 1)], n=n)
        c.js(f"document.getElementById('diff3d-payload').textContent = {json.dumps(json.dumps(msg))}")
        c.sleep(0.5)
        c.js(f"""(() => {{ const D = {F}.document;
            D.getElementById('show-virtual').click();
            D.querySelectorAll('#virtual-list input').forEach(cb => cb.click());
            D.getElementById('show-bz-points').click();
            D.getElementById('show-axes').click();
            D.getElementById('show-recip').click(); }})()""")
        c.sleep(0.6)
        dbg = c.js(f"{F}.__diffApp.debugCounts()")
        report["drawing"] = {"virtualObjects": dbg["virtualObjects"], "bzPoints": dbg["bzPoints"],
                             "bz info": c.js(f"{F}.document.getElementById('bz-points-info').textContent"),
                             "hkl at the solution": dbg["measuredHkl"]}
        c.screenshot(str(OUT / "viewer_psic.png"))
        # a geometry change keeps working (k6c), focus on a circle
        n += 1
        c.js(f"document.getElementById('diff3d-payload').textContent = "
             f"{json.dumps(json.dumps(message(Geometry.builtin('k6c'), {'kappa': 40, 'komega': 20}, n=n, focus='kappa')))}")
        c.sleep(0.6)
        c.screenshot(str(OUT / "viewer_k6c_focus.png"))
        report["after k6c"] = c.js(f"{F}.__diffApp.debugCounts().circleNames")
        report["slider rows"] = c.js(f"{F}.document.querySelectorAll('#sample-circles .angle-row, #detector-circles .angle-row').length")
    finally:
        report["page errors"] = c.errors
        c.close()
    print(json.dumps(report, indent=1, default=str))


main()
