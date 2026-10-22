import sys; sys.path.insert(0, "/home/lemke_h/mypy/bernina-scripts")
import math, numpy as np
import gi; gi.require_version("Hkl", "5.0"); from gi.repository import Hkl
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from bernina_prep import diffraction as D
from bernina_prep import two_reflections as T
U = Hkl.UnitEnum.USER
P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)     # diffcalc vec -> libhkl lab (x beam)

def libhkl_for(ub_d, lattice, E):
    f = Hkl.factories()["E6C"]; g = f.create_new_geometry(); el = f.create_new_engine_list()
    s = Hkl.Sample.new("s"); a, b, c, al, be, ga = lattice
    s.lattice_set(Hkl.Lattice.new(a, b, c, *map(math.radians, (al, be, ga))))
    m = s.UB_get(); m.init(*(P @ np.asarray(ub_d.UB, float)).ravel()); s.UB_set(m)
    g.wavelength_set(D.en2lam(E), U)
    el.init(g, Hkl.Detector.factory_new(Hkl.DetectorType(0)), s)
    return g, el, s

lat = (4.13, 4.13, 28.6, 90, 90, 120); E = 5830
ubd = D.build_ubcalc("x", *lat, ["(0,0,1) > (-1,0,0)", "(1,0,0) > (0,0,1)"], "(0,0,1)", frame="swissfel")
g, el, s = libhkl_for(ubd, lat, E)
ub_back = np.array([[s.UB_get().get(i, j) for j in range(3)] for i in range(3)])
print("UB taken as given:", np.allclose(ub_back, P @ np.asarray(ubd.UB, float)), " max diff", np.abs(ub_back - P @ ubd.UB).max())
calc = HklCalculation(ubd, Constraints({})); lam = D.en2lam(E)
eng = el.engine_get_by_name("hkl")
rng = np.random.default_rng(3); worst = 0
for _ in range(200):
    mu, eta, chi, phi, nu, de = rng.uniform(-120, 120, 6)
    g.axis_values_set([mu, eta, chi, phi, nu, de], U); el.get()
    hl = np.array(eng.pseudo_axis_values_get(U))
    hd = np.array(calc.get_hkl(Position(mu=mu, delta=de, nu=nu, eta=eta, chi=chi, phi=phi), lam))
    worst = max(worst, np.abs(hl - hd).max())
print(f"forward model, 200 random settings (mu,eta=omega,chi,phi,nu=gamma,delta): max |hkl_libhkl - hkl_diffcalc| = {worst:.1e}")

# inverse: Bernina 'eta, chi, phi fixed' = libhkl lifting_detector_mu
def sols_libhkl(mode, hkl, start):
    eng.current_mode_set(mode); g.axis_values_set(start, U); el.get()
    out = []
    try:
        for it in eng.pseudo_axis_values_set(list(hkl), U).items():
            out.append(np.array(it.geometry_get().axis_values_get(U)))
    except Exception as ex:
        return str(ex).split(":")[-1].strip()
    return out
for hkl, fixed in (((2, 0, 14), dict(eta=0, chi=0, phi=0)), ((1, 0, 10), dict(eta=0, chi=0, phi=30)), ((0, 1, 12), dict(eta=10, chi=5, phi=-40))):
    dfd = D.solve_hkl(ubd, D.make_constraints(fixed), *hkl, E)
    dset = [np.array([dfd[c][n] for n in ("mu", "eta", "chi", "phi", "gamma", "delta")]) for c in dfd.columns]
    lset = sols_libhkl("lifting_detector_mu", hkl, [0, fixed["eta"], fixed["chi"], fixed["phi"], 0, 0])
    if isinstance(lset, str): print(hkl, "libhkl:", lset); continue
    wrap = lambda a: (a + 180) % 360 - 180
    matched = sum(any(np.abs(wrap(l - d)).max() < 1e-4 for d in dset) for l in lset)
    print(f"{hkl} eta,chi,phi={fixed['eta']},{fixed['chi']},{fixed['phi']} fixed: diffcalc {len(dset)} solutions, libhkl {len(lset)}, libhkl ones also in diffcalc: {matched}")

# double diffraction: libhkl puts the detector on h1 and brings h2 into Bragg.
# Cross-check with the two_reflections solver: fix mu and chi at libhkl's
# values; its (eta, phi) and reflection-1 detector angles must contain libhkl's.
si_lat = (5.43, 5.43, 5.43, 90, 90, 90); E2 = 9000
ub_si = D.build_ubcalc("si", *si_lat, ["(1,1,1) > (1,0,0)", "(1,1,-1) > (0,1,0)"], "(1,1,1)", frame="swissfel")
for mode, start in (("double_diffraction_vertical", [0, 10, 20, 10, 0, 20]), ("double_diffraction_horizontal", [10, 0, 20, 10, 20, 0])):
    g, el, s = libhkl_for(ub_si, si_lat, E2); eng = el.engine_get_by_name("hkl"); calc = HklCalculation(ub_si, Constraints({}))
    eng.current_mode_set(mode)
    for name, v in zip(("h2", "k2", "l2"), (2, 2, 0)):
        p = eng.parameter_get(name); p.value_set(v, U); eng.parameter_set(name, p)
    g.axis_values_set(start, U); el.get()
    try:
        items = eng.pseudo_axis_values_set([1, 1, 1], U).items()
    except Exception as ex:
        print(mode, "libhkl error:", ex); continue
    print(f"{mode}: libhkl returned {len(items)} solution(s)")
    for it in items:
        a = np.array(it.geometry_get().axis_values_get(U)); mu, om, chi, phi, ga, de = a
        h1 = calc.get_hkl(Position(mu=mu, eta=om, chi=chi, phi=phi, nu=ga, delta=de), D.en2lam(E2))
        fx = {"mu": mu, "chi": chi} if mode.endswith("vertical") else {"eta": om, "chi": chi}
        df = T.solve_two_reflections(ub_si, (1, 1, 1), (2, 2, 0), E2, fx)
        hit = [c for c in df.columns if abs(((df[c].eta - om) + 180) % 360 - 180) < 1e-3 and abs(((df[c].mu - mu) + 180) % 360 - 180) < 1e-3 and abs(((df[c].phi - phi) + 180) % 360 - 180) < 1e-3]
        det_ok = [c for c in hit if abs(((df[c]["delta 1"] - de) + 180) % 360 - 180) < 1e-3 and abs(((df[c]["gamma 1"] - ga) + 180) % 360 - 180) < 1e-3]
        print("   libhkl", np.round(a, 3), "h1 via diffcalc", np.round(h1, 4), "| two_reflections has it:", bool(hit), " with same detector on h1:", bool(det_ok))
