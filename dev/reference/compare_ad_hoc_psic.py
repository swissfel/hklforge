import sys, time, warnings; sys.path.insert(0, "/home/lemke_h/mypy/bernina-scripts")
warnings.simplefilter("ignore")
import numpy as np
import ad_hoc_diffractometer as ahd
from ad_hoc_diffractometer import ConstraintSet, SampleConstraint, DetectorConstraint, ReferenceConstraint
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from bernina_prep import diffraction as D, two_reflections as T
wrap = lambda a: (np.asarray(a) + 180) % 360 - 180
NAMES = ("mu", "eta", "chi", "phi", "nu", "delta")

def ahd_for(ubd, E):
    g = ahd.make_geometry("psic")
    g.wavelength = D.en2lam(E)
    ahd.ub_identity(g.sample)
    g.sample.UB = np.array(ubd.UB, float)
    for n in NAMES:
        g.stage(n).limits = (-400.0, 400.0)
    return g

lat = (4.13, 4.13, 28.6, 90, 90, 120); E = 5830
ubd = D.build_ubcalc("x", *lat, ["(0,0,1) > (-1,0,0)", "(1,0,0) > (0,0,1)"], "(0,0,1)", frame="swissfel")
g = ahd_for(ubd, E); calc = HklCalculation(ubd, Constraints({})); lam = D.en2lam(E)
print("type:", type(g).__name__, "modes:", len(g.modes))

rng = np.random.default_rng(5); worst = 0
for _ in range(200):
    a = dict(zip(NAMES, rng.uniform(-120, 120, 6)))
    hd = np.array(calc.get_hkl(Position(**a), lam)); ha = np.array(g.inverse(a)); worst = max(worst, np.abs(hd - ha).max())
print(f"1) angles->hkl, 200 random settings: max |ahd - diffcalc| = {worst:.1e}")

def compare(mode_cs, cons, hkls, label):
    g.modes[label] = mode_cs; g.mode_name = label
    for hkl in hkls:
        t = time.time(); sa = g.forward(*hkl); ta = time.time() - t
        t = time.time()
        try:
            dfd = D.solve_hkl(ubd, D.make_constraints(cons), *hkl, E)
            dset = [np.array([dfd[c][n if n != "nu" else "gamma"] for n in NAMES]) for c in dfd.columns]
        except Exception:
            dset = []
        td = time.time() - t
        aset = [np.array([s[n] for n in NAMES]) for s in sa]
        in_d = sum(any(np.abs(wrap(x - d)).max() < 1e-5 for d in dset) for x in aset)
        in_a = sum(any(np.abs(wrap(x - d)).max() < 1e-5 for x in aset) for d in dset)
        print(f"   {hkl}: diffcalc {len(dset)} ({td*1e3:.0f} ms), ad_hoc {len(aset)} ({ta*1e3:.0f} ms); ad_hoc in diffcalc {in_d}/{len(aset)}, diffcalc in ad_hoc {in_a}/{len(dset)}")

print("2) custom mode eta, chi, phi fixed (Bernina default) -> mu, nu, delta")
compare(ConstraintSet([SampleConstraint("eta", 0.0), SampleConstraint("chi", 0.0), SampleConstraint("phi", 0.0)], computed=["mu", "nu", "delta"]),
        dict(eta=0, chi=0, phi=0), [(2, 0, 14), (1, 0, 10), (0, 1, 12)], "bernina_etachiphi")
print("   same, eta=10 chi=5 phi=-40")
compare(ConstraintSet([SampleConstraint("eta", 10.0), SampleConstraint("chi", 5.0), SampleConstraint("phi", -40.0)], computed=["mu", "nu", "delta"]),
        dict(eta=10, chi=5, phi=-40), [(2, 0, 14), (0, 1, 12)], "bernina2")
print("3) mu, eta, chi fixed (GIC-like) -> phi, nu, delta")
compare(ConstraintSet([SampleConstraint("mu", 2.0), SampleConstraint("eta", 0.0), SampleConstraint("chi", 0.0)], computed=["phi", "nu", "delta"]),
        dict(mu=2, eta=0, chi=0), [(2, 0, 14), (1, 1, 10), (1, 0, 3)], "gic")
print("4) surface: mu=0, nu=0, incidence 5 deg  vs diffcalc mu=0, nu=0, betain=5")
g.surface_normal = (0, 0, 1)
compare(ConstraintSet([SampleConstraint("mu", 0.0), DetectorConstraint("nu", 0.0), ReferenceConstraint("incidence", 5.0)], computed=["eta", "chi", "phi", "delta"],
        extras={"n_hat": ahd.REQUIRED}), dict(mu=0, nu=0, betain=5), [(1, 0, 10), (2, 0, 14)], "inc5")
