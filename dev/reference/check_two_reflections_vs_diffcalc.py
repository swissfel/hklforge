import sys; sys.path.insert(0, ".")
import itertools, time, numpy as np
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position, get_rotation_matrices
from diffcalc.util import x_rotation, y_rotation, z_rotation
from bernina_prep import diffraction as D0, two_reflections as T
class _NS: pass
D = _NS(); D.__dict__.update({**vars(D0), **vars(T)})

# vectorised rotations == diffcalc's
for ax, f in (("x", x_rotation), ("y", y_rotation), ("z", z_rotation)):
    assert np.allclose(D._rot(ax, 0.7), f(0.7)), ax
p = Position(mu=12, delta=33, nu=-21, eta=40, chi=-70, phi=111)
MU, DE, NU, ET, CH, PH = get_rotation_matrices(p)
assert np.allclose(D._sample_rotation(12, 40, -70, 111), MU @ ET @ CH @ PH)
u = (NU @ DE @ np.array([[0], [1], [0]])).ravel()
assert np.allclose(D._detector_angles(u), (33, -21))
print("matrix conventions match diffcalc")

def check(ub, h1, h2, E, fixed, label):
    t = time.time(); df = D.solve_two_reflections(ub, h1, h2, E, fixed); dt = time.time() - t
    calc = HklCalculation(ub, Constraints({}))
    lam = D.en2lam(E); worst = 0; wv = 0
    for c in df.columns:
        s = df[c]
        for i, h in ((1, h1), (2, h2)):
            pos = Position(mu=s.mu, delta=s[f"delta {i}"], nu=s[f"gamma {i}"], eta=s.eta, chi=s.chi, phi=s.phi)
            worst = max(worst, np.abs(np.array(calc.get_hkl(pos, lam)) - h).max())
            va = calc.get_virtual_angles(pos)
            wv = max(wv, abs(va["betain"] - s.betain), abs(va["betaout"] - s[f"betaout {i}"]))
    print(f"{label}: {len(df.columns)} cols in {dt*1e3:.0f} ms, max hkl err {worst:.1e}, max betain/out err {wv:.1e}")
    return df

ub = D.build_ubcalc("x", 4.13, 4.13, 28.6, 90, 90, 120, ["(0,0,1) > (-1,0,0)", "(1,0,0) > (0,0,1)"], "(0,0,1)", frame="swissfel")
df = check(ub, (2, 0, 14), (0, 2, 14), 5830, {"eta": 0, "chi": 0}, "hex (2,0,14)+(0,2,14), eta/chi fixed")
import pandas as pd
with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200): print(df.iloc[:, :4])
si = D.build_ubcalc("si", 5.43, 5.43, 5.43, 90, 90, 90, ["(1,1,1) > (1,0,0)", "(1,1,-1) > (0,1,0)"], "(1,1,1)", frame="swissfel")
for pair in itertools.combinations(["mu", "eta", "chi", "phi"], 2):
    try: check(si, (1, 1, 1), (2, 2, 0), 9000, {n: 0 for n in pair}, f"Si 111+220, {pair} fixed")
    except ValueError as e: print(pair, "->", e)
for h2 in [(2, 2, 2), (1, 1, -1)]:
    try: check(si, (1, 1, 1), h2, 9000, {"eta": 0, "chi": 0}, f"Si 111+{h2}")
    except ValueError as e: print(h2, "->", e)
