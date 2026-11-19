"""The photon energy as one more unknown, within limits (mode["energy"] = (lo, hi)).

As in test_solve_generic: random geometries, random true angles and a random
true energy; the mode holds circles and constraints at their true values and
the energy is free in a window around the truth. The truth must be among the
solutions (completeness), and every solution must give hkl and the
constraints at its own energy through the forward model (correctness).
Then two cross-checks that do not use the new code paths: each free-energy
solution must also come out of the fixed-energy solver at its energy with
one held circle released, and for the one-detector-circle case a brute-force
scan of the sample circle through the forward model must find as many
solutions.
"""
import numpy as np
import pytest

from hklforge import Degenerate, Diffractometer, Geometry, ModeError, wavevector
from hklforge.geometry import HC_EV_ANGSTROM
from hklforge.lattice import b_matrix
from hklforge.rotation import solve_trig_linear, solve_trig_poly, unit

from test_solve_generic import ANGLES, angle_diff, close, random_geometry

rng = np.random.default_rng(11)

# (free detector circles, free sample circles, constraints): one per free-energy case
CASES = [
    # detector held: k' fixed, energy from 2theta
    (0, 2, ()), (0, 3, ("betain",)), (0, 3, ("betaout",)), (0, 3, ("psi",)), (0, 3, ("h2",)),
    (0, 3, ("naz",)),
    # one detector circle, one sample circle
    (1, 1, ()),
    # sample first, every (x constraints, naz) combination
    (2, 0, ()), (2, 1, ("betain",)), (2, 1, ("betaout",)), (2, 1, ("beta",)), (2, 1, ("psi",)),
    (2, 1, ("h2",)), (2, 1, ("naz",)),
    (2, 2, ("betain", "h2")), (2, 2, ("psi", "betaout")), (2, 2, ("alpha", "beta")),
    (2, 2, ("betain", "naz")), (2, 2, ("h2", "naz")),
    (2, 3, ("betain", "h2", "naz")), (2, 3, ("psi", "alpha", "naz")),
]


def trial(nD, nS, extras, omega=False):
    if omega:
        geom = random_geometry(4, int(rng.integers(max(nD, 1), 3)), bisect_circle="s1")
        sample_free = ["s0"] + list(rng.choice(["s1", "s2", "s3"], 2, replace=False))
    else:
        geom = random_geometry(int(rng.integers(max(nS, 1), 5)), int(rng.integers(max(nD, 1), 3)))
        sample_free = list(rng.choice(geom.sample_names, nS, replace=False))
    ub = rng.normal(size=(3, 3)) * 1.5
    energy = rng.uniform(4000, 20000)
    k = wavevector(energy)
    d = Diffractometer(geom, ub, energy=9000.0, surface=rng.normal(size=3),   # 9 keV: not used
                       reference=rng.normal(size=3))
    truth = {n: float(rng.uniform(-180, 180)) for n in geom.names}
    free = sample_free + list(rng.choice(geom.detector_names, nD, replace=False))
    mode = {n: truth[n] for n in geom.names if n not in free}
    truth["energy"] = energy
    R = geom.sample_matrix(truth)
    hkl = np.linalg.solve(ub, geom.q_phi(truth, k))
    for extra in extras:
        if extra == "h2":
            mode["h2"] = np.linalg.solve(ub, R.T @ (k * (unit(rng.normal(size=3)) - np.asarray(geom.beam))))
        else:
            mode[extra] = d.pseudo(truth)[extra]
    mode["energy"] = (energy - rng.uniform(1, 3000), energy + rng.uniform(1, 3000))
    try:
        sols = d.solve(hkl, mode)
    except Degenerate:
        return None
    assert any(close(s, truth, 1e-6) for s in sols), (mode, truth, sols)
    for s in sols:
        lo, hi = mode["energy"]
        assert lo - 1e-6 <= s["energy"] <= hi + 1e-6
        ks = wavevector(s["energy"])
        assert np.allclose(d.hkl(s), hkl, atol=1e-8)
        if "h2" in extras:
            Q = geom.sample_matrix(s) @ ub @ mode["h2"]
            assert abs(np.linalg.norm(ks * np.asarray(geom.beam) + Q) - ks) < 1e-8 * ks
        pa = d.pseudo(s)
        for extra in extras:
            if extra in ANGLES:
                assert angle_diff(pa[extra], mode[extra]) < 1e-6, (extra, pa[extra], mode[extra])
    return sols, d, mode, hkl


@pytest.mark.parametrize("nD,nS,extras", CASES, ids=lambda c: "-".join(c) if isinstance(c, tuple) else str(c))
def test_free_energy_recovers_truth(nD, nS, extras):
    counts = [trial(nD, nS, extras) for _ in range(40)]
    assert sum(c is not None for c in counts) >= 38       # Degenerate only by bad luck


def test_free_energy_omega_detector_held():
    counts = [trial(0, 3, ("omega",), omega=True) for _ in range(40)]
    assert sum(c is not None for c in counts) >= 38


def test_free_energy_matches_fixed_energy_solver():
    """Each free-energy solution, at its own energy and with one held sample
    circle released, is a solution of the (already verified) fixed-energy
    planner. Covers the sample-first combinations and one detector circle."""
    checked = 0
    for nD, nS, extras in [c for c in CASES if c[0] >= 1]:
        for _ in range(15):
            res = trial(nD, nS, extras)
            if res is None:
                continue
            sols, d, mode, hkl = res
            release = next((n for n in d.geometry.sample_names if n in mode), None)
            if release is None:
                continue
            fixed_mode = {n: v for n, v in mode.items() if n not in ("energy", release)}
            for s in sols:
                d2 = Diffractometer(d.geometry, d.ub, energy=s["energy"], surface=d.surface,
                                    reference=d.reference)
                try:
                    ref = d2.solve(hkl, fixed_mode)
                except (Degenerate, ModeError):
                    continue
                sa = {n: v for n, v in s.items() if n != "energy"}
                assert any(close(sa, r, 1e-5) for r in ref), (fixed_mode, sa, ref)
                checked += 1
    assert checked > 200


def test_one_detector_circle_against_scan():
    """e_q1 (degree-2 trigonometric polynomial) against a brute-force scan:
    step the free sample circle, get Q-hat from the forward model, the
    energy from Bragg and k' as the mirror image of the beam; count where
    k' crosses the detector circle's cone, inside the energy window."""
    g = Geometry.builtin("psic")
    d = Diffractometer(g, b_matrix(5.431, 5.431, 5.431), energy=9000)
    b = np.asarray(g.beam)
    delta_axis = np.asarray(g.circle("delta").axis)
    nu = g.circle("nu")
    found = scanned = degenerate = 0
    for hkl in [(1, 1, 1), (2, 2, 0), (1, 1, 3), (0, 0, 4), (3, 1, 1), (2, 2, 4)]:
        modes = [{"mu": 0, "chi": 10, "phi": 20, "nu": 0}, {"mu": 0, "chi": 90, "phi": 0, "nu": 0}]
        modes += [{n: float(v) for n, v in zip(("mu", "chi", "phi", "nu"), rng.uniform(-60, 60, 4))}
                  for _ in range(6)]
        for mode in modes:
            window = (1000, 60000)
            try:
                sols = d.solve(hkl, mode | {"energy": window})
            except Degenerate:
                sols = None
            q = d.ub @ np.asarray(hkl, float)
            etas = np.linspace(-180, 180, 360001)
            cone_axis = nu.matrix(mode["nu"]) @ delta_axis        # delta's lab axis
            cone = delta_axis @ b                                 # k' = rot(nu) rot(delta) b
            # sample_matrix = M_mu rot(eta) M_chi M_phi, vectorised over eta
            inner = g.sample_matrix(mode | {"eta": 0.0, "mu": 0.0}) @ q   # rot(eta)=I at 0
            outer = g.circle("mu").matrix(mode["mu"])
            a = np.asarray(g.circle("eta").axis)
            t = np.radians(etas)[:, None]
            rotated = (np.cos(t) * inner + np.sin(t) * np.cross(a, inner)
                       + (1 - np.cos(t)) * (a @ inner) * a)       # Rodrigues
            Qh = rotated @ outer.T
            Qh /= np.linalg.norm(Qh, axis=1)[:, None]
            bQ = Qh @ b
            kout = b - 2 * bQ[:, None] * Qh
            vals = kout @ cone_axis - cone
            with np.errstate(divide="ignore"):
                e = np.where(bQ < 0, HC_EV_ANGSTROM * (-np.linalg.norm(q) / (2 * bQ)) / (2 * np.pi), -1)
            ok = (window[0] <= e) & (e <= window[1])
            if sols is None:      # a continuum: k' on the cone for every eta (eta, delta coaxial)
                assert np.abs(vals).max() < 1e-12 and ok.any(), (hkl, mode)
                degenerate += 1
                continue
            crossings = int(np.sum((np.sign(vals[:-1]) != np.sign(vals[1:])) & ok[:-1] & ok[1:]))
            assert crossings == len(sols), (hkl, mode, crossings, sols)
            found += len(sols)
            scanned += 1
    assert found > 20 and scanned > 40 and degenerate < 6, (found, scanned, degenerate)


def test_solve_trig_poly_products():
    """Roots of a product of two linear trig equations = union of each one's."""
    for _ in range(300):
        a1, b1, c1, a2, b2, c2 = rng.normal(size=6)
        f = lambda t: (a1 * np.cos(t) + b1 * np.sin(t) - c1) * (a2 * np.cos(t) + b2 * np.sin(t) - c2)  # noqa: E731
        want = solve_trig_linear(a1, b1, c1) + solve_trig_linear(a2, b2, c2)
        got = solve_trig_poly(f, 2)
        for t in want:
            assert min(abs((t - u + np.pi) % (2 * np.pi) - np.pi) for u in got) < 1e-6
        for u in got:
            assert abs(f(u)) < 1e-8 * max(1, abs(a1) + abs(b1) + abs(c1)) * max(1, abs(a2) + abs(b2) + abs(c2))
    with pytest.raises(Degenerate):
        solve_trig_poly(lambda t: 0.0, 2)


# --- psic examples and the window ---------------------------------------------

def psic():
    return Diffractometer(Geometry.builtin("psic"), b_matrix(5.431, 5.431, 5.431), energy=9000,
                          surface_hkl=(0, 0, 1), reference_hkl=(0, 0, 1))


def test_window_limits_the_energy():
    d = psic()
    truth = {"mu": 3, "eta": 20, "chi": 10, "phi": 20, "nu": 5, "delta": 35}
    hkl = np.linalg.solve(d.ub, d.geometry.q_phi(truth, wavevector(11000)))
    mode = {"mu": 3, "chi": 10, "phi": 20, "nu": 5}
    wide = d.solve(hkl, mode | {"energy": (None, None)})
    assert any(abs(s["energy"] - 11000) < 1e-6 for s in wide)
    assert d.solve(hkl, mode | {"energy": (11500, 20000)}) == [
        s for s in wide if 11500 <= s["energy"] <= 20000]
    narrow = d.solve(hkl, mode | {"energy": (10999, 11001)})
    assert narrow and all(abs(s["energy"] - 11000) < 1e-6 for s in narrow)


def test_positions_carry_their_energy():
    d = psic()
    sols = d.solve((1, 1, 1), {"eta": 10, "chi": 30, "phi": 45, "mu": 0, "energy": (1000, 50000)})
    assert sols
    for s in sols:
        assert np.allclose(d.hkl(s), (1, 1, 1), atol=1e-9)
        assert abs(d.spots(s, [(1, 1, 1)])["off_axis"][0]) < 1e-6     # detector on the reflection
    # fixed-energy modes are unchanged: no energy key
    assert all("energy" not in s for s in d.solve((1, 1, 1), {"eta": 10, "chi": 30, "phi": 45}))


@pytest.mark.parametrize("mode,msg", [
    ({"eta": 0, "chi": 0, "phi": 0, "energy": (1, 2e4)}, "already fixes the position"),
    ({"chi": 0, "phi": 0, "mu": 0, "qaz": 90, "energy": (1, 2e4)}, "qaz with a free energy"),
    ({"eta": 0, "chi": 0, "phi": 0, "energy": 9000}, "limits"),
    ({"eta": 0, "chi": 0, "phi": 0, "energy": (2e4, 1e4)}, "lo <= hi"),
    ({"eta": 0, "chi": 0, "phi": 0, "mu": 0, "nu": 0, "energy": (1, 2e4)}, "overdetermined"),
    ({"nu": 0, "delta": 30, "chi": 10, "omega": 1, "energy": (1, 2e4)}, "outside it"),
    ({"eta": 0, "chi": 0, "phi": 0, "mu": 0}, "let the energy vary"),
    ({"nu": 0, "delta": 0, "eta": 0}, "let the energy vary"),
])
def test_free_energy_modes_are_explained(mode, msg):
    with pytest.raises(ModeError, match=msg):
        psic().plan(mode)


def test_describe_free_energy():
    text = psic().plan({"mu": 0, "chi": 10, "phi": 20, "nu": 0, "energy": (5000, 20000)}).describe()
    assert "[e_q1]" in text and "energy free in [5000, 20000] eV" in text
